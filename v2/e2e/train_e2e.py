"""End-to-end track training on Colab: raw 5-min bars + raw daily bars -> score. No factor formulas, no Ceres.

Data (from e2e/build_e2e.py + base panels) under --data:
  axes.npz, intra.npy [T,N,48,5] fp16, daily.npy [T,N,10] fp16, univ.npy, y10.npy, y5.npy, industry.npy, log_circ_mv.npy
Model: per-day intraday Transformer over 48 bars -> attention-pooled day vector (last --days_intra days);
       GRU over the last --lookback days of [daily bar embedding, intraday day vector, has-intraday flag] -> score.
Loss: -IC against the rank-gauss label (optionally industry+size neutralised, optionally top-quintile weighted),
same windows / label / output format as colab/train.py so evaluation is shared with the factor track.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.special import ndtri

H = 11
BLOCKS = [  # name, test_start, test_end, val_start, val_label_end, train_label_end  (same as colab/train.py)
    ('2024H2', 20240701, 20241231, 20230703, 20240628, 20230630),
    ('2025H1', 20250101, 20250630, 20240101, 20241231, 20231229),
    ('2025H2', 20250701, 20251231, 20240701, 20250630, 20240628),
    ('2026H1', 20260101, 20260630, 20250101, 20251231, 20241231),
    ('2022H2', 20220701, 20221231, 20210701, 20220630, 20210630),
    ('2023H1', 20230101, 20230630, 20220101, 20221231, 20211231),
    ('2023H2', 20230701, 20231231, 20220701, 20230630, 20220630),
    ('2024H1', 20240101, 20240630, 20230101, 20231231, 20221231),
    ('2026H2', 20260701, 20260731, 20250701, 20260630, 20250630),
]
t0 = time.time()
def log(*a): print(f'[{time.time()-t0:7.0f}s]', *a, flush=True)


class Data:
    def __init__(self, root, label, neutral, dev, lookback, days_intra, flow=False, intra=True):
        r = Path(root)
        ax = np.load(r / 'axes.npz', allow_pickle=True)
        self.dates = pd.DatetimeIndex(ax['dates']); self.codes = ax['codes']
        self.dint = self.dates.strftime('%Y%m%d').astype(int).to_numpy()
        self.T, self.N = len(self.dates), len(self.codes)
        self.univ = np.load(r / 'univ.npy'); self.ys = [np.load(r / f'{l}.npy') for l in label.split('+')]; self.y = self.ys[-1]
        self.ind = np.load(r / 'industry.npy'); self.lmv = np.load(r / 'log_circ_mv.npy')
        self.neutral = neutral; self.dev = dev; self.L = lookback; self.Di = days_intra
        self.lend = np.array([self.dint[t + H] if t + H < self.T else 99999999 for t in range(self.T)])
        t1 = time.time()
        self.intra = torch.from_numpy(np.load(r / 'intra.npy', mmap_mode='r')[:]).to(dev) if intra else \
            torch.zeros((self.T, 1, 48, 5), dtype=torch.float16, device=dev).expand(-1, self.N, -1, -1)       # fp16 [T,N,48,5]
        self.daily = torch.from_numpy(np.load(r / 'daily.npy')).to(dev)                           # fp16 [T,N,10]
        self.flow = torch.from_numpy(np.load(r / 'flow.npy', mmap_mode='r')[:]).to(dev) if flow else None   # fp16 [T,N,48,2]
        log('data on gpu', tuple(self.intra.shape), tuple(self.daily.shape), f'{time.time()-t1:.0f}s')
        self._cache = {}; self.train_start = 0; self.train_pool = 0
        self.idxm = np.load(r / 'index_member.npy', mmap_mode='r') if (r / 'index_member.npy').exists() else None

    def day(self, t, need_label=True):
        key = (t, need_label)
        if key in self._cache: return self._cache[key]
        m = self.univ[t].copy()
        if need_label:
            for yk in self.ys: m &= np.isfinite(yk[t])
            if self.train_pool: m &= (self.idxm[t] & self.train_pool) > 0      # training / validation restricted to index members
        idx = np.where(m)[0]; out = {'t': t, 'idx': idx}
        if need_label and len(idx) >= 100:
            z = np.mean([ndtri((pd.Series(yk[t, idx]).rank().to_numpy() - 0.5) / len(idx)) for yk in self.ys], 0)
            z = (z - z.mean()) / (z.std() + 1e-9)
            if self.neutral:
                X = [np.ones(len(idx))]; ind = self.ind[t, idx]
                for k in np.unique(ind[ind >= 0]): X.append((ind == k).astype(float))
                lm = self.lmv[t, idx]; lm = np.where(np.isfinite(lm), lm, np.nanmean(lm)); X.append((lm - lm.mean()) / (lm.std() + 1e-9))
                X = np.stack(X, 1); beta, *_ = np.linalg.lstsq(X, z, rcond=None); z = z - X @ beta; z = z / (z.std() + 1e-9)
            out['yraw'] = self.y[t, idx]; out['z'] = z.astype(np.float32)
        self._cache[key] = out
        return out

    def x(self, t, idx):
        ii = torch.as_tensor(idx, device=self.dev)
        d = self.daily[t - self.L + 1:t + 1, ii].permute(1, 0, 2).float()                      # [B,L,10]
        it = self.intra[t - self.Di + 1:t + 1, ii].permute(1, 0, 2, 3).float()                 # [B,Di,48,5]
        if self.flow is not None:
            f = self.flow[t - self.Di + 1:t + 1, ii].permute(1, 0, 2, 3).float()               # [B,Di,48,2]
            has = (f.abs().sum((-1, -2), keepdim=True) > 0).float().expand(-1, -1, 48, 1)
            it = torch.cat([it, f, has], -1)                                                    # [B,Di,48,8]
        return d, it

    def split(self, blk, stride):
        name, fs, fe, vs, ve, te = blk
        cand = np.arange(max(self.L, self.Di), self.T)
        if self.train_start: cand = cand[self.dint[cand] >= self.train_start]
        tr = cand[(self.lend[cand] <= te)][::stride]
        va = cand[(self.dint[cand] >= vs) & (self.lend[cand] <= ve)][::stride]
        tv = cand[(self.lend[cand] <= ve)][::stride]
        test = cand[(self.dint[cand] >= fs) & (self.dint[cand] <= fe)]
        return tr, va, tv, test


class IntraEnc(nn.Module):
    def __init__(self, d=32, layers=2, heads=4, cin=5):
        super().__init__()
        self.inp = nn.Linear(cin, d); self.pos = nn.Parameter(torch.zeros(48, d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.q = nn.Parameter(torch.randn(d) / d ** 0.5); self.norm = nn.LayerNorm(d)
    def forward(self, x):                               # x [M,48,5] -> [M,d]
        x = torch.cat([x[..., :4] * 30.0, x[..., 4:5] * 48.0 - 1.0, x[..., 5:]], -1)
        h = self.enc(self.inp(x) + self.pos)
        a = torch.softmax(h @ self.q / h.shape[-1] ** 0.5, dim=1)
        return self.norm((a.unsqueeze(-1) * h).sum(1))


class E2E(nn.Module):
    def __init__(self, d=32, h=64, layers=2, intra=True, cin=5):
        super().__init__()
        self.use_intra = intra; self.cin = cin; self.no_daily = False
        self.intra = IntraEnc(d, layers, cin=cin) if intra else None
        self.dproj = nn.Linear(10, d)
        self.gru = nn.GRU(2 * d + 1 if intra else d, h, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Dropout(0.1), nn.Linear(h, 1))
    def forward(self, d, it):                           # d [B,L,10], it [B,Di,48,5]
        B, L, _ = d.shape
        dd = torch.cat([d[..., :4] * 30.0, d[..., 4:]], -1)
        if self.no_daily: dd = torch.zeros_like(dd)
        z = self.dproj(dd)
        if self.use_intra:
            Di = it.shape[1]
            e = self.intra(it.reshape(B * Di, 48, self.cin)).reshape(B, Di, -1)
            has = (it[..., 4].abs().sum(-1, keepdim=True) > 0).float()                      # [B,Di,1]
            e = e * has
            pad = torch.zeros(B, L - Di, e.shape[-1] + 1, device=d.device, dtype=e.dtype)
            z = torch.cat([z, torch.cat([pad, torch.cat([e, has], -1)], 1)], -1)
        o, _ = self.gru(z)
        return self.head(o[:, -1]).squeeze(-1)


def corr_loss(p, y, w):
    w = w / w.sum(); pc = p - (w * p).sum(); yc = y - (w * y).sum()
    return -(w * pc * yc).sum() / (torch.sqrt((w * pc * pc).sum() * (w * yc * yc).sum()) + 1e-8)


def rank_ic(p, y):
    return float(np.corrcoef(pd.Series(p).rank().to_numpy(), pd.Series(y).rank().to_numpy())[0, 1])


def run(D, blk, seed, args):
    tr, va, tv, test = D.split(blk, args.stride)
    torch.manual_seed(seed); np.random.seed(seed)
    ac = dict(device_type=D.dev.type, dtype=torch.bfloat16, enabled=bool(args.amp))

    @torch.no_grad()
    def predict(model, t, idx):
        model.eval(); out = []
        for s in range(0, len(idx), args.pred_batch):
            d, it = D.x(t, idx[s:s + args.pred_batch])
            with torch.autocast(**ac): out.append(model(d, it).float().cpu().numpy())
        return np.concatenate(out)

    def fit(days, epochs, val_days=None):
        model = E2E(args.d, args.hidden, args.layers, not args.no_intra, cin=8 if args.flow else 5).to(D.dev)
        model.no_daily = bool(args.no_daily)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        best, best_state, best_ep, bad, hist = -1e9, None, 0, 0, []
        rng = np.random.default_rng(seed)
        for ep in range(epochs):
            model.train(); losses = []; te = time.time()
            for t in rng.permutation(days):
                q = D.day(t)
                if 'z' not in q: continue
                n = len(q['idx']); sel = rng.choice(n, min(n, args.max_batch), replace=False)
                d, it = D.x(t, q['idx'][sel])
                w = np.ones(n, np.float32)
                if args.top_w > 0:
                    u = (pd.Series(q['z']).rank().to_numpy() - 0.5) / n; w *= 1.0 + args.top_w * (u >= 0.8)
                if args.tail_w > 0:         # emphasise both tails of the label distribution
                    u = (pd.Series(q['z']).rank().to_numpy() - 0.5) / n; w *= 1.0 + args.tail_w * ((u >= 0.8) | (u < 0.2))
                if args.cap_w > 0:          # emphasise large caps: weight grows with the circulating-cap percentile
                    lm = D.lmv[t, q['idx']]; us = (pd.Series(lm).rank().to_numpy() - 0.5) / n
                    w *= 1.0 + args.cap_w * np.nan_to_num(us, nan=0.5) ** 2
                with torch.autocast(**ac): p = model(d, it)
                loss = corr_loss(p.float(), torch.from_numpy(q['z'][sel]).to(D.dev), torch.from_numpy(w[sel]).to(D.dev))
                opt.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 3.0); opt.step(); losses.append(float(loss.detach()))
            rec = {'ep': ep + 1, 'loss': float(np.mean(losses)), 'sec': round(time.time() - te)}
            if val_days is not None:
                vic = float(np.mean([rank_ic(predict(model, t, D.day(t)['idx']), D.day(t)['yraw']) for t in val_days if 'z' in D.day(t)]))
                rec['val_ic'] = vic
                if vic > best: best, best_ep, bad = vic, ep + 1, 0; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else: bad += 1
            hist.append(rec); log(blk[0], 'seed', seed, rec)
            if val_days is not None and bad >= args.patience: break
        if best_state is not None: model.load_state_dict(best_state)
        return model, best, best_ep, hist

    model, vic, bep, hist = fit(tr, args.epochs, va)
    info = {'val_ic': vic, 'best_epoch': bep, 'hist': hist, 'n_train_days': len(tr)}
    if args.refit: model, _, _, _ = fit(tv, max(1, int(round(bep * 1.1))))
    preds, meta = [], []
    for t in test:
        q = D.day(t, need_label=False); preds.append(predict(model, t, q['idx'])); meta.append((t, q['idx']))
    return np.concatenate(preds), meta, info


def unpack_flow(data, parts='/content/up/flow.tgz.part'):
    import subprocess, os
    d = Path(data); lock = Path(str(d) + '.flowlock')
    while not (d / 'flow.npy').exists():
        if not Path('/content/up/FLOW_DONE').exists(): time.sleep(30); continue
        try: lock.mkdir()
        except FileExistsError: time.sleep(20); continue
        subprocess.run(f'mkdir -p /content/_flow_tmp && cat {parts}* | tar -xzf - -C /content/_flow_tmp', shell=True, check=True)
        os.replace('/content/_flow_tmp/flow.npy', d / 'flow.npy'); log('flow unpacked')


def unpack_dir(data, parts):
    """Extract a tgz-split bundle (top dir = basename of --data) into its parent once, under a mkdir lock."""
    import subprocess
    d = Path(data); lock = Path(str(d) + '.unpacking')
    while not (d / 'READY').exists():
        try: lock.mkdir()
        except FileExistsError: time.sleep(20); continue
        try:
            subprocess.run(f'rm -rf {d} && cat {parts}* | tar -xzf - -C {d.parent} && touch {d}/READY', shell=True, check=True); log('unpacked', d)
        finally:
            lock.rmdir()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/content/e2e'); ap.add_argument('--out', default='/content/out/e2e')
    ap.add_argument('--name', default='e2e'); ap.add_argument('--blocks', default='2024H2,2025H2')
    ap.add_argument('--seeds', default='7'); ap.add_argument('--label', default='y10'); ap.add_argument('--neutral', type=int, default=0)
    ap.add_argument('--top_w', type=float, default=0); ap.add_argument('--refit', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=30); ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--lr', type=float, default=5e-4); ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--max_batch', type=int, default=2500); ap.add_argument('--pred_batch', type=int, default=3000)
    ap.add_argument('--lookback', type=int, default=60); ap.add_argument('--days_intra', type=int, default=10)
    ap.add_argument('--d', type=int, default=32); ap.add_argument('--hidden', type=int, default=64); ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--stride', type=int, default=2); ap.add_argument('--amp', type=int, default=1)
    ap.add_argument('--no_intra', type=int, default=0); ap.add_argument('--device', default='cuda')
    ap.add_argument('--train_pool', type=int, default=0, help='index bits (1=CSI300, 2=CSI500, 4=CSI1000) restricting training samples; test predicts all')
    ap.add_argument('--tail_w', type=float, default=0, help='extra loss weight on top and bottom quintile label stocks')
    ap.add_argument('--cap_w', type=float, default=0, help='extra loss weight rising with size percentile (x^2)')
    ap.add_argument('--no_daily', type=int, default=0, help='zero the daily-bar inputs (intraday-only diagnostic)')
    ap.add_argument('--train_start', type=int, default=0, help='first feature date used for training (e.g. 20200101)')
    ap.add_argument('--unpack', default='', help='tgz part prefix to extract into dirname(--data) on first use')
    ap.add_argument('--flow', type=int, default=0, help='add the 15s order-flow channels (flow.npy, 2020+)')
    a = ap.parse_args()
    out = Path(a.out) / a.name; out.mkdir(parents=True, exist_ok=True)
    (out / 'args.json').write_text(json.dumps(vars(a), indent=1))
    if a.unpack: unpack_dir(a.data, a.unpack)
    if a.flow: unpack_flow(a.data)
    D = Data(a.data, a.label, a.neutral, torch.device(a.device), a.lookback, a.days_intra, bool(a.flow), not a.no_intra)
    D.train_start = a.train_start; D.train_pool = a.train_pool
    for blk in [b for b in BLOCKS if b[0] in a.blocks.split(',')]:
        dst = out / f'{blk[0]}_pred.npz'
        if dst.exists(): log('skip', dst); continue
        P, infos = [], []
        for s in [int(x) for x in a.seeds.split(',')]:
            p, meta, info = run(D, blk, s, a)
            o = 0; pz = np.empty_like(p)
            for t, idx in meta:
                v = p[o:o + len(idx)]; pz[o:o + len(idx)] = (v - v.mean()) / (v.std() + 1e-9); o += len(idx)
            P.append(pz); infos.append({'seed': s, **info})
            log(blk[0], 'seed', s, 'val_ic', round(info['val_ic'], 4), 'best_epoch', info['best_epoch'])
        pred = np.mean(P, 0)
        fd = np.concatenate([np.full(len(i), D.dint[t]) for t, i in meta]).astype(np.int32)
        st = np.concatenate([i for _, i in meta]).astype(np.int32)
        np.savez(dst, pred=pred.astype(np.float32), feature_date=fd, stock=st, stock_code=D.codes[st].astype('<U12'), seed_preds=np.stack(P).astype(np.float32))
        (out / f'{blk[0]}_log.json').write_text(json.dumps(infos, indent=1, default=str))
        ics, o = [], 0
        for t, idx in meta:
            y = D.y[t, idx]; m = np.isfinite(y)
            if m.sum() > 100: ics.append(rank_ic(pred[o:o + len(idx)][m], y[m]))
            o += len(idx)
        log('==>', a.name, blk[0], 'test rankIC', round(float(np.mean(ics)), 4), 'ICIR', round(float(np.mean(ics) / np.std(ics)), 3))
    log('all done')


if __name__ == '__main__':
    main()
