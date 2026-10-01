"""Rolling training on Colab GPU: XGBoost (tabular), MLP (tabular), GRU (daily sequence), Fusion (GRU + MLP).

Data layout (from pack.py) under --data:
  axes.npz, grid.npz, tab.npy [G,N,K] uint8 (255 = missing), tab_cols.json, seq.npy [T,N,F] fp16,
  univ.npy, y10.npy, y5.npy, industry.npy, log_circ_mv.npy
Outputs under --out/<model>/: <block>_pred.npz (test rows), <block>_log.json.
"""
import argparse, json, time, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.special import ndtri

BLOCKS = [  # name, test_start, test_end, val_start, val_label_end, train_label_end  (same as the baseline)
    ('2024H2', 20240701, 20241231, 20230703, 20240628, 20230630),
    ('2025H1', 20250101, 20250630, 20240101, 20241231, 20231229),
    ('2025H2', 20250701, 20251231, 20240701, 20250630, 20240628),
    ('2026H1', 20260101, 20260630, 20250101, 20251231, 20241231),
    # earlier windows (Ceres factors start 2020, so these have 1.5-3.5 years of training data)
    ('2022H2', 20220701, 20221231, 20210701, 20220630, 20210630),
    ('2023H1', 20230101, 20230630, 20220101, 20221231, 20211231),
    ('2023H2', 20230701, 20231231, 20220701, 20230630, 20220630),
    ('2024H1', 20240101, 20240630, 20230101, 20231231, 20221231),
    # untouched holdout: July 2026 (last month with Ceres factor coverage)
    ('2026H2', 20260701, 20260731, 20250701, 20260630, 20250630),
]
MAIN = ['2024H2', '2025H1', '2025H2', '2026H1']
H = 11          # a 10-day label realised at open(t+11)
LOOKBACK = 40   # sequence window (baseline daily branch used 40 days)

t0 = time.time()
def log(*a): print(f'[{time.time()-t0:7.0f}s]', *a, flush=True)


# ------------------------------------------------------------------ data
class Data:
    def __init__(self, root, label, neutral, dev):
        r = Path(root)
        ax = np.load(r / 'axes.npz', allow_pickle=True)
        self.dates = pd.DatetimeIndex(ax['dates']); self.codes = ax['codes']
        self.dint = self.dates.strftime('%Y%m%d').astype(int).to_numpy()
        g = np.load(r / 'grid.npz'); self.grid = g['grid']; self.on_sparse = g['on_sparse']
        self.T, self.N = len(self.dates), len(self.codes)
        self.univ = np.load(r / 'univ.npy'); self.ys = [np.load(r / f'{l}.npy') for l in label.split('+')]; y = self.ys[-1]
        self.ind = np.load(r / 'industry.npy'); self.lmv = np.load(r / 'log_circ_mv.npy')
        self.tab_cols = json.loads((r / 'tab_cols.json').read_text())
        self.dev = dev
        self.root = r
        self.tab_np = np.load(r / 'tab.npy', mmap_mode='r')                              # uint8 [G,N,K], CPU
        self._tab = self._seq = None                                                     # GPU copies, loaded on first NN use
        self.F = np.load(r / 'seq.npy', mmap_mode='r').shape[2]; self.K = self.tab_np.shape[2]
        self.gpos = {int(t): i for i, t in enumerate(self.grid)}
        # label end per date (open of t+11)
        self.lend = np.array([self.dint[min(t + H, self.T - 1)] if t + H < self.T else 99999999 for t in range(self.T)])
        # per-date sample sets: universe stocks with a finite label
        self.y = y; self.neutral = neutral
        self._cache = {}
        log('data', self.T, self.N, 'grid', len(self.grid), 'K', self.K, 'F', self.F)

    def day(self, t, need_label=True):
        """Index set, raw label and training target (rank-gauss, optionally industry+size neutralised) for date index t."""
        key = (t, need_label)
        if key in self._cache: return self._cache[key]
        m = self.univ[t].copy()
        if need_label:
            for yk in self.ys: m &= np.isfinite(yk[t])
        idx = np.where(m)[0]
        out = {'t': t, 'idx': idx}
        if need_label and len(idx) >= 100:
            yr = self.y[t, idx]
            z = np.mean([ndtri((pd.Series(yk[t, idx]).rank().to_numpy() - 0.5) / len(idx)) for yk in self.ys], 0)
            z = (z - z.mean()) / (z.std() + 1e-9)
            if self.neutral:
                X = [np.ones(len(idx))]
                ind = self.ind[t, idx]
                for k in np.unique(ind[ind >= 0]): X.append((ind == k).astype(float))
                lm = self.lmv[t, idx]; lm = np.where(np.isfinite(lm), lm, np.nanmean(lm)); X.append((lm - lm.mean()) / (lm.std() + 1e-9))
                X = np.stack(X, 1)
                beta, *_ = np.linalg.lstsq(X, z, rcond=None)
                z = z - X @ beta; z = z / (z.std() + 1e-9)
            out['yraw'] = yr; out['z'] = z.astype(np.float32)
        self._cache[key] = out
        return out

    @property
    def tab(self):
        if self._tab is None: self._tab = torch.from_numpy(np.ascontiguousarray(self.tab_np)).to(self.dev)
        return self._tab

    @property
    def seq(self):
        if self._seq is None:
            s = np.nan_to_num(np.load(self.root / 'seq.npy'), nan=0.0); self._seq = torch.from_numpy(s).to(self.dev)
        return self._seq

    def tab_np_x(self, t, idx):
        x = self.tab_np[self.gpos[t]][idx].astype(np.float32)
        x[x == 255] = np.nan
        return (x + 0.5) / 254.0 - 0.5

    def tab_x(self, t, idx, nan_for_missing=False):
        x = self.tab[self.gpos[t], torch.as_tensor(idx, device=self.dev)]
        miss = x == 255
        xf = (x.float() + 0.5) / 254.0 - 0.5
        xf = torch.where(miss, torch.full_like(xf, float('nan') if nan_for_missing else 0.0), xf)
        return xf

    def seq_x(self, t, idx):
        return self.seq[t - LOOKBACK + 1:t + 1, torch.as_tensor(idx, device=self.dev)].permute(1, 0, 2).float()

    def split(self, blk, sparse_train=True, grid_only=True, start=None):
        name, fs, fe, vs, ve, te = blk
        cand = self.grid if grid_only else np.arange(self.T)
        if start is not None: cand = cand[self.dint[cand] >= start]
        cand = cand[cand >= LOOKBACK]
        sp = np.isin(cand, self.grid[self.on_sparse]) if grid_only and sparse_train else np.ones(len(cand), bool)
        tr = cand[(self.lend[cand] <= te) & sp]
        va = cand[(self.dint[cand] >= vs) & (self.lend[cand] <= ve) & sp]
        tv = cand[(self.lend[cand] <= ve) & sp]
        test = self.grid[(self.dint[self.grid] >= fs) & (self.dint[self.grid] <= fe)]
        return tr, va, tv, test


def row_weight(D, t, q, days, args):
    """Loss weight per stock on training date t: time decay (half-life in trading days) x top-quintile emphasis."""
    w = np.ones(len(q['idx']), np.float32)
    if args.time_hl > 0: w *= 0.5 ** ((days.max() - t) / args.time_hl)
    if args.top_w > 0:
        u = (pd.Series(q['z']).rank().to_numpy() - 0.5) / len(q['z'])
        w *= 1.0 + args.top_w * (u >= 0.8)
    return w


def rank_ic(p, y):
    a = pd.Series(p).rank().to_numpy(); b = pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(a, b)[0, 1])


# ------------------------------------------------------------------ XGBoost
def run_xgb(D, blk, seed, refit, args):
    import xgboost as xgb
    tr, va, tv, test = D.split(blk)
    rank = args.xgb_obj == 'rank'
    def mat(days, label=True, weight=False):
        Xs, ys, ws, qs, meta = [], [], [], [], []
        for t in days:
            q = D.day(t, need_label=label)
            if label and 'z' not in q: continue
            Xs.append(D.tab_np_x(t, q['idx']))
            if label:
                if rank:
                    u = (pd.Series(q['z']).rank().to_numpy() - 0.5) / len(q['z'])
                    ys.append(np.floor(u * args.rank_levels).astype(np.float32)); qs.append(np.full(len(u), t))
                else: ys.append(q['z'])
            if weight and not rank: ws.append(row_weight(D, t, q, days, args))
            meta.append((t, q['idx']))
        X = np.concatenate(Xs); y = np.concatenate(ys) if label else None
        w = np.concatenate(ws) if ws else None
        mat.qid = np.concatenate(qs) if qs else None
        return X, y, meta, w
    Xtr, ytr, _, wtr = mat(tr, weight=True)
    params = dict(tree_method='hist', device='cuda', objective='reg:squarederror', max_depth=args.xgb_depth,
                  learning_rate=args.xgb_lr, subsample=0.7, colsample_bytree=0.5, min_child_weight=args.xgb_mcw,
                  reg_lambda=10.0, max_bin=128, seed=seed, eval_metric='rmse')
    if rank:
        params.update(objective='rank:ndcg', lambdarank_pair_method='mean', lambdarank_num_pair_per_sample=args.rank_pairs,
                      eval_metric=f'ndcg@{args.rank_k}', min_child_weight=1.0)
    qtr = mat.qid if rank else None
    Xva_, yva_, mva_, _ = mat(va); qva = mat.qid if rank else None
    Xva, yva, mva = Xva_, yva_, mva_
    dtr = xgb.QuantileDMatrix(Xtr, ytr, weight=wtr, qid=qtr, max_bin=128); dva = xgb.QuantileDMatrix(Xva, yva, qid=qva, ref=dtr, max_bin=128)
    del Xtr
    bst = xgb.train(params, dtr, num_boost_round=args.xgb_rounds, evals=[(dva, 'val')], early_stopping_rounds=args.xgb_es, verbose_eval=False)
    best = bst.best_iteration + 1
    pv = bst.predict(dva, iteration_range=(0, best))
    val_ic = np.mean([rank_ic(pv[o:o + len(i)], D.day(t)['yraw']) for (t, i), o in zip(mva, np.cumsum([0] + [len(i) for _, i in mva])[:-1])])
    if refit:
        del dtr, dva; import gc; gc.collect()
        Xtv, ytv, _, wtv = mat(tv, weight=True)
        bst = xgb.train(params, xgb.QuantileDMatrix(Xtv, ytv, weight=wtv, qid=mat.qid if rank else None, max_bin=128), num_boost_round=int(best * 1.1))
        best = int(best * 1.1)
    Xte, _, mte, _ = mat(test, label=False)
    pte = bst.predict(xgb.DMatrix(Xte), iteration_range=(0, best))
    imp = bst.get_score(importance_type='gain')
    top = sorted(((D.tab_cols[int(k[1:])], v) for k, v in imp.items()), key=lambda x: -x[1])[:30]
    return pte, mte, {'best_iter': best, 'val_ic': float(val_ic), 'n_train': len(ytr), 'top_gain': top}


# ------------------------------------------------------------------ neural nets
class MLP(nn.Module):
    def __init__(self, K, h=256, out=1, p=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(K, h), nn.SiLU(), nn.Dropout(p), nn.Linear(h, h // 2), nn.SiLU(), nn.Dropout(p), nn.Linear(h // 2, out))
    def forward(self, xt, xs=None): return self.net(xt).squeeze(-1)

class GRUNet(nn.Module):
    def __init__(self, F, h=64, layers=1, head=True):
        super().__init__()
        self.gru = nn.GRU(F, h, num_layers=layers, batch_first=True, dropout=0.1 if layers > 1 else 0.0)
        self.norm = nn.LayerNorm(h); self.head = nn.Linear(h, 1) if head else None
    def embed(self, xs):
        z, _ = self.gru(xs); return self.norm(z[:, -1])
    def forward(self, xt, xs): return self.head(self.embed(xs)).squeeze(-1)

class Fusion(nn.Module):
    def __init__(self, K, F, h=64):
        super().__init__()
        self.g = GRUNet(F, h, head=False)
        self.m = nn.Sequential(nn.Linear(K, 256), nn.SiLU(), nn.Dropout(0.1), nn.Linear(256, h), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(2 * h, h), nn.SiLU(), nn.Dropout(0.1), nn.Linear(h, 1))
    def forward(self, xt, xs):
        return self.head(torch.cat([self.g.embed(xs), self.m(xt)], 1)).squeeze(-1)

def corr_loss(p, y, w=None):
    if w is None: w = torch.ones_like(p)
    w = w / w.sum()
    pc = p - (w * p).sum(); yc = y - (w * y).sum()
    return -(w * pc * yc).sum() / (torch.sqrt((w * pc * pc).sum() * (w * yc * yc).sum()) + 1e-8)

def run_nn(D, blk, seed, refit, kind, args):
    grid_only = kind != 'gru'
    tr, va, tv, test = D.split(blk, grid_only=grid_only, start=None if grid_only else 20190101)
    torch.manual_seed(seed); np.random.seed(seed)
    make = {'mlp': lambda: MLP(D.K), 'gru': lambda: GRUNet(D.F, args.gru_hidden), 'fusion': lambda: Fusion(D.K, D.F, args.gru_hidden)}[kind]
    need_t, need_s = kind != 'gru', kind != 'mlp'
    def inputs(t, idx):
        return (D.tab_x(t, idx) if need_t else None), (D.seq_x(t, idx) if need_s else None)
    def fit(days, epochs, val_days=None):
        model = make().to(D.dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        best, best_state, best_ep, bad, hist = -1e9, None, 0, 0, []
        rng = np.random.default_rng(seed)
        for ep in range(epochs):
            model.train(); losses = []
            for t in rng.permutation(days):
                q = D.day(t)
                if 'z' not in q: continue
                idx = q['idx']
                if len(idx) > args.max_batch: idx_sel = rng.choice(len(idx), args.max_batch, replace=False)
                else: idx_sel = np.arange(len(idx))
                xt, xs = inputs(t, idx[idx_sel])
                y = torch.from_numpy(q['z'][idx_sel]).to(D.dev)
                rw = row_weight(D, t, q, days, args)
                w = torch.from_numpy(rw[idx_sel] / rw.max()).to(D.dev)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bool(args.amp)):
                    p = model(xt, xs)
                loss = corr_loss(p.float(), y, w) * float(rw.max())
                opt.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 3.0); opt.step(); losses.append(float(loss.detach()))
            rec = {'ep': ep + 1, 'loss': float(np.mean(losses))}
            if val_days is not None:
                vic = np.mean([rank_ic(predict(model, t, D.day(t)['idx']), D.day(t)['yraw']) for t in val_days if 'z' in D.day(t)])
                rec['val_ic'] = float(vic)
                if vic > best: best, best_ep, bad = vic, ep + 1, 0; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else: bad += 1
            hist.append(rec)
            if val_days is not None and bad >= args.patience: break
        if best_state is not None: model.load_state_dict(best_state)
        return model, best, best_ep, hist
    @torch.no_grad()
    def predict(model, t, idx):
        model.eval(); xt, xs = inputs(t, idx)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bool(args.amp)):
            return model(xt, xs).float().cpu().numpy()
    model, vic, bep, hist = fit(tr, args.epochs, va)
    info = {'val_ic': float(vic), 'best_epoch': bep, 'hist': hist, 'n_train_days': len(tr)}
    if refit:
        model, _, _, _ = fit(tv, max(1, int(round(bep * 1.1))))
    preds, meta = [], []
    for t in test:
        q = D.day(t, need_label=False)
        preds.append(predict(model, t, q['idx'])); meta.append((t, q['idx']))
    return np.concatenate(preds), meta, info


# ------------------------------------------------------------------ main
def shift3(d):
    y, m, dd = d // 10000, d // 100 % 100 + 3, d % 100
    if m > 12: y, m = y + 1, m - 12
    return y * 10000 + m * 100 + dd


def sub_blocks(blk, quarterly):
    """Half-year block, or two quarters where the second quarter's cutoffs move forward by 3 months."""
    if not quarterly: return [blk]
    name, fs, fe, vs, ve, te = blk
    mid = shift3(fs)
    return [(name, fs, mid - 1, vs, ve, te), (name, mid, fe, shift3(vs), shift3(ve), shift3(te))]


def prepare_data(data, parts):
    """Extract an alternative tab.npy bundle once (mkdir lock), linking every other file from /content/data."""
    import os, subprocess
    d = Path(data); lock = Path(str(d) + '.lock')
    while not (d / 'READY').exists():
        try:
            lock.mkdir()
        except FileExistsError:
            time.sleep(20); continue
        tmp = Path('/content/_unpack'); subprocess.run(f'rm -rf {tmp} && mkdir -p {tmp} && cat {parts}* | tar -xzf - -C {tmp}', shell=True, check=True)
        src = next(tmp.iterdir()); d.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir(): os.replace(f, d / f.name)
        for f in Path('/content/data').iterdir():
            if not (d / f.name).exists(): os.symlink(f, d / f.name)
        (d / 'READY').touch(); log('unpacked', d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/content/data'); ap.add_argument('--out', default='/content/out/run')
    ap.add_argument('--models', default='xgb,mlp,gru,fusion'); ap.add_argument('--blocks', default='all')
    ap.add_argument('--seeds', default='7,11,42'); ap.add_argument('--label', default='y10')
    ap.add_argument('--neutral', type=int, default=0); ap.add_argument('--refit', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=40); ap.add_argument('--patience', type=int, default=6)
    ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--max_batch', type=int, default=6000); ap.add_argument('--gru_hidden', type=int, default=64)
    ap.add_argument('--amp', type=int, default=0)
    ap.add_argument('--time_hl', type=float, default=0, help='half-life (trading days) of training-date weights; 0 = flat')
    ap.add_argument('--top_w', type=float, default=0, help='extra loss weight on top-quintile label stocks')
    ap.add_argument('--quarterly', type=int, default=0, help='retrain every quarter instead of every half year')
    ap.add_argument('--xgb_obj', default='reg', choices=['reg', 'rank']); ap.add_argument('--rank_levels', type=int, default=5)
    ap.add_argument('--rank_pairs', type=int, default=8); ap.add_argument('--rank_k', type=int, default=400)
    ap.add_argument('--xgb_es', type=int, default=200)
    ap.add_argument('--xgb_depth', type=int, default=7); ap.add_argument('--xgb_lr', type=float, default=0.03)
    ap.add_argument('--xgb_mcw', type=float, default=300.0); ap.add_argument('--xgb_rounds', type=int, default=3000)
    ap.add_argument('--unpack', default='', help='glob prefix of tgz parts holding <dir>/tab.npy etc. to extract into --data on first use')
    a = ap.parse_args()
    if a.unpack: prepare_data(a.data, a.unpack)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'args.json').write_text(json.dumps(vars(a), indent=1))
    D = Data(a.data, a.label, a.neutral, torch.device('cuda'))
    blocks = [b for b in BLOCKS if b[0] in MAIN] if a.blocks == 'all' else [b for b in BLOCKS if b[0] in a.blocks.split(',')]
    seeds = [int(s) for s in a.seeds.split(',')]
    for kind in a.models.split(','):
        (out / kind).mkdir(exist_ok=True)
        for blk in blocks:
            dst = out / kind / f'{blk[0]}_pred.npz'
            if dst.exists(): log('skip', dst); continue
            P, infos = [], []
            for s in seeds:
                ps, meta, info = [], [], {}
                for sb in sub_blocks(blk, a.quarterly):
                    p_, m_, i_ = run_xgb(D, sb, s, a.refit, a) if kind == 'xgb' else run_nn(D, sb, s, a.refit, kind, a)
                    ps.append(p_); meta += m_; info = {**i_, 'val_ic': i_['val_ic'] if not info else (info['val_ic'] + i_['val_ic']) / 2}
                p = np.concatenate(ps)
                # z-score per date before averaging seeds
                o = 0; pz = np.empty_like(p)
                for t, idx in meta:
                    v = p[o:o + len(idx)]; pz[o:o + len(idx)] = (v - v.mean()) / (v.std() + 1e-9); o += len(idx)
                P.append(pz); infos.append({'seed': s, **{k: v for k, v in info.items()}})
                log(kind, blk[0], 'seed', s, 'val_ic', round(info['val_ic'], 4), {k: info[k] for k in info if k in ('best_iter', 'best_epoch')})
            pred = np.mean(P, 0)
            fd = np.concatenate([np.full(len(i), D.dint[t]) for t, i in meta]).astype(np.int32)
            st = np.concatenate([i for _, i in meta]).astype(np.int32)
            np.savez(dst, pred=pred.astype(np.float32), feature_date=fd, stock=st, stock_code=D.codes[st].astype('<U12'),
                     seed_preds=np.stack(P).astype(np.float32))
            (out / kind / f'{blk[0]}_log.json').write_text(json.dumps(infos, indent=1, default=str))
            # quick test IC on y10 for the log (eval proper happens locally)
            ics = []
            o = 0
            for t, idx in meta:
                y = D.y[t, idx]; m = np.isfinite(y)
                if m.sum() > 100: ics.append(rank_ic(pred[o:o + len(idx)][m], y[m]))
                o += len(idx)
            log('==>', kind, blk[0], 'test rankIC', round(float(np.mean(ics)), 4), 'ICIR', round(float(np.mean(ics) / np.std(ics)), 3))
    log('all done')


if __name__ == '__main__':
    main()
