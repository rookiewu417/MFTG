"""End-to-end track training on Colab: raw 5-min bars + raw daily bars -> score. No factor formulas, no Ceres.

Data (from mftg/build_mftg.py + base panels) under --data:
  axes.npz, intra.npy [T,N,48,5] fp16, daily.npy [T,N,10] fp16, univ.npy, y10.npy, y5.npy, industry.npy, log_circ_mv.npy
  (--extra: extra.npy [T,N,K] fp16 from build_extra.py, appended to the daily inputs)
Model: per-day intraday Transformer over 48 bars -> attention-pooled day vector (last --days_intra days);
       GRU over the last --lookback days of [daily bar embedding, intraday day vector, has-intraday flag] -> score.
       Options: --rnn patch (PatchTST encoder instead of the GRU), --graph k (correlation-graph peer aggregation before the head),
       --pretrain n (next-day-bar self-supervised epochs first), --sam rho (sharpness-aware minimisation),
       --weekly n (extra GRU over n weekly bars), --demean 1 (market-relative daily price inputs).
Loss: -IC against the rank-gauss label (optionally industry+size neutralised, optionally top-quintile weighted,
optionally plus --pool_w x -IC inside the predicted top --pool_q, optionally risk-adjusted: --vol_label),
same windows / label / output format as weave/train.py so evaluation is shared with the Weave track.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.special import ndtri

H = 11
LABEL_H = {'y5': 6, 'y10': 11, 'y20': 21}   # label end = open of t+H; split() keeps every label inside its window
BLOCKS = [  # name, test_start, test_end, val_start, val_label_end, train_label_end  (same as weave/train.py)
    ('2024H2', 20240701, 20241231, 20230703, 20240628, 20230630),
    ('2025H1', 20250101, 20250630, 20240101, 20241231, 20231229),
    ('2025H2', 20250701, 20251231, 20240701, 20250630, 20240628),
    ('2026H1', 20260101, 20260630, 20250101, 20251231, 20241231),
    ('2022H2', 20220701, 20221231, 20210701, 20220630, 20210630),
    ('2023H1', 20230101, 20230630, 20220101, 20221231, 20211231),
    ('2023H2', 20230701, 20231231, 20220701, 20230630, 20220630),
    ('2024H1', 20240101, 20240630, 20230101, 20231231, 20221231),
    ('2026H2', 20260701, 20260731, 20250701, 20260630, 20250630),
    ('2026H2X', 20260701, 20260910, 20250701, 20260630, 20250630),   # same fit as 2026H2, test extended to the end of the data
]


def month_blocks(name):
    """Split a half-year block into monthly refits: test = one month, validation = the 12 months before it, training labels
    end where validation starts (same layout as BLOCKS, shifted month by month)."""
    b = next(x for x in BLOCKS if x[0] == name)
    out = []
    for m0 in pd.date_range(pd.Timestamp(str(b[1])), pd.Timestamp(str(b[2])), freq='MS'):
        m1 = min(m0 + pd.offsets.MonthEnd(0), pd.Timestamp(str(b[2])))
        vs = m0 - pd.DateOffset(years=1); di = lambda x: int(x.strftime('%Y%m%d'))
        out.append((f'M{m0:%Y%m}', di(m0), di(m1), di(vs), di(m0 - pd.Timedelta(days=1)), di(vs - pd.Timedelta(days=1))))
    return out


t0 = time.time()
def log(*a): print(f'[{time.time()-t0:7.0f}s]', *a, flush=True)


class Data:
    def __init__(self, root, label, neutral, dev, lookback, days_intra, flow=False, intra=True, mkt=False, extra=False):
        r = Path(root)
        ax = np.load(r / 'axes.npz', allow_pickle=True)
        self.dates = pd.DatetimeIndex(ax['dates']); self.codes = ax['codes']
        self.dint = self.dates.strftime('%Y%m%d').astype(int).to_numpy()
        self.T, self.N = len(self.dates), len(self.codes)
        self.univ = np.load(r / 'univ.npy'); self.ys = [np.load(r / f'{l}.npy') for l in label.split('+')]; self.y = self.ys[-1]
        self.H = max(LABEL_H[l] for l in label.split('+'))
        self.ind = np.load(r / 'industry.npy'); self.lmv = np.load(r / 'log_circ_mv.npy')
        self.neutral = neutral; self.dev = dev; self.L = lookback; self.Di = days_intra
        self.lend = np.array([self.dint[t + self.H] if t + self.H < self.T else 99999999 for t in range(self.T)])
        t1 = time.time()
        self.intra = torch.from_numpy(np.load(r / 'intra.npy', mmap_mode='r')[:]).to(dev) if intra else \
            torch.zeros((self.T, 1, 48, 5), dtype=torch.float16, device=dev).expand(-1, self.N, -1, -1)       # fp16 [T,N,48,5]
        self.daily = torch.from_numpy(np.load(r / 'daily.npy')).to(dev)                           # fp16 [T,N,10]
        self.flow = torch.from_numpy(np.load(r / 'flow.npy', mmap_mode='r')[:]).to(dev) if flow else None   # fp16 [T,N,48,2]
        self.mkt = torch.from_numpy(np.load(r / 'market.npy')).to(dev) if mkt else None             # fp32 [T,K] (build_market.py)
        self.extra = torch.from_numpy(np.load(r / 'extra.npy')).to(dev) if extra else None            # fp16 [T,N,K] (build_extra.py)
        self.nx = self.extra.shape[2] if extra else 0
        log('data on gpu', tuple(self.intra.shape), tuple(self.daily.shape), f'{time.time()-t1:.0f}s')
        self._cache = {}; self.train_start = 0; self.train_pool = 0; self.vol_label = 0
        self.LX = lookback; self.mm = None    # LX: days sliced per sample (> L with --weekly); mm: [T,4] market mean (--demean)
        self.idxm = np.load(r / 'index_member.npy', mmap_mode='r') if (r / 'index_member.npy').exists() else None
        self.sty = np.load(r / 'styles.npy', mmap_mode='r') if neutral == 2 else None     # beta, volatility, liquidity, momentum

    def day(self, t, need_label=True):
        key = (t, need_label)
        if key in self._cache: return self._cache[key]
        m = self.univ[t].copy()
        if need_label:
            for yk in self.ys: m &= np.isfinite(yk[t])
            if self.train_pool: m &= (self.idxm[t] & self.train_pool) > 0      # training / validation restricted to index members
        idx = np.where(m)[0]; out = {'t': t, 'idx': idx}
        if need_label and len(idx) >= 100:
            sd = 1.0
            if self.vol_label:      # risk-adjusted label: return / std of the last vol_label daily close-to-close log returns
                r = self.daily[t - self.vol_label + 1:t + 1, torch.as_tensor(idx, device=self.dev), 3].float()
                sd = np.maximum(r.std(0).cpu().numpy(), 0.005)
            z = np.mean([ndtri((pd.Series(yk[t, idx] / sd).rank().to_numpy() - 0.5) / len(idx)) for yk in self.ys], 0)
            z = (z - z.mean()) / (z.std() + 1e-9)
            if self.neutral:
                X = [np.ones(len(idx))]; ind = self.ind[t, idx]
                for k in np.unique(ind[ind >= 0]): X.append((ind == k).astype(float))
                lm = self.lmv[t, idx]; lm = np.where(np.isfinite(lm), lm, np.nanmean(lm)); X.append((lm - lm.mean()) / (lm.std() + 1e-9))
                if self.sty is not None: X += [np.asarray(self.sty[t, idx, k], np.float64) for k in range(self.sty.shape[2])]
                X = np.stack(X, 1); beta, *_ = np.linalg.lstsq(X, z, rcond=None); z = z - X @ beta; z = z / (z.std() + 1e-9)
            out['yraw'] = self.y[t, idx]; out['z'] = z.astype(np.float32)
        self._cache[key] = out
        return out

    def x(self, t, idx):
        ii = torch.as_tensor(idx, device=self.dev)
        L = self.LX
        d = self.daily[t - L + 1:t + 1, ii].permute(1, 0, 2).float()                           # [B,L,10]
        if self.mm is not None: d = torch.cat([d[..., :4] - self.mm[t - L + 1:t + 1], d[..., 4:]], -1)   # relative to the market
        if self.extra is not None: d = torch.cat([d, self.extra[t - L + 1:t + 1, ii].permute(1, 0, 2).float()], -1)   # [B,L,10+nx]
        if self.mkt is not None: d = torch.cat([d, self.mkt[t - L + 1:t + 1].expand(len(idx), -1, -1)], -1)   # [B,L,10+K]
        it = self.intra[t - self.Di + 1:t + 1, ii].permute(1, 0, 2, 3).float()                 # [B,Di,48,5]
        if self.flow is not None:
            f = self.flow[t - self.Di + 1:t + 1, ii].permute(1, 0, 2, 3).float()               # [B,Di,48,2]
            has = (f.abs().sum((-1, -2), keepdim=True) > 0).float().expand(-1, -1, 48, 1)
            it = torch.cat([it, f, has], -1)                                                    # [B,Di,48,8]
        return d, it, torch.as_tensor(np.asarray(self.ind[t, idx], np.int64) + 1, device=self.dev).clamp(min=0)

    def split(self, blk, stride):
        name, fs, fe, vs, ve, te = blk
        cand = np.arange(max(self.LX, self.Di), self.T)
        if self.train_start: cand = cand[self.dint[cand] >= self.train_start]
        tr = cand[(self.lend[cand] <= te)][::stride]
        va = cand[(self.dint[cand] >= vs) & (self.lend[cand] <= ve)][::stride]
        tv = cand[(self.lend[cand] <= ve)][::stride]
        test = cand[(self.dint[cand] >= fs) & (self.dint[cand] <= fe)]
        return tr, va, tv, test


class IntraEnc(nn.Module):
    def __init__(self, d=32, layers=2, heads=4, cin=5, center=0):
        super().__init__()
        self.center = center
        self.inp = nn.Linear(cin, d); self.pos = nn.Parameter(torch.zeros(48, d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.q = nn.Parameter(torch.randn(d) / d ** 0.5); self.norm = nn.LayerNorm(d)
    def forward(self, x):                               # x [M,48,5] -> [M,d]
        px = x[..., :4]                                 # ln(price / previous close): every bar carries the day's offset
        if self.center == 1: px = px - x[:, -1:, 3:4]                       # relative to the day's last close
        elif self.center == 2: px = px - x[..., 3:4].mean(1, keepdim=True)  # relative to the day's mean close
        x = torch.cat([px * 30.0, x[..., 4:5] * 48.0 - 1.0, x[..., 5:]], -1)
        h = self.enc(self.inp(x) + self.pos)
        a = torch.softmax(h @ self.q / h.shape[-1] ** 0.5, dim=1)
        return self.norm((a.unsqueeze(-1) * h).sum(1))


class RankGLU(nn.Module):
    """Score head after RankGLU (arXiv 2606.08930): a linear scoring path plus a bounded gated bottleneck branch."""
    def __init__(self, h, r=None):
        super().__init__()
        r = r or max(4, h // 4)
        self.norm = nn.LayerNorm(h); self.lin = nn.Linear(h, 1)
        self.a = nn.Linear(h, r); self.g = nn.Linear(h, r); self.out = nn.Linear(r, 1); self.drop = nn.Dropout(0.1)
    def forward(self, x):
        z = self.norm(x)
        return self.lin(z) + self.out(self.drop(torch.tanh(self.a(z) * torch.sigmoid(self.g(z)))))


class PatchEnc(nn.Module):
    """PatchTST-style temporal encoder (Nie et al., ICLR 2023): non-overlapping patches of `patch` days -> tokens ->
    Transformer; returns the token sequence like an RNN would (last token = newest patch)."""
    def __init__(self, cin, h, patch=5, layers=2, max_len=60):
        super().__init__()
        self.p = patch; self.emb = nn.Linear(cin * patch, h); self.pos = nn.Parameter(torch.zeros(max_len // patch + 1, h))
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(h, 4, 2 * h, 0.1, batch_first=True, norm_first=True), layers)
        self.norm = nn.LayerNorm(h)
    def forward(self, z):                               # z [B,L,cin] -> ([B,L//p,h], None)
        B, L, C = z.shape; L0 = L - L % self.p
        x = self.emb(z[:, L - L0:].reshape(B, L0 // self.p, self.p * C))
        return self.norm(self.enc(x + self.pos[-x.shape[1]:])), None


class CorrGraph(nn.Module):
    """Peer aggregation over a dynamic price-correlation graph (HIST / THGNN family): each stock attends to its k most
    correlated stocks of the day (60-day daily returns); the update uses the neighbour summary and the stock's deviation from it."""
    def __init__(self, h, k=10):
        super().__init__()
        self.k = k; self.q = nn.Linear(h, h); self.kv = nn.Linear(h, h)
        self.out = nn.Sequential(nn.Linear(2 * h, h), nn.SiLU(), nn.Dropout(0.1), nn.Linear(h, h))
        nn.init.zeros_(self.out[-1].weight); nn.init.zeros_(self.out[-1].bias)   # starts as the plain model
    def forward(self, h, r):                            # h [B,H], r [B,L] daily log returns
        with torch.autocast(device_type=r.device.type, enabled=False):
            r = r.float(); r = r - r.mean(1, keepdim=True); r = r / (r.norm(dim=1, keepdim=True) + 1e-6)
            C = r @ r.T; C.fill_diagonal_(-2.0)
        k = min(self.k, h.shape[0] - 1)
        c, j = C.topk(k, dim=1)                         # [B,k]
        hk = self.kv(h)[j]                              # [B,k,H]
        a = torch.softmax((self.q(h).unsqueeze(1) * hk).sum(-1) / h.shape[-1] ** 0.5 + 5.0 * c, dim=1)
        s = (a.unsqueeze(-1) * hk).sum(1)
        return h + self.out(torch.cat([s, h - s], -1))


class MFTG(nn.Module):
    def __init__(self, d=32, h=64, layers=2, intra=True, cin=5, xs=0, n_ind=0, center=0, mkt=0, nmk=10, head='mlp', rnn='gru', nx=0,
                 patch=5, graph=0, lookback=60, ssl=False, weekly=0):
        super().__init__()
        self.use_intra = intra; self.cin = cin; self.no_daily = False
        self.xs = None
        if xs:      # cross-sectional layer: each stock attends to the other stocks of the same day (+ industry embedding)
            self.iemb = nn.Embedding(n_ind + 1, h)
            self.xs = nn.TransformerEncoder(nn.TransformerEncoderLayer(h, 4, 2 * h, 0.1, batch_first=True, norm_first=True), xs)
        self.intra = IntraEnc(d, layers, cin=cin, center=center) if intra else None
        self.mkt = mkt; self.nx = nx                                   # nx extra per-stock daily inputs (after the 10 bars)
        self.dproj = nn.Linear(10 + nx + (nmk if mkt == 1 else 0), d)  # mkt 1: market features as extra inputs
        if mkt == 2:                                                   # mkt 2: market state gates the stock features (starts open)
            self.mgate = nn.Linear(nmk, 10); nn.init.zeros_(self.mgate.weight); nn.init.zeros_(self.mgate.bias)
        gin = 2 * d + 1 if intra else d
        self.gru = PatchEnc(gin, h, patch, max_len=lookback) if rnn == 'patch' else (nn.LSTM if rnn == 'lstm' else nn.GRU)(gin, h, batch_first=True)
        self.graph = CorrGraph(h, graph) if graph else None
        self.dec = nn.Linear(h, 10) if ssl else None   # next-day daily bar from the GRU state (--pretrain)
        self.L = lookback; self.weekly = weekly         # weekly: number of 5-day bars in the long-horizon branch
        if weekly:
            self.wproj = nn.Linear(10, d); self.wgru = nn.GRU(d, h // 2, batch_first=True); self.wout = nn.Linear(h // 2, h)
            nn.init.zeros_(self.wout.weight); nn.init.zeros_(self.wout.bias)    # starts as the plain model
        self.head = RankGLU(h) if head == 'glu' else \
            nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Dropout(0.1), nn.Linear(h, 1))
    def forward(self, d, it, ind=None):                 # d [B,L,10], it [B,Di,48,5], ind [B] industry id + 1 (0 = none)
        hw = None
        if self.weekly:                                 # d holds 5 * weekly days: weekly bars -> small GRU, main path keeps the last L
            w = d[:, -5 * self.weekly:, :10].reshape(d.shape[0], self.weekly, 5, 10)
            wb = torch.cat([w[..., 0, 0:1], w[..., 1].amax(2, keepdim=True), w[..., 2].amin(2, keepdim=True),
                            w[..., 3].sum(2, keepdim=True)], -1) * 30.0            # first gap, max high, min low, week log return
            ow, _ = self.wgru(self.wproj(torch.cat([wb, w[..., 4:].mean(2)], -1)))
            hw = self.wout(ow[:, -1]); d = d[:, -self.L:]
        B, L, _ = d.shape
        k = 10 + self.nx
        dd = torch.cat([d[..., :4] * 30.0, d[..., 4:k]], -1)
        if self.no_daily: dd = torch.zeros_like(dd)
        if self.mkt == 1: dd = torch.cat([dd, d[..., k:]], -1)
        elif self.mkt == 2: dd = torch.cat([dd[..., :10] * 2 * torch.sigmoid(self.mgate(d[..., k:])), dd[..., 10:]], -1)
        z = self.dproj(dd)
        if self.use_intra:
            Di = it.shape[1]
            e = self.intra(it.reshape(B * Di, 48, self.cin)).reshape(B, Di, -1)
            has = (it[..., 4].abs().sum(-1, keepdim=True) > 0).float()                      # [B,Di,1]
            e = e * has
            pad = torch.zeros(B, L - Di, e.shape[-1] + 1, device=d.device, dtype=e.dtype)
            z = torch.cat([z, torch.cat([pad, torch.cat([e, has], -1)], 1)], -1)
        o, _ = self.gru(z)
        h = o[:, -1]
        if hw is not None: h = h + hw
        if self.graph is not None: h = self.graph(h, d[..., 3])
        if self.xs is not None: h = self.xs((h + self.iemb(ind)).unsqueeze(0))[0]
        return self.head(h).squeeze(-1)

    def ssl_loss(self, d):                              # self-supervised: predict day s+1's 10 bar features from the state at s
        dd = torch.cat([d[..., :4] * 30.0, d[..., 4:10 + self.nx]], -1)
        o, _ = self.gru(self.dproj(dd))
        return ((self.dec(o[:, :-1]).float() - dd[:, 1:, :10].float()) ** 2).mean()


def epoch_days(days, rng, half_life=0):
    """Training days for one epoch: a permutation, or with half_life > 0 (trading days) the same number of days drawn
    with replacement, weight halving every half_life days back from the newest training day."""
    days = np.asarray(days)
    if half_life <= 0: return rng.permutation(days)
    w = 0.5 ** ((days.max() - days) / half_life)
    return rng.choice(days, size=len(days), replace=True, p=w / w.sum())


def corr_loss(p, y, w):
    w = w / w.sum(); pc = p - (w * p).sum(); yc = y - (w * y).sum()
    return -(w * pc * yc).sum() / (torch.sqrt((w * pc * pc).sum() * (w * yc * yc).sum()) + 1e-8)


def rank_ic(p, y):
    return float(np.corrcoef(pd.Series(p).rank().to_numpy(), pd.Series(y).rank().to_numpy())[0, 1])


def run(D, blk, seed, args):
    tr, va, tv, test = D.split(blk, args.stride)
    cand = np.arange(max(D.LX, D.Di), D.T); pre = cand[D.dint[cand] < blk[1]][::args.stride]   # pretraining days (before the test)
    torch.manual_seed(seed); np.random.seed(seed)
    ac = dict(device_type=D.dev.type, dtype=torch.bfloat16, enabled=bool(args.amp))

    @torch.no_grad()
    def predict(model, t, idx):
        model.eval(); out = []
        bs = len(idx) if model.graph is not None else args.pred_batch     # the graph needs the whole day's cross-section
        for s in range(0, len(idx), bs):
            d, it, ind = D.x(t, idx[s:s + bs])
            with torch.autocast(**ac): out.append(model(d, it, ind).float().cpu().numpy())
        return np.concatenate(out)

    def fit(days, epochs, val_days=None):
        model = MFTG(args.d, args.hidden, args.layers, not args.no_intra, cin=8 if args.flow else 5, xs=args.xs,
                    n_ind=int(D.ind.max()) + 1, center=args.intra_center, mkt=args.mkt,
                    nmk=D.mkt.shape[1] if D.mkt is not None else 10, head=args.head, rnn=args.rnn, nx=D.nx,
                    patch=args.patch, graph=args.graph, lookback=args.lookback, ssl=args.pretrain > 0, weekly=args.weekly).to(D.dev)
        model.no_daily = bool(args.no_daily)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        best, best_state, best_ep, bad, hist = -1e9, None, 0, 0, []
        rng = np.random.default_rng(seed)
        for pe in range(args.pretrain):     # features only (no labels): every day before the test window, incl. the label gap
            model.train(); losses = []; te = time.time()
            for t in rng.permutation(pre):
                idx = np.where(D.univ[t])[0]
                if len(idx) < 100: continue
                d, _, _ = D.x(t, idx[rng.choice(len(idx), min(len(idx), args.max_batch), replace=False)])
                with torch.autocast(**ac): loss = model.ssl_loss(d)
                opt.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 3.0); opt.step(); losses.append(float(loss.detach()))
            log(blk[0], 'seed', seed, {'pretrain': pe + 1, 'loss': float(np.mean(losses)), 'sec': round(time.time() - te)})
        for ep in range(epochs):
            model.train(); losses = []; te = time.time()
            for t in epoch_days(days, rng, args.half_life):
                q = D.day(t)
                if 'z' not in q: continue
                n = len(q['idx']); sel = rng.choice(n, min(n, args.max_batch), replace=False)
                d, it, ind = D.x(t, q['idx'][sel])
                w = np.ones(n, np.float32)
                if args.top_w > 0:
                    u = (pd.Series(q['z']).rank().to_numpy() - 0.5) / n; w *= 1.0 + args.top_w * (u >= 0.8)
                if args.tail_w > 0:         # emphasise both tails of the label distribution
                    u = (pd.Series(q['z']).rank().to_numpy() - 0.5) / n; w *= 1.0 + args.tail_w * ((u >= 0.8) | (u < 0.2))
                if args.cap_w > 0:          # emphasise large caps: weight grows with the circulating-cap percentile
                    lm = D.lmv[t, q['idx']]; us = (pd.Series(lm).rank().to_numpy() - 0.5) / n
                    w *= 1.0 + args.cap_w * np.nan_to_num(us, nan=0.5) ** 2
                with torch.autocast(**ac): p = model(d, it, ind)
                yb, wb = torch.from_numpy(q['z'][sel]).to(D.dev), torch.from_numpy(w[sel]).to(D.dev)
                loss = corr_loss(p.float(), yb, wb)
                if args.pool_w > 0:         # ordering inside the model's own top pool (where the long-only optimiser picks)
                    pm = p.detach() >= torch.quantile(p.detach().float(), 1 - args.pool_q)
                    loss = loss + args.pool_w * corr_loss(p.float()[pm], yb[pm], wb[pm])
                opt.zero_grad(set_to_none=True); loss.backward()
                if args.sam > 0:            # sharpness-aware minimisation: gradient at w + rho * g/|g|, applied at w
                    ps = [v for v in model.parameters() if v.grad is not None]
                    gn = torch.norm(torch.stack([v.grad.norm() for v in ps])) + 1e-12
                    eps = [args.sam * v.grad / gn for v in ps]
                    with torch.no_grad():
                        for v, e in zip(ps, eps): v.add_(e)
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast(**ac): p2 = model(d, it, ind)
                    corr_loss(p2.float(), yb, wb).backward()
                    with torch.no_grad():
                        for v, e in zip(ps, eps): v.sub_(e)
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
    ap.add_argument('--seeds', default='7'); ap.add_argument('--label', default='y10'); ap.add_argument('--neutral', type=int, default=0, help='1 = industry+size neutral label, 2 = also beta/volatility/liquidity/momentum')
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
    ap.add_argument('--xs', type=int, default=0, help='cross-sectional attention layers over the stocks of a day (prediction uses the whole day)')
    ap.add_argument('--monthly', type=int, default=0, help='refit every month inside each block (outputs M<yyyymm>_pred.npz)')
    ap.add_argument('--flow', type=int, default=0, help='add the 15s order-flow channels (flow.npy, 2020+)')
    ap.add_argument('--intra_center', type=int, default=0, help='remove the day offset from intraday price channels: 1 = vs last close, 2 = vs mean close')
    ap.add_argument('--half_life', type=float, default=0, help='recency-weighted day sampling, half-life in trading days (0 = uniform)')
    ap.add_argument('--mkt', type=int, default=0, help='market-state features (market.npy): 1 = extra daily inputs, 2 = gate on the stock features')
    ap.add_argument('--head', default='mlp', choices=['mlp', 'glu'], help='score head: mlp or RankGLU')
    ap.add_argument('--rnn', default='gru', choices=['gru', 'lstm', 'patch'], help='temporal encoder (patch = PatchTST-style Transformer)')
    ap.add_argument('--patch', type=int, default=5, help='days per patch for --rnn patch')
    ap.add_argument('--graph', type=int, default=0, help='k neighbours of the 60-day return-correlation graph (0 = off; set --pred_batch above N)')
    ap.add_argument('--pool_w', type=float, default=0, help='weight of the extra IC loss inside the predicted top --pool_q of the batch')
    ap.add_argument('--pool_q', type=float, default=0.3)
    ap.add_argument('--sam', type=float, default=0, help='sharpness-aware minimisation radius rho (0 = plain AdamW)')
    ap.add_argument('--pretrain', type=int, default=0, help='epochs of next-day-bar self-supervised pretraining before the IC loss (GRU, no intraday)')
    ap.add_argument('--weekly', type=int, default=0, help='long-horizon branch over this many 5-day bars (52 = one year; 0 = off)')
    ap.add_argument('--demean', type=int, default=0, help='subtract the day\'s tradable-universe mean from the daily price channels')
    ap.add_argument('--vol_label', type=int, default=0, help='risk-adjusted label: divide returns by the std of the last N daily returns (0 = off)')
    ap.add_argument('--extra', type=int, default=0, help='append the extra per-stock daily inputs (extra.npy: money flow, margin, reports, valuation)')
    a = ap.parse_args()
    out = Path(a.out) / a.name; out.mkdir(parents=True, exist_ok=True)
    (out / 'args.json').write_text(json.dumps(vars(a), indent=1))
    if a.unpack: unpack_dir(a.data, a.unpack)
    if a.flow: unpack_flow(a.data)
    D = Data(a.data, a.label, a.neutral, torch.device(a.device), a.lookback, a.days_intra, bool(a.flow), not a.no_intra, bool(a.mkt), bool(a.extra))
    D.train_start = a.train_start; D.train_pool = a.train_pool; D.vol_label = a.vol_label
    if a.weekly: D.LX = max(a.lookback, 5 * a.weekly)
    if a.demean:
        u = torch.from_numpy(D.univ).to(D.dev).unsqueeze(-1).float()                          # [T,N,1]
        D.mm = (D.daily[..., :4].float() * u).sum(1) / u.sum(1).clamp(min=1)                  # [T,4]
    blocks = [b for b in BLOCKS if b[0] in a.blocks.split(',')]
    if a.monthly: blocks = [m for b in blocks for m in month_blocks(b[0])]
    for blk in blocks:
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
