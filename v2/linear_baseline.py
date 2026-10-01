"""Does ML add value over a linear composite of the same 278 features?

Per window, weights come only from dates whose 10-day label ends by that window's validation cutoff (same cutoff the
ML models used for refit): 'icw' = mean rank IC per feature; 'topw' = mean top-decile excess per feature (long-side).
Scores are written in the same npz format as the ML runs, so evaluate.py / backtest.py treat them identically.
"""
import json
import numpy as np
import pandas as pd
from common import OUT, load_axes

BLOCKS = [('2022H2', 20220701, 20221231, 20220630), ('2023H1', 20230101, 20230630, 20221231), ('2023H2', 20230701, 20231231, 20230630),
          ('2024H1', 20240101, 20240630, 20231231), ('2024H2', 20240701, 20241231, 20240628), ('2025H1', 20250101, 20250630, 20241231),
          ('2025H2', 20250701, 20251231, 20250630), ('2026H1', 20260101, 20260630, 20251231), ('2026H2', 20260701, 20260731, 20260630)]
P = OUT / 'pack'
dates, codes = load_axes(); dint = dates.strftime('%Y%m%d').astype(int).to_numpy(); T = len(dates)
g = np.load(P / 'grid.npz'); grid, sparse = g['grid'], g['on_sparse']
tab = np.load(P / 'tab.npy', mmap_mode='r'); cols = json.loads((P / 'tab_cols.json').read_text()); K = len(cols)
univ = np.load(P / 'univ.npy'); y10 = np.load(P / 'y10.npy')
lend = np.array([dint[min(t + 11, T - 1)] if t + 11 < T else 99999999 for t in grid])

# per grid date, per feature: rank IC and top-decile excess vs the 10-day label (training statistics)
ic = np.full((len(grid), K), np.nan, np.float32); tx = np.full((len(grid), K), np.nan, np.float32)
for gi, t in enumerate(grid):
    if not sparse[gi] or lend[gi] > 20260630: continue
    m = univ[t] & np.isfinite(y10[t]); j = np.where(m)[0]
    if len(j) < 200: continue
    x = tab[gi][j].astype(np.float32); x[x == 255] = np.nan
    yr = pd.Series(y10[t, j]).rank().to_numpy(); yc = yr - yr.mean(); y = y10[t, j]
    for k in range(K):
        v = x[:, k]; ok = np.isfinite(v)
        if ok.sum() < 200: continue
        a = v[ok] - v[ok].mean(); b = yc[ok] - yc[ok].mean()
        ic[gi, k] = (a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum())
        thr = np.quantile(v[ok], 0.9); tx[gi, k] = np.nanmean(y[ok][v[ok] >= thr]) - np.nanmean(y[ok])
    if gi % 100 == 0: print('stats', gi, len(grid), flush=True)

for kind, stat in (('icw', ic), ('topw', tx)):
    out = OUT.parent / 'runs' / 'linear' / kind; out.mkdir(parents=True, exist_ok=True)
    for name, fs, fe, cut in BLOCKS:
        w = np.nanmean(stat[lend <= cut], axis=0); w = np.nan_to_num(w)
        # long-side variant: only features whose top decile beat the universe, weight by that edge (both directions via sign)
        test = np.where((dint[grid] >= fs) & (dint[grid] <= fe))[0]
        preds, fd, st = [], [], []
        for gi in test:
            t = grid[gi]; j = np.where(univ[t])[0]
            x = tab[gi][j].astype(np.float32); x[x == 255] = 126.5
            s = ((x - 126.5) / 254.0) @ w
            preds.append(s); fd.append(np.full(len(j), dint[t])); st.append(j)
        st = np.concatenate(st)
        np.savez(out / f'{name}_pred.npz', pred=np.concatenate(preds).astype(np.float32), feature_date=np.concatenate(fd).astype(np.int32),
                 stock=st.astype(np.int32), stock_code=codes[st].astype('<U12'))
    print('written', kind, flush=True)
