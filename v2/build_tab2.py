"""pack2 = pack + industry-relative versions of the 40 strongest features (by selection-period |ICIR|, 2020-2023H1)
+ market-state features (time-series percentile over the trailing 250 trading days, same value for every stock that day).
Writes build/pack2 and split upload parts build/upload2/."""
import json, shutil, subprocess
import numpy as np
import pandas as pd
from common import OUT, LAKE, con, load_axes

P, P2 = OUT / 'pack', OUT / 'pack2'
if P2.exists(): shutil.rmtree(P2)
P2.mkdir()
dates, codes = load_axes(); T, N = len(dates), len(codes)
g = np.load(P / 'grid.npz'); grid = g['grid']; G = len(grid)
cols = json.loads((P / 'tab_cols.json').read_text())
sel = json.loads((OUT / 'selected_features.json').read_text())
top40 = sorted(sel, key=lambda k: -abs(sel[k]['icir']))[:40]
tab = np.load(P / 'tab.npy', mmap_mode='r')
univ = np.load(OUT / 'base' / 'univ.npy'); ind = np.load(OUT / 'base' / 'industry.npy')

# ---- industry-relative: rank within (date, industry) among universe names
rel = np.full((G, N, len(top40)), 255, np.uint8)
for gi, t in enumerate(grid):
    u = univ[t]; ii = ind[t]
    for fk, f in enumerate(top40):
        x = tab[gi, :, cols.index(f)].astype(np.float32); x[(x == 255) | ~u | (ii < 0)] = np.nan
        s = pd.Series(x); r = s.groupby(ii).rank(pct=True)
        v = r.to_numpy(); ok = np.isfinite(v)
        rel[gi, ok, fk] = np.minimum((v[ok] * 254 - 0.5).clip(0).astype(np.int32), 253)
print('industry-relative done', flush=True)

# ---- market state (PIT: percentile of today's value within the previous 250 trading days)
B = OUT / 'base'
aclose = np.load(B / 'aclose.npy'); lc = pd.DataFrame(np.log(aclose)).ffill()
r1 = lc.diff().where(univ)
lmv = np.load(B / 'log_circ_mv.npy'); amt = pd.DataFrame(np.load(B / 'amount.npy'))
up_hit = np.load(B / 'seq.npy', mmap_mode='r')[:, :, 7]
mk = {}
ew = r1.mean(1)
for h in (5, 20, 60): mk[f'm_ret{h}'] = ew.rolling(h).sum()
small = pd.DataFrame(np.where(lmv <= np.nanpercentile(lmv, 30, axis=1, keepdims=True), 1.0, np.nan))
big = pd.DataFrame(np.where(lmv >= np.nanpercentile(lmv, 70, axis=1, keepdims=True), 1.0, np.nan))
smb = (r1 * small).mean(1) - (r1 * big).mean(1)
mk['m_smb20'] = smb.rolling(20).sum(); mk['m_smb5'] = smb.rolling(5).sum()
mk['m_disp20'] = r1.std(1).rolling(20).mean()
mk['m_breadth20'] = (r1 > 0).where(univ).mean(1).rolling(20).mean()
tot_amt = amt.where(univ).sum(1); mk['m_amt_ratio'] = tot_amt.rolling(5).mean() / tot_amt.rolling(60).mean()
mk['m_limitup5'] = pd.Series(np.nanmean(np.where(univ, np.asarray(up_hit, dtype=np.float32), np.nan), axis=1)).rolling(5).mean()
idx = con().sql(f"""select trade_date, close from read_parquet('{LAKE}/index/daily_000300/*/*/*.parquet', hive_partitioning=false)""").df()
i300 = pd.Series(idx.close.to_numpy(), index=pd.to_datetime(idx.trade_date)).sort_index().reindex(dates).ffill()
mk['m_csi300_ret20'] = np.log(i300).diff(20).reset_index(drop=True)
mkt = np.full((G, N, len(mk)), 255, np.uint8)
for k, (name, s) in enumerate(mk.items()):
    s = pd.Series(np.asarray(s, dtype=float))
    pct = s.rolling(250, min_periods=120).apply(lambda w: (w[:-1] < w[-1]).mean() if np.isfinite(w[-1]) else np.nan, raw=True)
    for gi, t in enumerate(grid):
        v = pct.iloc[t]
        if np.isfinite(v): mkt[gi, univ[t], k] = min(int(v * 253), 253)
print('market state done', list(mk), flush=True)

full = np.concatenate([np.asarray(tab), rel, mkt], axis=2)
np.save(P2 / 'tab.npy', full); del full
(P2 / 'tab_cols.json').write_text(json.dumps(cols + [f'r_{f}' for f in top40] + list(mk)))
for f in ['axes.npz', 'grid.npz', 'meta.json']: shutil.copy(P / f, P2 / f)
U = OUT / 'upload2'
if U.exists(): shutil.rmtree(U)
U.mkdir()
subprocess.run(f"tar -C {OUT} -cf - pack2/tab.npy pack2/tab_cols.json | pigz -3 -p 12 | split -b 60m -d -a 3 - {U}/pack2.tgz.part", shell=True, check=True)
print(subprocess.run(f'ls {U} | wc -l; du -sh {U}', shell=True, capture_output=True, text=True).stdout)
