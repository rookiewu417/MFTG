"""Style exposures for style-neutral labels: build/mftg/styles.npy [T, N, 4] fp16 = beta60, volatility (vol20),
liquidity (turnover20), momentum (ret240-20), winsorised (1/99 pct) and z-scored per day on the tradable universe.
The definitions mirror the style block of risk.py in the portfolio code (risk.py); the two sides share data files, not code."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'shared'))
from common import OUT, load_axes, SHARED

B = SHARED / 'base'
dates, codes = load_axes(); T, N = len(dates), len(codes)
univ = np.load(B / 'univ.npy')
lc = pd.DataFrame(np.log(np.load(B / 'aclose.npy'))).ffill().to_numpy()
def lag(a, h): return np.vstack([np.full((h, N), np.nan), a[:-h]])
r1 = lc - lag(lc, 1)
R1 = pd.DataFrame(r1); M = pd.Series(np.nanmean(np.where(univ, r1, np.nan), axis=1))
rm = R1.sub(R1.rolling(60, min_periods=30).mean()); mm = M - M.rolling(60, min_periods=30).mean()
with np.errstate(all='ignore'):
    STY = {'beta': rm.mul(mm, axis=0).rolling(60, min_periods=30).mean().div((mm ** 2).rolling(60, min_periods=30).mean(), axis=0).to_numpy(),
           'volatility': R1.rolling(20, min_periods=10).std().to_numpy(),
           'liquidity': np.log(pd.DataFrame(np.load(B / 'ind_turnover_rate_f.npy')).rolling(20, min_periods=10).mean().to_numpy()),
           'momentum': (lc - lag(lc, 240)) - (lc - lag(lc, 20))}

NAMES = ['beta', 'volatility', 'liquidity', 'momentum']
out = np.zeros((T, N, len(NAMES)), np.float16)
for k, name in enumerate(NAMES):
    X = STY[name]
    for t in range(T):
        x = np.where(univ[t] & np.isfinite(X[t]), X[t], np.nan)
        if np.isfinite(x).sum() < 100: continue
        lo, hi = np.nanpercentile(x, [1, 99]); x = np.clip(x, lo, hi)
        out[t, :, k] = np.nan_to_num((x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)).astype(np.float16)
np.save(OUT / 'mftg' / 'styles.npy', out)
print('styles', out.shape, 'nonzero share on univ (last day)', [round(float((out[-1, univ[-1], k] != 0).mean()), 3) for k in range(len(NAMES))])
