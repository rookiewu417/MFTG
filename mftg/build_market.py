"""Market-state features per day (same for every stock) -> build/mftg/market.npy [T, 10] float32.

All known at the close of day t: index log returns (CSI300 / CSI500 / CSI1000), equal-weight universe return,
cross-sectional dispersion, breadth, up/down-limit shares, total amount vs its trailing 20-day mean, trailing 20-day
equal-weight return and volatility. Scaled to roughly unit size. Used by train_mftg.py --mkt.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'shared'))
from common import LAKE, OUT, con, load_axes, SHARED

B = SHARED / 'base'; E = OUT / 'mftg'
dates, codes = load_axes(); T = len(dates)
D = np.load(E / 'daily.npy').astype(np.float32)          # [T,N,10]: ln O/H/L/C vs prev close, amt, tov, mv (csz), up, dn, traded
traded = D[..., 9] > 0
r = np.where(traded, D[..., 3], np.nan)
ix = con().sql(f"select ts_code, trade_date, close, pre_close from '{LAKE}/index/price.parquet'").df()
ix['trade_date'] = pd.to_datetime(ix.trade_date)
ixr = {c: g.set_index('trade_date').reindex(dates).pipe(lambda x: np.log(x.close / x.pre_close)).to_numpy() for c, g in ix.groupby('ts_code')}
amt = np.load(B / 'amount.npy'); tot = np.log(np.nansum(np.where(traded, amt, 0), 1) + 1.0)
with np.errstate(all='ignore'):
    ew = np.nanmean(r, 1)
    s = pd.Series(ew)
    F = np.stack([
        30 * ixr['000300.SH'], 30 * ixr['000905.SH'], 30 * ixr['000852.SH'],
        30 * ew,
        30 * np.nanstd(r, 1),
        4 * ((np.nan_to_num(r) > 0).sum(1) / np.maximum(traded.sum(1), 1) - 0.5),   # share of traded stocks up
        10 * np.nanmean(np.where(traded, D[..., 7], np.nan), 1),
        10 * np.nanmean(np.where(traded, D[..., 8], np.nan), 1),
        tot - pd.Series(tot).rolling(20, min_periods=5).mean().to_numpy(),
        10 * s.rolling(20, min_periods=5).sum().to_numpy(),
    ], 1)
F = np.nan_to_num(np.clip(F, -5, 5)).astype(np.float32)
np.save(E / 'market.npy', F)
print('market', F.shape, 'mean', F.mean(0).round(2), 'std', F.std(0).round(2))
