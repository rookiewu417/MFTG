"""Add open->open labels y<h> to build/shared/base without rebuilding the other panels.

Same definition as build_base.py: y_h = open(t+1) -> open(t+1+h) on adjusted opens, NaN unless buyable at the t+1 open
(traded and not opened at the up limit). Before writing, y5 and y10 are recomputed and must match the saved files.
Usage: python shared/add_labels.py 20 [30 ...]
"""
import sys
import numpy as np
import pandas as pd
from common import LAKE, CAL_START, CAL_END, con, to_panel, load_axes, SHARED

B = SHARED / 'base'
dates, codes = load_axes(); c = con()
px = c.sql(f"""select trade_date, ts_code, open, close, vol from read_parquet('{LAKE}/daily/price/*/*.parquet', hive_partitioning=false)
               where trade_date between '{CAL_START}' and '{CAL_END}' and ts_code not like '%.BJ'""").df()
P = to_panel(px, dates, codes, ['open', 'close', 'vol']); del px
adj = to_panel(c.sql(f"""select trade_date, ts_code, adj_factor from read_parquet('{LAKE}/daily/adj_factor/*/*.parquet', hive_partitioning=false)
                          where trade_date between '{CAL_START}' and '{CAL_END}'""").df(), dates, codes, ['adj_factor'])['adj_factor']
adj = pd.DataFrame(adj).ffill().bfill().to_numpy(np.float32)
up = to_panel(c.sql(f"""select trade_date, ts_code, up_limit from read_parquet('{LAKE}/daily/limit/*/*.parquet', hive_partitioning=false)
                         where trade_date between '{CAL_START}' and '{CAL_END}'""").df(), dates, codes, ['up_limit'])['up_limit']
traded = np.isfinite(P['close']) & (np.nan_to_num(P['vol']) > 0) & np.isfinite(P['open']) & (np.nan_to_num(P['open']) > 0)
aopen = np.where(traded, P['open'] * adj, np.nan)

def shift(a, k):
    out = np.full_like(a, np.nan if a.dtype.kind == 'f' else False)
    if k > 0: out[:-k] = a[k:]
    return out

up_hit_open = np.nan_to_num(P['open']) >= np.nan_to_num(up, nan=np.inf) - 0.005
buyable = shift(traded & ~up_hit_open, 1).astype(bool)
def fwd(h):
    r = shift(aopen, 1 + h) / shift(aopen, 1) - 1
    return np.where(buyable, r, np.nan).astype(np.float32)

for h in (5, 10):
    old = np.load(B / f'y{h}.npy'); new = fwd(h)
    same = np.array_equal(np.isnan(old), np.isnan(new)) and np.allclose(old[np.isfinite(old)], new[np.isfinite(new)], rtol=1e-5, atol=1e-6)
    assert same, f'y{h} recomputed from the lake differs from the saved panel; rebuild base instead'
    print(f'y{h} matches the saved panel')
for h in map(int, sys.argv[1:]):
    y = fwd(h); np.save(B / f'y{h}.npy', y); print(f'wrote y{h}', y.shape, 'finite share', round(float(np.isfinite(y).mean()), 4))
