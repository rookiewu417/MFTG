"""End-to-end track: order-flow channels from the 15-second bars (2020+), aggregated to the 48 five-minute bars.

build/mftg/flow.npy [T, N, 48, 2] fp16:
  0: active volume imbalance (buy_vol - sell_vol) / (buy_vol + sell_vol) in the bar (0 if no volume)
  1: ln(average trade size in the bar / average trade size of the day) (0 if no trades)
The 09:25 call auction is folded into the first bar and the 15:00 closing auction into the last. Days before 2020 = 0.
"""
import sys
import multiprocessing as mp
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'shared'))
from common import OUT, LAKE, con, load_axes, SHARED

E = OUT / 'mftg'
dates, codes = load_axes(); T, N = len(dates), len(codes)
CPOS = pd.Index(codes)


def job(args):
    t, path = args
    df = con().sql(f"""
        with b as (
          select ts_code,
                 case when hour(trade_time) < 12 then least(greatest((hour(trade_time) * 60 + minute(trade_time) - 570) // 5, 0), 23)
                      else 24 + least(greatest((hour(trade_time) * 60 + minute(trade_time) - 780) // 5, 0), 23) end as bar,
                 buy_volume, sale_volume, amount, transactions
          from read_parquet('{path}'))
        select ts_code, bar, sum(buy_volume) bv, sum(sale_volume) sv, sum(amount) amt, sum(transactions) n
        from b group by 1, 2""").df()
    j = CPOS.get_indexer(df.ts_code); ok = j >= 0; df = df[ok]; j = j[ok]
    X = np.zeros((N, 48, 2), np.float32)
    with np.errstate(all='ignore'):
        tot = df.bv + df.sv
        X[j, df.bar.to_numpy(), 0] = np.where(tot > 0, (df.bv - df.sv) / tot, 0.0)
        size = df.amt / df.n
        day = df.assign(j=j).groupby('j')[['amt', 'n']].transform('sum')
        rel = np.log(size / (day.amt / day.n))
        X[j, df.bar.to_numpy(), 1] = np.where(np.isfinite(rel), np.clip(rel, -5, 5), 0.0)
    mm = np.load(E / 'flow.npy', mmap_mode='r+'); mm[t] = X.astype(np.float16); mm.flush(); del mm
    return t, len(df)


if __name__ == '__main__':
    np.lib.format.open_memmap(E / 'flow.npy', mode='w+', dtype=np.float16, shape=(T, N, 48, 2))
    tpos = {d.strftime('%Y-%m-%d'): i for i, d in enumerate(dates)}
    jobs = []
    for p in sorted((LAKE / 'flow_bars' / '15s').glob('date=*')):
        d = p.name[5:]
        if d in tpos: jobs.append((tpos[d], str(p / 'data.parquet')))
    print('days', len(jobs), flush=True)
    n = 0
    with mp.get_context('fork').Pool(8) as pool:
        for t, k in pool.imap_unordered(job, jobs):
            n += 1
            if n % 100 == 0: print(n, dates[t].date(), k, flush=True)
    print('done')
