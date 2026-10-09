"""Build the report reproduction's FullBase inputs for the early windows (2022H2-2024H1), using the reproduction's own
15-minute aggregation of the 1-minute bars (copied from baseline_src/reproduce.py: aggregate_month / FEATS / adj scaling).

out/full_base/minute_base.float32 [T, N, 16, 6] + meta.json (shape, dates, codes); out/data/daily/post_weighted.parquet
(subset) and out/data/meta/stock_basic.parquet, so FullBase (fullbase_dataset.py, patched to read axes from meta.json) works.
"""
import json, re, sys, time
import multiprocessing as mp
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl

LAKE = Path('/home/ivenwu/projects/data/ashare'); OUT = Path('/home/ivenwu/projects/MFTG/build/repro')
FEATS = ["open", "high", "low", "close", "vwap", "vol"]
START, END = '2016-01-04', '2024-07-31'


def aggregate_month(path, codes):          # verbatim logic of baseline_src/reproduce.py::aggregate_month
    lf = pl.read_parquet(path, columns=["ts_code", "trade_time", "open", "high", "low", "close", "vol", "amount"])
    lf = lf.filter(pl.col("ts_code").is_in(list(codes)))
    lf = lf.with_columns([pl.col("trade_time").dt.date().alias("trade_date"),
      (pl.col("trade_time").dt.hour().cast(pl.Int32) * 60 + pl.col("trade_time").dt.minute().cast(pl.Int32)).alias("minute")])
    lf = lf.with_columns(pl.when((pl.col("minute") >= 571) & (pl.col("minute") <= 690)).then((pl.col("minute") - 571) // 15)
       .when((pl.col("minute") >= 781) & (pl.col("minute") <= 900)).then(8 + (pl.col("minute") - 781) // 15)
       .otherwise(-1).alias("slot")).filter(pl.col("slot") >= 0)
    out = (lf.sort(["ts_code", "trade_time"]).group_by(["ts_code", "trade_date", "slot"], maintain_order=False)
      .agg([pl.col("open").first().alias("open"), pl.col("high").max().alias("high"),
        pl.col("low").min().alias("low"), pl.col("close").last().alias("close"),
        pl.col("vol").sum().alias("vol"), pl.col("amount").sum().alias("amount")])
      .with_columns((pl.col("amount") / pl.col("vol").replace(0, None)).alias("vwap"))
      .select(["ts_code", "trade_date", "slot"] + FEATS))
    return out.to_pandas()


def job(path):
    x = aggregate_month(path, CODES)
    x["trade_date"] = pd.to_datetime(x.trade_date); x = x[x.trade_date.isin(DIDX)]
    if not len(x): return path, 0
    ii = x.trade_date.map(DIDX).to_numpy(); jj = x.ts_code.map(CIDX).to_numpy(); kk = x.slot.to_numpy()
    vals = x[FEATS].to_numpy(np.float32); vals[:, :5] *= np.nan_to_num(ADJ[ii, jj, None], nan=1.0)
    mm = np.memmap(OUT / 'full_base/minute_base.float32', dtype='float32', mode='r+', shape=SHAPE)
    mm[ii, jj, kk, :] = vals; mm.flush(); del mm
    return path, len(x)


if __name__ == '__main__':
    (OUT / 'full_base').mkdir(parents=True, exist_ok=True); (OUT / 'data/daily').mkdir(parents=True, exist_ok=True); (OUT / 'data/meta').mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(LAKE / 'daily/post_weighted.parquet', columns=['trade_date', 'ts_code'] + FEATS + ['adj_factor'])
    d['trade_date'] = pd.to_datetime(d.trade_date)
    dates = pd.DatetimeIndex(sorted(d.trade_date.unique())); dates = dates[(dates >= START) & (dates <= END)]
    sb = pd.read_parquet(LAKE / 'meta/stock_basic.parquet', columns=['ts_code'])
    codes = np.array(sorted(set(sb.ts_code.dropna()) | set(d.ts_code.dropna().unique())), dtype=object)
    SHAPE = (len(dates), len(codes), 16, 6); print('shape', SHAPE, flush=True)
    sub = d[(d.trade_date >= dates[0] - pd.Timedelta(days=120)) & (d.trade_date <= dates[-1])]
    sub.assign(trade_date=sub.trade_date.dt.date).to_parquet(OUT / 'data/daily/post_weighted.parquet', index=False)
    sb.to_parquet(OUT / 'data/meta/stock_basic.parquet', index=False)
    (OUT / 'full_base/meta.json').write_text(json.dumps({'shape': SHAPE, 'dates': [str(x.date()) for x in dates], 'codes': list(codes)}))
    dd = sub[sub.trade_date.isin(dates)]
    ADJ = dd.pivot_table(index='trade_date', columns='ts_code', values='adj_factor', aggfunc='last').reindex(index=dates, columns=codes).to_numpy(np.float32)
    CODES = set(codes); CIDX = {c: i for i, c in enumerate(codes)}; DIDX = {x: i for i, x in enumerate(dates)}
    mm = np.memmap(OUT / 'full_base/minute_base.float32', dtype='float32', mode='w+', shape=SHAPE); mm[:] = np.nan; mm.flush(); del mm
    paths = []
    for p in sorted((LAKE / 'minute/bars_1min').glob('year=*/month=*/data.parquet')):
        ym = re.search(r"year=(\d{4})/month=(\d{2})", str(p)); pm = pd.Timestamp(ym.group(1) + "-" + ym.group(2) + "-01")
        if pm + pd.offsets.MonthEnd(1) >= dates[0] and pm <= dates[-1]: paths.append(str(p))
    print('months', len(paths), flush=True); t0 = time.time()
    for path in paths:          # sequential: polars is multi-threaded itself and deadlocks in forked workers
        p, n = job(path); print(re.search(r"year=\d+/month=\d+", p).group(0), n, f'{time.time()-t0:.0f}s', flush=True)
    print('done')
