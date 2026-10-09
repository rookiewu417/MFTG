"""Shared paths and panel helpers for the dataset build (runs locally, next to the data lake)."""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb

LAKE = Path('/home/ivenwu/projects/data/ashare')
CERES = Path('/home/ivenwu/projects/ceres/data')
CATALOG = Path('/home/ivenwu/projects/ceres/catalog')
OUT = Path(os.environ.get('MFTG_OUT', '/home/ivenwu/projects/MFTG/build'))   # build/mftg/long/ holds the MFTG long history
SHARED = OUT / 'shared'   # base/ and axes.npz: panels every line uses

CAL_START = os.environ.get('MFTG_CAL_START', '2018-07-01')   # 6 months of look-back before the first sequence sample (2019-01)
CAL_END = '2026-09-10'     # last day with daily price in the lake
TAB_START = '2020-01-02'   # Ceres factor coverage starts here
GRID_EVERY = 2             # tabular rows are built on every 2nd trading day ...
DENSE_START = '2024-07-01' # ... and on every trading day from the first test block on


def con():
    c = duckdb.connect()
    c.execute('SET threads TO 12')
    return c


def load_axes():
    d = np.load(SHARED / 'axes.npz', allow_pickle=True)
    return pd.DatetimeIndex(d['dates']), d['codes']


def to_panel(df, dates, codes, value_cols, date_col='trade_date', code_col='ts_code', dtype=np.float32):
    """Scatter a long frame into [T, N] (or [T, N, K]) arrays; unmatched rows are dropped, duplicates keep last."""
    di = pd.Index(dates).get_indexer(pd.to_datetime(df[date_col]))
    ci = pd.Index(codes).get_indexer(df[code_col])
    ok = (di >= 0) & (ci >= 0)
    out = {}
    for c in value_cols:
        a = np.full((len(dates), len(codes)), np.nan, dtype)
        a[di[ok], ci[ok]] = df[c].to_numpy()[ok]
        out[c] = a
    return out
