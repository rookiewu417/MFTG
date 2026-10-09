"""End-to-end track inputs (raw market data only, no factor formulas).

build/mftg/intra.npy  [T, N, 48, 5] fp16  per 5-min bar: ln(open/pc), ln(high/pc), ln(low/pc), ln(close/pc), vol share of day
                                          (pc = previous close; missing bars carry the last close, vol share 0; no data = 0)
build/mftg/daily.npy  [T, N, 10] fp16     ln(o/pc), ln(h/pc), ln(l/pc), ln(c/pc), z(ln amount), z(ln turnover_f), z(ln circ_mv),
                                          close at up limit, close at down limit, traded
Axes = build/shared/axes.npz (same panel the Weave project uses), so labels / universe / windows are shared.
"""
import sys
import multiprocessing as mp
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'shared'))
from common import OUT, LAKE, con, load_axes, to_panel, SHARED

E = OUT / 'mftg'; E.mkdir(exist_ok=True)
B = SHARED / 'base'
dates, codes = load_axes(); T, N = len(dates), len(codes)
HM = [f'{h:02d}:{m:02d}' for h, m in [(9, 35 + 5 * i) if 35 + 5 * i < 60 else (10, 35 + 5 * i - 60) for i in range(5)]]
HM = pd.date_range('2000-01-01 09:35', '2000-01-01 11:30', freq='5min').strftime('%H:%M').tolist() + \
     pd.date_range('2000-01-01 13:05', '2000-01-01 15:00', freq='5min').strftime('%H:%M').tolist()
assert len(HM) == 48
BAR = {h: i for i, h in enumerate(HM)}


def daily_raw():
    df = con().sql(f"""select trade_date, ts_code, open, high, low, close, pre_close, vol from read_parquet('{LAKE}/daily/price/*/*.parquet', hive_partitioning=false)
                       where trade_date >= '{dates[0].date()}' and trade_date <= '{dates[-1].date()}'""").df()
    return to_panel(df, dates, codes, ['open', 'high', 'low', 'close', 'pre_close', 'vol'], dtype=np.float64)


def month_job(args):
    path, = args
    df = con().sql(f"select ts_code, trade_time, open, high, low, close, vol from read_parquet('{path}')").df()
    return path, write_bars(df)


def bars_from_15s(day):
    """5-min bars rebuilt from the 15s snapshot bars (start-labelled; 09:25 auction -> first bar, 11:30:00 -> 11:30 bar,
    15:00 closing auction -> last bar). Matches bars_5min with corr > 0.999 on prices and volume shares."""
    f = con().sql(f"select ts_code, trade_time, open, high, low, close, vol from read_parquet('{LAKE}/flow_bars/15s/date={day}/*.parquet')").df()
    if f.empty: return f
    D = pd.Timestamp(day); t = f.trade_time
    bar = t.dt.floor('5min') + pd.Timedelta('5min')
    bar = bar.where(t != D + pd.Timedelta('11h30min'), D + pd.Timedelta('11h30min'))
    bar = bar.where(bar > D + pd.Timedelta('9h35min'), D + pd.Timedelta('9h35min'))
    bar = bar.where(~((bar > D + pd.Timedelta('11h30min')) & (bar <= D + pd.Timedelta('13h05min'))), D + pd.Timedelta('13h05min'))
    bar = bar.where(bar <= D + pd.Timedelta('15h'), D + pd.Timedelta('15h'))
    f['trade_time'] = bar
    f = f.sort_values(['ts_code', 'trade_time'], kind='stable')
    return f.groupby(['ts_code', 'trade_time']).agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'),
                                                     close=('close', 'last'), vol=('vol', 'sum')).reset_index()


def write_bars(df):
    if df.empty: return 0
    d = pd.to_datetime(df.trade_time.dt.date)
    ti = pd.Index(dates).get_indexer(d); ji = pd.Index(codes).get_indexer(df.ts_code)
    bi = df.trade_time.dt.strftime('%H:%M').map(BAR).fillna(-1).astype(int).to_numpy()
    ok = (ti >= 0) & (ji >= 0) & (bi >= 0)
    df, ti, ji, bi = df[ok], ti[ok], ji[ok], bi[ok]
    if not len(df): return 0
    pc = PC[ti, ji]
    keys = np.unique(ti * N + ji); kpos = np.searchsorted(keys, ti * N + ji)
    X = np.full((len(keys), 48, 5), np.nan, np.float64)
    with np.errstate(all='ignore'):
        for c, name in enumerate(['open', 'high', 'low', 'close']):
            X[kpos, bi, c] = np.log(df[name].to_numpy() / pc)
        X[kpos, bi, 4] = df.vol.to_numpy()
    # carry the last close into missing bars (flat bar, zero volume); leading missing bars use 0 (= previous close)
    miss = np.isnan(X[:, :, 3])
    cl = pd.DataFrame(X[:, :, 3]).ffill(axis=1).fillna(0.0).to_numpy()
    for c in range(4): X[:, :, c] = np.where(miss, cl, X[:, :, c])
    v = np.nan_to_num(X[:, :, 4]); tot = v.sum(1, keepdims=True)
    X[:, :, 4] = np.where(tot > 0, v / np.maximum(tot, 1e-9), 0.0)
    X = np.nan_to_num(np.clip(X, -0.5, 1.0))
    mm = np.load(E / 'intra.npy', mmap_mode='r+')
    kt, kj = keys // N, keys % N
    for t in np.unique(kt):
        s = kt == t
        mm[t, kj[s]] = X[s].astype(np.float16)
    mm.flush(); del mm
    return len(keys)


def init(pc):
    global PC; PC = pc


if __name__ == '__main__':
    raw = daily_raw()
    pc = raw['pre_close']
    if '--daily_only' not in sys.argv and (not (E / 'intra.npy').exists() or '--force' in sys.argv):
        np.lib.format.open_memmap(E / 'intra.npy', mode='w+', dtype=np.float16, shape=(T, N, 48, 5))
        lo, hi = dates[0], dates[-1]
        files = sorted(p for p in (LAKE / 'minute' / 'bars_5min').glob('year=*/month=*/data.parquet')
                       if pd.Timestamp(int(p.parts[-3][5:]), int(p.parts[-2][6:]), 1) >= lo.replace(day=1) and pd.Timestamp(int(p.parts[-3][5:]), int(p.parts[-2][6:]), 1) <= hi)
        print('months', len(files), flush=True)
        with mp.get_context('fork').Pool(6, initializer=init, initargs=(pc,)) as pool:
            for path, n in pool.imap_unordered(month_job, [(str(f),) for f in files]):
                print(Path(path).parts[-3:-1], n, flush=True)
    if '--fill15s' in sys.argv:          # days missing from bars_5min: rebuild their bars from the 15s snapshot bars
        init(pc)
        it = np.load(E / 'intra.npy', mmap_mode='r')
        todo = [t for t in range(T) if dates[t] >= pd.Timestamp('2020-01-01') and not np.abs(it[t, :, :, 4]).sum() > 0
                and (LAKE / 'flow_bars' / '15s' / f'date={dates[t].date()}').exists()]
        del it
        for t in todo: print('15s ->', dates[t].date(), write_bars(bars_from_15s(str(dates[t].date()))), flush=True)
    # ---- daily raw panel
    traded = np.load(B / 'traded.npy')
    with np.errstate(all='ignore'):
        r = [np.log(raw[k] / pc) for k in ['open', 'high', 'low', 'close']]
        def csz(x):
            x = np.where(traded & np.isfinite(x), x, np.nan)
            return (x - np.nanmean(x, 1, keepdims=True)) / (np.nanstd(x, 1, keepdims=True) + 1e-9)
        amt = csz(np.log(np.load(B / 'amount.npy'))); tov = csz(np.log(np.load(B / 'ind_turnover_rate_f.npy'))); mv = csz(np.load(B / 'log_circ_mv.npy'))
        cr = np.load(B / 'close_raw.npy'); up = np.load(B / 'up_limit.npy'); dn = np.load(B / 'down_limit.npy')
        uph = (cr >= up - 0.005) & np.isfinite(up); dnh = (cr <= dn + 0.005) & np.isfinite(dn)
    D = np.stack([*[np.clip(x, -0.5, 1.0) for x in r], amt, tov, mv, uph, dnh, traded], -1).astype(np.float32)
    D[~traded] = 0.0
    D = np.nan_to_num(D).astype(np.float16)
    np.save(E / 'daily.npy', D)
    if '--daily_only' in sys.argv: print('daily', D.shape); sys.exit()
    it = np.load(E / 'intra.npy', mmap_mode='r')
    cov = [(traded[t].sum(), (np.abs(it[t, :, :, 4]).sum(-1) > 0).sum()) for t in range(0, T, 100)]
    print('daily', D.shape, 'coverage (traded, has intraday) every 100 days:', cov)
