"""Tabular features on the grid dates (every GRID_EVERY trading days from TAB_START).

Each feature is cross-sectionally ranked over the day's universe and stored as uint8 codes 0..253 (255 = missing).
Feature selection uses ONLY feature dates whose 10-day label ends by 2023-06-30 (the first test block's training cutoff).
"""
import json, os, sys, time, glob
from multiprocessing import Pool
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from common import CERES, CATALOG, OUT, TAB_START, GRID_EVERY, DENSE_START, LAKE, load_axes, con, to_panel

t0 = time.time()
def log(*a): print(f'[{time.time()-t0:6.0f}s]', *a, flush=True)

B = OUT / 'base'; F = OUT / 'fac'; F.mkdir(parents=True, exist_ok=True)
dates, codes = load_axes()
T, N = len(dates), len(codes)
_g0 = dates.searchsorted(pd.Timestamp(TAB_START)); _d = dates.searchsorted(pd.Timestamp(DENSE_START))
grid = np.unique(np.concatenate([np.arange(_g0, T, GRID_EVERY), np.arange(_d, T)]))   # every 2nd day, daily from DENSE_START
on_sparse = (grid - _g0) % GRID_EVERY == 0   # training rows use only the regular every-2nd-day grid
G = len(grid)
SEL_END = 20230630


def encode(panel_g, univ_g):
    """[G, N] float -> uint8 cross-sectional rank codes within the universe."""
    out = np.full(panel_g.shape, 255, np.uint8)
    for g in range(panel_g.shape[0]):
        m = univ_g[g] & np.isfinite(panel_g[g])
        n = int(m.sum())
        if n < 50: continue
        r = pd.Series(panel_g[g, m]).rank(method='average').to_numpy()
        out[g, m] = np.minimum(((r - 0.5) / n * 254).astype(np.int32), 253)
    return out


def stats(code, y_g, sel_rows, late_rows):
    """Rank IC vs y10 on the selection period, plus coverage in the selection and the late period."""
    ics = []
    for g in sel_rows:
        m = (code[g] != 255) & np.isfinite(y_g[g])
        if m.sum() < 200: continue
        a = code[g, m].astype(np.float64); b = pd.Series(y_g[g, m]).rank().to_numpy()
        ics.append(np.corrcoef(a, b)[0, 1])
    ics = np.array(ics)
    cov = lambda rows: float((code[rows] != 255).sum() / max(1, UNIV_CNT[rows].sum()))
    return {'ic': float(np.nanmean(ics)) if len(ics) else np.nan,
            'icir': float(np.nanmean(ics) / (np.nanstd(ics) + 1e-9)) if len(ics) else np.nan,
            'n_ic': int(len(ics)), 'cov_sel': cov(sel_rows), 'cov_late': cov(late_rows)}


def _init():
    global UNIV_G, Y_G, SEL_ROWS, LATE_ROWS, UNIV_CNT, GRID_DATES
    univ = np.load(B / 'univ.npy'); y10 = np.load(B / 'y10.npy')
    UNIV_G = univ[grid]; Y_G = y10[grid]; UNIV_CNT = UNIV_G.sum(1)
    di = dates[grid].strftime('%Y%m%d').astype(int).to_numpy()
    lend = np.array([int(dates[min(i + 11, T - 1)].strftime('%Y%m%d')) for i in grid])
    SEL_ROWS = np.where(lend <= SEL_END)[0]
    LATE_ROWS = np.where((di >= 20250701) & (di <= 20260630))[0]
    GRID_DATES = dates[grid]


def work(job):
    name, path, col = job
    dst = F / f'{name}.npy'
    try:
        if dst.exists():
            code = np.load(dst)
        else:
            df = pq.read_table(path, columns=['date', 'scmc', col]).to_pandas()
            df = df[pd.to_datetime(df['date']).isin(GRID_DATES)]
            pan = to_panel(df, GRID_DATES, codes, [col], date_col='date', code_col='scmc')[col]
            pan[~np.isfinite(pan)] = np.nan
            code = encode(pan, UNIV_G)
            np.save(dst, code)
        return name, stats(code, Y_G, SEL_ROWS, LATE_ROWS)
    except Exception as e:
        return name, {'error': repr(e)}


def ceres_jobs():
    prim = {}
    for f in ['factors.csv', 'factors_calendar.csv', 'factors_direct.csv', 'factors_relation.csv']:
        d = pd.read_csv(CATALOG / f)
        for r in d.itertuples():
            if isinstance(getattr(r, 'primary_variant', None), str): prim[r.slug] = r.primary_variant
    d = pd.read_csv(CATALOG / 'factors_dict.csv')
    for r in d.itertuples():
        if isinstance(r.parquet, str): prim.setdefault(r.slug, os.path.basename(r.parquet).replace('.parquet', ''))
    jobs = []
    for slug in sorted(os.listdir(CERES)):
        p = CERES / slug
        if not p.is_dir() or slug.startswith('server_builtin'): continue
        files = sorted(p.glob('*.parquet'))
        if not files: continue
        want = p / f'{prim.get(slug, "complete")}.parquet'
        path = want if want.exists() else (p / 'complete.parquet' if (p / 'complete.parquet').exists() else files[0])
        jobs.append((f'c_{slug}', str(path), 'f'))
    panel = CERES / 'server_builtin_panel.parquet'
    for col in pq.ParquetFile(panel).schema_arrow.names:
        if col.startswith('fp_') and col not in ('fp_Industry',):
            jobs.append((f'b_{col[3:]}', str(panel), col))
    return jobs


def derived():
    """Own features from base panels (price, indicator, flow, margin, northbound, intraday means)."""
    L = lambda k: np.load(B / f'{k}.npy')
    aclose = L('aclose'); traded = L('traded'); amt = L('amount')
    out = {}
    with np.errstate(all='ignore'):
        lc = np.log(aclose)
        lc_ff = pd.DataFrame(lc).ffill().to_numpy()
        ret1 = np.vstack([np.full((1, N), np.nan), np.diff(lc_ff, axis=0)])
        ret1[~traded] = np.nan
        R = pd.DataFrame(ret1)
        for h in (1, 5, 10, 20, 60, 120, 240):
            out[f'd_ret{h}'] = lc_ff - np.vstack([np.full((h, N), np.nan), lc_ff[:-h]])
        out['d_ret240_20'] = out['d_ret240'] - out['d_ret20']
        for h in (5, 20, 60):
            out[f'd_vol{h}'] = R.rolling(h, min_periods=max(3, h // 2)).std().to_numpy()
        out['d_skew20'] = R.rolling(20, min_periods=10).skew().to_numpy()
        out['d_max20'] = R.rolling(20, min_periods=10).max().to_numpy()
        out['d_min20'] = R.rolling(20, min_periods=10).min().to_numpy()
        C = pd.DataFrame(lc_ff)
        out['d_dist_hi60'] = lc_ff - C.rolling(60, min_periods=20).max().to_numpy()
        out['d_dist_lo60'] = lc_ff - C.rolling(60, min_periods=20).min().to_numpy()
        out['d_dist_hi250'] = lc_ff - C.rolling(250, min_periods=120).max().to_numpy()
        out['d_ma_gap20'] = lc_ff - C.rolling(20, min_periods=10).mean().to_numpy()
        A = pd.DataFrame(np.where(traded, amt, np.nan))
        out['d_amihud20'] = (pd.DataFrame(np.abs(ret1)) / A).rolling(20, min_periods=10).mean().to_numpy()
        out['d_lamt20'] = np.log(A.rolling(20, min_periods=10).mean().to_numpy())
        out['d_amt_ratio5_60'] = A.rolling(5, min_periods=3).mean().to_numpy() / A.rolling(60, min_periods=20).mean().to_numpy()
        tf = pd.DataFrame(np.where(traded, L('ind_turnover_rate_f'), np.nan))
        out['d_turn5'] = tf.rolling(5, min_periods=3).mean().to_numpy()
        out['d_turn20'] = tf.rolling(20, min_periods=10).mean().to_numpy()
        out['d_turn_ratio5_60'] = out['d_turn5'] / tf.rolling(60, min_periods=20).mean().to_numpy()
        out['d_turn_std20'] = tf.rolling(20, min_periods=10).std().to_numpy()
        out['d_volume_ratio'] = L('ind_volume_ratio')
        out['d_log_mv'] = np.log(L('ind_total_mv'))
        out['d_log_circ_mv'] = np.log(L('ind_circ_mv'))
        out['d_ep'] = 1 / L('ind_pe_ttm'); out['d_bp'] = 1 / L('ind_pb'); out['d_sp'] = 1 / L('ind_ps_ttm'); out['d_dv'] = L('ind_dv_ttm')
        # beta / idio vol vs equal-weight market over 60d
        mkt = np.nanmean(np.where(np.isfinite(ret1), ret1, np.nan), axis=1)
        M = pd.Series(mkt)
        rm = R.sub(R.rolling(60, min_periods=30).mean())
        mm = M - M.rolling(60, min_periods=30).mean()
        beta = (rm.mul(mm, axis=0)).rolling(60, min_periods=30).mean().div((mm ** 2).rolling(60, min_periods=30).mean(), axis=0)
        out['d_beta60'] = beta.to_numpy()
        resid = R.sub(beta.mul(M, axis=0))
        out['d_ivol60'] = resid.rolling(60, min_periods=30).std().to_numpy()
    c = con()
    grid_dates = dates[grid]
    mf = c.sql(f"""select trade_date, ts_code, net_mf_amount, buy_elg_amount+buy_lg_amount-sell_elg_amount-sell_lg_amount as big_net
                   from read_parquet('{LAKE}/flow/moneyflow/*/*/*.parquet', hive_partitioning=false) where trade_date >= '{dates[0].date()}'""").df()
    MF = to_panel(mf, dates, codes, ['net_mf_amount', 'big_net'])
    Aamt = A
    for h in (5, 20):
        out[f'd_mf_net{h}'] = pd.DataFrame(MF['net_mf_amount']).rolling(h, min_periods=h // 2).sum().to_numpy() / Aamt.rolling(h, min_periods=h // 2).sum().to_numpy()
        out[f'd_mf_big{h}'] = pd.DataFrame(MF['big_net']).rolling(h, min_periods=h // 2).sum().to_numpy() / Aamt.rolling(h, min_periods=h // 2).sum().to_numpy()
    mg = c.sql(f"""select trade_date, ts_code, rzye, rzmre, rqye from read_parquet('{LAKE}/flow/margin_detail/*/*/*.parquet', hive_partitioning=false)
                   where trade_date >= '{dates[0].date()}'""").df()
    MG = to_panel(mg, dates, codes, ['rzye', 'rzmre', 'rqye'])
    cmv = L('ind_circ_mv')
    rz = pd.DataFrame(MG['rzye']).ffill(limit=5)
    out['d_rz_mv'] = rz.to_numpy() / cmv
    out['d_rz_chg5'] = np.log(rz / rz.shift(5)).to_numpy()
    out['d_rz_chg20'] = np.log(rz / rz.shift(20)).to_numpy()
    out['d_rzmre_amt5'] = pd.DataFrame(MG['rzmre']).rolling(5, min_periods=3).sum().to_numpy() / Aamt.rolling(5, min_periods=3).sum().to_numpy()
    out['d_rq_mv'] = pd.DataFrame(MG['rqye']).ffill(limit=5).to_numpy() / cmv
    hk = c.sql(f"""select trade_date, ts_code, ratio from read_parquet('{LAKE}/flow/hk_hold/*/*/*.parquet', hive_partitioning=false)
                   where trade_date >= '{dates[0].date()}'""").df()
    HK = pd.DataFrame(to_panel(hk, dates, codes, ['ratio'])['ratio']).ffill(limit=10)
    out['d_hk_ratio'] = HK.to_numpy()
    out['d_hk_chg20'] = (HK - HK.shift(20)).to_numpy()
    out['d_hk_chg5'] = (HK - HK.shift(5)).to_numpy()
    iv = c.sql(f"select * from read_parquet('{LAKE}/features/v1/5min/*/*/*.parquet', hive_partitioning=false) where trade_date >= '{dates[0].date()}'").df()
    ivc = [x for x in iv.columns if x.startswith('i_')]
    IV = to_panel(iv, dates, codes, ivc)
    for k in ivc:
        out[f'd_{k}_m20'] = pd.DataFrame(IV[k]).rolling(20, min_periods=10).mean().to_numpy()
    return out


if __name__ == '__main__':
    _init()
    jobs = ceres_jobs()
    log('ceres jobs', len(jobs), 'grid', G, dates[grid[0]].date(), dates[grid[-1]].date(), 'sel rows', len(SEL_ROWS))
    res = {}
    with Pool(10, initializer=_init) as pool:
        for k, (name, st) in enumerate(pool.imap_unordered(work, jobs, chunksize=2)):
            res[name] = st
            if (k + 1) % 50 == 0: log('encoded', k + 1, '/', len(jobs))
    log('derived features')
    D = derived()
    for name, pan in D.items():
        pan = np.where(np.isfinite(pan), pan, np.nan)
        code = encode(pan[grid].astype(np.float32), UNIV_G)
        np.save(F / f'{name}.npy', code)
        res[name] = stats(code, Y_G, SEL_ROWS, LATE_ROWS)
    np.savez(OUT / 'grid.npz', grid=grid, on_sparse=on_sparse)
    (OUT / 'feature_stats.json').write_text(json.dumps(res, indent=1))
    errs = {k: v for k, v in res.items() if 'error' in v}
    log('done', len(res), 'errors', len(errs), list(errs.items())[:5])
