"""Extra daily inputs from data the end-to-end line has not used: money flow, margin, dragon-tiger list, holder count,
financial reports / forecasts / express reports, valuation.

build/mftg/extra.npy        [T, N, K] fp16  per-day cross-sectional rank-gauss (traded names), missing = 0
build/mftg/extra_names.json K names

Point in time: a value is used at the close of day t (trades at the t+1 open).
  * money flow, dragon-tiger list, valuation: published the evening of the trade date -> day t.
  * margin detail: published the next morning -> lagged one trading day.
  * reports / forecasts / express / holder count: from the first trading day on or after ann_date; the first disclosure
    of each (stock, period) is kept (later revisions are ignored) and a record never replaces a newer period.
Northbound holdings are left out: since 2024-09 only quarter-ends cover all names (otherwise ~900), so the test windows
would see a different feature than training.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtri
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'shared'))
from common import OUT, LAKE, con, load_axes, to_panel, SHARED

E = OUT / 'mftg'
B = SHARED / 'base'
dates, codes = load_axes(); T, N = len(dates), len(codes)
D0 = str(dates[0].date())


def lag(a, k=1): return np.vstack([np.full((k, a.shape[1]), np.nan, a.dtype), a[:-k]])
def rsum(a, k): return pd.DataFrame(a).rolling(k, min_periods=max(1, k // 2)).sum().to_numpy()


def csrank(x, ok):
    """Per-day rank -> standard normal over names with ok & finite x; everything else 0."""
    x = np.where(ok & np.isfinite(x), x, np.nan)
    r = pd.DataFrame(x).rank(axis=1).to_numpy(); n = np.isfinite(x).sum(1, keepdims=True)
    with np.errstate(all='ignore'): z = ndtri((r - 0.5) / np.maximum(n, 1))
    return np.clip(np.nan_to_num(z), -3, 3).astype(np.float32)


def asof_panel(df, cols, limit, date_col='ann_date', period_col='end_date'):
    """Long frame of announcements -> [T, N] panels: value from the first trading day >= date_col, forward-filled up to
    `limit` trading days. Keeps the first disclosure per (stock, period); drops records older than a period already seen."""
    df = df.dropna(subset=[date_col]).copy()
    df[date_col] = pd.to_datetime(df[date_col])
    if period_col:
        df[period_col] = pd.to_datetime(df[period_col])
        df = df.sort_values([date_col, period_col]).drop_duplicates(['ts_code', period_col], keep='first')
        prev = df.groupby('ts_code')[period_col].transform(lambda s: s.cummax().shift())
        df = df[prev.isna() | (df[period_col] >= prev)]
    ti = dates.searchsorted(df[date_col].to_numpy(), 'left')
    ci = pd.Index(codes).get_indexer(df.ts_code)
    ok = (ti < T) & (ci >= 0)
    df = df[ok].assign(ti=ti[ok], ci=ci[ok]).sort_values(['ti'] + ([period_col] if period_col else []))
    out = {}
    for c in cols:
        a = np.full((T, N), np.nan, np.float32)
        a[df.ti.to_numpy(), df.ci.to_numpy()] = df[c].to_numpy(np.float32)   # duplicates: last (newest period) wins
        out[c] = pd.DataFrame(a).ffill(limit=limit).to_numpy()
    return out


def event_age(df, date_col='ann_date'):
    """[T, N] trading days since the latest announcement (inf before the first)."""
    ti = dates.searchsorted(pd.to_datetime(df[date_col]).to_numpy(), 'left'); ci = pd.Index(codes).get_indexer(df.ts_code)
    ok = (ti < T) & (ci >= 0)
    last = np.full((T, N), np.nan); last[ti[ok], ci[ok]] = ti[ok]
    last = pd.DataFrame(last).ffill().to_numpy()
    return np.arange(T)[:, None] - last


def lake(sub, cols, where):
    return con().sql(f"""select {cols} from read_parquet('{LAKE}/{sub}', hive_partitioning=false, union_by_name=true)
                         where {where}""").df()


def build():
    traded = np.load(B / 'traded.npy'); amount = np.load(B / 'amount.npy')          # yuan * 1e3
    mv = np.exp(np.load(B / 'log_circ_mv.npy'))                                     # yuan * 1e4
    F = {}
    with np.errstate(all='ignore'):
        # ---- money flow (yuan * 1e4)
        m = lake('flow/moneyflow/*/*/*.parquet', '*', f"trade_date >= '{D0}'")
        m = to_panel(m, dates, codes, ['net_mf_amount', 'buy_elg_amount', 'sell_elg_amount', 'buy_lg_amount', 'sell_lg_amount'])
        amt = np.where(traded, amount / 10, np.nan)                                 # -> yuan * 1e4
        big = m['buy_elg_amount'] - m['sell_elg_amount'] + m['buy_lg_amount'] - m['sell_lg_amount']
        F['mf_net1'] = m['net_mf_amount'] / amt
        F['mf_elg1'] = (m['buy_elg_amount'] - m['sell_elg_amount']) / amt
        F['mf_lg1'] = (m['buy_lg_amount'] - m['sell_lg_amount']) / amt
        F['mf_big5'] = rsum(big, 5) / rsum(amt, 5)
        F['mf_big20'] = rsum(big, 20) / rsum(amt, 20)
        F['mf_elgshare5'] = rsum(m['buy_elg_amount'] + m['sell_elg_amount'], 5) / rsum(amt, 5)
        # ---- margin (yuan), published next morning -> lag 1
        g = lake('flow/margin_detail/*/*/*.parquet', 'ts_code, trade_date, rzye, rqye, rzmre', f"trade_date >= '{D0}'")
        g = {k: lag(v) for k, v in to_panel(g, dates, codes, ['rzye', 'rqye', 'rzmre']).items()}
        cap = mv * 1e4
        F['mg_rz'] = g['rzye'] / cap
        F['mg_drz5'] = (g['rzye'] - lag(g['rzye'], 5)) / cap
        F['mg_drz20'] = (g['rzye'] - lag(g['rzye'], 20)) / cap
        F['mg_rq'] = g['rqye'] / cap
        F['mg_buy5'] = rsum(g['rzmre'], 5) / rsum(lag(amt) * 1e4, 5)
        F['mg_has'] = np.isfinite(g['rzye']).astype(np.float32)
        # ---- dragon-tiger list (yuan)
        tl = lake('flow/top_list/*/*/*.parquet', 'ts_code, trade_date, net_amount', f"trade_date >= '{D0}'")
        tl = tl.groupby(['ts_code', 'trade_date'], as_index=False).net_amount.sum().assign(on=1.0)
        tl = to_panel(tl, dates, codes, ['net_amount', 'on'])
        F['tl_cnt20'] = rsum(np.nan_to_num(tl['on']), 20)
        F['tl_net5'] = rsum(np.nan_to_num(tl['net_amount']), 5) / rsum(amt * 1e4, 5)
        # ---- holder count change (latest report vs the previous one)
        h = lake('flow/holdernumber/*/*/*.parquet', 'ts_code, ann_date, end_date, holder_num', "ann_date >= '2016-01-01'")
        h = h.dropna(subset=['ann_date', 'holder_num']).sort_values(['ann_date', 'end_date']).drop_duplicates(['ts_code', 'end_date'])
        h = h.sort_values(['ts_code', 'end_date'])
        h['hn_chg'] = np.log(h.holder_num / h.groupby('ts_code').holder_num.shift())
        F['hn_chg'] = asof_panel(h, ['hn_chg'], 250)['hn_chg']
        # ---- financial indicators / express / forecast
        fi = lake('fundamental/fina_indicator/*/*/*.parquet', '*', "ann_date >= '2016-01-01'")
        fcols = ['roe', 'netprofit_yoy', 'or_yoy', 'grossprofit_margin', 'debt_to_assets']
        F.update({f'fi_{k}': v for k, v in asof_panel(fi, fcols, 300).items()})
        ex = lake('fundamental/express/*/*/*.parquet', 'ts_code, ann_date, end_date, yoy_net_profit', "ann_date >= '2016-01-01'")
        F['ex_yoy'] = asof_panel(ex, ['yoy_net_profit'], 120)['yoy_net_profit']
        fc = lake('fundamental/forecast/*/*/*.parquet', 'ts_code, ann_date, end_date, p_change_min, p_change_max', "ann_date >= '2016-01-01'")
        fc['mid'] = fc[['p_change_min', 'p_change_max']].mean(1)
        F['fc_mid'] = asof_panel(fc, ['mid'], 120)['mid']
        anns = pd.concat([d[['ts_code', 'ann_date']] for d in [fi, ex, fc]]).dropna()
        F['ann_recent'] = np.exp(-event_age(anns) / 5.0)                            # 1 on the announcement day, 0 long after
        # ---- valuation (daily indicator)
        v = lake('daily/indicator/*/*.parquet', 'ts_code, trade_date, pe_ttm, pb, ps_ttm, dv_ttm, volume_ratio', f"trade_date >= '{D0}'")
        v = to_panel(v, dates, codes, ['pe_ttm', 'pb', 'ps_ttm', 'dv_ttm', 'volume_ratio'])
        have = np.isfinite(v['pb'])
        F['ep'] = np.where(np.isfinite(v['pe_ttm']), 1 / v['pe_ttm'], np.where(have, -0.05, np.nan))   # null pe_ttm = loss
        F['bp'] = 1 / v['pb']
        F['sp'] = 1 / v['ps_ttm']
        F['dv'] = np.where(have, np.nan_to_num(v['dv_ttm']), np.nan)
        F['vol_ratio'] = v['volume_ratio']
    names = list(F)
    X = np.stack([csrank(F[k], traded) for k in names], -1).astype(np.float16)
    np.save(E / 'extra.npy', X)
    (E / 'extra_names.json').write_text(json.dumps(names))
    cov = {k: round(float(np.isfinite(F[k])[traded].mean()), 3) for k in names}
    print('extra', X.shape, 'coverage over traded:', cov, flush=True)


if __name__ == '__main__':
    build()
