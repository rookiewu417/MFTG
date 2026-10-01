"""Diagnostics beyond return: style exposure, overlap with simple factors, signal decay, drawdown, capacity, long/short split.

Uses the same evaluation dates as evaluate.py (baseline test dates, every 5 trading days).
"""
import numpy as np
import pandas as pd
from common import OUT
import evaluate as E

B = OUT / 'base'
aclose = np.load(B / 'aclose.npy'); amount = np.load(B / 'amount.npy'); y20 = np.load(B / 'y20.npy')
beta_ok = True
lc = pd.DataFrame(np.log(aclose)).ffill().to_numpy()
ret20 = lc - np.vstack([np.full((20, lc.shape[1]), np.nan), lc[:-20]])
ret5 = lc - np.vstack([np.full((5, lc.shape[1]), np.nan), lc[:-5]])
r1 = np.vstack([np.full((1, lc.shape[1]), np.nan), np.diff(lc, axis=0)])
vol20 = pd.DataFrame(r1).rolling(20, min_periods=10).std().to_numpy()
amt20 = pd.DataFrame(amount).rolling(20, min_periods=10).mean().to_numpy()
turn = np.load(B / 'ind_turnover_rate_f.npy')
turn20 = pd.DataFrame(turn).rolling(20, min_periods=10).mean().to_numpy()
SIMPLE = {'size': E.lmv, 'rev20': ret20, 'rev5': ret5, 'vol20': vol20, 'turn20': turn20, 'liq(amt20)': np.log(amt20)}

MODELS = {
    'base_incremental': {b: f'/home/ivenwu/projects/ml-factor/baseline_preds/formal_ensembles/incremental_{b}_ensemble.npz' for b in E.BLOCKS},
    'xgb': {b: str(OUT.parent / f'runs/runA_xgb/xgb/{b}_pred.npz') for b in E.BLOCKS},
    'xgb_top': {b: str(OUT.parent / f'runs/exp_xgb_top/xgb/{b}_pred.npz') for b in E.BLOCKS},
}
eval_dates = {b: sorted(np.unique(np.load(f'/home/ivenwu/projects/ml-factor/baseline_preds/formal_ensembles/daily_{b}_ensemble.npz')['feature_date'])) for b in E.BLOCKS}


def pct(a):
    return (pd.Series(a).rank().to_numpy() - 0.5) / len(a)


def run(name, files):
    rec = []; prev = None
    for b in E.BLOCKS:
        df = E.load_pred(files[b])
        for d in eval_dates[b]:
            t = E.tpos[d]; g = df[df.fd == d]
            j = g.j.to_numpy(); p = g.pred.to_numpy()
            m = E.univ[t, j] & np.isfinite(E.y10[t, j]); j, p = j[m], p[m]
            if len(j) < 100: continue
            k = len(j) // 10; o = np.argsort(p); top, bot = j[o[-k:]], j[o[:k]]
            r = {'date': d, 'block': b}
            # overlap with simple factors (rank corr) and top-decile style percentile (0.5 = neutral)
            for f, arr in SIMPLE.items():
                v = arr[t, j]; ok = np.isfinite(v)
                r[f'corr_{f}'] = np.corrcoef(pct(p[ok]), pct(v[ok]))[0, 1]
                pv = np.full(len(j), np.nan); pv[ok] = pct(v[ok])
                r[f'toppct_{f}'] = np.nanmean(pv[o[-k:]])
            # industry concentration of the top decile (largest single-industry share)
            ii = E.ind[t, top]; r['top_ind_max'] = pd.Series(ii[ii >= 0]).value_counts(normalize=True).max()
            # decay: IC vs 5/10/20-day forward labels
            for h, yy in (('5', E.y5), ('10', E.y10), ('20', y20)):
                r[f'ic_y{h}'] = E.ric(p, yy[t, j])
            # long / short legs vs universe mean (10-day)
            mu = np.nanmean(E.y10[t, j])
            r['long_ex'] = np.nanmean(E.y10[t, top]) - mu; r['short_ex'] = mu - np.nanmean(E.y10[t, bot])
            # capacity: median 20d avg amount of top-decile names, share under 50m yuan/day
            a = amt20[t, top]; r['top_amt_med_m'] = np.nanmedian(a) / 1e6; r['top_amt_lt50m'] = np.nanmean(a < 5e7)
            # rebalance excess (5-day, 10bp)
            h5 = np.nan_to_num(E.y5[t, top]); bench = np.nanmean(E.y5[t, j])
            to = 1.0 if prev is None else 1 - len(np.intersect1d(top, prev)) / len(top); prev = top
            r['ex5'] = (1 + h5.mean() - 0.002 * to) / (1 + bench) - 1
            rec.append(r)
    return pd.DataFrame(rec)


out = {}
for name, files in MODELS.items():
    r = run(name, files)
    nav = np.cumprod(1 + r.ex5); dd = (nav / np.maximum.accumulate(nav) - 1).min()
    r['month'] = (r.date // 100)
    mon = r.groupby('month').ex5.apply(lambda x: np.prod(1 + x) - 1)
    s = {c: r[c].mean() for c in r.columns if c.startswith(('corr_', 'toppct_', 'ic_y', 'long_ex', 'short_ex', 'top_amt', 'top_ind'))}
    s['excess_maxdd'] = dd; s['month_win'] = (mon > 0).mean(); s['worst_month'] = mon.min()
    s['long_share'] = r.long_ex.mean() / (r.long_ex.mean() + r.short_ex.mean())
    out[name] = s
res = pd.DataFrame(out)
pd.set_option('display.width', 200)
print(res.round(3).to_string())
res.to_csv(OUT.parent / 'results' / 'diagnostics.csv')
