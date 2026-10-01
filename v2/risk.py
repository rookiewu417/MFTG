"""Risk & stability: style exposures, Fama-MacBeth style/industry attribution of the top-decile excess, regime split.

Style factors (cross-sectional z-scores on the tradable universe): size, beta60, momentum (ret240-20), reversal (ret20),
volatility (vol20), liquidity (turnover20), value (bp). Portfolio = long-only backtest periods from backtest.py.
"""
from dataclasses import replace
import numpy as np
import pandas as pd
import backtest as bt
from common import OUT

B = OUT / 'base'
BL = ['2024H2', '2025H1', '2025H2', '2026H1']
R = '/home/ivenwu/projects/ml-factor/v2/runs'
lc = pd.DataFrame(np.log(bt.aclose)).ffill().to_numpy()
def lag(a, h): return np.vstack([np.full((h, bt.N), np.nan), a[:-h]])
r1 = lc - lag(lc, 1)
mkt = np.nanmean(np.where(bt.univ, r1, np.nan), axis=1)
R1 = pd.DataFrame(r1); M = pd.Series(mkt)
rm = R1.sub(R1.rolling(60, min_periods=30).mean()); mm = M - M.rolling(60, min_periods=30).mean()
beta = rm.mul(mm, axis=0).rolling(60, min_periods=30).mean().div((mm ** 2).rolling(60, min_periods=30).mean(), axis=0).to_numpy()
with np.errstate(all='ignore'):
    STY = {'size': bt.lmv, 'beta': beta, 'momentum': (lc - lag(lc, 240)) - (lc - lag(lc, 20)), 'reversal': lc - lag(lc, 20),
           'volatility': R1.rolling(20, min_periods=10).std().to_numpy(),
           'liquidity': np.log(pd.DataFrame(np.load(B / 'ind_turnover_rate_f.npy')).rolling(20, min_periods=10).mean().to_numpy()),
           'value': np.log(1 / np.load(B / 'ind_pb.npy'))}
names = list(STY)


def zs(x):
    x = np.where(np.isfinite(x), x, np.nan); med = np.nanmedian(x); x = np.clip(x, *np.nanpercentile(x, [1, 99]))
    return np.nan_to_num((x - np.nanmean(x)) / (np.nanstd(x) + 1e-12))


def attribute(model_path, cfg):
    S = bt.score_panel(bt.load_scores({b: f'{R}/{model_path}/{b}_pred.npz' for b in BL}), cfg.ema_hl)
    reb = bt.rebal_dates(S, 20240701, 20260630, cfg.hold)
    per = bt.simulate(S, reb, cfg)
    rows = []; held = None
    # re-derive holdings to compute exposures (same rule as simulate, deterministic)
    P = bt.P_open
    for k, t in enumerate(per.t.to_numpy()):
        t_exit = per.t.to_numpy()[k + 1] if k + 1 < len(per) else t + cfg.hold
        e = np.where(bt.univ[t] & bt.buyable[t] & np.isfinite(S[t]))[0]
        n = int(round(cfg.q * len(e))); top = e[np.argsort(-S[t, e])[:n]]
        ret = P[t_exit + 1, e] / P[t + 1, e] - 1; ok = np.isfinite(ret)
        X = np.column_stack([zs(STY[f][t, e]) for f in names])
        ind = bt.industry[t, e]; D = np.column_stack([(ind == j).astype(float) for j in np.unique(ind[ind >= 0])])
        A = np.column_stack([X, D])
        f, *_ = np.linalg.lstsq(A[ok], ret[ok], rcond=None)          # Fama-MacBeth factor returns this period
        w = np.isin(e, top).astype(float); w /= w.sum(); u = np.ones(len(e)) / len(e)
        expo = (w - u) @ A
        contrib_style = expo[:len(names)] * f[:len(names)]; contrib_ind = expo[len(names):] @ f[len(names):]
        active = np.nansum((w - u) * np.nan_to_num(ret))
        rows.append({'date': bt.dint[t], 'active': active, 'industry': contrib_ind, **{f'c_{n_}': c for n_, c in zip(names, contrib_style)},
                     **{f'x_{n_}': x for n_, x in zip(names, expo[:len(names)])},
                     'mkt': np.nanmean(ret), 'small_minus_big': np.nanmean(ret[bt.lmv[t, e] < np.nanmedian(bt.lmv[t, e])]) - np.nanmean(ret[bt.lmv[t, e] >= np.nanmedian(bt.lmv[t, e])]),
                     'dispersion': np.nanstd(ret)})
    a = pd.DataFrame(rows)
    a['specific'] = a.active - a.industry - a[[f'c_{n_}' for n_ in names]].sum(1)
    return a


def ann(x, per_year=252 / 5): return x.mean() * per_year


if __name__ == '__main__':
    pd.set_option('display.width', 200)
    cfg = replace(bt.Cfg(), ema_hl=0)
    out = {}
    for label, path in {'xgb': 'runA_xgb/xgb', 'xgb_neu': 'runB/xgb', 'xgb_top': 'exp_xgb_top/xgb', 'fusion_top': 'exp_fusion_top/fusion'}.items():
        a = attribute(path, cfg)
        s = {'active (gross, ann.)': ann(a.active), 'industry': ann(a.industry), **{n_: ann(a[f'c_{n_}']) for n_ in names},
             'specific (pure alpha)': ann(a.specific), 'specific IR': a.specific.mean() / a.specific.std() * np.sqrt(252 / 5)}
        s.update({f'exposure_{n_}': a[f'x_{n_}'].mean() for n_ in names})
        # regimes: split periods by market direction, size rotation and dispersion (terciles of each)
        for reg in ['mkt', 'small_minus_big', 'dispersion']:
            q = pd.qcut(a[reg], 3, labels=['low', 'mid', 'high'])
            for lvl in ['low', 'high']:
                s[f'{reg}={lvl}: active'] = ann(a.active[q == lvl])
        out[label] = s
        a.to_csv(f'/home/ivenwu/projects/ml-factor/v2/results/attrib_{label}.csv', index=False)
    res = pd.DataFrame(out)
    print(res.round(3).to_string())
    res.to_csv('/home/ivenwu/projects/ml-factor/v2/results/risk_attribution.csv')
