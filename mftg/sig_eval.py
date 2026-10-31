"""Signal-level evaluation of MFTG ensembles as a factor (no portfolio optimisation).

Ensemble = equal-weight mean of per-date z-scored members (members = paths under runs/mftg, joined on shared stocks,
same as ens_eval.py). Scored on weekly cross-sections (every 5th prediction date of each block) on the tradable universe.

Writes to --out (CSV per table):
  blocks.csv    per ensemble x block: RankIC vs y10, neutral RankIC (signal ranked, residual on SW-1 industry dummies +
                log circ mv + beta / volatility / liquidity / momentum), head IC (inside the signal's top 30%)
  summary.csv   per ensemble over the in-sample blocks: IC mean / std / ICIR / share > 0, neutral IC and ICIR,
                head IC, out-of-sample IC (2026H2X), half-years with IC above the reference ensemble (first spec)
  deciles.csv   10 groups by score: mean y10 excess over the universe, annualised (x 252/10); long-short, monotonicity
  topbook.csv   fixed sanity check: top 10% equal weight, y5 hold, rebalanced every 5 days (non-overlapping),
                excess over the universe, gross / net of 10bp one-way; bottom 10% the same; turnover per rebalance
  pools.csv     RankIC inside CSI300 / CSI500 / CSI1000 / outside the three, and in up / down market weeks
  decay.csv     RankIC of the signal at t vs y5 starting at t+k (k = 0, 5, 10, 20), and rank autocorrelation at lag 5
  styles.csv    mean cross-sectional rank corr of the signal with size and the four styles; R^2 of industry + styles
  incr.csv      vs the reference ensemble: corr, and RankIC of the residual after regressing on the reference
                plus industry + styles (new information the reference does not carry)
Usage: python mftg/sig_eval.py NAME=m1+m2+... [NAME=...] --out DIR [--jobs 8]
"""
import argparse, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'shared'))
from common import OUT, load_axes, SHARED

RUNS = OUT.parent / 'runs' / 'mftg'
B = SHARED / 'base'
IS_BLOCKS = ['2022H2', '2023H1', '2023H2', '2024H1', '2024H2', '2025H1', '2025H2', '2026H1']
OOS = '2026H2X'
dates, codes = load_axes(); dint = dates.strftime('%Y%m%d').astype(int).to_numpy(); tpos = {d: i for i, d in enumerate(dint)}
T = len(dates)
univ = np.load(B / 'univ.npy'); Y10 = np.load(B / 'y10.npy'); Y5 = np.load(B / 'y5.npy')
IND = np.load(B / 'industry.npy'); LMV = np.load(B / 'log_circ_mv.npy'); IDX = np.load(B / 'index_member.npy')
STY = np.load(OUT / 'mftg' / 'styles.npy', mmap_mode='r')     # [T, N, 4] beta, volatility, liquidity, momentum
STY_NAMES = ['size', 'beta', 'volatility', 'liquidity', 'momentum']


def rank(a): return pd.Series(a).rank().to_numpy()
def ric(p, y):
    m = np.isfinite(p) & np.isfinite(y)
    return np.corrcoef(rank(p[m]), rank(y[m]))[0, 1] if m.sum() > 50 else np.nan


def member(m, b):
    f = RUNS / m / f'{b}_pred.npz'
    if not f.exists(): return None
    z = np.load(f)
    d = pd.DataFrame({'t': [tpos[x] for x in z['feature_date']], 'j': z['stock'].astype(int), 'p': z['pred'].astype(float)})
    d['p'] = d.groupby('t').p.transform(lambda v: (v - v.mean()) / (v.std() + 1e-12))
    return d.set_index(['t', 'j']).p


def ensemble(members, b):
    ps = [member(m, b) for m in members]
    if any(p is None for p in ps): return None
    return pd.concat(ps, axis=1).dropna().mean(1)


def design(t, j):
    """Industry dummies + size + 4 styles (no intercept column needed: the dummies span it; stocks without industry get their own)."""
    ii = IND[t, j]; cols = [(ii == k).astype(float) for k in np.unique(ii)]
    lm = LMV[t, j]; lm = np.where(np.isfinite(lm), lm, np.nanmean(lm)); cols.append((lm - lm.mean()) / (lm.std() + 1e-12))
    s = np.nan_to_num(np.asarray(STY[t, j, :], dtype=float))
    return np.column_stack(cols + [s[:, k] for k in range(s.shape[1])])


def resid(z, X):
    beta, *_ = np.linalg.lstsq(X, z, rcond=None)
    return z - X @ beta


def zrank(p): r = rank(p); return (r - r.mean()) / (r.std() + 1e-12)


def eval_ens(args):
    name, members, ref_members = args
    out = {k: [] for k in ['week', 'dec', 'book', 'decay', 'sty', 'incr']}
    for b in IS_BLOCKS + [OOS]:
        e = ensemble(members, b)
        if e is None: continue
        r = ensemble(ref_members, b) if ref_members and ref_members != members else None
        ts = np.sort(e.index.get_level_values(0).unique())[::5]
        prev_top = prev_bot = None; prev_sig = None
        for t in ts:
            g = e.loc[t]; j0 = g.index.to_numpy(); p0 = g.to_numpy(); u = univ[t, j0]; j0, p0 = j0[u], p0[u]
            y = Y10[t, j0]; m = np.isfinite(y)
            if m.sum() < 100: continue                         # label not known yet (end of the data)
            j, p, y = j0[m], p0[m], y[m]
            X = design(t, j); zp = zrank(p); n = len(j)
            ic = ric(p, y); nic = ric(resid(zp, X), y)
            top = np.argsort(p)[-int(n * 0.3):]; head = ric(p[top], y[top])
            bits = IDX[t, j]; mk = np.nanmean(y)
            row = {'ens': name, 'block': b, 't': t, 'ic': ic, 'nic': nic, 'head_ic': head, 'mkt': mk,
                   'ic300': ric(p[(bits & 1) > 0], y[(bits & 1) > 0]), 'ic500': ric(p[(bits & 2) > 0], y[(bits & 2) > 0]),
                   'ic1000': ric(p[(bits & 4) > 0], y[(bits & 4) > 0]), 'ic_rest': ric(p[bits == 0], y[bits == 0])}
            out['week'].append(row)
            # deciles on y10
            q = np.minimum((rank(p) - 1) * 10 // n, 9).astype(int); ex = y - y.mean()
            out['dec'].append({'ens': name, 'block': b, 't': t, **{f'D{k + 1}': ex[q == k].mean() for k in range(10)}})
            # style exposure
            s = {'size': ric(p, LMV[t, j])}
            for k, nm in enumerate(STY_NAMES[1:]): s[nm] = ric(p, np.asarray(STY[t, j, k], dtype=float))
            fit = zp - resid(zp, X); s['r2'] = fit.var() / zp.var()
            out['sty'].append({'ens': name, 'block': b, 't': t, **s})
            # decay: signal at t vs y5 starting at t+k, on stocks tradable at t
            dk = {'ens': name, 'block': b, 't': t}
            for k in (0, 5, 10, 20):
                if t + k < T: dk[f'k{k}'] = ric(p0, Y5[t + k, j0])
            if prev_sig is not None:
                a = prev_sig.reindex(j0).to_numpy(); dk['autocorr5'] = ric(a, p0)
            prev_sig = pd.Series(p0, index=j0)
            out['decay'].append(dk)
            # fixed top / bottom 10% book on y5 (non-overlapping: dates are 5 trading days apart)
            y5 = Y5[t, j0]; m5 = np.isfinite(y5); jj, pp, yy = j0[m5], p0[m5], y5[m5]; k10 = len(jj) // 10
            if k10 > 10:
                o = np.argsort(pp); tp, bt = set(jj[o[-k10:]]), set(jj[o[:k10]])
                to_t = 1.0 if prev_top is None else len(tp - prev_top) / len(tp)
                to_b = 1.0 if prev_bot is None else len(bt - prev_bot) / len(bt)
                out['book'].append({'ens': name, 'block': b, 't': t, 'top_ex': yy[o[-k10:]].mean() - yy.mean(),
                                    'bot_ex': yy[o[:k10]].mean() - yy.mean(), 'to_top': to_t, 'to_bot': to_b})
                prev_top, prev_bot = tp, bt
            # increment over the reference ensemble
            if r is not None:
                rr = r.reindex(pd.MultiIndex.from_arrays([np.full(n, t), j])).to_numpy(); mm = np.isfinite(rr)
                if mm.sum() > 100:
                    Xr = np.column_stack([X[mm], zrank(rr[mm])])
                    out['incr'].append({'ens': name, 'block': b, 't': t, 'corr_ref': ric(p[mm], rr[mm]),
                                        'incr_ic': ric(resid(zp[mm], Xr), y[mm])})
    return {k: pd.DataFrame(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('specs', nargs='+'); ap.add_argument('--out', required=True)
    ap.add_argument('--jobs', type=int, default=8)
    a = ap.parse_args(); od = Path(a.out); od.mkdir(parents=True, exist_ok=True)
    specs = [(s.split('=')[0], s.split('=')[1].split('+')) for s in a.specs]; ref = specs[0][1]
    with Pool(min(a.jobs, len(specs))) as pool: res = pool.map(eval_ens, [(n, ms, ref) for n, ms in specs])
    cat = {k: pd.concat([r[k] for r in res if len(r[k])], ignore_index=True) for k in res[0]}
    w = cat['week']; IS = w.block.isin(IS_BLOCKS)
    have = set(w[IS].groupby('ens').block.nunique().loc[lambda v: v == len(IS_BLOCKS)].index)
    order = [n for n, _ in specs if n in have]
    if len(order) < len(specs): print('skipped (missing blocks):', [n for n, _ in specs if n not in have])
    for k in cat: cat[k] = cat[k][cat[k].ens.isin(order)]
    w = cat['week']; IS = w.block.isin(IS_BLOCKS)

    blocks = w.groupby(['ens', 'block'])[['ic', 'nic', 'head_ic']].mean().reset_index()
    blocks.to_csv(od / 'blocks.csv', index=False)
    piv = blocks.pivot(index='block', columns='ens', values='ic')[order]
    g = w[IS].groupby('ens')
    s = pd.DataFrame({'ic': g.ic.mean(), 'ic_std': g.ic.std(), 'nic': g.nic.mean(), 'nic_std': g.nic.std(),
                      'ic_pos': g.ic.apply(lambda v: (v > 0).mean()), 'head_ic': g.head_ic.mean(), 'n': g.size()}).loc[order]
    s['icir'] = s.ic / s.ic_std; s['nicir'] = s.nic / s.nic_std
    s['oos_ic'] = w[w.block == OOS].groupby('ens').ic.mean().reindex(order)
    s['oos_nic'] = w[w.block == OOS].groupby('ens').nic.mean().reindex(order)
    s['d_ic'] = s.ic - s.ic.iloc[0]; s['d_nic'] = s.nic - s.nic.iloc[0]
    pn = blocks[blocks.block.isin(IS_BLOCKS)].pivot(index='block', columns='ens', values='nic')[order]
    s['wins_ic'] = [(piv.loc[IS_BLOCKS, c] > piv.loc[IS_BLOCKS, order[0]]).sum() for c in order]
    s['wins_nic'] = [(pn[c] > pn[order[0]]).sum() for c in order]
    s['min_half_ic'] = piv.loc[IS_BLOCKS].min().reindex(order)
    s = s[['ic', 'icir', 'ic_pos', 'nic', 'nicir', 'head_ic', 'oos_ic', 'oos_nic', 'd_ic', 'd_nic', 'wins_ic', 'wins_nic', 'min_half_ic', 'n']]
    s.to_csv(od / 'summary.csv')

    d = cat['dec']; dc = [f'D{k}' for k in range(1, 11)]
    dd = d[d.block.isin(IS_BLOCKS)].groupby('ens')[dc].mean().loc[order] * 252 / 10
    dd['LS'] = dd.D10 - dd.D1; dd['mono'] = [np.corrcoef(np.arange(10), rank(dd.loc[e, dc].to_numpy()))[0, 1] for e in order]
    oo = d[d.block == OOS].groupby('ens')[dc].mean().reindex(order) * 252 / 10; dd['oos_D10'] = oo.D10; dd['oos_D1'] = oo.D1
    dd.to_csv(od / 'deciles.csv')

    bk = cat['book']; rows = []
    for e in order:
        for per, sel in [('IS', bk.block.isin(IS_BLOCKS)), ('OOS', bk.block == OOS)]:
            x = bk[(bk.ens == e) & sel]
            if not len(x): continue
            net = x.top_ex - 2 * 0.001 * x.to_top
            rows.append({'ens': e, 'period': per, 'top_gross': x.top_ex.mean() * 252 / 5, 'top_net10': net.mean() * 252 / 5,
                         'top_ir': net.mean() / net.std() * np.sqrt(252 / 5), 'top_turnover': x.to_top.iloc[1:].mean(),
                         'bot_gross': x.bot_ex.mean() * 252 / 5, 'cum_top_net': (1 + net).prod() - 1,
                         'neg_halves': int((x.assign(n=net).groupby('block').n.sum() < 0).sum())})
    pd.DataFrame(rows).to_csv(od / 'topbook.csv', index=False)

    up = w.mkt > w[IS].mkt.median()
    pl = w[IS].groupby('ens')[['ic300', 'ic500', 'ic1000', 'ic_rest']].mean().loc[order]
    pl['ic_up'] = w[IS & up].groupby('ens').ic.mean(); pl['ic_down'] = w[IS & ~up].groupby('ens').ic.mean()
    pl.to_csv(od / 'pools.csv')

    dk = cat['decay']; dk[dk.block.isin(IS_BLOCKS)].groupby('ens')[['k0', 'k5', 'k10', 'k20', 'autocorr5']].mean().loc[order].to_csv(od / 'decay.csv')
    sty = cat['sty']; sty[sty.block.isin(IS_BLOCKS)].groupby('ens')[STY_NAMES + ['r2']].mean().loc[order].to_csv(od / 'styles.csv')
    if len(cat['incr']):
        ic_ = cat['incr']; ic_['is'] = ic_.block.isin(IS_BLOCKS)
        x = ic_.groupby(['ens', 'is'])[['corr_ref', 'incr_ic']].mean().unstack('is')
        x.columns = [f'{a}_{"IS" if b else "OOS"}' for a, b in x.columns]
        gi = ic_[ic_['is']].groupby('ens').incr_ic; x['incr_t'] = gi.mean() / gi.std() * np.sqrt(gi.size())
        x.reindex([o for o in order if o in x.index]).to_csv(od / 'incr.csv')
    pd.set_option('display.width', 250)
    for f in ['summary', 'deciles', 'topbook', 'pools', 'decay', 'styles', 'incr']:
        if (od / f'{f}.csv').exists(): print(f'\n== {f}'); print(pd.read_csv(od / f'{f}.csv').round(4).to_string(index=False))
    print('\n== IC by block'); print(piv.round(4).to_string())


if __name__ == '__main__':
    main()
