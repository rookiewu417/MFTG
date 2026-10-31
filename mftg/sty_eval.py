"""Style exposure of MFTG ensembles (signal level, no portfolio). Same ensembles / weekly cross-sections / universe as sig_eval.py.

Styles (per date: winsorised 1/99 pct, z-scored on the cross-section, missing = 0):
  size (log circ mv), size_nl (size^3 orthogonal to size), beta, volatility, liquidity, momentum (styles.npy),
  rev20 / rev5 (past 20 / 5 day log return), bp, ep, dv, roe, leverage (debt to assets), growth (net profit yoy)
  (the last six from build/mftg/extra.npy, point in time, already rank-gauss).

Writes to --out:
  expo.csv        univariate exposure: mean weekly rank corr of score with each style (in sample), t, share of weeks with
                  the same sign, and the out-of-sample mean
  expo_blocks.csv the same mean per half-year (stability)
  multi.csv       weekly OLS of the ranked score on industry dummies + all styles: mean coefficient and t per style;
                  R^2 of industry only / industry + 5 base styles / industry + all styles
  icdec.csv       IC decomposition: raw IC, IC after removing industry, + 5 base styles, + all styles; IC of the fitted
                  (industry + style) part; per style the IC lost when that style alone is added to industry ('add_') and the
                  IC that only that style carries given all others ('drop_')
  styic.csv       RankIC of each style with y10 in the same weeks (did the tilts pay?), in sample and out of sample
  fret.csv        Fama-MacBeth on y5 (non-overlapping weeks): raw factor return of the ranked score alone, pure return
                  with industry + all styles in the regression; annualised, IR, t; and the time-series regression of the
                  raw weekly factor return on the weekly style returns (alpha, R^2)
  ind.csv         mean ranked score per SW-1 industry (industry tilt)
  weekly.csv.gz / fret_weekly.csv.gz   per-week rows behind the tables above
Usage: python mftg/sty_eval.py NAME=m1+m2+... [NAME=...] --out DIR [--jobs 8]
"""
import argparse, json
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from sig_eval import (ensemble, rank, ric, resid, zrank, univ, Y10, Y5, IND, LMV, STY, IS_BLOCKS, OOS, OUT, B, T)

EX = np.load(OUT / 'mftg' / 'extra.npy', mmap_mode='r')
EXN = json.load(open(OUT / 'mftg' / 'extra_names.json'))
EXS = {'bp': 'bp', 'ep': 'ep', 'dv': 'dv', 'roe': 'fi_roe', 'leverage': 'fi_debt_to_assets', 'growth': 'fi_netprofit_yoy'}
LC = pd.DataFrame(np.log(np.load(B / 'aclose.npy'))).ffill().to_numpy()
IND_CODES = json.load(open(B / 'meta.json'))['ind_codes']
BASE = ['size', 'beta', 'volatility', 'liquidity', 'momentum']
STYLES = ['size', 'size_nl', 'beta', 'volatility', 'liquidity', 'momentum', 'rev20', 'rev5'] + list(EXS)


def zs(x):
    x = np.where(np.isfinite(x), x, np.nan)
    if np.isfinite(x).sum() < 50: return np.zeros(len(x))
    lo, hi = np.nanpercentile(x, [1, 99]); x = np.clip(x, lo, hi)
    return np.nan_to_num((x - np.nanmean(x)) / (np.nanstd(x) + 1e-12))


def styles(t, j):
    s = {'size': zs(LMV[t, j])}
    c = s['size'] ** 3; c = c - s['size'] * (c @ s['size']) / (s['size'] @ s['size']); s['size_nl'] = zs(c)
    for k, nm in enumerate(['beta', 'volatility', 'liquidity', 'momentum']): s[nm] = zs(np.asarray(STY[t, j, k], dtype=float))
    s['rev20'] = zs(LC[t, j] - LC[t - 20, j]); s['rev5'] = zs(LC[t, j] - LC[t - 5, j])
    for nm, src in EXS.items(): s[nm] = zs(np.asarray(EX[t, j, EXN.index(src)], dtype=float))
    return pd.DataFrame(s)[STYLES]


def inddum(t, j):
    ii = IND[t, j]; u = np.unique(ii)
    return np.column_stack([(ii == k).astype(float) for k in u]), ii


def r2(z, X): f = z - resid(z, X); return f.var() / z.var()


def eval_ens(args):
    name, members = args
    rows, frs, inds = [], [], []
    for b in IS_BLOCKS + [OOS]:
        e = ensemble(members, b)
        if e is None: continue
        for t in np.sort(e.index.get_level_values(0).unique())[::5]:
            g = e.loc[t]; j = g.index.to_numpy(); p = g.to_numpy(); u = univ[t, j]; j, p = j[u], p[u]
            y = Y10[t, j]; m = np.isfinite(y)
            if m.sum() < 100: continue
            j, p, y = j[m], p[m], y[m]
            S = styles(t, j); D, ii = inddum(t, j); zp = zrank(p); Sa = S.to_numpy()
            Xb = np.column_stack([D, S[BASE].to_numpy()]); Xa = np.column_stack([D, Sa])
            r = {'ens': name, 'block': b, 't': t, 'ic': ric(p, y)}
            for k in STYLES: r[f'x_{k}'] = ric(p, S[k].to_numpy()); r[f'sic_{k}'] = ric(S[k].to_numpy(), y)
            coef, *_ = np.linalg.lstsq(Xa, zp, rcond=None)
            for i, k in enumerate(STYLES): r[f'b_{k}'] = coef[D.shape[1] + i]
            r['r2_ind'] = r2(zp, D); r['r2_base'] = r2(zp, Xb); r['r2_all'] = r2(zp, Xa)
            ra = resid(zp, Xa); ri = resid(zp, D)
            r['nic_ind'] = ric(ri, y); r['nic_base'] = ric(resid(zp, Xb), y); r['nic_all'] = ric(ra, y)
            r['ic_fit'] = ric(zp - ra, y)
            for i, k in enumerate(STYLES):
                r[f'add_{k}'] = ric(resid(zp, np.column_stack([D, Sa[:, i]])), y) - r['nic_ind']
                r[f'drop_{k}'] = ric(resid(zp, np.column_stack([D, np.delete(Sa, i, 1)])), y) - r['nic_all']
            rows.append(r)
            # Fama-MacBeth on y5 (dates are 5 trading days apart -> non-overlapping)
            y5 = Y5[t, j]; m5 = np.isfinite(y5)
            if m5.sum() > 100:
                lo, hi = np.nanpercentile(y5[m5], [1, 99]); yy = np.clip(y5[m5], lo, hi); yy = yy - yy.mean()
                z5 = zrank(p[m5]); X5 = np.column_stack([D[m5], Sa[m5]])
                f = {'ens': name, 'block': b, 't': t, 'raw': (z5 @ yy) / (z5 @ z5)}
                cf, *_ = np.linalg.lstsq(np.column_stack([X5, z5]), yy, rcond=None); f['pure'] = cf[-1]
                cs, *_ = np.linalg.lstsq(X5, yy, rcond=None)
                for i, k in enumerate(STYLES): f[f'f_{k}'] = cs[D.shape[1] + i]
                frs.append(f)
            if b in IS_BLOCKS:
                inds.append(pd.Series(zp).groupby(ii).mean().rename(lambda k: IND_CODES[k] if k >= 0 else 'none'))
    ind = pd.concat(inds, axis=1).mean(1).rename(name) if inds else pd.Series(name=name, dtype=float)
    return pd.DataFrame(rows), pd.DataFrame(frs), ind


def tstat(v): v = v.dropna(); return v.mean() / v.std() * np.sqrt(len(v))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('specs', nargs='+'); ap.add_argument('--out', required=True)
    ap.add_argument('--jobs', type=int, default=8)
    a = ap.parse_args(); od = Path(a.out); od.mkdir(parents=True, exist_ok=True)
    specs = [(s.split('=')[0], s.split('=')[1].split('+')) for s in a.specs]
    with Pool(min(a.jobs, len(specs))) as pool: res = pool.map(eval_ens, specs)
    w = pd.concat([r[0] for r in res], ignore_index=True); fr = pd.concat([r[1] for r in res], ignore_index=True)
    have = set(w[w.block.isin(IS_BLOCKS)].groupby('ens').block.nunique().loc[lambda v: v == len(IS_BLOCKS)].index)
    order = [n for n, _ in specs if n in have]
    if len(order) < len(specs): print('skipped (missing blocks):', [n for n, _ in specs if n not in have])
    w = w[w.ens.isin(order)]; fr = fr[fr.ens.isin(order)]
    w.to_csv(od / "weekly.csv.gz", index=False); fr.to_csv(od / "fret_weekly.csv.gz", index=False)     # per-week detail
    IS = w.block.isin(IS_BLOCKS); gi = w[IS].groupby('ens'); go = w[w.block == OOS].groupby('ens')

    ex = []
    for e in order:
        x = w[IS & (w.ens == e)]
        for k in STYLES:
            v = x[f'x_{k}']; mu = v.mean()
            ex.append({'ens': e, 'style': k, 'mean': mu, 't': tstat(v), 'std_w': v.std(), 'same_sign': (np.sign(v) == np.sign(mu)).mean(),
                       'oos': go[f'x_{k}'].mean().get(e, np.nan)})
    ex = pd.DataFrame(ex); ex.to_csv(od / 'expo.csv', index=False)
    xb = w.groupby(['ens', 'block'])[[f'x_{k}' for k in STYLES]].mean().reset_index(); xb.to_csv(od / 'expo_blocks.csv', index=False)

    mu = pd.DataFrame({k: gi[f'b_{k}'].mean() for k in STYLES}).loc[order]
    mt = pd.DataFrame({k: gi[f'b_{k}'].apply(tstat) for k in STYLES}).loc[order]
    mult = pd.concat({'coef': mu, 't': mt}, axis=1)
    for c in ['r2_ind', 'r2_base', 'r2_all']: mult[('r2', c)] = gi[c].mean()
    mult.to_csv(od / 'multi.csv')

    dc = pd.DataFrame({c: gi[c].mean() for c in ['ic', 'nic_ind', 'nic_base', 'nic_all', 'ic_fit']}).loc[order]
    dc['nic_all_ir'] = gi.nic_all.mean() / gi.nic_all.std()
    dc['oos_ic'] = go.ic.mean(); dc['oos_nic_all'] = go.nic_all.mean(); dc['oos_ic_fit'] = go.ic_fit.mean()
    for k in STYLES: dc[f'add_{k}'] = gi[f'add_{k}'].mean()
    for k in STYLES: dc[f'drop_{k}'] = gi[f'drop_{k}'].mean()
    dc.to_csv(od / 'icdec.csv')

    one = w[w.ens == order[0]]; si = []
    for k in STYLES:
        v = one[one.block.isin(IS_BLOCKS)][f'sic_{k}']
        si.append({'style': k, 'ic': v.mean(), 't': tstat(v), 'ic_pos': (v > 0).mean(),
                   'oos_ic': one[one.block == OOS][f'sic_{k}'].mean(),
                   **{bk: one[one.block == bk][f'sic_{k}'].mean() for bk in IS_BLOCKS}})
    pd.DataFrame(si).to_csv(od / 'styic.csv', index=False)

    ann = 252 / 5; out = []
    for e in order:
        for per, sel in [('IS', fr.block.isin(IS_BLOCKS)), ('OOS', fr.block == OOS)]:
            x = fr[(fr.ens == e) & sel].dropna()
            if len(x) < 5: continue
            o = {'ens': e, 'period': per, 'n': len(x)}
            for c in ['raw', 'pure']:
                o[f'{c}_ann'] = x[c].mean() * ann; o[f'{c}_ir'] = x[c].mean() / x[c].std() * np.sqrt(ann); o[f'{c}_t'] = tstat(x[c])
            Fm = np.column_stack([np.ones(len(x)), x[[f'f_{k}' for k in STYLES]].to_numpy()])
            cf, *_ = np.linalg.lstsq(Fm, x.raw.to_numpy(), rcond=None); res_ = x.raw.to_numpy() - Fm @ cf
            o['ts_alpha_ann'] = cf[0] * ann; o['ts_r2'] = 1 - res_.var() / x.raw.var(ddof=0)
            o['corr_raw_pure'] = np.corrcoef(x.raw, x.pure)[0, 1]
            out.append(o)
    pd.DataFrame(out).to_csv(od / 'fret.csv', index=False)
    sf = fr[fr.block.isin(IS_BLOCKS) & (fr.ens == order[0])]
    pd.DataFrame({'ann': sf[[f'f_{k}' for k in STYLES]].mean() * ann,
                  'ir': sf[[f'f_{k}' for k in STYLES]].mean() / sf[[f'f_{k}' for k in STYLES]].std() * np.sqrt(ann)}
                 ).to_csv(od / 'style_ret.csv')

    pd.concat([r[2] for r in res if r[2].name in order], axis=1)[order].to_csv(od / 'ind.csv')

    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
    print('== exposure (mean rank corr, IS)'); print(ex.pivot(index='style', columns='ens', values='mean').loc[STYLES, order].round(3).to_string())
    print('\n== exposure t'); print(ex.pivot(index='style', columns='ens', values='t').loc[STYLES, order].round(1).to_string())
    print('\n== same-sign share'); print(ex.pivot(index='style', columns='ens', values='same_sign').loc[STYLES, order].round(2).to_string())
    print('\n== exposure OOS'); print(ex.pivot(index='style', columns='ens', values='oos').loc[STYLES, order].round(3).to_string())
    print('\n== multivariate coef'); print(mu.T.round(3).to_string())
    print('\n== multivariate t'); print(mt.T.round(1).to_string())
    print('\n== R2'); print(gi[['r2_ind', 'r2_base', 'r2_all']].mean().loc[order].round(3).to_string())
    print('\n== IC decomposition'); print(dc[['ic', 'nic_ind', 'nic_base', 'nic_all', 'nic_all_ir', 'ic_fit', 'oos_ic', 'oos_nic_all', 'oos_ic_fit']].round(4).to_string())
    print('\n== add_ (IC change when only this style is neutralised, beyond industry)'); print(dc[[f'add_{k}' for k in STYLES]].T.round(4).to_string())
    print('\n== drop_ (IC only this style carries, given all others)'); print(dc[[f'drop_{k}' for k in STYLES]].T.round(4).to_string())
    print('\n== style IC vs y10'); print(pd.read_csv(od / 'styic.csv').round(4).to_string(index=False))
    print('\n== style factor returns (FM, IS)'); print(pd.read_csv(od / 'style_ret.csv', index_col=0).round(4).to_string())
    print('\n== factor return of score'); print(pd.read_csv(od / 'fret.csv').round(4).to_string(index=False))
    print('\n== exposure by half-year (CAND-like = 2nd spec)')
    print(xb[xb.ens == order[min(1, len(order) - 1)]].set_index('block')[[f'x_{k}' for k in STYLES]].round(3).T.to_string())
    print('\n== industry tilt (top / bottom 5 of first two specs)')
    ind = pd.read_csv(od / 'ind.csv', index_col=0)
    for e in order[:2]: s = ind[e].sort_values(); print(e, s.head(5).round(3).to_dict(), s.tail(5).round(3).to_dict())


if __name__ == '__main__':
    main()
