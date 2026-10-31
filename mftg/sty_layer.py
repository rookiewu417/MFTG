"""Per-style shares and layered tests of MFTG ensembles (same ensembles / weeks / universe / styles as sty_eval.py).

Exact additive split each week, with zp = ranked score and ry = ranked y10, both standardised, so IC = mean(zp * ry)
(Pearson on ranks = RankIC); zp = industry part + sum_k b_k S_k + residual (OLS on industry dummies + 14 styles):
  IC        = mean(ind_part * ry) + sum_k b_k mean(S_k * ry) + mean(resid * ry)
  var share = cov(part, zp) / var(zp)   (sums to R^2 + residual share = 1)
Writes to --out:
  share.csv        per ensemble x component: IC contribution (IS mean, t, OOS) and share of IC; variance share
  share_blocks.csv per ensemble x half-year x component: IC contribution
  deciles.csv      10 groups of the raw score and of the pure residual: mean y10 excess over the universe, annualised
                   (x 252/10), long-short, monotonicity; IS and OOS
  layers.csv       universe split into 5 layers by each main style: RankIC of the score inside the layer and
                   top-minus-bottom quintile y10 spread (annualised) inside the layer
  *_weekly.csv.gz  per-week rows behind share / deciles / layers
Usage: python mftg/sty_layer.py NAME=m1+m2+... [NAME=...] --out DIR [--jobs 8]
"""
import argparse
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from sig_eval import ensemble, rank, ric, resid, zrank, univ, Y10, IS_BLOCKS, OOS
from sty_eval import styles, inddum, STYLES

LAYER_STY = ['size', 'beta', 'volatility', 'liquidity', 'momentum', 'rev20', 'rev5', 'bp', 'ep', 'growth']
COMP = ['industry'] + STYLES + ['pure']


def std(x): return (x - x.mean()) / (x.std() + 1e-12)


def deciles(s, ex, n):
    q = np.minimum((rank(s) - 1) * 10 // n, 9).astype(int)
    return [ex[q == k].mean() for k in range(10)]


def eval_ens(args):
    name, members = args
    sh, dec, lay = [], [], []
    for b in IS_BLOCKS + [OOS]:
        e = ensemble(members, b)
        if e is None: continue
        for t in np.sort(e.index.get_level_values(0).unique())[::5]:
            g = e.loc[t]; j = g.index.to_numpy(); p = g.to_numpy(); u = univ[t, j]; j, p = j[u], p[u]
            y = Y10[t, j]; m = np.isfinite(y)
            if m.sum() < 100: continue
            j, p, y = j[m], p[m], y[m]; n = len(j)
            S = styles(t, j); D, _ = inddum(t, j); Sa = S.to_numpy()
            zp = std(zrank(p)); ry = std(rank(y)); X = np.column_stack([D, Sa])
            coef, *_ = np.linalg.lstsq(X, zp, rcond=None); nd = D.shape[1]
            parts = {'industry': D @ coef[:nd]}
            for i, k in enumerate(STYLES): parts[k] = Sa[:, i] * coef[nd + i]
            parts['pure'] = zp - X @ coef
            r = {'ens': name, 'block': b, 't': t, 'ic': (zp * ry).mean()}
            for k, v in parts.items(): r[f'ic_{k}'] = (v * ry).mean(); r[f'var_{k}'] = (v * zp).mean() / (zp * zp).mean()
            sh.append(r)
            ex = y - y.mean()
            dec.append({'ens': name, 'block': b, 't': t, 'kind': 'raw', **{f'D{k + 1}': v for k, v in enumerate(deciles(p, ex, n))}})
            dec.append({'ens': name, 'block': b, 't': t, 'kind': 'pure', **{f'D{k + 1}': v for k, v in enumerate(deciles(parts['pure'], ex, n))}})
            for k in LAYER_STY:
                s = S[k].to_numpy(); L = np.minimum((rank(s) - 1) * 5 // n, 4).astype(int)
                for l in range(5):
                    mm = L == l
                    if mm.sum() < 100: continue
                    pl, yl = p[mm], y[mm]; ql = rank(pl) / len(pl)
                    lay.append({'ens': name, 'block': b, 't': t, 'style': k, 'layer': l + 1, 'ic': ric(pl, yl),
                                'spread': yl[ql > 0.8].mean() - yl[ql <= 0.2].mean(), 'ret': yl.mean() - y.mean()})
    return pd.DataFrame(sh), pd.DataFrame(dec), pd.DataFrame(lay)


def tstat(v): v = v.dropna(); return v.mean() / v.std() * np.sqrt(len(v))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('specs', nargs='+'); ap.add_argument('--out', required=True)
    ap.add_argument('--jobs', type=int, default=8)
    a = ap.parse_args(); od = Path(a.out); od.mkdir(parents=True, exist_ok=True)
    specs = [(s.split('=')[0], s.split('=')[1].split('+')) for s in a.specs]; order = [n for n, _ in specs]
    with Pool(min(a.jobs, len(specs))) as pool: res = pool.map(eval_ens, specs)
    sh, dec, lay = (pd.concat([r[i] for r in res], ignore_index=True) for i in range(3))
    for nm, df in [("share", sh), ("deciles", dec), ("layers", lay)]: df.to_csv(od / f"{nm}_weekly.csv.gz", index=False)     # per-week detail
    IS = sh.block.isin(IS_BLOCKS)

    rows = []
    for e in order:
        x = sh[IS & (sh.ens == e)]; xo = sh[(sh.block == OOS) & (sh.ens == e)]
        for c in COMP:
            rows.append({'ens': e, 'comp': c, 'ic': x[f'ic_{c}'].mean(), 't': tstat(x[f'ic_{c}']),
                         'ic_share': x[f'ic_{c}'].mean() / x.ic.mean(), 'var_share': x[f'var_{c}'].mean(),
                         'oos_ic': xo[f'ic_{c}'].mean(), 'oos_share': xo[f'ic_{c}'].mean() / xo.ic.mean()})
        rows.append({'ens': e, 'comp': 'total', 'ic': x.ic.mean(), 't': tstat(x.ic), 'ic_share': 1.0,
                     'var_share': sum(x[f'var_{c}'].mean() for c in COMP), 'oos_ic': xo.ic.mean(), 'oos_share': 1.0})
    share = pd.DataFrame(rows); share.to_csv(od / 'share.csv', index=False)
    sb = sh.groupby(['ens', 'block'])[['ic'] + [f'ic_{c}' for c in COMP]].mean().reset_index(); sb.to_csv(od / 'share_blocks.csv', index=False)

    dc = [f'D{k}' for k in range(1, 11)]; out = []
    for (e, kind), x in dec.groupby(['ens', 'kind']):
        for per, sel in [('IS', x.block.isin(IS_BLOCKS)), ('OOS', x.block == OOS)]:
            v = x[sel][dc].mean() * 252 / 10
            out.append({'ens': e, 'kind': kind, 'period': per, **v.to_dict(), 'LS': v.D10 - v.D1,
                        'mono': np.corrcoef(np.arange(10), rank(v.to_numpy()))[0, 1],
                        'LS_t': tstat((x[sel].D10 - x[sel].D1))})
    dd = pd.DataFrame(out); dd.to_csv(od / 'deciles.csv', index=False)

    li = lay[lay.block.isin(IS_BLOCKS)]
    ly = li.groupby(['ens', 'style', 'layer']).agg(ic=('ic', 'mean'), spread=('spread', 'mean'), ret=('ret', 'mean')).reset_index()
    ly['ic_t'] = li.groupby(['ens', 'style', 'layer']).ic.apply(tstat).to_numpy()
    ly['spread'] *= 252 / 10; ly['ret'] *= 252 / 10
    lo = lay[lay.block == OOS].groupby(['ens', 'style', 'layer']).ic.mean().rename('oos_ic').reset_index()
    ly = ly.merge(lo, on=['ens', 'style', 'layer'], how='left'); ly.to_csv(od / 'layers.csv', index=False)

    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
    print('== IC contribution (IS mean)'); print(share.pivot(index='comp', columns='ens', values='ic').loc[COMP + ['total'], order].round(4).to_string())
    print('\n== share of IC'); print(share.pivot(index='comp', columns='ens', values='ic_share').loc[COMP + ['total'], order].round(3).to_string())
    print('\n== t of contribution'); print(share.pivot(index='comp', columns='ens', values='t').loc[COMP + ['total'], order].round(1).to_string())
    print('\n== share of score variance'); print(share.pivot(index='comp', columns='ens', values='var_share').loc[COMP + ['total'], order].round(3).to_string())
    print('\n== OOS IC contribution'); print(share.pivot(index='comp', columns='ens', values='oos_ic').loc[COMP + ['total'], order].round(4).to_string())
    for e in order:
        print(f'\n== IC contribution by half-year: {e}')
        x = sb[sb.ens == e].set_index('block')
        style_sum = x[[f'ic_{k}' for k in STYLES]].sum(1)
        print(pd.DataFrame({'ic': x.ic, 'industry': x.ic_industry, 'styles': style_sum, 'volatility': x.ic_volatility,
                            'liquidity': x.ic_liquidity, 'size': x.ic_size, 'rev20': x.ic_rev20, 'rev5': x.ic_rev5,
                            'bp': x.ic_bp, 'pure': x.ic_pure}).round(4).to_string())
    print('\n== deciles (annualised y10 excess)'); print(dd.round(3).to_string(index=False))
    for e in order:
        print(f'\n== layers IS: {e} (RankIC inside each style layer, 1 = lowest)')
        print(ly[ly.ens == e].pivot(index='style', columns='layer', values='ic').loc[LAYER_STY].round(3).to_string())
        print('top-bottom quintile spread inside layer (annualised)')
        print(ly[ly.ens == e].pivot(index='style', columns='layer', values='spread').loc[LAYER_STY].round(3).to_string())
        print('OOS RankIC inside layer')
        print(ly[ly.ens == e].pivot(index='style', columns='layer', values='oos_ic').loc[LAYER_STY].round(3).to_string())


if __name__ == '__main__':
    main()
