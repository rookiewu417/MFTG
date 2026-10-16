"""Offline comparison of equal-weight ensembles of MFTG runs (member = path under runs/mftg, e.g. Daily-Neu or e2e/dL510).

Each member is z-scored per date, members are averaged over the stocks they share. Scored on weekly cross-sections
(every 5th prediction date) of each block: RankIC vs y10 on the tradable universe and the top-decile y10 excess
(annualised, no costs) and the RankIC inside the signal's top 30% (pool_ic). Usage: python mftg/ens_eval.py NAME=m1+m2+... [NAME=...] [--blocks ...]
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'shared'))
from common import OUT, load_axes, SHARED

RUNS = OUT.parent / 'runs' / 'mftg'
BLOCKS = ['2022H2', '2023H1', '2023H2', '2024H1', '2024H2', '2025H1', '2025H2', '2026H1']
dates, codes = load_axes(); dint = dates.strftime('%Y%m%d').astype(int).to_numpy(); tpos = {d: i for i, d in enumerate(dint)}
univ = np.load(SHARED / 'base' / 'univ.npy'); y10 = np.load(SHARED / 'base' / 'y10.npy')
rank = lambda a: pd.Series(a).rank().to_numpy()


def member(m, b):
    z = np.load(RUNS / m / f'{b}_pred.npz')
    d = pd.DataFrame({'t': [tpos[x] for x in z['feature_date']], 'j': z['stock'].astype(int), 'p': z['pred'].astype(float)})
    d['p'] = d.groupby('t').p.transform(lambda v: (v - v.mean()) / (v.std() + 1e-12))
    return d.set_index(['t', 'j']).p


def score(members, b):
    e = pd.concat([member(m, b) for m in members], axis=1).dropna().mean(1)
    ts = np.sort(e.index.get_level_values(0).unique())[::5]; ics, tops, pools = [], [], []
    for t in ts:
        g = e.loc[t]; j = g.index.to_numpy(); p = g.to_numpy(); m = univ[t, j] & np.isfinite(y10[t, j]); j, p = j[m], p[m]
        if len(j) < 100: continue      # label not known yet (end of the data)
        y = y10[t, j]; ics.append(np.corrcoef(rank(p), rank(y))[0, 1]); k = len(j) // 10; tops.append(y[np.argsort(p)[-k:]].mean() - y.mean())
        top = np.argsort(p)[-int(len(j) * 0.3):]; pools.append(np.corrcoef(rank(p[top]), rank(y[top]))[0, 1])   # inside the long-only candidate pool
    return np.mean(ics), np.mean(tops) * 252 / 10, np.mean(pools)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('specs', nargs='+'); ap.add_argument('--blocks', default=','.join(BLOCKS))
    a = ap.parse_args(); blocks = a.blocks.split(','); rows = []
    for spec in a.specs:
        name, ms = spec.split('=')
        for b in blocks:
            ic, top, pool = score(ms.split('+'), b); rows.append({'ens': name, 'block': b, 'ic': ic, 'top': top, 'pool_ic': pool})
    df = pd.DataFrame(rows); base = a.specs[0].split('=')[0]
    w = df.pivot(index='block', columns='ens', values='ic')[[s.split('=')[0] for s in a.specs]]
    print(w.round(4).to_string())
    s = df.groupby('ens')[['ic', 'top', 'pool_ic']].mean().loc[w.columns]
    s['d_ic'] = s.ic - s.loc[base, 'ic']; s['wins'] = [(w[c] > w[base]).sum() for c in w.columns]
    print(); print(s.round(4).to_string())


if __name__ == '__main__':
    main()
