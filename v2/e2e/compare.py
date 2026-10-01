"""Compare prediction sets (factor track vs e2e track vs blends) on shared windows.

usage: pixi run python e2e/compare.py BLOCKS name=path[+path...] ...
  BLOCKS e.g. 2024H2,2025H2; path relative to runs/ (dir holding <block>_pred.npz); '+' = per-date z-score average.
Reports per-block RankIC / ICIR on y10 (tradable universe), and the all-A long-only portfolio (pf.py, daily rebalance,
EW top-decile with buffer, and the style-constrained optimiser) over the same blocks.
"""
import sys
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import backtest as bt
import pf

R = Path('/home/ivenwu/projects/ml-factor/v2/runs')
Y = np.load('/home/ivenwu/projects/ml-factor/v2/build/base/y10.npy', mmap_mode='r')
SPAN = {'2022H2': (20220701, 20221231), '2023H1': (20230101, 20230630), '2023H2': (20230701, 20231231), '2024H1': (20240101, 20240630),
        '2024H2': (20240701, 20241231), '2025H1': (20250101, 20250630), '2025H2': (20250701, 20251231), '2026H1': (20260101, 20260630),
        '2026H2': (20260701, 20260731)}


def load(spec, blocks):
    frames = [bt.load_scores({b: str(R / p / f'{b}_pred.npz') for b in blocks}).set_index(['fd', 'j']).pred for p in spec.split('+')]
    return pd.concat(frames, axis=1).dropna().mean(1).rename('pred').reset_index()


def ic_table(df):
    rows = []
    for fd, g in df.groupby('fd'):
        t = bt.tpos[fd]; y = Y[t, g.j.to_numpy()]; m = np.isfinite(y) & bt.univ[t, g.j.to_numpy()]
        if m.sum() > 100: rows.append((fd, pd.Series(g.pred.to_numpy()[m]).rank().corr(pd.Series(y[m]).rank())))
    return pd.DataFrame(rows, columns=['fd', 'ic'])


if __name__ == '__main__':
    blocks = sys.argv[1].split(','); specs = dict(a.split('=', 1) for a in sys.argv[2:])
    out = []
    corr_ref = {}
    for name, spec in specs.items():
        df = load(spec, blocks); ic = ic_table(df)
        row = {'model': name}
        for b in blocks:
            s, e = SPAN[b]; x = ic[(ic.fd >= s) & (ic.fd <= e)].ic
            row[f'ic_{b}'] = x.mean(); row[f'icir_{b}'] = x.mean() / x.std()
        row['ic'] = ic.ic.mean(); row['icir'] = ic.ic.mean() / ic.ic.std()
        S = bt.score_panel(df, 3.0)
        s0, e0 = SPAN[blocks[0]][0], SPAN[blocks[-1]][1]
        for label, cfg in {'ew': replace(pf.PCfg(), hold=1), 'opt': replace(pf.PCfg(), hold=1, weight='opt', max_w=0.005, tc_pen=0.5,
                                                                              sty_caps=(('size', 0.05), ('beta', 0.05), ('volatility', 0.05)))}.items():
            r = pf.run(S, s0, e0, cfg)
            r = r[r.date.isin([d for d in r.date if any(SPAN[b][0] <= d <= SPAN[b][1] for b in blocks)])]
            st = pf.stats(r)
            row[f'{label}_ann'] = st['ann']; row[f'{label}_ir'] = st['ir']; row[f'{label}_dd'] = st['maxdd']
        corr_ref[name] = df.set_index(['fd', 'j']).pred
        out.append(row); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    res = pd.DataFrame(out)
    pd.set_option('display.width', 250); print(res.round(3).to_string(index=False))
    # average cross-sectional correlation between the prediction sets
    names = list(corr_ref)
    if len(names) > 1:
        M = pd.concat(corr_ref, axis=1).dropna()
        C = M.groupby(level=0).corr().groupby(level=1).mean().loc[names, names]
        print('mean daily cross-sectional correlation\n', C.round(2).to_string())
