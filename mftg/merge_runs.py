"""Merge runs of the same configuration trained with different seed sets into one member (more seeds, less noise).

Each run's pred is z-scored per date; the merged pred is their mean over the common (date, stock) rows, seed_preds are
concatenated. Writes runs/mftg/_merge/<name>/<block>_pred.npz for every block all runs have.
Usage: python mftg/merge_runs.py NAME run1 run2 [...]   (runs relative to runs/mftg, e.g. e2e/dDMV e2e/dDMV_s2)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'shared'))
from common import OUT

RUNS = OUT.parent / 'runs' / 'mftg'


def load(run, b):
    z = np.load(RUNS / run / f'{b}_pred.npz')
    d = pd.DataFrame({'fd': z['feature_date'], 'stock': z['stock'], 'code': z['stock_code'], 'p': z['pred'].astype(float)})
    d['p'] = d.groupby('fd').p.transform(lambda v: (v - v.mean()) / (v.std() + 1e-12))
    s = pd.DataFrame(z['seed_preds'].T, columns=[f'{run}:{i}' for i in range(z['seed_preds'].shape[0])])
    return pd.concat([d, s], axis=1).set_index(['fd', 'stock', 'code'])


def main():
    name, runs = sys.argv[1], sys.argv[2:]
    blocks = sorted(set.intersection(*[{f.name[:-9] for f in (RUNS / r).glob('*_pred.npz')} for r in runs]))
    dst = RUNS / '_merge' / name; dst.mkdir(parents=True, exist_ok=True)
    for b in blocks:
        fr = [load(r, b) for r in runs]
        d = pd.concat([f.p.rename(i) for i, f in enumerate(fr)], axis=1, join='inner')
        seeds = pd.concat([f.drop(columns='p') for f in fr], axis=1, join='inner').loc[d.index]
        idx = d.index.to_frame(index=False)
        np.savez(dst / f'{b}_pred.npz', pred=d.mean(1).to_numpy(np.float32), feature_date=idx.fd.to_numpy(np.int32),
                 stock=idx.stock.to_numpy(np.int32), stock_code=idx.code.to_numpy().astype('<U12'),
                 seed_preds=seeds.to_numpy(np.float32).T)
        print(b, len(d), 'rows,', seeds.shape[1], 'seeds')


if __name__ == '__main__':
    main()
