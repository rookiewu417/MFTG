"""Post-hoc style/industry neutralisation of prediction files: residualise each day's scores on the risk.py styles
(+ industry dummies, optionally a subset of styles) and write them as a new run under runs/neut_<styles>/<model>/."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import backtest as bt
from risk import STY, zs

R = Path('/home/ivenwu/projects/ml-factor/v2/runs')
ap = argparse.ArgumentParser(); ap.add_argument('models', nargs='+'); ap.add_argument('--styles', default='all'); ap.add_argument('--ind', type=int, default=1)
a = ap.parse_args()
styles = list(STY) if a.styles == 'all' else a.styles.split(',')
tag = f"neut_{'all' if a.styles == 'all' else '+'.join(styles)}{'' if a.ind else '_noind'}"
for m in a.models:
    src = R / m; dst = R / tag / m.replace('/', '_'); dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.glob('*_pred.npz')):
        z = np.load(f, allow_pickle=True); fd = z['feature_date'].astype(int); pred = z['pred'].astype(float).copy()
        j_all = np.array([bt.cpos.get(c, -1) for c in z['stock_code']])
        for d in np.unique(fd):
            ii = np.where((fd == d) & (j_all >= 0))[0]; t = bt.tpos[d]; j = j_all[ii]
            X = [np.ones(len(j))] + [zs(STY[s][t, j]) for s in styles]
            if a.ind:
                ind = bt.industry[t, j]; X += [(ind == k).astype(float) for k in np.unique(ind[ind >= 0])]
            X = np.column_stack(X); p = pred[ii]
            beta, *_ = np.linalg.lstsq(X, p, rcond=None); pred[ii] = p - X @ beta
        np.savez(dst / f.name, pred=pred.astype(np.float32), feature_date=z['feature_date'], stock=z['stock'], stock_code=z['stock_code'])
    print('written', dst, flush=True)
