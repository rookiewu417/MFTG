"""Materialise an ensemble as a normal run: per-date z-scored average of member predictions on shared (date, stock).
usage: make_ens.py NAME run/model[:weight] run/model[:weight] ...   -> runs/ens/NAME/<block>_pred.npz"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

R = Path('/home/ivenwu/projects/ml-factor/v2/runs')
name, members = sys.argv[1], sys.argv[2:]
parts = [(m.split(':')[0], float(m.split(':')[1]) if ':' in m else 1.0) for m in members]
dst = R / 'ens' / name; dst.mkdir(parents=True, exist_ok=True)
blocks = sorted(set.intersection(*[{f.name.split('_')[0] for f in (R / p).glob('*_pred.npz')} for p, _ in parts]))
for b in blocks:
    frames = []
    for p, w in parts:
        z = np.load(R / p / f'{b}_pred.npz', allow_pickle=True)
        d = pd.DataFrame({'fd': z['feature_date'].astype(int), 'code': z['stock_code'], 'pred': z['pred'].astype(float)})
        d['pred'] = w * d.groupby('fd').pred.transform(lambda v: (v - v.mean()) / (v.std() + 1e-12))
        frames.append(d.set_index(['fd', 'code']).pred)
    m = pd.concat(frames, axis=1).dropna()
    s = m.sum(1) / sum(w for _, w in parts)
    np.savez(dst / f'{b}_pred.npz', pred=s.to_numpy(np.float32), feature_date=s.index.get_level_values(0).to_numpy(np.int32),
             stock=np.zeros(len(s), np.int32), stock_code=s.index.get_level_values(1).to_numpy().astype('<U12'))
print(name, blocks)
