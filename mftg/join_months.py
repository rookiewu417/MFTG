"""Assemble monthly-refit outputs (M<yyyymm>_pred.npz) into half-year block files, and extract a single seed from a
3-seed half-year run so the two can be compared at equal ensemble size.
usage: pixi run python mftg/join_months.py months <run_dir>            -> <run_dir>/<block>_pred.npz   (run_dir relative to runs/, e.g. mftg/dNH_M1)
       pixi run python mftg/join_months.py seed <run_dir> <k> <out_dir> -> seed index k of every block file"""
import sys
from pathlib import Path
import numpy as np

R = Path('/home/ivenwu/projects/MFTG/runs')
SPAN = {'2022H2': (20220701, 20221231), '2023H1': (20230101, 20230630), '2023H2': (20230701, 20231231), '2024H1': (20240101, 20240630),
        '2024H2': (20240701, 20241231), '2025H1': (20250101, 20250630), '2025H2': (20250701, 20251231), '2026H1': (20260101, 20260630),
        '2026H2X': (20260701, 20260910)}
KEYS = ['pred', 'feature_date', 'stock', 'stock_code']
if sys.argv[1] == 'months':
    d = R / sys.argv[2]
    for b, (s, e) in SPAN.items():
        fs = [f for f in sorted(d.glob('M*_pred.npz')) if s // 100 <= int(f.name[1:7]) <= e // 100]
        if not fs: continue
        z = [np.load(f) for f in fs]
        np.savez(d / f'{b}_pred.npz', **{k: np.concatenate([x[k] for x in z]) for k in KEYS})
        print(b, len(fs), 'months')
else:
    d, k, o = R / sys.argv[2], int(sys.argv[3]), R / sys.argv[4]; o.mkdir(parents=True, exist_ok=True)
    for f in sorted(d.glob('*_pred.npz')):
        if f.name.startswith('M'): continue
        z = np.load(f); np.savez(o / f.name, pred=z['seed_preds'][k], **{x: z[x] for x in KEYS[1:]})
        print(f.name)
