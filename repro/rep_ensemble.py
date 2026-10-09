"""Seed-ensemble the early-window reproduction runs (runs/repro/<model>_s<seed>/<model>_<block>_pred.npz) like
baseline_src/ensemble_predictions.py (per-date z-score, then mean), and assemble runs/repro/rep8/<model>/<block>_pred.npz:
early windows from these ensembles, 2024H2-2026H1 from the formal reproduction ensembles."""
from pathlib import Path
import numpy as np
import pandas as pd

R = Path('/home/ivenwu/projects/MFTG/runs'); BASE = Path('/home/ivenwu/projects/MFTG/baseline_preds/formal_ensembles')
EARLY = ['2022H2', '2023H1', '2023H2', '2024H1']; MAIN = ['2024H2', '2025H1', '2025H2', '2026H1']
for m in ['incremental', 'daily', 'mixed', 'gru15']:
    out = R / 'repro' / 'rep8' / m; out.mkdir(parents=True, exist_ok=True)
    for b in EARLY:
        fr = []
        for s in (7, 11, 42):                     # mixed / gru15 early windows only have seed 7
            f = R / 'repro' / f'{m}_s{s}' / f'{m}_{b}_pred.npz'
            if not f.exists(): continue
            z = np.load(f, allow_pickle=True)
            d = pd.DataFrame({'fd': z['feature_date'].astype(int), 'code': z['stock_code'].astype(str), 'p': z['pred'].astype(float)})
            d['p'] = d.groupby('fd').p.transform(lambda v: (v - v.mean()) / (v.std() + 1e-12))
            fr.append(d.set_index(['fd', 'code']).p.rename(s))
        e = pd.concat(fr, axis=1).dropna().mean(1)
        np.savez(out / f'{b}_pred.npz', pred=e.to_numpy(np.float32), feature_date=e.index.get_level_values(0).to_numpy(np.int32),
                 stock_code=e.index.get_level_values(1).to_numpy().astype('<U16'))
    for b in MAIN:
        dst = out / f'{b}_pred.npz'
        if not dst.exists(): dst.symlink_to(BASE / f'{m}_{b}_ensemble.npz')
    print(m, 'ok', {b: sorted(p.parent.name for p in (R / 'repro').glob(f'{m}_s*/{m}_{b}_pred.npz')) for b in EARLY}, flush=True)
