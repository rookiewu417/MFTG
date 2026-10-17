"""Uncertainty-penalised ensemble from seed disagreement (deep-ensemble epistemic uncertainty).

The members' per-seed predictions (seed_preds, each z-scored per date) form a 3 x n_members ensemble. Per stock and date:
m = mean over all seed predictions, s = their std; signal = z(m) - kappa * z(s) (z = per-date z-score).
Writes runs/mftg/_unc/<name>_k<kappa>/<block>_pred.npz (single-member signals; kappa 0 is the plain ensemble written
through the same path, the scale-matched control) and prints, inside the top 30% of m, the RankIC of -s vs y10.
Usage: python mftg/ens_unc.py [--name neu4] [--members Daily-Neu,...] [--kappas 0,0.1,0.25,0.5]
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'shared'))
from common import OUT, load_axes, SHARED

RUNS = OUT.parent / 'runs' / 'mftg'
BLOCKS = ['2022H2', '2023H1', '2023H2', '2024H1', '2024H2', '2025H1', '2025H2', '2026H1', '2026H2', '2026H2X']
NEU4 = 'Daily-Neu,Daily-Neu-H256,Daily-Neu-CSI800,Full-Neu'


def zs(g):
    return (g - g.mean()) / (g.std() + 1e-12)


def block(members, b):
    fr = []
    for k, m in enumerate(members):
        z = np.load(RUNS / m / f'{b}_pred.npz')
        d = pd.DataFrame(z['seed_preds'].T, columns=[f'{k}_{s}' for s in range(z['seed_preds'].shape[0])])
        d['fd'] = z['feature_date']; d['stock'] = z['stock']; d['code'] = z['stock_code']
        fr.append(d.set_index(['fd', 'stock', 'code']))
    d = pd.concat(fr, axis=1, join='inner')
    d = d.groupby(level='fd').transform(zs)          # every seed z-scored on the common stocks
    out = pd.DataFrame({'m': d.mean(1), 's': d.std(1)})
    g = out.groupby(level='fd')
    out['zm'] = g.m.transform(zs); out['zs'] = g.s.transform(zs)
    return out.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='neu4'); ap.add_argument('--members', default=NEU4)
    ap.add_argument('--kappas', default='0,0.1,0.25,0.5')
    a = ap.parse_args(); members = a.members.split(','); kappas = [float(k) for k in a.kappas.split(',')]
    dates, _ = load_axes(); tpos = {d: i for i, d in enumerate(dates.strftime('%Y%m%d').astype(int))}
    y10 = np.load(SHARED / 'base' / 'y10.npy'); univ = np.load(SHARED / 'base' / 'univ.npy')
    rows = []
    for b in BLOCKS:
        d = block(members, b)
        for kp in kappas:
            dst = RUNS / '_unc' / f'{a.name}_k{kp:g}'; dst.mkdir(parents=True, exist_ok=True)
            p = (d.zm - kp * d.zs).astype(np.float32).to_numpy()
            np.savez(dst / f'{b}_pred.npz', pred=p, feature_date=d.fd.to_numpy(np.int32), stock=d.stock.to_numpy(np.int32),
                     stock_code=d.code.to_numpy().astype('<U12'))
        ics, corr = [], []                               # weekly: inside the top 30% of m, does low disagreement mean higher y10?
        for fd, g in list(d.groupby('fd'))[::5]:
            t = tpos[fd]; j = g.stock.to_numpy(); y = y10[t, j]; ok = univ[t, j] & np.isfinite(y)
            if ok.sum() < 100: continue
            g, y = g[ok], y[ok]; top = g.zm.to_numpy() >= np.quantile(g.zm, 0.7)
            ics.append(pd.Series(-g.s.to_numpy()[top]).corr(pd.Series(y[top]), method='spearman'))
            corr.append(pd.Series(g.s.to_numpy()).corr(pd.Series(g.zm.abs().to_numpy()), method='spearman'))
        rows.append({'block': b, 'pool_ic_neg_std': np.mean(ics), 'corr_std_absm': np.mean(corr), 'n': len(ics)})
    r = pd.DataFrame(rows); print(r.round(4).to_string(index=False))
    print('mean over the 8 in-sample blocks:', r[r.block.isin(BLOCKS[:8])][['pool_ic_neg_std', 'corr_std_absm']].mean().round(4).to_dict())


if __name__ == '__main__':
    main()
