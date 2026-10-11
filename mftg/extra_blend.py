"""Linear test of the extra inputs before any training: walk-forward combination of the features' incremental ICs,
added to the MFTG-Ens-Neu score; writes prediction files the portfolio layer reads (runs/mftg/xblend/<name>/<block>_pred.npz).

For day t the feature weights are the mean incremental ICs (extra_ic.py definition: feature and label residualised on
industry, size and the ensemble score) over every 5th day d <= t - GAP whose y10 label has ended (d + 11 trading days < t),
from the first test day on; features whose running |t| < TMIN get weight 0. Before 120 such days, weight 0 (pure ensemble).
combo_t = sum_k w_k x_k / sum_k |w_k|, residualised on industry + size + ensemble, z-scored; score = z(ens) + lam * combo.
z(ens) keeps the ensemble's own shape (strongly left-skewed: bottom tail to -6, top ~1): the optimiser's turnover penalty
is in these units, so a rank-gauss version trades far more (control b0 must reproduce the recommended book).

--pool top: weights from the incremental ICs inside the top 30% of the ensemble only (the long-only candidate pool);
the full-cross-section weights are dominated by features that only sort the middle / bottom (e.g. dragon-tiger listings).

usage: pixi run python mftg/extra_blend.py [--runs DIR] [--lams 0,0.05,0.1,0.2,0.4] [--pool all|top]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'shared'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, load_axes, SHARED
from extra_ic import ens_scores, rg, HALVES

BLOCKS = {b: None for b in HALVES + ['2026H2', '2026H2X']}
H, MIN_DAYS, TMIN = 11, 24, 2.0       # MIN_DAYS of every-5th-day ICs ~ 120 trading days


def block_of(d):
    if d > 20260630: return None
    return f'{d // 10000}H{1 if d % 10000 < 701 else 2}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default=str(Path(__file__).resolve().parents[1] / 'runs' / 'mftg'))
    ap.add_argument('--lams', default='0,0.05,0.1,0.2,0.4')
    ap.add_argument('--pool', default='all', choices=['all', 'top'])
    a = ap.parse_args(); lams = [float(x) for x in a.lams.split(',')]
    dates, codes = load_axes(); dint = dates.strftime('%Y%m%d').astype(int).to_numpy(); cpos = {c: i for i, c in enumerate(codes)}
    Bd = SHARED / 'base'
    univ = np.load(Bd / 'univ.npy'); y10 = np.load(Bd / 'y10.npy'); ind = np.load(Bd / 'industry.npy'); lmv = np.load(Bd / 'log_circ_mv.npy')
    X = np.load(OUT / 'mftg' / 'extra.npy', mmap_mode='r'); names = json.loads((OUT / 'mftg' / 'extra_names.json').read_text())
    K = len(names)
    # the ensemble over every block incl. the July holdout and the extended OOS window
    import extra_ic; extra_ic.HALVES = list(BLOCKS)
    ens = ens_scores(a.runs, dint, cpos)
    days = sorted(ens)
    res = lambda v, M: v - M @ np.linalg.lstsq(M, v, rcond=None)[0]
    hist_t, hist_ic = [], []          # every-5th-day incremental ICs, with their label end index
    out = {lam: {} for lam in lams}
    wlog = []
    for n, t in enumerate(days):
        j, s = ens[t]
        lm = lmv[t, j]; lm = np.where(np.isfinite(lm), lm, np.nanmean(lm)); ii = ind[t, j]
        C = np.stack([np.ones(len(j)), (lm - lm.mean()) / lm.std()] + [(ii == k).astype(float) for k in np.unique(ii[ii >= 0])], 1)
        Cs = np.c_[C, rg(s)]
        xs = np.asarray(X[t, j], np.float64)
        # weights from labels already known at t
        known = [i for i, te in enumerate(hist_t) if te < t]
        w = np.zeros(K)
        if len(known) >= MIN_DAYS:
            M = np.array([hist_ic[i] for i in known]); mu = np.nanmean(M, 0); sd = np.nanstd(M, 0) + 1e-12
            tt = mu / sd * np.sqrt(np.isfinite(M).sum(0)); w = np.where(np.abs(tt) >= TMIN, mu, 0.0)
        if n % 20 == 0: wlog.append({'date': dint[t], 'n_hist': len(known), **dict(zip(names, np.round(w, 4)))})
        z = (s - s.mean()) / (s.std() + 1e-12)
        if np.abs(w).sum() > 0:
            c = res(xs @ w / np.abs(w).sum(), Cs); c = c / (c.std() + 1e-12)
        else:
            c = np.zeros(len(j))
        for lam in lams: out[lam].setdefault(t, (j, z + lam * c))
        # this day's incremental ICs (every 5th day), usable once the label has ended
        if n % 5 == 0 and dint[t] <= 20260630:
            m = univ[t, j] & np.isfinite(y10[t, j])
            if a.pool == 'top': m &= pd.Series(s).rank(pct=True).to_numpy() > 0.7
            if m.sum() > (100 if a.pool == 'top' else 300):
                y = rg(y10[t, j][m]); Cm = Cs[m]; yr = res(y, Cm)
                ic = np.array([np.corrcoef(res(xs[m, k], Cm), yr)[0, 1] if xs[m, k].std() > 1e-6 else np.nan for k in range(K)])
                hist_t.append(t + H); hist_ic.append(ic)
    for lam in lams:
        d = Path(a.runs) / 'xblend' / f'{"t" if a.pool == "top" else "b"}{lam:g}'; d.mkdir(parents=True, exist_ok=True)
        rows = [(dint[t], j, p) for t, (j, p) in out[lam].items()]
        for b in BLOCKS:
            if b == '2026H2X': sel = [r for r in rows if 20260701 <= r[0] <= 20260910]
            elif b == '2026H2': sel = [r for r in rows if 20260701 <= r[0] <= 20260731]
            else: sel = [r for r in rows if block_of(r[0]) == b]
            fd = np.concatenate([np.full(len(j), x) for x, j, _ in sel]).astype(np.int32)
            st = np.concatenate([j for _, j, _ in sel]).astype(np.int32)
            pr = np.concatenate([p for _, _, p in sel]).astype(np.float32)
            np.savez(d / f'{b}_pred.npz', pred=pr, feature_date=fd, stock=st, stock_code=codes[st].astype('<U12'))
    pd.DataFrame(wlog).to_csv(Path(a.runs) / 'xblend' / f'weights_{a.pool}.csv', index=False)
    print('written', [f'{"t" if a.pool == "top" else "b"}{l:g}' for l in lams], 'weight snapshots:', len(wlog))


if __name__ == '__main__':
    main()
