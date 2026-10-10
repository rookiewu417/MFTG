"""Evaluate prediction files (MFTG runs and the baseline ensembles) on one common protocol.

Dates: the baseline's own test feature dates (every 5 trading days), so every model is scored on identical cross-sections.
Universe: tradable universe (non-ST, listed >= 120 days, traded, non-BJ) and buyable at the t+1 open.
Metrics per block: RankIC vs y10 (open->open, tradable), ICIR, RankIC vs y10cc (baseline's close->close label),
industry+size neutral RankIC, top-decile excess 10d return vs the universe mean, and a 5-day-rebalance top-decile
portfolio (y5 hold returns, 10bp one-way cost) with excess over the equal-weight universe.
"""
import argparse, glob, json
from pathlib import Path
import numpy as np
import pandas as pd
from common import OUT, load_axes, SHARED

BLOCKS = ['2024H2', '2025H1', '2025H2', '2026H1']
B = SHARED / 'base'
dates, codes = load_axes()
dint = dates.strftime('%Y%m%d').astype(int).to_numpy()
tpos = {d: i for i, d in enumerate(dint)}
cpos = {c: i for i, c in enumerate(codes)}
univ = np.load(B / 'univ.npy'); y10 = np.load(B / 'y10.npy'); y5 = np.load(B / 'y5.npy'); y10cc = np.load(B / 'y10cc.npy')
ind = np.load(B / 'industry.npy'); lmv = np.load(B / 'log_circ_mv.npy'); idxm = np.load(B / 'index_member.npy')


def rank(a): return pd.Series(a).rank().to_numpy()
def ric(p, y):
    m = np.isfinite(p) & np.isfinite(y)
    return np.corrcoef(rank(p[m]), rank(y[m]))[0, 1] if m.sum() > 50 else np.nan

def neutralize(p, t, j):
    X = [np.ones(len(j))]
    ii = ind[t, j]
    for k in np.unique(ii[ii >= 0]): X.append((ii == k).astype(float))
    lm = lmv[t, j]; lm = np.where(np.isfinite(lm), lm, np.nanmean(lm)); X.append(lm - lm.mean())
    X = np.stack(X, 1); z = rank(p)
    beta, *_ = np.linalg.lstsq(X, z, rcond=None)
    return z - X @ beta


def load_pred(path):
    z = np.load(path, allow_pickle=True)
    fd = z['feature_date'].astype(int)
    st = np.array([cpos.get(c, -1) for c in z['stock_code']])
    return pd.DataFrame({'fd': fd, 'j': st, 'pred': z['pred'].astype(float)}).query('j >= 0')


def load_any(f):
    if isinstance(f, str): return load_pred(f)
    frames = []
    for k, p in enumerate(f):
        d = load_pred(p)
        d['pred'] = d.groupby('fd').pred.transform(lambda v: (v - v.mean()) / (v.std() + 1e-12))
        frames.append(d.set_index(['fd', 'j']).pred.rename(k))
    m = pd.concat(frames, axis=1).dropna()
    return pd.DataFrame({'fd': m.index.get_level_values(0), 'j': m.index.get_level_values(1), 'pred': m.mean(1).to_numpy()})


def eval_block(df, eval_dates, sub_mask=None, cost=0.001):
    rows = []; prev_top = None; nav = []
    for d in eval_dates:
        t = tpos[d]
        g = df[df.fd == d]
        if len(g) == 0: continue
        j = g.j.to_numpy(); p = g.pred.to_numpy()
        m = univ[t, j] & np.isfinite(y10[t, j])
        if sub_mask is not None: m &= sub_mask(t, j)
        j, p = j[m], p[m]
        if len(j) < 100: continue
        yy = y10[t, j]
        r = {'date': d, 'n': len(j), 'ic': ric(p, yy), 'ic_cc': ric(p, y10cc[t, j]), 'ic_neu': ric(neutralize(p, t, j), yy)}
        k = max(1, len(j) // 10); o = np.argsort(p)
        top, bot = j[o[-k:]], j[o[:k]]
        r['top_ex10'] = np.nanmean(y10[t, top]) - np.nanmean(yy)
        r['ls10'] = np.nanmean(y10[t, top]) - np.nanmean(y10[t, bot])
        h = np.nan_to_num(y5[t, top]); bench = np.nan_to_num(y5[t, j]).mean()   # missing hold return (suspended at t+6) counts as 0 on both sides
        to = 1.0 if prev_top is None else 1 - len(np.intersect1d(top, prev_top)) / len(top)
        r['turnover'] = to
        r['port_gross'] = h.mean(); r['port_net'] = h.mean() - 2 * cost * to; r['bench'] = bench
        prev_top = top
        rows.append(r)
    return pd.DataFrame(rows)


def summarize(r):
    ic = r.ic.dropna()
    nav = np.prod(1 + r.port_net); bnav = np.prod(1 + r.bench); ex = (1 + r.port_net) / (1 + r.bench) - 1
    per_year = 252 / 5
    return {'ic': ic.mean(), 'icir': ic.mean() / ic.std(), 'ic_cc': r.ic_cc.mean(), 'ic_neu': r.ic_neu.mean(),
            'icir_neu': r.ic_neu.mean() / r.ic_neu.std(),
            'top_ex10_ann': r.top_ex10.mean() * 252 / 10, 'ls10_ann': r.ls10.mean() * 252 / 10,
            'port_net_ann': nav ** (per_year / len(r)) - 1, 'excess_ann': np.prod(1 + ex) ** (per_year / len(r)) - 1,
            'excess_ir': ex.mean() / ex.std() * np.sqrt(per_year), 'turnover': r.turnover.iloc[1:].mean(), 'n_dates': len(r)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='*', help='run dirs (each with <model>/<block>_pred.npz)')
    ap.add_argument('--baseline', default='/home/ivenwu/projects/MFTG/baseline_preds/formal_ensembles')
    ap.add_argument('--universe', default='all', choices=['all', 'csi300', 'csi500', 'csi1000', 'csi800_1000'])
    ap.add_argument('--out', default=str(OUT.parent / 'results'))
    ap.add_argument('--ens', action='append', default=[], help='name=runA/xgb+runA/fusion (paths relative to runs/)')
    a = ap.parse_args()
    sub = {'all': None, 'csi300': lambda t, j: (idxm[t, j] & 1) > 0, 'csi500': lambda t, j: (idxm[t, j] & 2) > 0,
           'csi1000': lambda t, j: (idxm[t, j] & 4) > 0, 'csi800_1000': lambda t, j: idxm[t, j] > 0}[a.universe]
    # common evaluation dates = the baseline's test feature dates
    eval_dates = {b: sorted(np.unique(np.load(f'{a.baseline}/daily_{b}_ensemble.npz')['feature_date'])) for b in BLOCKS}
    cands = {f'base_{m}': {b: f'{a.baseline}/{m}_{b}_ensemble.npz' for b in BLOCKS} for m in ['daily', 'gru15', 'mixed', 'incremental']}
    for run in a.runs:
        for mdir in sorted(Path(run).glob('*/')):
            files = {b: str(mdir / f'{b}_pred.npz') for b in BLOCKS if (mdir / f'{b}_pred.npz').exists()}
            if files: cands[f'{Path(run).name}/{mdir.name}'] = files
    for spec in a.ens:
        name, parts = spec.split('=')
        files = {b: [str(OUT.parent / 'runs' / p / f'{b}_pred.npz') for p in parts.split('+')] for b in BLOCKS}
        cands[f'ens/{name}'] = {b: fs for b, fs in files.items() if all(Path(f).exists() for f in fs)}
    res = []; daily = {}
    for name, files in cands.items():
        allr = []
        for b, f in files.items():
            r = eval_block(load_any(f), eval_dates[b], sub); r['block'] = b; allr.append(r)
            res.append({'model': name, 'block': b, **summarize(r)})
        rr = pd.concat(allr); daily[name] = rr
        res.append({'model': name, 'block': 'ALL' if len(files) == 4 else f'ALL({len(files)})', **summarize(rr)})
    df = pd.DataFrame(res)
    outd = Path(a.out); outd.mkdir(parents=True, exist_ok=True)
    df.to_csv(outd / f'eval_{a.universe}.csv', index=False)
    pd.concat([v.assign(model=k) for k, v in daily.items()]).to_csv(outd / f'eval_{a.universe}_dates.csv', index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
    fmt = df.copy()
    for c in ['ic', 'ic_cc', 'ic_neu']: fmt[c] = fmt[c].map(lambda x: f'{x:.4f}')
    for c in ['icir', 'icir_neu', 'excess_ir']: fmt[c] = fmt[c].map(lambda x: f'{x:.2f}')
    for c in ['top_ex10_ann', 'ls10_ann', 'port_net_ann', 'excess_ann', 'turnover']: fmt[c] = fmt[c].map(lambda x: f'{x:.1%}')
    print(f'universe={a.universe}')
    print(fmt[fmt.block.str.startswith('ALL')].to_string(index=False))
    print(fmt[~fmt.block.str.startswith('ALL')].sort_values(['block', 'model']).to_string(index=False))


if __name__ == '__main__':
    main()
