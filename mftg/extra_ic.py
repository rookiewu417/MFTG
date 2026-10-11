"""Screen the extra inputs (build_extra.py) before training: per feature, RankIC vs y10 and the incremental IC after
removing industry, size and the current MFTG-Ens-Neu score (both sides residualised, rank-gauss), on the 8 test windows
2022H2..2026H1, every 5th trading day. 'top' = the same incremental IC inside the top 30% of the ensemble score (the
long-only optimiser's candidate pool), 'bot' = inside the bottom 30%.

usage: pixi run python mftg/extra_ic.py [--runs DIR]   -> build/mftg/extra_ic.csv
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtri
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'shared'))
from common import OUT, load_axes, SHARED

MEMBERS = ['Daily-Neu', 'Daily-Neu-H256', 'Daily-Neu-CSI800', 'Full-Neu']
HALVES = ['2022H2', '2023H1', '2023H2', '2024H1', '2024H2', '2025H1', '2025H2', '2026H1']


def rg(x):
    r = pd.Series(x).rank().to_numpy(); return ndtri((r - 0.5) / len(x))


def ens_scores(runs, dint, cpos):
    """{t: (j, z)} mean of the members' per-day z-scores over names every member scored."""
    fr = []
    for k, m in enumerate(MEMBERS):
        for b in HALVES:
            z = np.load(Path(runs) / m / f'{b}_pred.npz', allow_pickle=True)
            d = pd.DataFrame({'fd': z['feature_date'].astype(int), 'j': [cpos.get(c, -1) for c in z['stock_code']],
                              'p': z['pred'].astype(float), 'm': k})
            fr.append(d[d.j >= 0])
    d = pd.concat(fr)
    d['p'] = d.groupby(['m', 'fd']).p.transform(lambda v: (v - v.mean()) / (v.std() + 1e-12))
    w = d.pivot_table(index=['fd', 'j'], columns='m', values='p').dropna()
    s = w.mean(1).reset_index(name='s')
    tpos = {x: i for i, x in enumerate(dint)}
    return {tpos[fd]: (g.j.to_numpy(), g.s.to_numpy()) for fd, g in s.groupby('fd') if fd in tpos}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--runs', default=str(Path(__file__).resolve().parents[1] / 'runs' / 'mftg'))
    a = ap.parse_args()
    dates, codes = load_axes(); dint = dates.strftime('%Y%m%d').astype(int).to_numpy()
    cpos = {c: i for i, c in enumerate(codes)}
    Bd = SHARED / 'base'
    univ = np.load(Bd / 'univ.npy'); y10 = np.load(Bd / 'y10.npy'); ind = np.load(Bd / 'industry.npy'); lmv = np.load(Bd / 'log_circ_mv.npy')
    X = np.load(OUT / 'mftg' / 'extra.npy', mmap_mode='r'); names = json.loads((OUT / 'mftg' / 'extra_names.json').read_text())
    ens = ens_scores(a.runs, dint, cpos)
    days = sorted(ens)[::5]
    rows = []
    for t in days:
        j, s = ens[t]
        m = univ[t, j] & np.isfinite(y10[t, j]); j, s = j[m], s[m]
        if len(j) < 300: continue
        lm = lmv[t, j]; lm = np.where(np.isfinite(lm), lm, np.nanmean(lm)); ii = ind[t, j]
        C = np.stack([np.ones(len(j)), (lm - lm.mean()) / lm.std()] + [(ii == k).astype(float) for k in np.unique(ii[ii >= 0])], 1)
        Cs = np.c_[C, rg(s)]
        res = lambda v, M: v - M @ np.linalg.lstsq(M, v, rcond=None)[0]
        y = rg(y10[t, j]); y_c = res(y, C); y_cs = res(y, Cs)
        xs = np.asarray(X[t, j], np.float64)
        q = pd.Series(s).rank(pct=True).to_numpy(); top, bot = q > 0.7, q <= 0.3
        r = {'t': t, 'date': dint[t], 'ens_ic': np.corrcoef(rg(s), y)[0, 1]}
        for k, nm in enumerate(names):
            x = xs[:, k]
            if x.std() < 1e-6: r[f'{nm}|raw'] = r[f'{nm}|neu'] = r[f'{nm}|inc'] = np.nan; continue
            r[f'{nm}|raw'] = np.corrcoef(x, y)[0, 1]
            r[f'{nm}|neu'] = np.corrcoef(res(x, C), y_c)[0, 1]
            r[f'{nm}|inc'] = np.corrcoef(res(x, Cs), y_cs)[0, 1]
            for part, sel in [('top', top), ('bot', bot)]:
                xr = res(x[sel], Cs[sel]); yr = res(y[sel], Cs[sel])
                r[f'{nm}|{part}'] = np.corrcoef(xr, yr)[0, 1] if xr.std() > 1e-9 else np.nan
        rows.append(r)
    df = pd.DataFrame(rows)
    df['half'] = [f'{d // 10000}H{1 if d % 10000 < 701 else 2}' for d in df.date]
    out = []
    for nm in names:
        o = {'feature': nm}
        for kind in ['raw', 'neu', 'inc', 'top', 'bot']:
            v = df[f'{nm}|{kind}']; o[kind] = v.mean(); o[f'{kind}_t'] = v.mean() / v.std() * np.sqrt(v.count())
        hm = df.groupby('half')[f'{nm}|inc'].mean()
        o['inc_same_sign_halves'] = int((np.sign(hm) == np.sign(o['inc'])).sum())
        out.append(o)
    res_df = pd.DataFrame(out).sort_values('inc_t', key=np.abs, ascending=False)
    res_df.to_csv(OUT / 'mftg' / 'extra_ic.csv', index=False)
    print(f'{len(df)} dates, ens RankIC {df.ens_ic.mean():.4f}')
    pd.set_option('display.width', 200)
    print(res_df.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
