"""Head-to-head scorecard: e2e track vs the Huatai report model (our reproduction), on the report's own protocol.

Common cross-sections: the reproduction's weekly feature dates in 2024H2-2026H1 (its 4 test windows).
A. Report single-factor test (report ch.3): pools all-A / CSI300 / CSI500 / CSI1000, drop ST and names suspended or
   limit-up on the next day, factor winsorised (5 MAD) + industry & size neutralised + standardised, weekly rebalance,
   no cost, next-week return = open(t+1) -> open(t+6). RankIC mean / std / IC_IR / IC>0, 10 layers: TOP decile annualised
   excess vs the pool mean, TOP IR, TOP weekly win rate, TOP two-sided turnover; also long-short (L10 - L1).
B. Tradable top decile (raw score, all-A): weekly, 10 bp one-way, open execution, unsellable names carried (backtest.py).
C. Index enhancement with the report's construction (enhance.py): CSI300 / CSI500 / CSI1000, weekly two-sided turnover
   caps 30 / 40 / 50 %, VWAP, 0.4 % two-sided cost.
usage: pixi run python e2e/scorecard.py [--parts A,B,C] name=spec ...   (spec: runs-relative dirs joined by '+', or base:<model>)
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import backtest as bt
import enhance as en

R = Path('/home/ivenwu/projects/ml-factor/v2/runs'); BASE = Path('/home/ivenwu/projects/ml-factor/baseline_preds/formal_ensembles')
OUTD = Path('/home/ivenwu/projects/ml-factor/v2/results')
BL = ['2024H2', '2025H1', '2025H2', '2026H1']
Y5 = np.load('/home/ivenwu/projects/ml-factor/v2/build/base/y5.npy', mmap_mode='r')
DATES = sorted(set(np.concatenate([np.load(BASE / f'daily_{b}_ensemble.npz')['feature_date'] for b in BL]).astype(int)))
POOLS = {'all': None, 'csi300': 1, 'csi500': 2, 'csi1000': 4}
PER_YEAR = 252 / 5


def files(spec):
    if spec.startswith('base:'): return [{b: str(BASE / f'{spec[5:]}_{b}_ensemble.npz') for b in BL}]
    return [{b: str(R / p / f'{b}_pred.npz') for b in BL} for p in spec.split('+')]


def load(spec):
    """Per-date z-scored average of the members, restricted to the common weekly dates."""
    frames = [bt.load_scores(f).set_index(['fd', 'j']).pred for f in files(spec)]
    df = pd.concat(frames, axis=1).dropna().mean(1).rename('pred').reset_index()
    return df[df.fd.isin(DATES)]


def prep(x, t, j):
    """Report preprocessing: 5-MAD winsorise, industry + size neutralise, standardise."""
    med = np.median(x); mad = np.median(np.abs(x - med)) + 1e-12
    x = np.clip(x, med - 5 * 1.4826 * mad, med + 5 * 1.4826 * mad)
    X = [np.ones(len(j))]; ind = bt.industry[t, j]
    for k in np.unique(ind[ind >= 0]): X.append((ind == k).astype(float))
    lm = bt.lmv[t, j]; lm = np.where(np.isfinite(lm), lm, np.nanmean(lm)); X.append((lm - lm.mean()) / (lm.std() + 1e-12))
    X = np.stack(X, 1); beta, *_ = np.linalg.lstsq(X, x, rcond=None); r = x - X @ beta
    return (r - r.mean()) / (r.std() + 1e-12)


def part_a(df):
    rows = []
    g = {fd: x for fd, x in df.groupby('fd')}
    for pool, bit in POOLS.items():
        ics, top_ex, ls, prev = [], [], [], None; tos = []
        for fd in DATES:
            if fd not in g: continue
            t = bt.tpos[fd]; x = g[fd]; j = x.j.to_numpy(); s = x.pred.to_numpy()
            m = bt.univ[t, j] & bt.buyable[t, j] & np.isfinite(Y5[t, j])
            if bit: m &= (bt.idxm[t, j] & bit) > 0
            j, s = j[m], s[m]
            if len(j) < 100: continue
            f = prep(s, t, j); y = Y5[t, j]
            ics.append(pd.Series(f).rank().corr(pd.Series(y).rank()))
            q = pd.qcut(pd.Series(f).rank(method='first'), 10, labels=False).to_numpy()
            top = j[q == 9]; top_ex.append(y[q == 9].mean() - y.mean()); ls.append(y[q == 9].mean() - y[q == 0].mean())
            if prev is not None: tos.append(2 * (1 - len(np.intersect1d(prev, top)) / len(top)))
            prev = top
        ic = np.array(ics); te = np.array(top_ex); ls = np.array(ls)
        rows.append({'pool': pool, 'rankic': ic.mean(), 'ic_std': ic.std(), 'ic_ir': ic.mean() / ic.std(), 'ic_pos': (ic > 0).mean(),
                     'top_excess_ann': np.prod(1 + te) ** (PER_YEAR / len(te)) - 1, 'top_ir': te.mean() / te.std() * np.sqrt(PER_YEAR),
                     'top_win': (te > 0).mean(), 'top_turnover2': np.mean(tos), 'ls_ann': np.prod(1 + ls) ** (PER_YEAR / len(ls)) - 1,
                     'weeks': len(ic)})
    return rows


def part_b(df):
    S = bt.score_panel(df, 0)
    reb = [bt.tpos[d] for d in DATES if d in set(df.fd)]
    r = bt.simulate(S, reb, bt.Cfg())
    gaps = np.diff(r.t.to_numpy()); py = 252 / gaps.mean()
    ex = r.ex; nav = np.cumprod(1 + ex)
    return {'trad_excess_ann': np.prod(1 + ex) ** (py / len(ex)) - 1, 'trad_ir': ex.mean() / ex.std() * np.sqrt(py),
            'trad_maxdd': (nav / np.maximum.accumulate(nav) - 1).min(), 'trad_turnover': r.turnover.iloc[1:].mean()}


def part_c(df, caps):
    S = bt.score_panel(df, 0)
    dl = [bt.tpos[d] for d in DATES]
    out = []
    for idx in ['csi300', 'csi500', 'csi1000']:
        for cap in caps:
            r = en.run(S, idx, cap, dl)
            ex = r.ex.iloc[1:]; gaps = np.diff([bt.tpos[d] for d in r.date]); py = 252 / gaps.mean()
            nav = np.cumprod(1 + ex); port = np.cumprod(1 + r.port.iloc[1:])
            mon = pd.Series(ex.to_numpy(), index=pd.to_datetime(r.date.iloc[1:].astype(str))).groupby(pd.Grouper(freq='ME')).apply(lambda v: np.prod(1 + v) - 1)
            out.append({'index': idx, 'to_cap': cap, 'excess_ann': nav.iloc[-1] ** (py / len(ex)) - 1, 'te': ex.std() * np.sqrt(py),
                        'ir': ex.mean() / ex.std() * np.sqrt(py), 'excess_maxdd': (nav / nav.cummax() - 1).min(),
                        'month_win': (mon > 0).mean(), 'turnover2': r.turnover2.iloc[1:].mean(), 'abs_ann': port.iloc[-1] ** (py / len(ex)) - 1,
                        'member_w': r.member_w.mean()})
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--parts', default='A,B,C'); ap.add_argument('--caps', default='0.3,0.4,0.5')
    ap.add_argument('--tag', default='scorecard'); ap.add_argument('--blocks', default='', help='other windows: weekly dates = every 5th trading day')
    ap.add_argument('specs', nargs='+'); a = ap.parse_args()
    if a.blocks:
        BL[:] = a.blocks.split(',')
        SPAN2 = {'2024H2': (20240701, 20241231), '2025H1': (20250101, 20250630), '2025H2': (20250701, 20251231), '2026H1': (20260101, 20260630)}
        SPAN = {'2022H2': (20220701, 20221231), '2023H1': (20230101, 20230630), '2023H2': (20230701, 20231231), '2024H1': (20240101, 20240630),
                '2026H2': (20260701, 20260731)}
        main = [d for d in DATES if any(b in BL for b in ('2024H2', '2025H1', '2025H2', '2026H1'))
                and any(SPAN2[b][0] <= d <= SPAN2[b][1] for b in BL if b in SPAN2)]
        days = [d for d in bt.dint if any(SPAN[b][0] <= d <= SPAN[b][1] for b in BL if b in SPAN)]
        early = days[::5]
        rep_early = [R / 'rep8/incremental' / f'{b}_pred.npz' for b in BL if b in SPAN]
        if rep_early and all(f.exists() for f in rep_early):      # use the reproduction's own weekly dates when available
            early = sorted(set(np.concatenate([np.load(f)['feature_date'] for f in rep_early]).astype(int)))
        DATES[:] = sorted(early + main)        # reproduction dates inside its windows, every 5th day elsewhere
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    A, B, C = [], [], []
    for item in a.specs:
        name, spec = item.split('=', 1); df = load(spec)
        if 'A' in a.parts: A += [{'model': name, **r} for r in part_a(df)]
        if 'B' in a.parts: B.append({'model': name, **part_b(df)})
        if 'C' in a.parts: C += [{'model': name, **r} for r in part_c(df, [float(c) for c in a.caps.split(',')])]
        print('done', name, flush=True)
    for label, rows in [('A', A), ('B', B), ('C', C)]:
        if not rows: continue
        t = pd.DataFrame(rows); t.to_csv(OUTD / f'{a.tag}_{label}.csv', index=False)
        print(f'\n===== part {label}'); print(t.round(4).to_string(index=False))
