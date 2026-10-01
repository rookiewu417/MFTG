"""Index-enhancement portfolios with the Huatai report's construction (Figure 34/38/42):
maximise expected score s.t. constituent weight >= floor, |w - b| <= 0.8%, industry / size exposure caps,
weekly two-sided turnover cap; weekly rebalance at VWAP; 0.4% two-sided cost. Solved as an LP (HiGHS).
"""
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, vstack, hstack, identity
import backtest as bt
from common import LAKE, con

IDX = {'csi300': ('000300.SH', 1, 0.8, 0.01, 0.05), 'csi500': ('000905.SH', 2, 0.6, 0.01, 0.05), 'csi1000': ('000852.SH', 4, 0.8, 0.02, 0.2),   # code, bit, member floor, ind cap, size cap
       'csiall': ('000985.SH', None, 0.0, 0.02, 0.2)}
STYLE_CAPS = {}   # e.g. {'beta': 0.2, 'momentum': 0.2}; filled from --style
BL = ['2024H2', '2025H1', '2025H2', '2026H1']
R = '/home/ivenwu/projects/ml-factor/v2/runs'
BASE = '/home/ivenwu/projects/ml-factor/baseline_preds/formal_ensembles'


def bench_weights(code):
    w = con().sql(f"""select trade_date, ts_code, weight from read_parquet('{LAKE}/index/weight/*/*.parquet', hive_partitioning=false)
                      where index_code = '{code}' and trade_date >= '2022-01-01'""").df()
    w['t'] = bt.dates.searchsorted(pd.to_datetime(w.trade_date), side='right')     # usable from the next trading day
    w['j'] = [bt.cpos.get(c, -1) for c in w.ts_code]
    return w[w.j >= 0]


def solve(alpha, b, w_prev, cand, ind, zsize, can_buy, can_sell, member, floor, ind_cap, size_cap, to_cap, dev=0.008, extra=()):
    n = len(cand)
    lo = np.maximum(0, b - dev); hi = b + dev
    hi = np.where(can_buy, hi, np.minimum(hi, w_prev)); lo = np.where(can_sell, lo, np.maximum(lo, w_prev))
    lo = np.minimum(lo, hi)
    # variables: w (n), u (n) buys, v (n) sells;  w - u + v = w_prev
    I = identity(n, format='csr')
    A_eq = [hstack([I, -I, I]), csr_matrix(np.r_[np.ones(n), np.zeros(2 * n)])]
    b_eq = [w_prev, [1.0]]
    rows, rhs = [], []
    rows.append(csr_matrix(np.r_[-member.astype(float), np.zeros(2 * n)])); rhs.append([-floor])
    rows.append(csr_matrix(np.r_[np.zeros(n), np.ones(2 * n)])); rhs.append([to_cap])
    for k in np.unique(ind[ind >= 0]):
        m = (ind == k).astype(float); bk = (m * b).sum()
        rows.append(csr_matrix(np.r_[m, np.zeros(2 * n)])); rhs.append([bk + ind_cap])
        rows.append(csr_matrix(np.r_[-m, np.zeros(2 * n)])); rhs.append([-(bk - ind_cap)])
    for z, cap_ in extra:
        bz = (z * b).sum()
        rows.append(csr_matrix(np.r_[z, np.zeros(2 * n)])); rhs.append([bz + cap_])
        rows.append(csr_matrix(np.r_[-z, np.zeros(2 * n)])); rhs.append([-(bz - cap_)])
    bs = (zsize * b).sum()
    rows.append(csr_matrix(np.r_[zsize, np.zeros(2 * n)])); rhs.append([bs + size_cap])
    rows.append(csr_matrix(np.r_[-zsize, np.zeros(2 * n)])); rhs.append([-(bs - size_cap)])
    c = np.r_[-alpha, np.zeros(2 * n)]
    bounds = list(zip(lo, hi)) + [(0, None)] * (2 * n)
    res = linprog(c, A_ub=vstack(rows), b_ub=np.concatenate(rhs), A_eq=vstack(A_eq), b_eq=np.concatenate(b_eq), bounds=bounds, method='highs')
    return res.x[:n] if res.status == 0 else None


def run(S, idx, to_cap, dates_list):
    code, bit, floor, ind_cap, size_cap = IDX[idx]
    W = bench_weights(code)
    snaps = sorted(W.t.unique())
    w_prev_full = {}
    rows = []; n_fail = [0]
    for k, t in enumerate(dates_list[:-1]):
        t_exit = dates_list[k + 1]
        sn = [s for s in snaps if s <= t]
        if not sn: continue
        wb = W[W.t == sn[-1]]
        b_full = np.zeros(bt.N); b_full[wb.j.to_numpy()] = wb.weight.to_numpy(); b_full /= b_full.sum()
        s = S[t]
        cand = np.where((bt.univ[t] & np.isfinite(s)) | (b_full > 0) | np.isin(np.arange(bt.N), list(w_prev_full)))[0]
        alpha = np.nan_to_num(s[cand], nan=-3.0)
        b = b_full[cand]; w_prev = np.array([w_prev_full.get(j, 0.0) for j in cand])
        member = (bt.idxm[t, cand] & bit) > 0 if bit else b > 0
        ind = bt.industry[t, cand]
        lm = bt.lmv[t, cand]; lm = np.where(np.isfinite(lm), lm, np.nanmedian(lm)); zsize = (lm - lm.mean()) / lm.std()
        can_buy = bt.buyable[t, cand] & bt.univ[t, cand]
        can_sell = ~bt.unsellable_next[t, cand]
        cap = 2.0 if not w_prev_full else to_cap          # first rebalance builds the book freely
        extra = []
        for sname, scap in STYLE_CAPS.items():
            v = STY[sname][t, cand]; v = np.where(np.isfinite(v), v, np.nanmedian(v)); extra.append(((v - v.mean()) / (v.std() + 1e-12), scap))
        w = solve(alpha, b, w_prev, cand, ind, zsize, can_buy, can_sell, member, floor, ind_cap, size_cap, cap, extra=extra)
        if w is None:
            w = solve(alpha, b, w_prev, cand, ind, zsize, can_buy, can_sell, member, floor, ind_cap * 2, size_cap * 2, cap,
                      extra=[(z, c * 2) for z, c in extra])
        for k in (4, 8):          # still infeasible (e.g. many unsellable names): relax exposures, member floor and turnover
            if w is not None: break
            w = solve(alpha, b, w_prev, cand, ind, zsize, can_buy, can_sell, member, floor * (1 - 0.1 * k), ind_cap * k, size_cap * k,
                      cap * k / 2, extra=[(z, c * k) for z, c in extra])
        if w is None: w = w_prev if w_prev.sum() > 0 else b; n_fail[0] += 1
        r = bt.P_vwap[t_exit + 1, cand] / bt.P_vwap[t + 1, cand] - 1
        r = np.nan_to_num(r)
        trade = np.abs(w - w_prev).sum()
        port = (w * r).sum(); bench = (b * r).sum(); cost = 0.002 * trade if w_prev_full else 0.0
        rows.append({'date': bt.dint[t], 'port': port - cost, 'bench': bench, 'ex': port - cost - bench, 'turnover2': trade,
                     'n': int((w > 1e-5).sum()), 'member_w': float(w[member].sum())})
        drift = w * (1 + r); drift /= drift.sum()
        w_prev_full = {j: x for j, x in zip(cand, drift) if x > 1e-7}
    if n_fail[0]: print(f'  [{idx} cap {to_cap}] LP infeasible on {n_fail[0]} dates, kept previous weights', flush=True)
    return pd.DataFrame(rows)


def summarize(r):
    ex = r.ex.iloc[1:]
    gaps = np.diff([bt.tpos[d] for d in r.date]); per_year = 252 / (gaps.mean() if len(gaps) else 5)
    if len(ex) < 8: return {'excess_ann': np.prod(1 + ex) - 1, 'ir': np.nan, 'maxdd': np.nan, 'turnover2': r.turnover2.mean(), 'n': r.n.mean(), 'member_w': r.member_w.mean(), 'win_week': (ex > 0).mean()}
    nav = np.cumprod(1 + ex)
    return {'excess_ann': np.prod(1 + ex) ** (per_year / len(ex)) - 1, 'ir': ex.mean() / ex.std() * np.sqrt(per_year),
            'maxdd': (nav / np.maximum.accumulate(nav) - 1).min(), 'turnover2': r.turnover2.iloc[1:].mean(),
            'n': r.n.mean(), 'member_w': r.member_w.mean(), 'win_week': (ex > 0).mean()}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--to', default='0.3,0.5'); ap.add_argument('--index', default='csi500,csi1000')
    ap.add_argument('--style', default='', help='e.g. beta:0.2,momentum:0.2'); ap.add_argument('--models', default='')
    ap.add_argument('--out', default='enhance.csv'); ap.add_argument('--span', default='main', choices=['main', 'all', 'holdout']); ap.add_argument('--ema', type=float, default=0); a = ap.parse_args()
    if a.span == 'all': BL[:0] = ['2022H2', '2023H1', '2023H2', '2024H1']
    if a.span == 'holdout': BL[:] = ['2026H2']
    if a.style:
        import risk; globals()['STY'] = risk.STY
        STYLE_CAPS.update({k: float(v) for k, v in (x.split(':') for x in a.style.split(','))})
    import pathlib
    models_all = {'base_incremental (report model)': {b: f'{BASE}/incremental_{b}_ensemble.npz' for b in BL},
              'xgb': {b: f'{R}/runA_xgb/xgb/{b}_pred.npz' for b in BL},
              'xgb_top': {b: f'{R}/exp_xgb_top/xgb/{b}_pred.npz' for b in BL},
              'fusion_top': {b: f'{R}/exp_fusion_top/fusion/{b}_pred.npz' for b in BL},
              'xgb_neu': {b: f'{R}/runB/xgb/{b}_pred.npz' for b in BL}}
    import pathlib
    for d in sorted(pathlib.Path(R, 'ens').glob('*')):
        models_all[f'ens_{d.name}'] = {b: str(d / f'{b}_pred.npz') for b in BL}
    models = {k: v for k, v in models_all.items() if not a.models or k.split(' ')[0] in a.models.split(',')}
    if a.span == 'main':
        days = sorted(set(np.concatenate([np.load(f'{BASE}/daily_{b}_ensemble.npz')['feature_date'] for b in BL])))
        dl = [bt.tpos[d] for d in days]
    else:   # our own schedule: every >=5 trading days on dates where predictions exist
        S0 = bt.score_panel(bt.load_scores({b: f'{R}/runB/xgb/{b}_pred.npz' for b in BL}), 0)
        dl = bt.rebal_dates(S0, 20260701, 20260731, 5) if a.span == 'holdout' else bt.rebal_dates(S0, 20220701, 20260630, 5)
        if a.span == 'holdout': dl.append(min(dl[-1] + 5, bt.T - 2))
    models_all = {k: v for k, v in models_all.items() if all(pathlib.Path(f).exists() for f in v.values())}
    out = []
    for mname, files in models.items():
        S = bt.score_panel(bt.load_scores(files), a.ema)
        for idx in a.index.split(','):
            for to in map(float, a.to.split(',')):
                r = run(S, idx, to, dl)
                blk = {f'ex_{b}': summarize(r[(r.date >= int(b[:4] + ('0101' if b[4:] == 'H1' else '0701'))) & (r.date <= int(b[:4] + ('0630' if b[4:] == 'H1' else '1231')))])['excess_ann'] for b in BL}
                out.append({'model': mname, 'index': idx, 'turnover_cap': to, **summarize(r), **blk})
                print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[-1].items()}, flush=True)
    pd.DataFrame(out).to_csv(f'/home/ivenwu/projects/ml-factor/v2/results/{a.out}', index=False)
