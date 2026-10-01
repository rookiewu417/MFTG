"""Daily weight-level portfolio simulator for portfolio-construction studies (round 4).

Signal at close t -> trades executed at t+1 (open or VWAP); positions marked open-to-open, so the daily P&L of day t is
held from open t+1 to open t+2. Weights drift with prices between trades. Supports:
  * target construction: top-q equal / score / inverse-vol weights, or a cost-aware LP with industry + style exposure
    bounds relative to the equal-weight universe;
  * K staggered tranches (each rebalanced every K days on its own offset);
  * per-name position cap as a share of ADV20 (capacity), per-day participation cap (multi-day execution, residual
    trades carried forward), limit-up (cannot buy) / limit-down or suspended (cannot sell) constraints;
  * fixed cost + square-root impact at a given AUM;
  * benchmarks: equal-weight tradable universe (buy-and-hold, reset every 5 days) and CSI500 / CSI1000 index.
"""
from dataclasses import dataclass, replace, field
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, vstack, hstack, identity
import backtest as bt
from common import LAKE, con

T, N, dint = bt.T, bt.N, bt.dint
Pm = bt.P_open                                                   # mark price (open, ffilled)
with np.errstate(all='ignore'):
    R_oo = np.vstack([Pm[1:] / Pm[:-1] - 1, np.full((1, N), np.nan)])   # R_oo[t] = open t -> open t+1
    R_oo = np.nan_to_num(R_oo)
    VW_SLIP = np.nan_to_num(bt.P_vwap / Pm - 1)                          # vwap vs open, same day
_ix = con().sql(f"select ts_code, trade_date, open from read_parquet('{LAKE}/index/price.parquet')").df()
IDX_OO = {}
for code, name in [('000905.SH', 'csi500'), ('000852.SH', 'csi1000'), ('000300.SH', 'csi300')]:
    s = _ix[_ix.ts_code == code].set_index(pd.to_datetime(_ix[_ix.ts_code == code].trade_date)).open.reindex(bt.dates).ffill().to_numpy()
    IDX_OO[name] = np.r_[s[1:] / s[:-1] - 1, np.nan]

_sty = None
def styles():
    global _sty
    if _sty is None:
        import risk
        _sty = {k: risk.STY[k] for k in ['size', 'beta', 'momentum', 'volatility', 'liquidity', 'value', 'reversal']}
    return _sty


@dataclass(frozen=True)
class PCfg:
    hold: int = 5                 # rebalance period of each tranche
    tranches: int = 1             # K staggered books (hold should be a multiple of K for even spacing; offsets = hold/K)
    q: float = 0.1
    q_out: float = 0.2            # buffer
    weight: str = 'ew'            # ew / score / ivol / opt
    ema_hl: float = 3.0
    min_adv: float = 0.0
    exec_px: str = 'open'
    cost_bp: float = 10.0
    aum: float = 0.0
    impact_k: float = 1.0
    pos_adv: float = 0.0          # >0: position value <= pos_adv * ADV20 (needs aum)
    part: float = 0.0             # >0: daily trade per name <= part * ADV20 (needs aum)
    max_w: float = 0.02           # per-name weight cap for score / ivol / opt
    # optimiser
    ind_cap: float = 0.03         # |industry weight - bench| per industry
    sty_caps: tuple = ()          # (('size', 0.3), ('beta', 0.3), ...) in z-score units of exposure vs EW universe
    tc_pen: float = 0.0           # alpha units per unit one-way trade (turnover penalty)
    opt_pool: float = 0.3         # optimiser candidates: top opt_pool of eligible by score + current holdings
    universe: str = 'all'
    impact_pen: float = 0.0      # >0: add impact (sigma*sqrt(trade$/ADV), trade = 0.3% of AUM) / kappa to the per-name trade penalty
    kappa: float = 0.008         # expected excess return per unit z over the holding horizon (converts cost -> alpha units)
    bench_idx: str = ''          # '' = EW universe; 'csi500' / 'csi1000' = index enhancement against cap weights
    member_floor: float = 0.0


def eligible(t, s, cfg):
    e = bt.univ[t] & bt.buyable[t] & np.isfinite(s)
    if cfg.universe == 'csi500': e &= (bt.idxm[t] & 2) > 0
    elif cfg.universe == 'csi1000': e &= (bt.idxm[t] & 4) > 0
    elif cfg.universe == 'csi1800': e &= (bt.idxm[t] & 6) > 0
    if cfg.min_adv > 0: e &= np.nan_to_num(bt.adv20[t]) >= cfg.min_adv
    return np.where(e)[0]


def cap_redistribute(w, cap):
    """Water-fill: clip w to cap, spread the excess pro rata over uncapped names (keeps sum)."""
    w = w.copy(); tot = w.sum()
    for _ in range(50):
        over = w > cap + 1e-12
        if not over.any(): break
        ex = (w[over] - cap[over]).sum(); w[over] = cap[over]
        free = ~over & (w < cap - 1e-12)
        if not free.any() or w[free].sum() <= 0: break
        w[free] += ex * w[free] / w[free].sum()
    return w * (tot / max(w.sum(), 1e-12)) if w.sum() > tot else w


def target_top(t, s, e, held_w, cfg):
    rk = pd.Series(s[e]).rank(ascending=False).to_numpy()
    n = max(1, int(round(cfg.q * len(e))))
    rank_of = np.full(N, np.inf); rank_of[e] = rk
    held = np.where(held_w > 1e-9)[0]
    keep = held[rank_of[held] <= cfg.q_out * len(e)]
    order = e[np.argsort(rk)]
    new = order[~np.isin(order, keep)][:max(0, n - len(keep))]
    sel = np.r_[keep, new].astype(int)
    if cfg.weight == 'ew': w = np.ones(len(sel))
    elif cfg.weight == 'score': w = np.clip(s[sel] - np.nanpercentile(s[e], 100 * (1 - cfg.q_out)), 0.05, None)
    elif cfg.weight == 'ivol': w = 1 / np.clip(np.nan_to_num(bt.sig20[t, sel], nan=0.03), 0.01, None)
    w = w / w.sum()
    if cfg.weight != 'ew': w = cap_redistribute(w, np.full(len(w), cfg.max_w))
    out = np.zeros(N); out[sel] = w
    return out


def target_opt(t, s, e, held_w, cfg, capw=None, tc=None, bench=None):
    """max alpha'w - tc'|w - w_prev|  s.t. sum w = 1, 0<=w<=max_w (or b+max_w with an index bench), industry and
    style exposures relative to the benchmark bounded. bench=None -> equal weight over e; else index weights [N]."""
    zs = np.full(N, np.nan); zs[e] = (s[e] - s[e].mean()) / (s[e].std() + 1e-12)
    pool = e[np.argsort(-s[e])[:int(cfg.opt_pool * len(e))]]
    cand = np.union1d(pool, np.where(held_w > 1e-9)[0])
    if bench is not None: cand = np.union1d(cand, np.where(bench > 0)[0])
    n = len(cand); wp = held_w[cand]
    b = np.zeros(n)
    if bench is not None: b = bench[cand] / bench[cand].sum()
    alpha = np.nan_to_num(zs[cand], nan=-3.0)
    can_buy = np.isin(cand, e)
    can_sell = ~bt.unsellable_next[t, cand]
    mx = np.full(n, cfg.max_w) + b
    if capw is not None: mx = np.minimum(mx, capw[cand])
    hi = np.where(can_buy, mx, np.minimum(mx, wp)); lo = np.where(can_sell, 0.0, wp); hi = np.maximum(hi, lo)
    I = identity(n, format='csr')
    A_eq = vstack([hstack([I, -I, I]), csr_matrix(np.r_[np.ones(n), np.zeros(2 * n)])]); b_eq = np.r_[wp, 1.0]
    rows, rhs = [], []
    def band(vec, target, cap):
        rows.extend([csr_matrix(np.r_[vec, np.zeros(2 * n)]), csr_matrix(np.r_[-vec, np.zeros(2 * n)])]); rhs.extend([target + cap, -(target - cap)])
    ind_c = bt.industry[t, cand]
    if bench is None:
        ind_e = bt.industry[t, e]
        for k in np.unique(ind_e[ind_e >= 0]): band((ind_c == k).astype(float), (ind_e == k).mean(), cfg.ind_cap)
    else:
        for k in np.unique(ind_c[ind_c >= 0]): m = (ind_c == k).astype(float); band(m, m @ b, cfg.ind_cap)
        if cfg.member_floor > 0:
            rows.append(csr_matrix(np.r_[-(b > 0).astype(float), np.zeros(2 * n)])); rhs.append(-cfg.member_floor)
    S = styles()
    for name, capv in cfg.sty_caps:
        v = S[name][t]; ve = v[e]; mu, sd = np.nanmean(ve), np.nanstd(ve) + 1e-12
        z = np.clip(np.nan_to_num((v[cand] - mu) / sd), -3, 3)
        band(z, 0.0 if bench is None else z @ b, capv)
    tcv = np.full(n, cfg.tc_pen) if tc is None else tc[cand]
    c = np.r_[-alpha, tcv, tcv]
    if held_w.sum() < 1e-9: c[n:] = 0
    res = linprog(c, A_ub=vstack(rows) if rows else None, b_ub=np.array(rhs) if rows else None, A_eq=A_eq, b_eq=b_eq,
                  bounds=list(zip(lo, hi)) + [(0, None)] * (2 * n), method='highs')
    if res.status != 0 and bench is not None:          # relax exposure bands
        return target_opt(t, s, e, held_w, replace(cfg, ind_cap=cfg.ind_cap * 2, sty_caps=tuple((k, c_ * 2) for k, c_ in cfg.sty_caps)), capw, tc, bench)
    if res.status != 0:
        return target_top(t, s, e, held_w, replace(cfg, weight='ew'))
    out = np.zeros(N); out[cand] = np.clip(res.x[:n], 0, None)
    return out / out.sum()


_bw = {}
def index_weights(name):
    """[T, N] index weights, each snapshot usable from the next trading day, forward-filled."""
    if name not in _bw:
        import enhance
        W = enhance.bench_weights({'csi500': '000905.SH', 'csi1000': '000852.SH'}[name])
        A = np.full((T, N), np.nan, np.float32)
        for tt, g in W.groupby('t'):
            if tt < T: row = np.zeros(N, np.float32); row[g.j.to_numpy()] = g.weight.to_numpy(); A[tt] = row / row.sum()
        _bw[name] = pd.DataFrame(A).ffill().fillna(0).to_numpy()
    return _bw[name]


def run_book(S, start, end, cfg, offset=0, aum=0.0):
    """One book. Returns daily frame: port (net, marked open-to-open), gross, cost, turnover (one-way), bench_ew, cash."""
    days = np.where((dint >= start) & (dint <= end))[0]
    days = days[days < T - 2]
    w = np.zeros(N)                       # weights at the close of day t (post-drift), before trading at t+1 open
    bw = np.zeros(N); cash = 1.0
    target = None
    last_s = np.full(N, np.nan); last_t = -99
    rows = []
    for i, t in enumerate(days):
        if np.isfinite(S[t]).any(): last_s = S[t]; last_t = t
        s = last_s if t - last_t <= 2 else np.full(N, np.nan)
        e = eligible(t, s, cfg)
        if i % cfg.hold == offset % cfg.hold and len(e) >= 50:
            capw = cfg.pos_adv * np.nan_to_num(bt.adv20[t]) / aum if aum > 0 and cfg.pos_adv > 0 else None
            if cfg.weight == 'opt':
                tc = None
                if cfg.impact_pen > 0 and aum > 0:
                    imp = np.nan_to_num(bt.sig20[t], nan=0.03) * np.sqrt(0.003 * aum / np.maximum(np.nan_to_num(bt.adv20[t]), 1e5))
                    tc = cfg.tc_pen + cfg.impact_pen * np.minimum(imp, 0.05) / cfg.kappa
                target = target_opt(t, s, e, w, cfg, capw, tc, index_weights(cfg.bench_idx)[t] if cfg.bench_idx else None)
            else:
                target = target_top(t, s, e, w, cfg)
                if capw is not None:
                    sel = target > 0
                    target[sel] = cap_redistribute(target[sel], capw[sel])
            b_new = np.zeros(N); b_new[e] = 1 / len(e); bw = b_new
            if cfg.bench_idx: bw = index_weights(cfg.bench_idx)[t].astype(float).copy()
        # ---- trade at the t+1 open towards target
        trade = np.zeros(N); cost = 0.0
        if target is not None:
            d = target - w
            cannot_buy = ~bt.buyable[t]; cannot_sell = bt.unsellable_next[t]
            d = np.where((d > 0) & cannot_buy, 0, d); d = np.where((d < 0) & cannot_sell, 0, d)
            if aum > 0 and cfg.part > 0:
                lim = cfg.part * np.nan_to_num(bt.adv20[t]) / aum
                d = np.clip(d, -lim, lim)
            # keep the book fully funded: scale buys to available cash + sells
            sells = -d[d < 0].sum(); buys = d[d > 0].sum(); avail = cash + sells
            if buys > avail + 1e-12: d = np.where(d > 0, d * avail / buys, d)
            trade = d; w = w + d; cash = 1 - w.sum()
            tv = np.abs(d)
            cost = cfg.cost_bp * 1e-4 * tv.sum()
            if cfg.exec_px == 'vwap': cost += (d * VW_SLIP[t + 1]).sum()          # buy above / sell below the open mark
            if aum > 0:
                nz = tv > 1e-9
                part_ = tv[nz] * aum / np.maximum(np.nan_to_num(bt.adv20[t, nz]), 1e5)
                cost += (tv[nz] * cfg.impact_k * np.nan_to_num(bt.sig20[t, nz], nan=0.03) * np.sqrt(part_)).sum()
            if np.abs(target - w).sum() < 1e-6: target = target   # fully executed; keep for later days (no-op)
        r = R_oo[t + 1]
        gross = (w * r).sum()
        bench = (bw * r).sum() / max(bw.sum(), 1e-12) if bw.sum() > 0 else np.nan
        rows.append({'t': t, 'date': dint[t], 'gross': gross, 'cost': cost, 'port': gross - cost, 'bench_ew': bench,
                     'turnover': np.abs(trade).sum() / 2, 'n': int((w > 1e-6).sum()), 'cash': cash})
        # drift
        v = w * (1 + r); tot = v.sum() + cash
        w = v / tot; cash = cash / tot
        bv = bw * (1 + r); bw = bv / max(bv.sum(), 1e-12)
    return pd.DataFrame(rows)


def run(S, start, end, cfg):
    K = cfg.tranches
    books = [run_book(S, start, end, cfg, offset=k * cfg.hold // K, aum=cfg.aum / K) for k in range(K)]
    df = books[0][['t', 'date', 'bench_ew']].copy()
    for c in ['gross', 'cost', 'port', 'turnover', 'cash']: df[c] = np.mean([b[c].to_numpy() for b in books], axis=0)
    df['n'] = np.max([b.n.to_numpy() for b in books], axis=0)
    for k in IDX_OO: df[k] = IDX_OO[k][df.t.to_numpy() + 1]
    first = max(int(np.argmax(b.n.to_numpy() > 0)) for b in books)        # skip the build-up days
    return df.iloc[first:].reset_index(drop=True)


def stats(df, bench='bench_ew'):
    ex = (1 + df.port) / (1 + df[bench]) - 1
    if len(ex) < 2: return {}
    nav = np.cumprod(1 + ex)
    yrs = len(ex) / 252
    return {'ann': nav.iloc[-1] ** (1 / yrs) - 1 if yrs > 0.3 else nav.iloc[-1] - 1, 'ir': ex.mean() / ex.std() * np.sqrt(252),
            'maxdd': (nav / nav.cummax() - 1).min(), 'vol': ex.std() * np.sqrt(252), 'to_ann': df.turnover.mean() * 252,
            'cost_ann': df.cost.mean() * 252, 'n': df.n.mean(), 'cash': df.cash.mean()}


HALVES = [('2022H2', 20220701, 20221231), ('2023H1', 20230101, 20230630), ('2023H2', 20230701, 20231231),
          ('2024H1', 20240101, 20240630), ('2024H2', 20240701, 20241231), ('2025H1', 20250101, 20250630),
          ('2025H2', 20250701, 20251231), ('2026H1', 20260101, 20260630)]


def report(df, bench='bench_ew'):
    main = df[df.date <= 20260630]; jul = df[df.date > 20260630]
    out = stats(main, bench)
    halves = {}
    for b, s, e in HALVES:
        x = main[(main.date >= s) & (main.date <= e)]
        halves[b] = np.prod((1 + x.port) / (1 + x[bench])) - 1 if len(x) else np.nan
    h = np.array(list(halves.values()))
    out.update({'worst_half': np.nanmin(h), 'neg_halves': int((h < 0).sum()), 'july': np.prod((1 + jul.port) / (1 + jul[bench])) - 1})
    out.update({f'h_{k}': v for k, v in halves.items()})
    return out


def load_signal(paths, ema_hl):
    R = '/home/ivenwu/projects/ml-factor/v2/runs'
    B9 = [b for b, _, _ in HALVES] + ['2026H2']
    frames = [bt.load_scores({b: f'{R}/{p}/{b}_pred.npz' for b in B9}).set_index(['fd', 'j']).pred for p in paths]
    df = pd.concat(frames, axis=1).dropna().mean(1).rename('pred').reset_index()
    return bt.score_panel(df, ema_hl)
