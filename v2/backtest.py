"""Holdings-level long-only backtest for tradability studies.

Signal at close t -> trade at t+1 (open or VWAP). Equal-weight holdings; benchmark = equal-weight tradable universe
over the same holding interval. Options: rebalance period, EMA smoothing of daily scores, buffer (hysteresis),
liquidity floor, unsellable (limit-down / suspended) names carried, fixed cost + square-root impact at a given AUM.
"""
from dataclasses import dataclass, replace
import numpy as np
import pandas as pd
from common import OUT, LAKE, con, to_panel, load_axes

B = OUT / 'base'
dates, codes = load_axes()
T, N = len(dates), len(codes)
dint = dates.strftime('%Y%m%d').astype(int).to_numpy()
tpos = {d: i for i, d in enumerate(dint)}
cpos = {c: i for i, c in enumerate(codes)}
univ = np.load(B / 'univ.npy'); buyable = np.load(B / 'buyable.npy'); traded = np.load(B / 'traded.npy')
aopen = np.load(B / 'aopen.npy'); aclose = np.load(B / 'aclose.npy')
open_raw = np.load(B / 'open_raw.npy'); down_limit = np.load(B / 'down_limit.npy')
amount = np.load(B / 'amount.npy')
industry = np.load(B / 'industry.npy'); lmv = np.load(B / 'log_circ_mv.npy'); idxm = np.load(B / 'index_member.npy')

_vol = to_panel(con().sql(f"""select trade_date, ts_code, vol from read_parquet('{LAKE}/daily/price/*/*.parquet', hive_partitioning=false)
                               where trade_date >= '{dates[0].date()}'""").df(), dates, codes, ['vol'])['vol']
with np.errstate(all='ignore'):
    adj = aopen / open_raw
    avwap = np.where(traded, amount / _vol * adj, np.nan)
P_open = pd.DataFrame(np.where(traded, aopen, np.nan)).ffill().to_numpy()
P_vwap = pd.DataFrame(avwap).ffill().to_numpy()
adv20 = pd.DataFrame(np.where(traded, amount, np.nan)).rolling(20, min_periods=10).mean().to_numpy()
r1 = np.vstack([np.full((1, N), np.nan), np.diff(np.log(pd.DataFrame(aclose).ffill().to_numpy()), axis=0)])
sig20 = pd.DataFrame(r1).rolling(20, min_periods=10).std().to_numpy()
# a name cannot be sold at t+1 if it is suspended or opens at the down limit
unsellable_next = np.zeros((T, N), bool)
unsellable_next[:-1] = ~traded[1:] | (np.nan_to_num(open_raw[1:]) <= np.nan_to_num(down_limit[1:], nan=-1) + 0.005)


@dataclass(frozen=True)
class Cfg:
    hold: int = 5            # trading days between rebalances
    q: float = 0.1           # target fraction of eligible names held
    q_out: float = 0.1       # keep a holding while its score rank stays within the top q_out (buffer if > q)
    ema_hl: float = 0.0      # half-life (trading days) of EMA over daily cross-sectional z-scores; 0 = off
    min_adv: float = 0.0     # liquidity floor on 20-day average amount (yuan)
    exec_px: str = 'open'    # 'open' or 'vwap'
    cost_bp: float = 10.0    # one-way fixed cost
    aum: float = 0.0         # yuan; > 0 adds square-root impact k * sigma20 * sqrt(trade / adv20)
    impact_k: float = 1.0
    universe: str = 'all'    # all / csi500 / csi1000


def load_scores(files):
    """{block: npz path} -> DataFrame(fd, j, pred) with per-date z-scored predictions."""
    fr = []
    for f in files.values():
        z = np.load(f, allow_pickle=True)
        d = pd.DataFrame({'fd': z['feature_date'].astype(int), 'j': [cpos.get(c, -1) for c in z['stock_code']], 'pred': z['pred'].astype(float)})
        fr.append(d[d.j >= 0])
    d = pd.concat(fr)
    d['pred'] = d.groupby('fd').pred.transform(lambda v: (v - v.mean()) / (v.std() + 1e-12))
    return d


def score_panel(df, ema_hl):
    """Dense [T, N] score panel; optional EMA across available dates (missing = no update)."""
    S = np.full((T, N), np.nan, np.float32)
    ti = df.fd.map(tpos).to_numpy(); S[ti, df.j.to_numpy()] = df.pred.to_numpy()
    if ema_hl > 0:
        days = np.unique(ti); a = 1 - 0.5 ** (1 / ema_hl); cur = np.full(N, np.nan)
        for t in days:
            x = S[t]; m = np.isfinite(x)
            cur = np.where(m & np.isfinite(cur), (1 - a) * cur + a * x, np.where(m, x, cur))
            S[t] = np.where(m, cur, np.nan)
    return S


def simulate(S, rebal, cfg: Cfg):
    """S: [T,N] scores; rebal: sorted date indices (signal days). Returns per-period DataFrame."""
    P = P_vwap if cfg.exec_px == 'vwap' else P_open
    held = np.array([], int); rows = []
    for k, t in enumerate(rebal):
        t_exit = rebal[k + 1] if k + 1 < len(rebal) else min(t + cfg.hold, T - 2)
        if t_exit + 1 >= T: break
        s = S[t]
        elig = univ[t] & buyable[t] & np.isfinite(s)
        if cfg.universe == 'csi500': elig &= (idxm[t] & 2) > 0
        elif cfg.universe == 'csi1000': elig &= (idxm[t] & 4) > 0
        if cfg.min_adv > 0: elig &= np.nan_to_num(adv20[t]) >= cfg.min_adv
        e = np.where(elig)[0]
        if len(e) < 50: continue
        n = max(1, int(round(cfg.q * len(e))))
        rk = pd.Series(s[e]).rank(ascending=False).to_numpy()          # 1 = best
        rank_of = dict(zip(e, rk))
        keep = [j for j in held if rank_of.get(j, 1e9) <= cfg.q_out * len(e)]
        stuck = [j for j in held if j not in keep and unsellable_next[t, j]]
        new = [j for j in e[np.argsort(rk)] if j not in keep][:max(0, n - len(keep))]
        target = np.array(sorted(set(keep) | set(new) | set(stuck)), int)
        bought = np.setdiff1d(target, held); sold = np.setdiff1d(held, target)
        to = len(bought) / max(1, len(target))
        entry, exitp = P[t + 1], P[t_exit + 1]
        ret = exitp[target] / entry[target] - 1
        bench = np.nanmean(exitp[e] / entry[e] - 1)
        cost = cfg.cost_bp * 1e-4 * (len(bought) + len(sold)) / max(1, len(target))
        if cfg.aum > 0:
            tv = cfg.aum / max(1, len(target))
            traded_names = np.concatenate([bought, sold])
            part = tv / np.maximum(np.nan_to_num(adv20[t, traded_names]), 1e5)
            cost += np.sum(cfg.impact_k * np.nan_to_num(sig20[t, traded_names], nan=0.03) * np.sqrt(part)) / max(1, len(target))
        gross = np.nanmean(ret)
        rows.append({'t': t, 'date': dint[t], 'n': len(target), 'turnover': to, 'gross': gross, 'cost': cost, 'net': gross - cost,
                     'bench': bench, 'ex': (1 + gross - cost) / (1 + bench) - 1, 'adv_med': np.nanmedian(adv20[t, target]),
                     'stuck': len(stuck)})
        held = target
    return pd.DataFrame(rows)


def summary(r, hold):
    per_year = 252 / hold
    ex = r.ex; nav = np.cumprod(1 + ex)
    return {'excess_ann': np.prod(1 + ex) ** (per_year / len(r)) - 1, 'excess_ir': ex.mean() / ex.std() * np.sqrt(per_year),
            'maxdd': (nav / np.maximum.accumulate(nav) - 1).min(), 'turnover': r.turnover.iloc[1:].mean(),
            'cost_ann': r.cost.mean() * per_year, 'n': r.n.mean(), 'adv_med_m': r.adv_med.median() / 1e6, 'periods': len(r)}


def rebal_dates(S, start, end, hold):
    have = np.where(np.isfinite(S).any(1))[0]
    have = have[(dint[have] >= start) & (dint[have] <= end)]
    out, last = [], -10 ** 9
    for t in have:
        if t - last >= hold: out.append(t); last = t
    return out
