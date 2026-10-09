#!/usr/bin/env python3
"""Point-in-time decile portfolio backtest for model prediction NPZ files.

Signals observed after close on feature_date are traded at the next available
open.  By default portfolios are rebalanced every five trading days and held
through the next rebalance (no overlapping sleeves).  All ten equal-weighted
deciles are reported, with turnover and proportional transaction costs.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def simulate_layer(targets, dates, opens, closes, cost):
    """Simulate one layer with shares and cash marked at every close.

    ``targets`` maps a session index to the desired names effective from that
    session's open.  The desired set remains active until the next target;
    consequently a missing open defers a trade, while an existing holding is
    never silently removed.  Missing closes use the last observed close.
    """
    n = len(dates); shares = {}; cash = 1.0; last = {}; desired = set(); out = []
    for t in range(n):
        if t in targets:
            desired = set(targets[t])
        op=opens[t]; cl=closes[t]
        close_seen = set()
        for k,v in list(cl.items()):
            if np.isfinite(v) and v>0:
                last[k]=float(v); close_seen.add(k)
        # If today's close is unavailable, the open is the latest observable
        # mark and may be used for an existing holding (never a future price).
        for k in shares:
            if k not in close_seen and k in op and np.isfinite(op[k]) and op[k] > 0:
                last[k] = float(op[k])
        fees = 0.0; traded = 0.0
        sell=[k for k in shares if k not in desired and k in op and np.isfinite(op[k]) and op[k]>0]
        for k in sell:
            value=shares.pop(k)*float(op[k]); fees += value*cost; traded += value; cash += value*(1-cost)
        buy=[k for k in desired if k not in shares and k in op and np.isfinite(op[k]) and op[k]>0]
        # Divide by (1+cost) so the cash balance cannot become negative.
        budget=cash/(len(buy)*(1+cost)) if buy else 0.0
        for k in buy:
            qty=budget/float(op[k]); fees += budget*cost; traded += budget; cash-=budget*(1+cost); shares[k]=qty; last[k]=float(op[k])
        equity=cash+sum(qty*last[k] for k,qty in shares.items() if k in last)
        out.append((dates[t],equity,cash,len(shares),traded,fees))
    return pd.DataFrame(out,columns=['date','nav','cash','n_holdings','traded_value','cost_paid'])


def _date_int(x):
    return int(pd.Timestamp(x).strftime("%Y%m%d"))


def run_backtest(pred_path, data_path, out_dir, start=None, end=None,
                 cost_bps=10.0, rebalance_every=5, holding_days=5, codes_path=None):
    # ``holding_days`` is retained for API compatibility; positions are held
    # until the next valid signal or the clipped backtest end.
    z = np.load(pred_path, allow_pickle=False)
    required = {"pred", "feature_date", "valid", "stock"}
    missing = required - set(z.files)
    if missing:
        raise ValueError(f"prediction file missing keys: {sorted(missing)}")
    pred, fdate = np.asarray(z["pred"], float), np.asarray(z["feature_date"])
    valid = np.asarray(z["valid"], bool)
    direct_codes = "stock_code" in z.files
    if direct_codes:
        row_codes = np.asarray(z["stock_code"]).astype(str)
        stock = np.zeros(len(row_codes), dtype=int); pred_codes = np.unique(row_codes)
    elif "stock" not in z.files:
        raise ValueError("prediction requires stock_code or codes+stock")
    elif "codes" in z.files:
        pred_codes = np.asarray(z["codes"]).astype(str)
    elif codes_path:
        cp = Path(codes_path)
        pred_codes = np.asarray(np.load(cp, allow_pickle=False) if cp.suffix == ".npy" else
                                pd.read_csv(cp, header=None).iloc[:, 0]).astype(str)
    else:
        raise ValueError("prediction stock ids require explicit NPZ codes or --codes mapping")
    d = pd.read_parquet(data_path, columns=["trade_date", "ts_code", "open", "close", "adj_factor"])
    d["trade_date"] = pd.to_datetime(d["trade_date"])
    if start is not None:
        d = d[d.trade_date >= pd.Timestamp(start)]
    if end is not None:
        d = d[d.trade_date <= pd.Timestamp(end)]
    d = d.sort_values(["trade_date", "ts_code"])
    dates = pd.DatetimeIndex(sorted(d["trade_date"].unique()))
    market_codes = set(d["ts_code"].unique())
    codes = np.array(list(pred_codes) + sorted(market_codes - set(pred_codes)))
    code_to_id = {i: c for i, c in enumerate(codes)}
    op = d.pivot(index="trade_date", columns="ts_code", values="open").reindex(index=dates, columns=codes).to_numpy(float)
    cl = d.pivot(index="trade_date", columns="ts_code", values="close").reindex(index=dates, columns=codes).to_numpy(float)
    if not direct_codes and len(stock) and np.nanmax(stock) >= len(codes):
        raise ValueError("stock ids exceed sorted market-code universe")
    di = {_date_int(x): i for i, x in enumerate(dates)}
    lo = _date_int(start) if start else int(fdate.min())
    hi = _date_int(end) if end else int(fdate.max())
    signal_days = sorted(int(x) for x in np.unique(fdate[valid & np.isfinite(pred)])
                         if lo <= int(x) <= hi and int(x) in di)
    # The market panel is clipped above, so the final admitted signal runs only
    # to the requested backtest end and cannot trade beyond it.
    calendar_end_i = len(dates) - 1
    rows, holdings = [], [set() for _ in range(10)]
    cost = cost_bps / 10000.0
    for si, sd in enumerate(signal_days):
        i = di[sd]
        if i + 1 >= len(dates):
            continue
        q = np.flatnonzero(valid & (fdate == sd) & np.isfinite(pred))
        if len(q) < 10:
            continue
        order = q[np.argsort(pred[q], kind="mergesort")]
        groups = np.array_split(order, 10)
        new_holdings = [set(int(np.flatnonzero(codes == (row_codes[j] if direct_codes else codes[stock[j]]))[0]) for j in g) for g in groups]
        entry_i = i + 1
        next_i = di[signal_days[si + 1]] + 1 if si + 1 < len(signal_days) else calendar_end_i
        exit_i = min(next_i, calendar_end_i)
        if exit_i <= entry_i:
            continue
        for layer, (old, new) in enumerate(zip(holdings, new_holdings), 1):
            turnover = len(old.symmetric_difference(new)) / max(len(old) + len(new), 1)
            names = np.array(sorted(new), dtype=int)
            px0 = op[entry_i, names]; px1 = op[exit_i, names]
            ok = np.isfinite(px0) & np.isfinite(px1) & (px0 > 0) & (px1 > 0)
            gross = float(np.mean(px1[ok] / px0[ok] - 1.0)) if ok.any() else np.nan
            net = gross - cost * turnover if np.isfinite(gross) else np.nan
            rows.append({"feature_date": sd, "entry_date": _date_int(dates[entry_i]),
                         "exit_date": _date_int(dates[exit_i]), "layer": layer,
                         "n_signal": len(new), "n_traded": int(ok.sum()),
                         "gross_return": gross, "turnover": turnover,
                         "cost": cost * turnover, "net_return": net})
        holdings = new_holdings
    trades = pd.DataFrame(rows)
    if trades.empty:
        summary = {"n_rebalances": 0, "n_layers": 10, "cost_bps_one_way": cost_bps,
                   "rebalance_every": rebalance_every, "holding_days": holding_days}
        daily = pd.DataFrame(columns=["date", "layer", "gross_return", "turnover", "cost", "net_return", "nav"])
    else:
        trades["entry_date"] = pd.to_datetime(trades.entry_date.astype(str))
        # Build daily desired-target schedules and run the real cash/share
        # state machine.  NAV is marked from close prices on every session.
        target_maps = [dict() for _ in range(10)]
        for si, sd in enumerate(signal_days):
            q = np.flatnonzero(valid & (fdate == sd) & np.isfinite(pred))
            if len(q) < 10 or di[sd] + 1 >= len(dates): continue
            order = q[np.argsort(pred[q], kind="mergesort")]
            for layer, g in enumerate(np.array_split(order, 10)):
                target_maps[layer][di[sd] + 1] = set(int(stock[j]) for j in g)
        full = []
        for layer in range(10):
            sim = simulate_layer(target_maps[layer], dates, [dict(enumerate(op[t])) for t in range(len(dates))],
                                 [dict(enumerate(cl[t])) for t in range(len(dates))], cost)
            sim["layer"] = layer + 1
            sim["net_return"] = sim.nav.pct_change().fillna(sim.nav - 1)
            prev_nav = sim.nav.shift(1).fillna(1.0)
            sim["gross_return"] = sim.net_return + sim.cost_paid / prev_nav
            sim["turnover"] = sim.traded_value / sim.nav.shift(1).fillna(1.0)
            sim["cost"] = sim.cost_paid
            full.append(sim[["date", "layer", "gross_return", "turnover", "cost", "net_return", "nav"]])
        daily = pd.concat(full, ignore_index=True)
        ann_factor = 252.0
        summary = {"n_rebalances": int(trades.feature_date.nunique()), "n_layers": 10,
                   "first_entry": _date_int(trades.entry_date.min()), "last_exit": int(trades.exit_date.max()),
                   "cost_bps_one_way": cost_bps, "rebalance_every": rebalance_every,
                   "holding_days": holding_days, "execution": "signal at t close, next available open; fixed 5-trading-day rebalance",
                   "mean_net_by_layer": {str(k): float(v) for k, v in trades.groupby("layer").net_return.mean().items()},
                   "mean_turnover_by_layer": {str(k): float(v) for k, v in trades.groupby("layer").turnover.mean().items()}}
        for layer, g in daily.groupby("layer"):
            r = g.net_return.dropna()
            summary.setdefault("annualized_return_by_layer", {})[str(layer)] = float((1 + r).prod() ** (ann_factor / max(len(r), 1)) - 1) if len(r) else np.nan
            summary.setdefault("max_drawdown_by_layer", {})[str(layer)] = float((g.nav / g.nav.cummax() - 1).min()) if len(g) else np.nan
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out / "portfolio_trades.csv", index=False)
    daily.to_csv(out / "portfolio_daily.csv", index=False)
    (out / "portfolio_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    return trades, daily, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True); p.add_argument("--data", required=True); p.add_argument("--out", required=True)
    p.add_argument("--start"); p.add_argument("--end"); p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--rebalance-every", type=int, default=5); p.add_argument("--holding-days", type=int, default=5); p.add_argument("--codes")
    a = p.parse_args(); _, _, s = run_backtest(a.pred, a.data, a.out, a.start, a.end, a.cost_bps, a.rebalance_every, a.holding_days, a.codes)
    print(json.dumps(s, indent=2, allow_nan=True))

if __name__ == "__main__": main()
