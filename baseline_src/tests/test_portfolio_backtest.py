import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # baseline_src/
import numpy as np
import pandas as pd
from portfolio_backtest import run_backtest, simulate_layer


def test_missing_open_close_defers_and_marks_last():
    dates = pd.date_range("2024-01-01", periods=7)
    opens = [
        {"A": 10.0},
        {"A": np.nan, "B": 20.0},  # A cannot be sold; B can be bought
        {"A": 12.0, "B": np.nan},  # deferred A sale, B remains held
        {"A": 13.0, "B": 22.0},
        {"A": 13.0, "B": 22.0}, {"A": 13.0, "B": 22.0}, {"A": 13.0, "B": 22.0},
    ]
    closes = [
        {"A": 10.0, "B": 20.0},
        {"A": 11.0, "B": np.nan},
        {"A": np.nan, "B": 21.0},
        {"A": 13.0, "B": 22.0},
        {"A": 13.0, "B": 22.0}, {"A": 13.0, "B": 22.0}, {"A": 13.0, "B": 22.0},
    ]
    out = simulate_layer({0: {"A"}, 1: {"B"}}, dates, opens, closes, 0.001)
    assert out.n_holdings.tolist() == [1, 2, 1, 1, 1, 1, 1]
    assert out.cash.iloc[1] < 1.0
    assert np.isfinite(out.nav).all()
    assert out.cost_paid.iloc[2] > 0  # deferred sale executes on recovery
    # Three later sessions have no new target and therefore no implicit rebalance.
    assert (out.traded_value.iloc[4] == 0) and (out.cost_paid.iloc[4] == 0)


def test_next_open_and_cost(tmp_path):
    dates = pd.bdate_range("2024-01-01", periods=8)
    codes = [f"{i:06d}.SZ" for i in range(12)]
    rows = []
    for di, day in enumerate(dates):
        for si, code in enumerate(codes):
            # Opens rise by 2% per session; closes are irrelevant.
            rows.append({"trade_date": day, "ts_code": code, "open": 100 + di * 2 + si,
                         "close": 100 + di + si, "adj_factor": 1.0})
    data = tmp_path / "daily.parquet"
    pd.DataFrame(rows).to_parquet(data)
    pred = tmp_path / "pred.npz"
    # Signal on day 0; top decile (one stock) must enter day 1 open, exit day 5 open.
    np.savez(pred, pred=np.arange(12, dtype=float), feature_date=np.full(12, 20240101, dtype=np.int32),
             valid=np.ones(12, dtype=bool), stock=np.arange(12, dtype=np.int32), codes=np.array(codes))
    trades, daily, summary = run_backtest(pred, data, tmp_path / "out", start="2024-01-01", end="2024-01-10",
                                      cost_bps=10, rebalance_every=5, holding_days=5)
    assert summary["n_layers"] == 10
    top = trades[trades.layer == 10].iloc[0]
    # day 1 open to day 5 open proves the one-day trading delay.
    assert top.entry_date == pd.Timestamp("20240102")
    assert top.exit_date == 20240110
    assert daily.date.min() == pd.Timestamp("2024-01-01")
    assert daily.date.max() == pd.Timestamp("2024-01-10")
    assert top.gross_return > 0.05
    assert top.cost == 0.001  # initial 100% turnover * 10 bp
