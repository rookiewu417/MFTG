# Multi-frequency GRU report reproduction

This is an independent from-scratch implementation of the attached Huatai report. It reads only `/root/autodl-tmp/quant/data/ashare`; existing project code and old results are not inputs. The server minute source is 1-minute data, so the preparation step aggregates it into 16 non-crossing 15-minute bars per trading day using the 09:31-11:30 and 13:01-15:00 sessions. Daily data is the server's post-weighted parquet.

The common complete minute interval is 2016-01-04 through 2026-06-30. July 2026 is partial and excluded. The report's 20-day x 16-bar minute input has 320 x 6 values; its daily branch has 40 x 6 values. The label is the next ten-trading-day close return. For minute inputs, each stock/feature is divided by its own 20-day temporal mean and then cross-sectionally z-scored across stocks at each time/feature (`axis=1`, then `axis=0` for arrays shaped `[stock,time,feature]`). Daily inputs use the last value as denominator and the same cross-sectional z-score. Missing bars, missing daily windows, and missing labels are excluded by a valid mask.

The static split is by label end: train label_end <= 20230630; validation feature_date >= 20230703 and label_end <= 20240628; test feature_date >= 20240701 and label_end <= 20260630. The rolling schedule in `rolling_schedule.py` uses four half-year test blocks (2024H2, 2025H1, 2025H2, 2026H1) and a preceding one-year validation block with a disjoint label cutoff. The official exchange calendar in `trade_cal.parquet` is used for date arithmetic; a stock that is suspended remains masked rather than shortening the horizon.

## Rebuild and run

All official commands should run through pixi after the lock is finalized:

    ./bin/pixi install
    ./bin/pixi run python reproduce.py metadata --data-root /root/autodl-tmp/quant/data/ashare
    ./bin/pixi run python reproduce.py prepare --data-root /root/autodl-tmp/quant/data/ashare --out artifacts512 --max-stocks 512 --every 5
    ./bin/pixi run python reproduce.py train --out artifacts512 --model gru15 --epochs 100 --patience 10
    ./bin/pixi run python reproduce.py evaluate --out artifacts512 --model gru15

The four model choices are `daily`, `gru15`, `mixed`, and `incremental`. `incremental` trains the daily branch to its best validation checkpoint, freezes all daily parameters and BN buffers in evaluation mode, then trains the minute branch on `y - yhat1` with `yhat1` detached; final output is `yhat1 + yhat2`. `corr_loss` is negative Pearson correlation with explicit shape and finite checks. Hidden size is 30 and batch training uses one sampled feature date at a time in the technical implementation; the report's optimizer and learning rate are not specified, so Adam at 1e-3 is an implementation assumption.

## Full-market memory-mapped path

`prepare_full_base.py` creates `full_base/minute_base.float32` with shape `[2546,5893,16,6]` (about 5.4 GiB) from the union of `stock_basic` and historical daily codes. It does not materialize all N x 320 windows. `fullbase_dataset.py` loads daily features and constructs normalized windows per sampled date, applying the per-date cross-sectional mask. `rolling_full.py` is the bounded rolling fit/predict entry and writes one checkpoint and an NPZ prediction file per model and half-year block. Its defaults are `--epochs 100 --patience 10 --batch 5000 --every 5 --seed 7`; use `--max-train-days`, `--max-val-days`, and `--max-test-days` for bounded smoke runs. For example:

    ./bin/pixi run python prepare_full_base.py --data-root /root/autodl-tmp/quant/data/ashare --out full_base
    ./bin/pixi run python fullbase_dataset.py inspect --base-dir full_base
    ./bin/pixi run python rolling_full.py --model daily --rounds 2024H2 --epochs 1 --out full_rolling_daily_h2

The full-base daily 2024H2 one-epoch benchmark completed on CPU in 275.6 seconds for 355/46/25 sampled train/validation/test days, with validation IC 0.06248. The GPU environment must be checked with `torch.cuda.is_available()` and an actual CUDA tensor before using it for formal results.

The `artifacts32` and `artifacts512` directories are reduced technical benchmarks using the first sorted code subset, retained to validate tensor shapes and model wiring. They are not full-A-share performance results. Full-base coverage has 5,893 historical codes; no historical size, industry, ST, or delisting metadata sufficient for complete neutralization was available, so those report controls are not claimed.

## Portfolio output

`portfolio_backtest.py` consumes prediction NPZ files containing `pred`, `feature_date`, `label_end`, `stock`, and `valid`. It is a separate point-in-time diagnostic: signals observed after the feature-date close trade at the next available open, with explicit suspension masks, ten equal-weighted layers, fixed five-trading-day rebalance, and a configurable one-way cost. It must be run only after the prediction NPZ and execution date range are checked. Forward-return deciles from `reproduce.py evaluate` are diagnostics and are not portfolio net returns.

The report leaves optimizer/lr, exact minute bin endpoint convention, execution price/cost, and historical industry/size fields unspecified. These choices are recorded as assumptions in the run metadata and must not be presented as report facts.
