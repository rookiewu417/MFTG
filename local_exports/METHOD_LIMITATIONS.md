# Method limitations and fidelity notes

This note records documented differences and unresolved controls between this reproduction and the Huatai report. Statements about the report below are based on the attached 2023-05-11 PDF; statements about this run are based on `README.md`, `prepare_info.json`, and `daily_rolling_summary.json` in this directory.

## Time and sampling

- The report evaluates 2017-01-04 through 2023-04-28. This run uses complete source data through 2026-06-30 and tests 2024H2 onward; results are therefore outside the report's published period.
- The report says samples are taken every five trading days and models are retrained every half year. This run uses `sample_every=5` and a rolling schedule with half-year test blocks, but the README documents a preceding one-year validation block. That rolling training protocol is a reproduction choice and is not claimed to be identical to the report's undisclosed fold construction.
- The report's label is the next ten-trading-day return. The README specifies next ten-trading-day close return; any difference in price adjustment or exact endpoint is an implementation detail requiring explicit metadata.

## Inputs and preprocessing

- The report's minute input is the previous 20 trading days x 16 bars x 6 features (OHLC, VWAP, volume), and its daily branch is the previous 40 trading days x 6 features. This run documents those same window dimensions and aggregates source 1-minute data into 16 bars using 09:31-11:30 and 13:01-15:00.
- The report describes temporal normalization followed by cross-sectional z-score normalization. The README documents temporal-mean division for minute data and last-value division for daily data, followed by cross-sectional z-score. The exact minute bar endpoint convention is explicitly unspecified by the report and remains an implementation assumption.
- The report's tests use CSI 300, CSI 500, CSI 1000, and all-A universes, excluding ST/PT and stocks suspended or limit-up on the next trading day. The full-base run covers 5,893 historical codes but lacks sufficient historical size, industry, ST, and delisting metadata for complete implementation of those report controls. No index-universe or neutralization equivalence is claimed.
- The report applies factor winsorization, industry and market-cap neutralization, and standardization before testing. This run's prediction portfolio diagnostic does not claim those report controls; the README explicitly records the missing metadata.

## Models and training

- The report specifies one-layer GRU hidden/output dimension 30, batch size 5,000, maximum 100 epochs, patience 10, and retraining every half year. The README records hidden size 30, batch 5,000, 100/10 defaults, and Adam at learning rate 1e-3. The report does not specify optimizer or learning rate, so Adam/1e-3 is an implementation assumption.
- The report's four comparisons are 15-minute GRU, attention GRU, daily-plus-15-minute mixed, and two-stage incremental residual learning. It trains three random seeds and equally ensembles them. The current formal three-seed rolling training is running; interim technical artifacts and the one-epoch full rolling H2 benchmark remain single-run outputs and must not be compared to the report's ensemble results as like-for-like.
- The report describes incremental training as fitting the daily branch, freezing it, then fitting the minute residual. The README documents that behavior. The one-epoch full rolling daily H2 benchmark is a smoke/baseline run, not a converged reproduction of the report's training.

## Portfolio and costs

- The report's factor tests use weekly rebalancing and explicitly exclude transaction costs. Its index-enhancement examples report weekly two-sided turnover targets of 30%, 40%, and 50%.
- This run's `portfolio_backtest.py` is a separate execution diagnostic: a signal observed after the feature-date close is executed at the next available open, with ten equal-weighted layers, actual shares/cash, suspension deferral, daily close marking, and configurable one-way transaction cost (default 10 bp). These execution, price, fee, and layer-NAV conventions are not report facts.
- The portfolio diagnostic's 5-trading-day signal cadence is consistent with the report's weekly sampling convention, but it is not evidence that the report's turnover-controlled index-enhancement construction was reproduced. No benchmark-relative excess return, index membership, or report-style neutralization result should be inferred from these outputs.

## Available metadata post-processing

`factor_postprocess.py` is an independent downstream utility. It applies point-in-time listing/delisting and historical ST/PT name filters using `list_date`, `delist_date`, and `name.parquet` intervals constrained by `ann_date <= feature_date`; it never substitutes the current name for a historical name. It uses SW2021 L1 memberships for the 2024-2026 test period because `industry_criteria.parquet` explicitly labels SW2021/SW2014 and L1/L2/L3; overlapping higher levels are not treated as additional first-level industries. Within each date it applies a scaled-MAD cutoff of 5 across the full valid cross-section, removes each L1 industry's mean, then applies one unified cross-sectional z-score across all residuals. MAD=5 and the residualization order are implementation assumptions because the report specifies winsorization and industry/size neutralization but no exact algorithm/order. The output preserves `pred_raw`, writes filtered/industry-only `pred`, and records coverage counts. This is industry-only processing: market-cap neutralization and limit-up filtering remain unavailable because the source directory has no market-cap or limit-up fields.

## Explicit non-equivalences

The material fidelity limitations are: different evaluation dates; a separately documented rolling split; incomplete universe and historical metadata; no claimed ST/PT, suspension/limit-up, winsorization, industry/size neutralization equivalence; unspecified optimizer and learning rate in the source report; exact bar endpoint and execution conventions unspecified by the report; and single-run/one-epoch technical artifacts versus the report's three-seed equal-weight ensemble and converged training. These are limitations of comparability, not claims about the report's omitted implementation details.
