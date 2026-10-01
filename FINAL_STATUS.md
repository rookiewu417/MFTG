# Final reproduction status

The formal run identified by `repro_20260917_234522_f75533e15ade` completed all 48 scheduled jobs: four model variants (`daily`, `gru15`, `mixed`, `incremental`), four half-year rounds (2024H2 through 2026H1), and seeds 7, 11, and 42. Each job has a matching `.SUCCESS`, checkpoint, prediction file, configuration, and rolling summary.

The feature-valid pipeline completed all 16 model/round groups. Each group has a three-seed ensemble, raw feature-valid prediction rows, SW2021 L1 post-processing, and both zero-cost and one-way 10 bp portfolio diagnostics. The local bundle was read successfully and contains 1,015 entries, 48 checkpoints, 48 formal predictions, 16 feature ensembles, 70 portfolio summary files, `pixi.toml`, `pixi.lock`, source files, README, and method limitations.

The local bundle is:

`local_exports/RESULT_BUNDLE_repro_20260917_234522_f75533e15ade.tar.gz`

SHA-256: `a5d3f1f67d21194450a3ee143d86954ec354ce355344034d6b413f6a3fce7af8`

The finalization manifest SHA-256 is `696fc73753e8c8f5911141863b66f6661592cfd077b2b451687d9fa2b173b720`. The local receipt records the same run ID and both hashes. The reported portfolio figures are execution diagnostics under the documented next-open, five-trading-day rebalance, ten-layer, and cost assumptions. They are not claims of identical performance to the report: the test period is later, the historical universe and metadata controls differ, exact report fold details and optimizer are partly unspecified, and market-cap/limit-up neutralization is unavailable.

The remote instance connection closed immediately after the exact export and ARM gates were written, consistent with the authorized guard action. The platform's billing state was not independently verified.
