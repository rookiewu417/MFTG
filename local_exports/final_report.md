# Feature-only formal reproduction report

All rows below come from checkpoint feature-only inference using only as-of feature validity; future labels are retained only as evaluation metadata.

| model | round | ensemble rows | raw valid | post valid | rebalance (0bp) | rebalance (10bp) |
|---|---:|---:|---:|---:|---:|---:|
| daily | 2024H2 | 115993 | 115993 | 113952 | 24 | 24 |
| daily | 2025H1 | 114721 | 114721 | 112799 | 23 | 23 |
| daily | 2025H2 | 126463 | 126463 | 122860 | 25 | 25 |
| daily | 2026H1 | 106162 | 106162 | 103330 | 21 | 21 |
| gru15 | 2024H2 | 115993 | 115993 | 113952 | 24 | 24 |
| gru15 | 2025H1 | 114721 | 114721 | 112799 | 23 | 23 |
| gru15 | 2025H2 | 126463 | 126463 | 122860 | 25 | 25 |
| gru15 | 2026H1 | 106162 | 106162 | 103330 | 21 | 21 |
| mixed | 2024H2 | 115993 | 115993 | 113952 | 24 | 24 |
| mixed | 2025H1 | 114721 | 114721 | 112799 | 23 | 23 |
| mixed | 2025H2 | 126463 | 126463 | 122860 | 25 | 25 |
| mixed | 2026H1 | 106162 | 106162 | 103330 | 21 | 21 |
| incremental | 2024H2 | 115993 | 115993 | 113952 | 24 | 24 |
| incremental | 2025H1 | 114721 | 114721 | 112799 | 23 | 23 |
| incremental | 2025H2 | 126463 | 126463 | 122860 | 25 | 25 |
| incremental | 2026H1 | 106162 | 106162 | 103330 | 21 | 21 |

The model/date/seed outputs, post-processing, zero-cost and one-way 10bp portfolios are complete for all 16 groups.

Training, data, normalization, and method limitations are documented in METHOD_LIMITATIONS.md.
