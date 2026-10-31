# 信号口径评估结果（2026-10-31 至 11-01）

从这一轮起项目不做组合优化，只按信号评估。结论和表格写在 TRACKS.md 的"信号口径重新评估""风格暴露""每个风格的占比与分层"三节；这里是对应的原始输出。全部在远程机器镜像 `~/remote/projects/MFTG` 上运行，命令见 `run.sh`（`bash runs/mftg/_sig_eval/run.sh styles layers` 只重跑后两批）。

口径：8 个半年（2022H2–2026H1，197 个周截面）为样本内，2026H2X（2026-07-01 至 09-10，约 9 周）为样本外；可交易股票池，标签 y10；集成 = 成员逐日 z-score 后等权。

| 目录 | 脚本 | 远程机器输出 | 内容 |
|---|---|---|---|
| `main/` | `mftg/sig_eval.py` | `~/remote/sig_eval` | 27 个集成和单成员（含三组独立种子）：IC / 中性化 IC、十分组、头部 10% 组合、分股票池、衰减、5 风格暴露、相对 NEU4 的增量 IC |
| `prune/` | `mftg/sig_eval.py` | `~/remote/sig_eval2` | 第一候选去掉 / 加入成员（CAND-C、CAND-H、CAND-HC、CAND+F、CAND-C+dX） |
| `pt_mkt/` | `mftg/sig_eval.py` | `~/remote/sig_eval3` | PatchTST 与市场状态门控成员 |
| `styles/` | `mftg/sty_eval.py` | `~/remote/sty_eval` | 14 个风格：单变量 / 多元暴露、按半年的暴露、逐层中性化的 IC、风格自身 IC、FM 纯因子收益、行业偏离 |
| `layers/` | `mftg/sty_layer.py` | `~/remote/sty_layer` | IC 的可加拆分（行业 + 14 个风格 + 纯残差，按半年）、原始 / 纯残差十分组、按 10 个风格各分 5 层的层内 IC 和价差 |

每个目录里的 `stdout.log` 是运行时打印的汇总表，`*.csv` 是汇总，`*_weekly.csv.gz` / `weekly.csv.gz`（只有 styles、layers 有）是逐周明细，可以直接重算任何汇总而不必重跑。各 CSV 的列含义写在对应脚本开头的说明里。

集成名称：NEU4 = Daily-Neu + Daily-Neu-H256 + Daily-Neu-CSI800 + Full-Neu（基准）；CAND = dDMV9 + H256_6 + C800_6 + fVOL9（第一候选）；CAND-C = CAND 去掉 C800_6；`+dX` = 再加 e2e/dX；`m_` 开头为单成员。
