# MFTG

**A股多频率神经网络因子挖掘与机器学习因子进阶研究**

本项目包含两个部分：
1. **基线（多频率 GRU）**：日线 + 15 分钟线的多频率 GRU 模型，仅作为对比基线，不继续优化。
2. **端到端线 MFTG**（`mftg/`）：直接从原始 5 分钟 K 线与日线训练的神经网络，不使用人工因子公式。

因子线（Weave）已拆为独立项目，位于 `../Weave/`。

---

## 核心结果

### 因子线（Weave）结果与基线对比
在与基线完全相同的 4 个半年滚动测试窗口（2024H2–2026H1）上，**XGBoost + 多头加权损失**（`Weave-XGB-Top2`，因子线结果见 `../Weave`）表现最优：

| 口径（全A，5日调仓，前10%等权，单边10bp） | 基线 daily GRU | XGB（普通） | **XGB + 多头加权** |
|---|---|---|---|
| **RankIC / ICIR** | 0.085 / 0.61 | 0.129 / 0.88 | **0.105 / 0.91** |
| **行业市值中性 RankIC** | 0.076 | 0.110 | **0.085** |
| **前10%超额（10日持有，年化）** | 1.7% | 11.8% | **25.0%** |
| **组合相对等权股票池超额（扣费后）** | −1.8% | +12.1% | **+25.8%** |
| **超额 IR** | −0.16 | 1.30 | **3.37** |
| **每次调仓换手率** | 68% | 49% | **61%** |
| **超额为正的窗口数** | 2/4 | 3/4 | **4/4** |

在指数成分股内部同样稳健：
- **中证1000**：成分股内超额 **+19.0%**（4/4 窗口为正，基线为 −2.8%）
- **中证500**：成分股内超额 **+12.1%**（3/4 窗口为正，基线为 −6.7%）

> **多头加权核心机制**：在 IC 损失和 XGB 样本权重中，给标签排名前 20% 的股票赋予额外权重（`top_w=2`）。牺牲部分全局排序精度，换取显著更强的多头选股能力。

---

## 项目架构

```
MFTG/
├── README.md / LICENSE / .gitignore
├── pixi.toml / pixi.lock                        # 基线环境（含 PyTorch）；task：metadata / prepare / train / evaluate / rolling / backtest / test
│
├── docs/                                        # 文档
│   ├── FINAL_STATUS.md                          # 基线运行终态与哈希清单
│   └── METHOD_LIMITATIONS.md                    # 基线方法边界与实现假设（local_exports/ 里另有同内容副本）
│
├── baseline_src/                                # 基线代码（多频率 GRU，2024H2–2026H1 四个窗口）
│   ├── gru_baseline.py                             # 数据预处理/模型训练/单点评估入口
│   ├── rolling_full.py                          # 全市场滚动训练与推断
│   ├── fullbase_dataset.py                      # 内存映射高频+日频数据集构建
│   ├── feature_valid_inference.py               # 特征日截面推断流水线
│   ├── ensemble_predictions.py                  # 多种子模型集成
│   ├── factor_postprocess.py                    # 行业处理与 PIT 股票过滤（预测之后的后处理）
│   ├── asof_inference_export.py                 # 按 feature_date 导出 as-of 预测池
│   ├── portfolio_backtest.py                    # 严格 PIT 十分位组合回测
│   ├── compare_feature_valid.py / inspect_dates.py   # 预测文件对比、日期检查小工具
│   ├── tests/                                   # test_*.py（pixi run baseline-test 跑其中两个；test_portfolio_backtest.py 需 pytest）
│   └── pixi.toml                                # 在本目录内单独运行的环境副本（与根目录相同）
│
├── baseline_ops/                                # 基线训练的远端（AutoDL）运维脚本，已完成使命，仅留档
│   ├── auto_shutdown_guard{,_v2,_v3}.sh         # 结果导出后的关机守卫（v1 已被撤销为只做诊断）
│   ├── test_auto_shutdown_guard{,_v2,_v3}.sh    # 对应测试（用 mock 关机命令）
│   ├── export_and_arm_watch.sh                  # 经 SSH 把结果包拉到 local_exports/ 并"上保险"
│   ├── finalize_results.sh                      # 远端收尾：生成清单与结果包
│   ├── finalization/                            # 守卫测试产生的日志
│   └── profiling/                               # 远端性能小实验（sync_profile_temp.py / perf_memory_io.py，依赖已不在本地的缓存）
│
│
├── baseline_preds/                              # 【勿动】基线预测与回测记录（Portfolio 的 common.py 硬编码读取 formal_ensembles/）
│   ├── formal_ensembles/                        # 各模型各窗口的集成预测 npz
│   └── formal_feature_valid/                    # 各窗口(2024H2-2026H1)回测明细与指标
│
├── local_exports/                               # 基线实验导出结果包与报告（含 266MB 结果包，已 gitignore）
│   ├── README.md / RESULTS_INTERPRETATION.md / final_report.md / summary.json
│   └── RESULT_BUNDLE_*.tar.gz、清单与日志
│
├── TRACKS.md                                # 实验迭代路线图（两条线的记录同在此文件）
├── pixi.toml / pixi.lock                    # 环境（基线与端到端共用）；task：build-base / build-mftg / build-flow / build-styles / evaluate / rep-ensemble
│
├── shared/                                  # 共用代码
│   ├── common.py                            # 路径常量（LAKE / OUT / SHARED）与面板工具
│   ├── build_base.py                        # 基础面板：价格、可交易标记、行业、市值 → build/shared/base（脚本无 main 保护，import 即执行）
│   └── evaluate.py                          # IC 评估
│
├── mftg/                                    # MFTG 线：原始行情 → 日内 / 日线网络 → 预测
│   ├── build_mftg.py / build_flow.py / build_styles.py   # 数据准备
│   ├── train_mftg.py / driver_mftg.py       # 训练与 Colab 驱动
│   ├── join_months.py / bench.py            # 月度重训拼接、速度基准
│   └── upload_{mftg,flow,parts}.sh          # 上传脚本
│
├── baseline_early/                                   # 基线的早期窗口补训练（2022H2–2024H1）与 8 窗口拼接
│   └── build_fullbase.py / driver_baseline.py / fullbase_dataset.py / gru_baseline.py / rolling_full.py / rep_ensemble.py
│
├── colab/                                   # 云端运维（env / keeper / supervisor / monitor / sync / fetch / submit / status / watch / tail）
│
├── build/                                   # 【勿动】数据产物（已 gitignore）
│   ├── shared/                              # base/ 与 axes.npz
│   ├── mftg/                                # 端到端数据、打包目录、long/（长历史）
│   ├── repro/                               # 基线产物
│   └── keeper/                              # 云端任务守护状态
└── runs/                                    # 【勿动】预测文件（Portfolio 只读取这里）
    ├── mftg/                                # 端到端线变体（见下方命名对照）
    └── repro/                               # 基线（rep8/ 为拼接后的 8 窗口）
```

> **因子线与组合层已拆出**：因子线（Weave）是独立项目，位于 [`../Weave/`](../Weave/README.md)。组合层（组合构建、回测与评分）是独立项目，位于 [`../Portfolio/`](../Portfolio/README.md)。

---

## 严谨的研究与评估准则

1. **严格 Point-in-Time (PIT) 防未来函数**：
   - 信号于 $t$ 日收盘后生成，调仓严格在 $t+1$ 日开盘价执行。
   - 过滤 $t+1$ 日开盘一字涨停（无法买入）、停牌、ST 及上市未满 120 日次新股。
   - 所有横截面标准化与分位数仅在当日有效可买股票池内计算。
2. **滚动训练隔离 (Walk-Forward Validation)**：
   - 滚动划分为 4 个半年窗口：`2024H2`, `2025H1`, `2025H2`, `2026H1`。
   - 每个测试窗口前置 1 年独立验证集用于早停，严禁测试集信息泄露。
3. **真实交易摩擦扣除**：
   - 所有超额收益均已扣除单边 10bp 交易成本。
   - 基准对齐：以全市场可交易等权股票池作为超额基准，剥离纯小盘 Beta 收益。

---

## 快速开始

本项目使用 [pixi](https://pixi.sh/) 管理跨平台依赖与虚拟环境。

### 1. 安装环境
```bash
# 基线与端到端线共用一个环境（PyTorch、Polars、DuckDB、LightGBM 等）
pixi install
```

### 2. 基线模型流程
```bash
# 1. 生成元数据与特征映射
pixi run python baseline_src/gru_baseline.py metadata --data-root <path_to_ashare>

# 2. 准备全市场 15 分钟高频基底 (mmap)
pixi run python baseline_src/gru_baseline.py prepare --data-root <path_to_ashare> --out full_base

# 3. 运行滚动训练 (可选 daily / gru15 / mixed / incremental)
pixi run python baseline_src/rolling_full.py --model daily --rounds 2024H2 --epochs 100 --seed 7

# 4. 运行严格 PIT 组合回测
pixi run python baseline_src/portfolio_backtest.py --pred-file <path_to_pred.npz> --cost-bps 10
```

### 3. 端到端线（MFTG）流程
```bash
# 1. 数据准备（读取原始行情，写入 build/shared 与 build/mftg）
pixi run build-base      # = python shared/build_base.py
pixi run build-mftg
pixi run build-flow
pixi run build-styles

# 2. 训练：在 Colab 上通过 mftg/driver_mftg.py 与 mftg/upload_*.sh 完成（见 mftg/）
# 3. 结果回收：colab/sync.sh 把预测下载到 runs/$LINE/<run>/<model>/，LINE 默认 mftg，基线任务用 LINE=repro
#    会话名用 SES=<name>；临时文件在 /tmp/mftg_colab（可用 MFTG_SCRATCH 改）
```

因子线（Weave）的流程见 [`../Weave/README.md`](../Weave/README.md)。

---

## 命名对照（旧名 → 新名）

端到端线（`mftg/`）的名字整理过两次：先把 `e2e` 系列统一为 **MFTG**（Multi-Frequency Transformer-GRU），再给变体和集成起正式的名字。下表左列是改名前在 TRACKS 和代码里使用的名字。

**保留旧名的地方**：`results/` 下已生成的 csv / log / parquet 文件名（如 `r4_daily/long9_e2e_neu4_*.parquet`）、`runs/mftg/*/args.json` 里的 `name` 字段，以及远端 Colab 路径（`/content/e2e`、`e2e.tgz`）。这些是当时的产出记录或远端数据位置，改了就得重新上传数据，所以不动。

### 仓库

| 旧名 | 新名 | 说明 |
|---|---|---|
| `ml-factor` | MFTG | 仓库名（GitHub 与本地目录） |

### 模型家族

| 旧名 | 新名 | 说明 |
|---|---|---|
| `e2e` | MFTG | 端到端线的模型家族：日内 Transformer 编码 + 日间 GRU |

### 单模型

| 旧名 | 新名（规格名 `mftg/…`） | 输入 | 标签 | 其他 |
|---|---|---|---|---|
| `fA` | MFTG-Full（`mftg/Full`） | 日线 + 5 分钟 K 线 + 订单流 | 普通 | 隐藏层 64 |
| `fNH` | MFTG-Full-Neu（`mftg/Full-Neu`） | 日线 + 5 分钟 K 线 + 订单流 | 中性化 | 隐藏层 128 |
| `dNH` | MFTG-Daily-Neu（`mftg/Daily-Neu`） | 日线 | 中性化 | 隐藏层 128 |
| `dNH256` | MFTG-Daily-Neu-H256（`mftg/Daily-Neu-H256`） | 日线 | 中性化 | 隐藏层 256 |
| `dNH800` | MFTG-Daily-Neu-CSI800（`mftg/Daily-Neu-CSI800`） | 日线 | 中性化 | 隐藏层 128，训练池为沪深300 + 中证500（`--train_pool 3`） |
| `dNT1H` | MFTG-Daily-Neu-Top1（`mftg/Daily-Neu-Top1`） | 日线 | 中性化 | 隐藏层 128，多头加权 1 |

### 集成

| 旧名 | 新名 | 成员与权重 | 信号组（`round4.py`） |
|---|---|---|---|
| `E2E`（09-30 的"最终端到端模型"） | MFTG-Ens | MFTG-Full ×1 + MFTG-Daily-Neu ×1 + MFTG-Daily-Neu-Top1 ×2 | 无 |
| `OURS` | MFTG-Ens-Full | MFTG-Full ×2 + MFTG-Daily-Neu ×1 + MFTG-Daily-Neu-Top1 ×1 | `mftg_ens_full`（改名前为 `e2e_ours`） |
| `NEU4` | MFTG-Ens-Neu | MFTG-Daily-Neu、MFTG-Daily-Neu-H256、MFTG-Daily-Neu-CSI800、MFTG-Full-Neu 等权 | `mftg_ens_neu`（改名前为 `e2e_neu4`） |
| NEU4 与 OURS 各半 | MFTG-Ens-Both | — | `mftg_ens_both`（改名前为 `e2e_both`） |

目前推荐的集成是 **MFTG-Ens-Neu**（TRACKS 中的"推荐方案"）。

### 因子线（Weave）

因子线代码已移到独立项目 `../Weave`（见其 README），下表只记录命名对应关系。

| 旧名 | 新名（规格名 `weave/…`） | 配置 |
|---|---|---|
| 因子线（factor） | Weave（机器学习 + 因子组合） | 家族名，模型包括 XGBoost 与 MLP |
| `runA_xgb/xgb` | Weave-XGB（`weave/XGB`） | 普通标签 |
| `runB/xgb` | Weave-XGB-Neu（`weave/XGB-Neu`） | 中性化标签 |
| `exp_xgb_top/xgb` | Weave-XGB-Top2（`weave/XGB-Top2`） | 多头加权，top_w = 2 |
| `runA/mlp` | Weave-MLP（`weave/MLP`） | 普通标签 |
| `exp_mlp_top/mlp` | Weave-MLP-Top（`weave/MLP-Top`） | 多头加权，权重未留记录 |
| `ens/neuX_topX` | Weave-Ens-NeuTop（`ens/Weave-Ens-NeuTop`） | Weave-XGB-Neu 与 Weave-XGB-Top2 的组合 |

已退役的 fusion 系列（`runA/fusion`、`exp_fusion_top`、`fusion_neu*`）保留原名，只作对照。远端 Colab 的输出目录（如 `/content/out/runB`）也保持不变。

### 暂未改名

以下探索性运行没有用在集成中，暂保留原名：`dNT`、`dNT800`、`dNT1H800`、`dNTH`、`dNLH`、`dNW`、`dA`、`dAW`、`dN`、`dN5`、`dT`、`dSH`、`dSH800`、`dNH_M`、`dNH_M1`、`dNH800_M`、`dNH_s7`、`dNHB`、`dNHX`、`daily_*`、`intra_only`、`pilot*`、`smoke*`。需要统一的话可以接着改。

## 许可证
本项目代码采用 [MIT License](LICENSE)。
