# ml-factor

**A股多频率神经网络因子挖掘与机器学习因子进阶研究**

本项目包含两个部分：
1. **基线复现**：华泰证券金工研报《神经网络多频率因子挖掘模型》（2023-05-11）的独立端到端复现。
2. **v2 进阶**：在此基础上扩展的工业级机器学习因子体系（XGBoost / MLP / Fusion + 多头加权损失），在全A及宽基指数成分股内实现显著超额收益。

---

## 核心结果

### v2 进阶模型与基线对比
在与基线完全相同的 4 个半年滚动测试窗口（2024H2–2026H1）上，**XGBoost + 多头加权损失**（`exp_xgb_top`）表现最优：

| 口径（全A，5日调仓，前10%等权，单边10bp） | 基线 daily GRU | XGB（普通） | **XGB + 多头加权 (v2)** |
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
ml-factor/
├── 20230511-华泰证券-金工：神经网络多频率因子挖掘模型.pdf  # 原始研报
├── FINAL_STATUS.md                              # 基线复现审计终态与哈希清单
├── METHOD_LIMITATIONS.md                        # 复现方法边界与实现假设说明
│
├── baseline_src/                                # 基线复现核心源码
│   ├── reproduce.py                             # 数据预处理/模型训练/单点评估入口
│   ├── rolling_full.py                          # 全市场滚动训练与推断
│   ├── fullbase_dataset.py                      # 内存映射高频+日频数据集构建
│   ├── feature_valid_inference.py               # 特征日截面推断流水线
│   └── ensemble_predictions.py                  # 多种子模型集成
│
├── baseline_preds/                              # 基线评估结果与组合回测记录
│   └── formal_feature_valid/                    # 各窗口(2024H2-2026H1)回测明细与指标
│
├── local_exports/                               # 基线实验导出摘要与报告
│   ├── README.md                                # 基线复现说明文档
│   ├── RESULTS_INTERPRETATION.md                # 复现结果解读
│   └── summary.json                             # 完整评估汇总指标
│
└── v2/                                          # v2 进阶因子挖掘与组合优化
    ├── REPORT.md                                # v2 完整研究报告与实验结论
    ├── TRACKS.md                                # 实验迭代路线图
    ├── build_base.py                            # 序列特征工程 (31维)
    ├── build_tab.py / build_tab2.py             # 截面表格特征工程 (278维)
    ├── pack.py                                  # 紧凑二进制特征打包
    ├── evaluate.py / backtest.py                # 严格 PIT 回测引擎
    ├── pf.py / tradability.py                   # 组合构建、换手缓冲与可交易性过滤
    ├── risk.py / neutralize.py                  # Barra风格风险中性化与二次规划
    ├── dynamic.py / robust.py / round4.py       # 动态加权、稳健性检验与参数扫描
    ├── colab/                                   # 分布式/云端多卡训练运维套件
    │   ├── train.py                             # XGBoost-GPU / MLP / GRU / Fusion 训练
    │   └── driver.py / submit.sh                # 异步任务队列与状态同步
    └── results/                                 # 实验结果、评分卡与容量测算 CSV
```

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
# 安装根目录依赖（包含 PyTorch, Polars, DuckDB, NumPy, Scipy 等）
pixi install

# 或进入 v2 目录
cd v2 && pixi install
```

### 2. 基线模型复现流程
```bash
# 1. 生成元数据与特征映射
pixi run python baseline_src/reproduce.py metadata --data-root <path_to_ashare>

# 2. 准备全市场 15 分钟高频基底 (mmap)
pixi run python baseline_src/reproduce.py prepare --data-root <path_to_ashare> --out full_base

# 3. 运行滚动训练 (可选 daily / gru15 / mixed / incremental)
pixi run python baseline_src/rolling_full.py --model daily --rounds 2024H2 --epochs 100 --seed 7

# 4. 运行严格 PIT 组合回测
pixi run python portfolio_backtest.py --pred-file <path_to_pred.npz> --cost-bps 10
```

### 3. v2 进阶模型流程
```bash
cd v2

# 1. 特征构建与打包
pixi run python build_base.py
pixi run python build_tab.py
pixi run python pack.py

# 2. 模型训练（支持 XGBoost-GPU / MLP / Fusion）
pixi run python colab/train.py --models xgb --top_w 2 --blocks 2024H2,2025H1,2025H2,2026H1

# 3. 评分卡与风格中性化回测
pixi run python backtest.py
pixi run python risk.py
```

---

## 许可证
本项目代码采用 [MIT License](LICENSE)。
