# 回归算法分析

## 项目概述

本项目通过同一回归任务，比较不同建模思路在预测性能、误差分布与泛化表现上的差异，研究模型的适用场景与改进方法。

当前以共享单车租赁需求预测为应用场景，使用日期、时段、日历与天气特征，预测每小时租赁总量 `cnt`。

项目构建 Ridge、Poisson 回归、决策树和梯度提升树，在统一的数据划分、预处理与评价规则下进行训练和比较，为后续模型优化提供可复现的基础结果。

## 当前进度

| 成员 | 负责内容 | 当前成果与进度 |
| --- | --- | --- |
| A | 数据准备与预处理 | 已完成数据集选择、质量检查、EDA、特征定义、时间划分及预处理流水线 |
| B | 基础回归模型构建 | 已完成四种模型构建、原理说明、训练与时间验证、结果生成及基础分析 |
| C | 模型优化与拓展实验 | 待开展参数优化、特征工程和拓展实验；已准备模型接口、基线结果及交接说明 |

当前结果来自开发训练集内部的五折时间验证。独立测试集用于配置冻结后的最终评价。

## 数据集与预测设定

### 数据来源

采用 [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) 的小时数据，记录美国华盛顿特区 Capital Bikeshare 在 2011—2012 年的租赁情况。

- 原始文件：`data/raw/hour.csv`
- 数据规模：17,379 条记录、17 个字段
- 预测目标：每小时租赁总量 `cnt`
- 模型输入：12 个天气与日历特征

完整字段定义和引用要求见 [原始数据说明](data/raw/Readme.txt)。

### 输入特征

| 类型 | 特征 |
| --- | --- |
| 连续天气特征 | 温度 `temp`、体感温度 `atemp`、湿度 `hum`、风速 `windspeed` |
| 日历与天气类别 | 季节 `season`、年份 `yr`、月份 `mnth`、小时 `hr`、节假日 `holiday`、星期 `weekday`、工作日 `workingday`、天气类别 `weathersit` |

模型通过 `FEATURES` 白名单选择输入。`dteday` 用于排序、时间划分和结果追踪，`cnt` 作为训练目标。

## 回归算法路线

| 模型 | 建模思路 | 在实验中的作用 |
| --- | --- | --- |
| Ridge | 平方误差与 L2 正则化 | 建立加性线性基线 |
| Poisson 回归 | 通过对数链接建模计数目标的条件均值 | 研究计数建模对预测表现的影响 |
| 决策树 | 根据特征条件分裂形成分段预测 | 研究非线性与特征交互 |
| 梯度提升树 | 逐步增加树以改善目标损失 | 比较单树与树集成的泛化表现 |

模型使用 Scikit-learn 实现，基础参数保存在 `configs/baseline.toml`。

## 实验设计与评价指标

### 数据划分

按时间排序并保留完整日期，划分开发训练集与独立测试集：

| 数据部分 | 时间范围 | 样本数 | 用途 |
| --- | --- | ---: | --- |
| 开发训练集 | 2011-01-01 至 2012-08-07 | 13,915 | 数据探索、模型训练与内部验证 |
| 独立测试集 | 2012-08-08 至 2012-12-31 | 3,464 | 配置冻结后的最终评价 |

开发训练集内部采用五折扩展窗口时间验证。先按完整日期划分，再映射回小时记录，四种模型复用相同验证折。

### 预处理

- 连续特征使用中位数填充。
- 类别特征使用众数填充和独热编码。
- Ridge、Poisson 对连续变量标准化。
- 决策树、梯度提升树保留连续变量原始尺度。
- 梯度提升树使用稠密编码输出与固定迭代次数。
- 各折分别拟合完整的预处理与模型流水线。

### 评价指标

| 指标 | 用途 | 方向 |
| --- | --- | --- |
| RMSE | 主要选型指标，对大误差更敏感 | 越低越好 |
| MAE | 解释平均每小时的绝对预测偏差 | 越低越好 |
| R² | 辅助评价相对均值预测的平方误差改善 | 越高越好 |

结果表采用各折指标的算术平均，同时记录折间样本标准差。运行文件还保存 MSE、训练指标和预测值诊断信息，支持后续误差分析。

## 安装与运行

以下命令在 `projects/regression-analysis/` 执行，环境使用 Python 3.12 和 uv 管理。

### 安装环境

```shell
uv sync --locked
```

### 数据准备与探索

```shell
uv run --locked bike-prepare
uv run --locked bike-eda
```

数据准备命令生成划分文件、统计摘要与探索图；模型训练直接读取 `data/raw/hour.csv`。

### 独立训练四种算法

```shell
uv run --locked bike-ridge
uv run --locked bike-poisson
uv run --locked bike-decision-tree
uv run --locked bike-gradient-boosting
```

每个命令分别生成一个包含模型、指标、预测和图表的运行目录。

统一入口支持选择单个模型或依次训练全部模型：

```shell
uv run --locked bike-train --model ridge
uv run --locked bike-train --model all
```

另存实验配置后，通过 `--config` 指定配置文件：

```shell
uv run --locked bike-train --model all --config configs/experiment.toml
```

其中 `experiment.toml` 由实验者创建，默认配置为 `configs/baseline.toml`。

### 整合独立结果

```shell
uv run --locked bike-compare --inputs <Ridge运行目录> <Poisson运行目录> <决策树运行目录> <提升树运行目录>
```

将占位路径替换为训练命令实际输出的目录。汇总程序读取已有结果，核对数据、特征、依赖版本、验证边界与真实记录，生成统一指标表和对比图。

### 运行检查

```shell
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
```

## 项目结构

```text
.
├── README.md
├── pyproject.toml
├── uv.lock
├── .python-version
├── configs/
│   └── baseline.toml
├── src/
│   └── bike_sharing/
│       ├── __init__.py
│       ├── data.py                 # 特征定义与外层时间划分
│       ├── paths.py                # 项目路径
│       ├── preprocessing.py        # 预处理流水线
│       ├── prepare_data.py         # 基础数据探索
│       ├── extended_analysis.py    # 补充分组分析
│       ├── validation.py           # 整日时间验证
│       ├── metrics.py              # 统一评价指标
│       ├── train.py                # 公共训练流程
│       ├── results.py              # 指标汇总与保存
│       ├── plotting.py             # 结果绘图
│       ├── aggregate.py            # 独立结果整合
│       └── models/
│           ├── __init__.py         # 模型注册接口
│           ├── ridge.py
│           ├── poisson.py
│           ├── decision_tree.py
│           └── gradient_boosting.py
├── data/
│   ├── raw/                        # 原始数据与来源说明
│   └── processed/                  # 本地派生数据
├── artifacts/
│   ├── eda/                        # 本地探索图
│   ├── runs/                       # 各次训练产物
│   └── comparisons/                # 综合比较产物
├── docs/                           # 运行说明与代码导读
├── reports/                        # 实验报告与交接
└── tests/                          # 自动化检查
```

内部包名 `bike_sharing` 对应当前应用数据集，项目目录名 `regression-analysis` 描述研究目标。

## 基础实验结果

下表为相同五折时间验证下的指标均值：

| 模型 | RMSE ↓ | MAE ↓ | R² ↑ |
| --- | ---: | ---: | ---: |
| Ridge | 108.84 | 80.58 | 0.545 |
| Poisson | 110.46 | 76.05 | 0.542 |
| 决策树 | 115.34 | 82.91 | 0.476 |
| 梯度提升树 | **88.71** | **63.76** | **0.685** |

当前配置下，梯度提升树取得最低的平均 RMSE 和 MAE，平均 RMSE 相比 Ridge 降低约 18.49%。Poisson 的 MAE 为 76.05、RMSE 为 110.46，体现出与 Ridge 的误差结构差异。决策树的平均 RMSE 为 115.34，为后续树复杂度分析提供了对照。

完整结果、折间标准差和误差分析见 [基础模型报告](reports/baseline_report.md)。

## 结果文件与可视化

运行产物保存在本地 `artifacts/`，每次运行使用独立的时间戳目录。

```text
artifacts/
├── eda/
├── runs/
│   └── <时间戳>_<算法>/
│       ├── config.toml
│       ├── metadata.json
│       ├── fold_boundaries.csv
│       ├── fold_metrics.csv
│       ├── summary_metrics.csv
│       ├── validation_predictions.csv
│       ├── subgroup_metrics.csv
│       ├── models/
│       └── figures/
└── comparisons/
    └── <时间戳>/
```

| 产物 | 内容与用途 |
| --- | --- |
| `config.toml` | 保存本次训练配置 |
| `metadata.json` | 记录运行状态、依赖版本及数据、代码和锁文件哈希 |
| `fold_boundaries.csv` | 保存各折时间范围与样本数 |
| `fold_metrics.csv` | 保存各折训练、验证指标及拟合耗时 |
| `summary_metrics.csv` | 汇总验证指标均值、标准差和预测值诊断 |
| `validation_predictions.csv` | 保存日期小时、原始行号、真实值和预测值 |
| `subgroup_metrics.csv` | 展示工作日类型与逐小时分组误差 |
| `models/*.joblib` | 保存全开发训练集拟合的完整流水线 |

每种算法和综合比较均生成三类图：

| 图表 | 用途 |
| --- | --- |
| `metrics.png` | 展示 MAE、RMSE 的折均值与标准差 |
| `actual_vs_predicted.png` | 观察高估、低估及不同需求水平的误差 |
| `time_comparison.png` | 展示最后验证折前七天的真实值与预测曲线 |

R² 保存于指标文件和报告中。训练完成状态记录在 `metadata.json`；综合比较目录通过 `aggregation.json` 和输入配置快照追踪参与对比的运行。

A 阶段归档图保存在 `reports/figures/`，新生成的数据探索图保存在 `artifacts/eda/`。

## 文档与参考资料

### 项目文档

- [数据准备说明](docs/data_preparation.md)
- [项目迁移说明](docs/migration.md)
- [训练与汇总说明](docs/training.md)
- [代码导读](docs/code_walkthrough.md)
- [数据探索报告](reports/data_preparation.md)
- [基础模型报告](reports/baseline_report.md)
- [B→C 交接说明](reports/handoff_to_c.md)：模型接口、优化方向与最终评价安排

### 数据与工具

- [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)
- [原始数据说明与引用要求](data/raw/Readme.txt)
- [Scikit-learn 官方文档](https://scikit-learn.org/stable/)
- [uv 官方文档](https://docs.astral.sh/uv/)
