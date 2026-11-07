# 回归算法分析

## 项目概述

本项目以共享单车租赁需求预测为应用场景，比较 Ridge、Poisson 回归、决策树和梯度提升树，研究不同回归方法的预测表现及改进效果。

## 当前进度

| 成员 | 负责内容 | 当前进度 |
| --- | --- | --- |
| A | 数据准备与预处理 | 已完成质量检查、EDA、特征定义、时间划分和预处理 |
| B | 基础回归模型构建 | 已完成四模型训练、时间验证、图表及结果分析，归档为 v1 |
| C | 模型优化与拓展实验 | 在 v2 继续开展改进，已提供方向指导与诊断实验结论 |

## 版本结构

```text
regression-analysis/
├── README.md
├── v1/                         # 第一轮完整项目与基础结果
│   ├── README.md
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── .python-version
│   ├── configs/
│   ├── src/bike_sharing/
│   ├── data/
│   ├── artifacts/               # 本地运行产物
│   ├── docs/
│   ├── reports/
│   │   ├── baseline_report.md
│   │   └── handoff_to_c.md
│   └── tests/
└── v2/                         # 第二轮改进指导
    ├── README.md
    └── diagnostic_results.md
```

v1 保存第一轮的模型脚本、配置、数据、环境、文档和结果。v2 当前保存优化计划与小型诊断结论，由 C 继续实现与验证。后续 v3、v4 按相同方式建立独立目录，并记录相对 v1 与上一版本的改进效果。

## 版本索引

| 版本 | 内容 | 状态 | 文档 |
| --- | --- | --- | --- |
| v1 | 四种基础回归模型及统一时间验证 | 已完成 | [运行与结果](v1/README.md) |
| v2 | 交互特征、类别处理与后续优化指导 | 方案与诊断结论已整理，正式实现待 C 开展 | [改进指导](v2/README.md)、[诊断结论](v2/diagnostic_results.md) |

## 运行 v1

```shell
cd projects/regression-analysis/v1
uv sync --locked
uv run --locked bike-train --model all
```

四个单模型入口、结果汇总方式及完整目录说明见 [v1 README](v1/README.md)。各版本在自身目录管理环境和运行结果，便于保留每轮实验条件。

## 文档与交接

- [v1 基础模型报告](v1/reports/baseline_report.md)
- [B→C 交接文件](v1/reports/handoff_to_c.md)
- [v2 改进指导](v2/README.md)
- [v2 小型诊断实验结论](v2/diagnostic_results.md)
