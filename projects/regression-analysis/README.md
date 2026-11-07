# 回归算法分析

## 项目概述

本项目以共享单车租赁需求预测为应用场景，比较 Ridge、Poisson 回归、决策树和梯度提升树，研究不同回归方法的预测表现及改进效果。

## 当前进度

| 成员 | 负责内容 | 当前进度 |
| --- | --- | --- |
| A | 数据准备与预处理 | 已完成质量检查、EDA、特征定义、时间划分和预处理 |
| B | 基础回归模型构建 | 已完成四模型训练、时间验证、图表及结果分析，归档为 v1 |
| C | 模型优化与拓展实验 | 已完成两项诊断复现、六组参数搜索、五模型独立测试和课堂材料 |

## 版本结构

各版本分别维护代码、配置、实验结果和报告。v1 的[基线报告](v1/reports/baseline_report.md)同时指向五折复现明细与后续独立测试记录。D 从[课堂汇报入口](presentation/README.md)取用图表、两张成绩与分组表，以及逐页制作指南。

## 版本索引

| 版本 | 内容 | 状态 | 文档 |
| --- | --- | --- | --- |
| v1 | 四种基础回归模型及统一时间验证 | 已完成 | [运行与结果](v1/README.md) |
| v2 | 特征交互、原生类别与提升树调参 | 开发集选型及独立测试已完成 | [C 实验报告](v2/C_report.md)、[课堂摘要](v2/C_pre_summary.md) |
| v3 | PyTorch MLP 深度学习拓展 | 预设网络已完成五折时间验证及留出测试 | [MLP 报告、图表与复现命令](v3/README.md) |

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
