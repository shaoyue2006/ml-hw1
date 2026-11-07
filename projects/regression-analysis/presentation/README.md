# 从经典回归到深度学习：共享单车小时租赁需求预测

**11 月 9 日课堂 Pre，约 11 页、9 分钟。** 用 12 个天气和日历字段预测每小时租赁量，讲清特征表示、模型复杂度和需求水平怎样影响预测误差。

**做幻灯片**时，按[逐页交接指南](D_PPT_Handoff.md)确定每页讲什么、用哪张图和讲多久，再从[主图速选](figures/selected_figures.md)取图。**准备主讲和答疑**时，查[实验技术手册](D_Experiment_Technical_Guide.md)，核对时间划分、模型原理、早停与选型规则，以及每项结论的数值来源。

核对数字用下面两张表，最终可编辑 PPTX 放入 [ppt/](ppt/)。

| 要回答的问题 | 取数位置 |
|---|---|
| 8 个模型在五折验证和独立测试中各得多少分？ | [模型成绩总表](tables/performance_summary.csv) |
| 在低、中、高需求时段，经过超参数优化的梯度提升树与 MLP 各自更擅长哪一组？ | [独立测试需求分组表](tables/test_demand_groups.csv) |

## 三段实验怎么讲

- **v1 · 建立基线。** 对比 Ridge、Poisson、决策树和梯度提升树。梯度提升树的五折平均 RMSE 最低，为 **88.71**。
- **v2 · 模型优化：特征表示改进与超参数调整。** 第 6 页比较 Ridge 的组合特征优化与梯度提升树的类别特征表示优化。组合小时与是否工作日后，Ridge 的五折平均 RMSE 从 **108.84** 降至 **88.73**（降约 18.48%）；梯度提升树从独热编码改为原生类别处理后，从 **88.71** 降至 **78.06**（降约 12.01%）。第 7 页在原生类别提升树上搜索学习率与最大叶节点数，五折平均 RMSE 进一步降至 **75.10**。这项搜索对应课程的参数优化要求；第 8 页报告冻结方案的独立测试 RMSE **65.50**。
- **v3 · PyTorch MLP 深度学习回归拓展。** 后续 PyTorch MLP 的独立测试 RMSE 为 **61.81**。高需求组的测试 MAE 从优化后的梯度提升树的 **62.57** 降至 MLP 的 **51.77**，低、中需求组则是提升树误差更低。

## 素材维护与核数

[图片清单](figures/assets_manifest.csv)的 `destination` 相对于本目录，`source` 和两张数据表里的来源路径相对于上一级 `regression-analysis/`。优先选用本地真实实验图，维护素材时逐项核对文件、来源与评价范围。缺少来源的成绩留空。五折成绩取各折指标的算术平均，独立测试成绩单列。Poisson 和决策树的补评记录见 [v1 基线报告](../v1/reports/baseline_report.md)。

第 6 页的梯度提升树均为 `HistGradientBoostingRegressor`。蓝色条是开发集内部五个时间验证折的指标算术平均，绿色条是用完整开发集拟合冻结配置后在留出测试集上的指标。v2 只依据五折平均验证 RMSE 选型，独立测试用于报告冻结配置的表现。

从 `regression-analysis/` 在 v3 的 PyTorch 环境运行以下命令。第一条从已保存的结果表绘制基线图、v2/v3 比较图和 v2 测试首周曲线，第二条从需求分组表绘制 PNG 和 SVG。只生成 MLP 架构图可用 `architecture`，生成以上全部素材可用 `all`。

```powershell
python -B presentation/scripts/make_presentation_figures.py comparisons
python -B presentation/scripts/make_presentation_figures.py demand
```
