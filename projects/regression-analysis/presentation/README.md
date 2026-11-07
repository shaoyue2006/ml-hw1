# 从经典回归到深度学习：共享单车小时租赁需求预测

**11 月 9 日课堂 Pre，约 11 页、9 分钟。** 用 12 个天气和日历字段预测每小时租赁量，讲清特征表示、模型复杂度和需求水平怎样影响预测误差。

**做幻灯片**时，按[逐页交接指南](D_PPT_Handoff.md)确定每页讲什么、用哪张图和讲多久，再从[主图速选](figures/selected_figures.md)取图。**准备主讲和答疑**时，查[实验技术手册](D_Experiment_Technical_Guide.md)，核对时间划分、模型原理、早停与选型规则，以及每项结论的数值来源。

核对数字用下面两张表，最终可编辑 PPTX 放入 [ppt/](ppt/)。

| 要回答的问题 | 取数位置 |
|---|---|
| 8 个模型在五折验证和独立测试中各得多少分？ | [模型成绩总表](tables/performance_summary.csv) |
| 在低、中、高需求时段，调参提升树与 MLP 各自更擅长哪一组？ | [独立测试需求分组表](tables/test_demand_groups.csv) |

## 三段实验怎么讲

- **v1 · 建立基线。** 对比 Ridge、Poisson、决策树和梯度提升树。梯度提升树的五折平均 RMSE 最低，为 **88.71**。
- **v2 · 改表示，再调参。** Ridge 加入“小时 × 工作日”交互后，验证 RMSE 从 **108.84** 降至 **88.73**。提升树改用原生类别后从 **88.71** 降至 **78.06**，调参后进一步降至 **75.10**；冻结方案的独立测试 RMSE 为 **65.50**。
- **v3 · 深度学习拓展。** 后续 PyTorch MLP 的独立测试 RMSE 为 **61.81**。高需求组的测试 MAE 从调参树的 **62.57** 降至 MLP 的 **51.77**，低、中需求组则是提升树误差更低。

## 素材维护与核数

[图片清单](figures/assets_manifest.csv)的 `destination` 相对于本目录，`source` 和两张数据表里的来源路径相对于上一级 `regression-analysis/`。优先选用本地真实实验图，维护素材时逐项核对文件、来源与评价范围。缺少来源的成绩留空。五折成绩取各折指标的算术平均，独立测试成绩单列。Poisson 和决策树的补评记录见 [v1 基线报告](../v1/reports/baseline_report.md)。

第 9、10 页的架构图和需求分组图同时提供 PNG 和 SVG。架构图读取 v3 最终权重、变换器和配置；分组图直接读取独立测试需求分组表。需要重新制图时，从 `regression-analysis/` 在 v3 的 PyTorch 环境执行以下命令。参数可选 `architecture`、`demand` 或 `all`。

```powershell
python -B presentation/scripts/make_presentation_figures.py all
```
