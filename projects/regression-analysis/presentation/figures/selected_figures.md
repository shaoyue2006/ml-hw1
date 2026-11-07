# 课堂主图速选｜8 张，约 9 分钟

按讲述顺序排列。链接打开可直接插入 PPT 的 PNG。第 1、4、11 页分别放封面、时间划分和结论。各图的原始文件路径及评价范围见[来源清单](assets_manifest.csv)。

| 页 | 取图 | 来源 | 这一页要回答的问题 |
|---:|---|---|---|
| 2 | [租赁量分布](01_data/v1_rental_distribution.png) | v1 数据探索 | 开发集目标为何有长右尾？ |
| 3 | [工作日与非工作日逐小时曲线](01_data/v1_hourly_pattern.png) | v1 数据探索 | 同一小时的需求是否随日期类型改变？ |
| 5 | [四模型验证误差](02_baselines/v1_four_model_metrics.png) | v1 四模型固定配置复现 | 哪个基线的五折平均 RMSE 最低？ |
| 6 | [特征表示改进：Ridge 的组合特征优化与梯度提升树的类别特征表示优化](03_optimization/v2_model_comparison.png) | v2 五候选比较 | 读左图蓝色条，先比较 Ridge 组合小时与工作日前后，再比较提升树采用独热编码与原生类别处理时的五折平均 RMSE。 |
| 7 | [梯度提升树超参数优化：学习率与最大叶节点数](03_optimization/v2_sensitivity.png) | v2 六组配置的五折时间验证 | 固定学习率比较 15、31 两个叶节点上限，分别读训练与验证 RMSE，解释为何在六组中选 0.10、15。 |
| 8 | [测试首周](03_optimization/v2_test_week.png) | v2 冻结后测试 | 真实曲线与传统模型预测在哪些时段分开？整期指标另行标注。 |
| 9 | [MLP 网络结构](04_mlp/mlp_architecture.png) | v3 最终权重与配套预处理器 | 12 个原始字段怎样变为 61 维输入？早停问题可调出[训练曲线](04_mlp/v3_early_stopping.png)。 |
| 10 | [需求分组对比](04_mlp/demand_group_comparison.png) | [独立测试分组表](../tables/test_demand_groups.csv) | MLP 在哪个需求区间降低 MAE？[全模型对比](04_mlp/v3_model_comparison.png)可用于答疑。 |

**第 6 页图注。** 梯度提升树采用 `HistGradientBoostingRegressor`。五折验证均值是开发集内五个时间验证折的算术平均。v2 以五折平均验证 RMSE 选定配置；冻结配置后，用全部开发集拟合，再在留出测试集上计算独立测试指标。第 6 页读蓝色验证条比较输入表示，第 8 页用独立测试展示冻结方案的表现。

[相关性矩阵](01_data/v1_correlations.png)供答疑或正式报告使用。[返回 D 交接指南](../D_PPT_Handoff.md)
