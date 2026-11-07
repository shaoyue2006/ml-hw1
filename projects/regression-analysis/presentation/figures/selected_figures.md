# 课堂主图速选｜8 张，约 9 分钟

按讲述顺序排列。链接打开可直接插入 PPT 的 PNG。第 1、4、11 页分别放封面、时间划分和结论。各图的原始文件路径及评价范围见[来源清单](assets_manifest.csv)。

| 页 | 取图 | 来源 | 这一页要回答的问题 |
|---:|---|---|---|
| 2 | [租赁量分布](01_data/v1_rental_distribution.png) | v1 数据探索 | 开发集目标为何有长右尾？ |
| 3 | [工作日与非工作日逐小时曲线](01_data/v1_hourly_pattern.png) | v1 数据探索 | 同一小时的需求是否随日期类型改变？ |
| 5 | [四模型验证误差](02_baselines/v1_four_model_metrics.png) | v1 四模型固定配置复现 | 哪个基线的五折平均 RMSE 最低？ |
| 6 | [表示优化对比](03_optimization/v2_model_comparison.png) | v2 五候选比较 | 分别改变 Ridge 的交互项与提升树的类别处理，验证误差怎样变化？指蓝色条。 |
| 7 | [参数敏感性](03_optimization/v2_sensitivity.png) | v2 六组参数搜索 | 更多叶节点降低训练误差，能否同时改善验证？ |
| 8 | [测试首周](03_optimization/v2_test_week.png) | v2 冻结后测试 | 真实曲线与传统模型预测在哪些时段分开？整期指标另行标注。 |
| 9 | [MLP 网络结构](04_mlp/mlp_architecture.png) | v3 最终权重与配套预处理器 | 12 个原始字段怎样变为 61 维输入？早停问题可调出[训练曲线](04_mlp/v3_early_stopping.png)。 |
| 10 | [需求分组对比](04_mlp/demand_group_comparison.png) | [独立测试分组表](../tables/test_demand_groups.csv) | MLP 在哪个需求区间降低 MAE？[全模型对比](04_mlp/v3_model_comparison.png)可用于答疑。 |

[相关性矩阵](01_data/v1_correlations.png)供答疑或正式报告使用。[返回 D 交接指南](../D_PPT_Handoff.md)
