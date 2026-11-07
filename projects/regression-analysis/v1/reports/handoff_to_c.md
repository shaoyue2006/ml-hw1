# B 到 C 的模型交接

## 版本交接安排

| 目录 | 内容 | 后续负责人 |
| --- | --- | --- |
| `v1/` | B 完成的第一轮模型、配置、数据、环境、报告及本地运行结果 | 保留为基准 |
| `v2/` | 改进方向与两组小型诊断结论 | C 接续正式实现与实验 |

v2 的优先工作是复现小时与工作日交互、提升树原生类别处理。后续修改放在 v2，v1 用作相同条件下的版本对照。每轮记录改动、配置、MAE/RMSE/R² 及改善比例，后续 v3、v4 延续同一记录方式。

- [v2 改进指导](../../v2/README.md)
- [小型诊断结论](../../v2/diagnostic_results.md)

当前命令在 `projects/regression-analysis/v1/` 执行，v2 的正式代码和运行方式由 C 补充。

## B 阶段成果

B 已建立 Ridge、Poisson 回归、决策树和直方图梯度提升树，完成共享时间划分下的训练验证、全开发集重拟合、结果生成和基础分析。

| 交付内容 | 位置 | 作用 |
| --- | --- | --- |
| 四种算法 | `src/bike_sharing/models/` | 支持独立训练和模型构建 |
| 基础参数 | `configs/baseline.toml` | 作为优化前的比较起点 |
| 时间验证接口 | `src/bike_sharing/validation.py` | 提供共享的整日验证折 |
| 统一指标 | `src/bike_sharing/metrics.py` | 计算 RMSE、MAE、MSE、R² |
| 结果汇总入口 | `bike-compare` | 生成统一指标与对比图 |
| 基础结果与分析 | [基础报告](baseline_report.md) | 提供优化效果的参照 |
| 运行说明 | [训练说明](../docs/training.md) | 说明训练命令和产物协议 |

完整模型、验证预测、分组误差与图表保存在各成员本地的 `artifacts/` 中，可以通过相应配置重新生成。

## C 阶段工作安排

| 方向 | 建议实验 | 观察内容 |
| --- | --- | --- |
| 参数优化 | 学习率、迭代次数、叶节点数、正则化 | 验证 RMSE、MAE 与训练验证差距 |
| 损失函数 | 提升树的平方误差与 Poisson 损失 | 高低需求场景的预测分布 |
| 特征工程 | Ridge、Poisson 增加“小时×工作日”交互 | 加性模型表达能力的变化 |
| 泛化分析 | 不同时间折、工作日类型和需求水平 | 模型在不同场景下的误差 |
| 误差诊断 | Poisson 最后一折的高估记录 | 结合特征与系数寻找可验证解释 |

为每组优化方案保存独立配置，并与基础配置在相同验证规则下比较。

## 可复用接口

下面的参数搜索示例供 C 阶段执行：

```python
from sklearn.model_selection import GridSearchCV
from threadpoolctl import threadpool_limits

from bike_sharing.data import FEATURES, TARGET, load_and_split
from bike_sharing.models import build_model
from bike_sharing.train import DEFAULT_CONFIG, load_config
from bike_sharing.validation import make_time_folds

config = load_config(DEFAULT_CONFIG)
train, held_out = load_and_split()
model = build_model("gradient_boosting", config["models"]["gradient_boosting"], config["seed"])
search = GridSearchCV(
    model,
    {"model__learning_rate": [0.03, 0.05, 0.1], "model__max_iter": [100, 200, 400]},
    cv=make_time_folds(train, config["n_splits"]),
    scoring="neg_root_mean_squared_error",
    refit=True,
    n_jobs=1,
)
with threadpool_limits(limits=config["threads"]):
    search.fit(train[FEATURES], train[TARGET])
```

预处理与估计器保留在完整 Pipeline 中，各折分别学习预处理和模型参数。搜索参数使用 `model__` 前缀，线程上限通过 `threadpool_limits` 控制。

## 统一比较约定

- 沿用原始数据、12 个基础输入特征和外层时间划分。
- 共用五个整日时间验证折，主选型依据为平均验证 RMSE，同时记录 MAE、R² 与折间标准差。
- 以原始预测值作为基础比较口径；新增特征或后处理时标记实验版本，并保留对应配置。
- 系统记录搜索范围、尝试方案、验证结果和选择依据。
- 需要早停时，在当前训练窗口内部安排靠后的时间段作为早停验证段。
- 模型诊断结合训练误差、验证误差、时间变化和分组表现展开；变量作用解释结合专门的对照实验。

## 最终评价与 D 阶段交接

模型、参数、特征和后处理方案确定后，在独立测试集上统一评价基础模型与优化模型。

独立测试集覆盖 2012-08-08 至 2012-12-31，共 3,464 条记录，承担最终泛化评价。开发训练集承担参数选择、特征选择和内部验证。

交给 D 的材料包括：

1. 基础与优化模型的最终指标及对比图。
2. 带日期小时的真实值、预测值和分组误差。
3. 对应实验配置、模型版本与复现命令。
4. 优化效果分析和需要进一步讨论的实验现象。
