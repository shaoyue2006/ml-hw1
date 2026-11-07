# 回归代码导读与交接

## 为什么既有独立算法文件，也有公共训练流程

四种算法分别放在 `models/ridge.py`、`models/poisson.py`、`models/decision_tree.py` 和 `models/gradient_boosting.py`。每个文件负责自己的估计器、预处理选择及运行入口。训练循环不复制四份，而由 `train.py` 统一执行；否则很容易出现某人修改划分、某人忘记同步指标的问题。

使用 Scikit-learn 的模型接口完成拟合；这些文件是课程算法的工程封装，不是从零推导求解器。算法基本原理与结果分析在基础报告中说明，关键实现决策在对应文件用中文注释。

## 一次 Ridge 训练的执行链

```text
bike-ridge / python -m bike_sharing.models.ridge
  → ridge.py 的 main()
  → train.py 的 main(model_name="ridge")
  → load_config() 读取 baseline.toml
  → data.load_and_split() 保留外层测试集
  → run_experiment() 运行开发训练集内部时间验证
      → validation.make_time_folds() 按整日构造同一组折
      → models.build_model() 转到 ridge.build_pipeline()
      → 每折 clone() 完整 Pipeline
      → fit() 学习本折预处理与模型参数
      → predict() 只预测本折验证行
      → metrics.regression_metrics() 计算统一指标
      → 全开发集重新 fit() 并保存 joblib
      → results.save_results() 输出指标、预测和图表
  → 保存配置快照、运行元信息及代码/数据哈希
```

另外三个算法的入口与这条链相同，变化只发生在对应 `build_pipeline()` 的模型与预处理选择中。所有模型的外层输入均为原始 12 列，不允许某个脚本自行加入 `casual` 或 `registered`。

## 四个算法文件应该怎样改

| 文件 | 主要实现 | 后续可能调整 |
| --- | --- | --- |
| `models/ridge.py` | Ridge + 连续变量标准化 + 类别独热编码 | alpha、交互项实验 |
| `models/poisson.py` | PoissonRegressor + 同样的标准化与编码 | alpha、计数模型诊断 |
| `models/decision_tree.py` | 单棵决策树，不标准化连续变量 | 深度、叶节点样本数、剪枝 |
| `models/gradient_boosting.py` | 直方图提升树，稠密编码，关闭随机早停 | 学习率、轮数、树复杂度、Poisson 损失 |

常规参数放在配置文件，避免直接改 Python。模型构建函数不调用 `fit()`，因此 GridSearchCV 可以在每折克隆整个流水线，并通过 `model__alpha` 等参数调优。工厂函数中的 `seed` 为统一接口，当前 Ridge/Poisson 确定性求解器不使用它。

## 独立运行后的结果如何整合

```text
四个独立命令 → 四份完成运行目录
  → aggregate.py 读取 metadata、配置、折边界、指标和预测
  → 检查数据哈希与实验规则
  → 检查各模型的行号、折号、真实值与日期小时完全对应
  → results.save_results() 生成统一指标表和对比图
```

`aggregate.py` 不调用模型 `fit()`，无需重新训练，也不会从各次运行中自动挑选最好的成绩。`results.py` 由训练和汇总共用，保证“单独运行时的指标”与“综合报告的指标”采用相同计算口径。

## 交接时最容易出错的位置

- `preprocessing.py`：只能构建流水线，在各折训练阶段拟合；不要提前全量 fit。
- `validation.py`：以日期为单位切分，返回 iloc 位置索引；不要随机打乱。
- `metrics.py`：保留原始预测，包括负数，不能为某个模型临时截断。
- `train.py`：每折必须 clone 未拟合的流水线；最后保存的模型是全开发集重拟合版本，不是任何一折的模型。
- `results.py`：报告的是各折指标算术平均，标准差不是置信区间，也不是拼接后的总体指标。
- `aggregate.py`：一次比较只选一个版本的每种模型，基础版本与优化版本应明确区分。
- 独立测试集仅在 B/C 配置冻结后统一评价，当前训练入口不输出测试指标。
