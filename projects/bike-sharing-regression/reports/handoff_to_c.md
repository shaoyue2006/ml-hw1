# B 到 C 的模型交接

## 已完成范围

B 已建立 Ridge、Poisson、决策树、直方图梯度提升树四个基础模型，完成同一时间划分下的验证、全开发集重拟合、结果生成与基础分析。未执行独立测试集模型评价；本报告不是最终测试报告。

基础参数保存在 `configs/baseline.toml`，实际结果见 [基础报告](baseline_report.md)，运行与产物协议见 [训练说明](../docs/training.md)。基线配置是预先固定的起点，没有经过系统调参。

## C 的复用接口

```python
from sklearn.model_selection import GridSearchCV

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
# 以下是 C 后续运行的优化示例；B 阶段没有执行参数搜索。
search.fit(train[FEATURES], train[TARGET])
```

预处理必须保留在 Pipeline 内，参数使用 `model__` 前缀。不要提前在全开发集拟合预处理器后再交叉验证。直方图提升树的线程由 OpenMP 控制，外层并行搜索时需限制线程，避免过度并行。

## 优先研究问题

1. 学习率、迭代次数、叶节点数与正则化对验证 RMSE 的影响。基础模型存在训练/验证差距，但不能只凭差距就把所有误差归因于过拟合，季节和需求变化也可能影响结果。
2. 同一提升树比较平方误差与 Poisson 损失。平方误差版本存在负预测，Poisson 损失可研究非负计数预测；不要只为计算某个指标临时改变预测后处理。
3. Ridge/Poisson 增加“小时×工作日”交互，检验加性结构不足。保持原始特征基线不变，另建实验配置。
4. 分析 Poisson 在最后验证折的退化，以及高峰时段误差。不要未经验证认定是某个天气或日历变量导致。

## 比较与最终交接规则

- 保持外层划分、五个验证时间段、主指标与原始预测政策一致。更改特征或后处理时单独标记实验。
- 多次搜索后的同一验证折最佳分数可能偏乐观，不能视为独立泛化结果；记录搜索范围和尝试。
- 先固定最终模型、参数、特征及后处理，再统一运行基础模型与优化模型的独立测试集评价。
- 测试集为 2012-08-08 至 2012-12-31，共 3,464 条，不用于调参、早停、选特征或选择截断规则。
- 向 D 交付统一指标、日期小时、真实值与预测值、实验配置及限制。模型的特征关联和重要性不是因果结论。
