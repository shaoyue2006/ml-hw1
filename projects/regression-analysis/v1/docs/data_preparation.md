# 共享单车租赁需求预测：数据准备与预处理

## 任务与数据来源

使用美国华盛顿特区 Capital Bikeshare 2011—2012 年的小时数据，根据日期、时段、天气和日历信息预测该小时的租赁总量 cnt。数据来自 UCI Bike Sharing Dataset：https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset 。原始文件和字段说明位于 `../data/raw/`。数据源要求引用 Fanaee-T 和 Gama 的论文，完整信息见原始 Readme.txt。

本实验假设预测时可获得该小时的天气信息。数据提供的是历史天气信息，因此测试结果不能直接解释成使用天气预报的实际预测效果。cnt 是非负整数，用回归预测数值；若预测值为负，应由建模组说明评价和后处理方式。

## 数据检查与划分

hour.csv 共 17,379 行、17 列；缺失值、完全重复行、重复日期和小时均为 0。cnt 是目标，casual 与 registered 的和就是 cnt，禁止作为输入。instant 是记录编号，不作为输入；dteday 用于排序和划分，年、月、小时等字段已提供给模型。

按日期排序，选取最接近 80% 记录数的整日边界：训练集 13,915 行（2011-01-01 至 2012-08-07），测试集 3,464 行（2012-08-08 至 2012-12-31）。保留完整日期，测试集仅用于最终评价。此切分是一次历史时间段评估，不能代表所有季节或未来年份的泛化效果；测试期的年份已出现在训练期。

预处理：四个连续天气特征用训练数据的中位数填充，按需标准化；季节、年、月、小时、节假日、星期、工作日及天气类别用众数填充后独热编码。原始温度等已按数据说明归一化，标准化仍可供 Ridge 等模型使用。所有填充值、标准化参数及编码类别都由流水线在拟合时学习。树模型可通过 `scale_numeric=False` 不做标准化。因为该数据集没有缺失值，填充步骤主要是为了流程完整性。

## 探索图（只使用训练集）

- `../reports/figures/01_rental_distribution.png`：租赁量分布。
- `../reports/figures/02_hourly_pattern.png`：工作日与非工作日逐小时平均租赁量。
- `../reports/figures/03_temperature.png`：温度与租赁量关系。原始 temp 按 41 缩放，图中乘回 41 显示摄氏度。
- `../reports/figures/04_correlations.png`：连续天气特征和目标的相关系数。

训练集的平均租赁量为 174.85，中位数为 131；这些是训练集描述统计，不是模型评价结果。图表标题采用英文，避免本地中文字体缺失。

## 运行和交接

在本项目目录执行 `uv sync --locked` 后运行：

```powershell
uv run --locked bike-prepare
```

脚本会重新生成 `data/processed/train.csv`、`data/processed/test.csv`、数据检查 JSON 和 `artifacts/eda/` 下的探索图；导出的 CSV 是未拟合预处理的原始列，包含目标及需排除的列，仅作为划分结果，建模时务必通过 FEATURES 选择特征。

B 和 C 使用已安装的 `bike_sharing` 包。示例（在本项目的 uv 环境运行）：

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from bike_sharing.data import load_and_split, FEATURES, TARGET
from bike_sharing.preprocessing import make_pipeline

train, test = load_and_split()
model = make_pipeline(Ridge(), scale_numeric=True)
cv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(
    model, train[FEATURES], train[TARGET], cv=cv, scoring="neg_mean_absolute_error"
)
print("交叉验证 MAE:", -scores.mean())
# 确定模型和参数后再用全部训练集拟合，最后只在测试集评估一次。
```

TimeSeriesSplit 基于已排序的训练行，可能在一天内部切分；如需保证每折按整天切分，可自行按日期构造折。调参参数名前加 `model__`，如 `model__alpha`。各组模型应使用同一划分和相同评价指标；不要在全量数据上提前 fit 预处理器。单独测试集不用于选特征或调参。

补充分析运行 `uv run --locked bike-eda`，生成工作日与小时热力图、月份与天气对比图，以及对应分组统计表。更详细的检查过程、分析结果和限制见 [数据探索与预处理报告](../reports/data_preparation.md)。
