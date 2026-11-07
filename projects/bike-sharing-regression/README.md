# 共享单车租赁需求回归

使用 UCI Bike Sharing 的 2011—2012 年小时数据，根据天气与日历特征预测每小时租赁总量 `cnt`。已实现 Ridge、Poisson、决策树及直方图梯度提升树的基础训练与时间验证，独立测试集留待 B/C 配置冻结后统一评价。

## 环境与运行

在本目录执行，Python 3.12 由 `.python-version` 指定，实际依赖版本由 `uv.lock` 锁定：

```powershell
uv sync --locked
uv run --locked bike-prepare
uv run --locked bike-eda
uv run --locked bike-ridge
uv run --locked bike-poisson
uv run --locked bike-decision-tree
uv run --locked bike-gradient-boosting
uv run --locked bike-train --model ridge
uv run --locked bike-train --model all
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
```

也可以运行 `uv run --locked python -m bike_sharing.prepare_data` 与 `uv run --locked python -m bike_sharing.extended_analysis`。代码导入不执行数据分析或生成文件，入口与当前工作目录无关。

新增依赖使用 `uv add`；开发依赖使用 `uv add --dev`；同时提交 `pyproject.toml` 与 `uv.lock`。不维护第二套手写 requirements 文件。

## 目录职责

```text
src/bike_sharing/   数据接口、预处理、分析入口；后续增加训练模块
data/raw/          原始 CSV 与原始数据说明，不修改
data/processed/    可重建的划分 CSV、检查 JSON、探索统计（忽略）
artifacts/eda/     运行生成的探索图（忽略）
reports/           已整理报告与保留的 A 阶段图表
docs/              数据运行说明和迁移记录
tests/             原始文件完整性、时间划分与特征契约检查
```

基础配置位于 `configs/baseline.toml`；模型运行产物位于 `artifacts/runs/<run_id>/`，默认不提交 Git。`reports/figures/` 保留原有已提交图表，探索脚本重新运行生成到 `artifacts/eda/`，不会覆盖报告归档。

## 数据与接口

原始文件为 `data/raw/hour.csv`，共 17,379 行。数据来源：[UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)，引用要求见 [原始说明](data/raw/Readme.txt)。

```python
from bike_sharing.data import FEATURES, TARGET, load_and_split
from bike_sharing.preprocessing import make_pipeline

train, test = load_and_split()
# 建模时仅使用 train[FEATURES]；预处理必须在相应训练数据内拟合。
```

沿用 A 的时间划分：训练集 13,915 行，2011-01-01 至 2012-08-07；独立测试集 3,464 行，2012-08-08 至 2012-12-31。目标泄漏字段 `casual`、`registered` 和编号 `instant` 不作为特征，日期只用于排序与划分。预测设定假定目标小时天气已知，不等同于使用天气预报的实际效果。

## 已确定的 B 开发约定

- 算法路线：Ridge → Poisson → 决策树 → 直方图梯度提升树。
- 评价：RMSE 为模型选择主指标，MAE 为实际误差指标，R² 为辅助指标。
- 训练集内部采用共享的时间验证折，每折单独拟合完整预处理流水线。
- 基础评价保留原始预测，不取整；负预测单独统计。
- B、C 配置冻结前，不使用独立测试集评价和调参。
- 四模型共享整日时间验证折，基础参数固定；C 在相同协议下优化。

详细数据分析见 [A 阶段报告](reports/data_preparation.md)，运行细节见 [数据准备说明](docs/data_preparation.md)。

模型运行见 [训练说明](docs/training.md)，实测结果见 [基础模型报告](reports/baseline_report.md)，优化接口见 [B→C 交接](reports/handoff_to_c.md)。

四种算法分别位于 `src/bike_sharing/models/` 下的独立文件；训练后可用 `bike-compare --inputs <运行目录...>` 单独汇总。代码调用关系和修改位置见 [代码导读](docs/code_walkthrough.md)。
