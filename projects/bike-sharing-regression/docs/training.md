# 基础回归模型训练

所有命令在 `projects/bike-sharing-regression/` 执行。

```powershell
uv sync --locked
uv run --locked bike-ridge
uv run --locked bike-poisson
uv run --locked bike-decision-tree
uv run --locked bike-gradient-boosting
```

四个算法各有独立 Python 文件和运行入口；也可通过公共入口选择模型：

```powershell
uv run --locked bike-train --model ridge
uv run --locked bike-train --model poisson
uv run --locked bike-train --model decision_tree
uv run --locked bike-train --model gradient_boosting
uv run --locked bike-train --model all
```

`--model all` 按 Ridge、Poisson、决策树、梯度提升树顺序训练。默认配置为 `configs/baseline.toml`，修改实验配置时另存新文件并使用 `--config configs/新实验.toml`，不覆盖已经记录的基础配置。`--output-root` 可指定本地产物目录，默认 `artifacts/runs/`。

## 模块职责

| 模块 | 职责 |
| --- | --- |
| `data.py` | 原始数据、特征白名单及外层时间划分 |
| `preprocessing.py` | 未拟合的填充、标准化与编码流水线 |
| `models/ridge.py` | Ridge 模型构建与独立运行入口 |
| `models/poisson.py` | Poisson 模型构建与独立运行入口 |
| `models/decision_tree.py` | 决策树模型构建与独立运行入口 |
| `models/gradient_boosting.py` | 梯度提升树模型构建与独立运行入口 |
| `models/__init__.py` | 注册模型，保留统一 build_model 接口 |
| `validation.py` | 按整日构造扩展窗口验证折，供 B/C 共用 |
| `metrics.py` | 在原始目标尺度上统一计算指标 |
| `train.py` | 配置检查、逐折训练、全开发集重拟合和产物保存 |
| `plotting.py` | 验证指标、预测散点及时间片段对比图 |
| `results.py` | 四模型统一的指标汇总和结果保存 |
| `aggregate.py` | 读取已有独立结果并检查可比性，生成综合对比，不训练模型 |

模型输入仅使用 `FEATURES`。Ridge/Poisson 标准化连续特征；树模型不标准化。四者均使用相同类别列的独热编码；梯度提升树使用稠密输出适配接口，实际类别集合仍仅由各训练折学习。基础版梯度提升树关闭自动早停，不引入随机验证子集。迭代模型若出现不收敛警告，实验会失败而不是悄悄记录结果。

## 验证与指标

保留 A 的 13,915 条开发训练记录，在其唯一日期序列上使用 5 折 `TimeSeriesSplit`，再映射回小时记录。验证折不重叠、训练窗口逐渐扩展，同一天不会跨训练与验证。早期训练期未出现的类别使用 `handle_unknown="ignore"` 处理，不从未来学习类别集合。

输出 RMSE、MAE、MSE、R²，记录训练指标和负预测数量。模型选择依据各折 RMSE 的算术平均；标准差采用样本标准差（ddof=1），不是置信区间。折间 R² 的平均也不等于拼接所有验证预测计算的总体 R²。RMSE 的折均值不等于折均 MSE 的平方根。预测不截断、不取整。

`bike-train` 不计算独立测试集指标。全开发集拟合出的模型用于交接，不应把其训练指标称为泛化结果。模型 `.score()` 不用于统一比较，以免不同估计器默认评分含义不同。

## 本地产物

每次运行创建新的 UTC 时间戳目录；已存在目录拒绝覆盖。只有包含 `metadata.json` 且 `status` 为 `complete` 的目录才视为 CLI 成功运行。失败目录可能留下部分文件，不应当作正式结果。

| 文件 | 内容 |
| --- | --- |
| `config.toml`、`metadata.json` | 配置快照、数据/代码/锁文件哈希、Git 状态、依赖版本、评价约定 |
| `fold_boundaries.csv` | 每折日期与样本数 |
| `fold_metrics.csv` | 各模型各折的训练与验证指标、拟合耗时 |
| `summary_metrics.csv` | 验证指标折均值、标准差及负预测数量 |
| `validation_predictions.csv` | 模型、折号、原始行索引、日期小时、真实值和预测值 |
| `subgroup_metrics.csv` | 按工作日类型、小时统计的验证误差及样本数 |
| `models/*.joblib` | 在全部开发训练集上重拟合的完整 Pipeline |
| `figures/*.png` | 指标对比、真实值与预测值、最后一折前七天预测图 |

验证预测只覆盖 11,608 条开发记录；最早用于首折训练的 2,307 条没有验证预测，不用训练预测补齐。小时记录存在缺口，时间图按真实日期小时绘制。

运行产物不提交 Git，整理后的方法与结果见 `reports/baseline_report.md`。不要加载来源不可信的 joblib 文件。

## 独立整合四种算法结果

四个单模型命令分别输出一个运行目录。将实际目录替换到下面命令；不要把占位路径直接运行：

```powershell
uv run --locked bike-compare --inputs artifacts/runs/<ridge运行目录> artifacts/runs/<poisson运行目录> artifacts/runs/<决策树运行目录> artifacts/runs/<提升树运行目录>
```

也支持对一个 `--model all` 产生的完成目录重新生成对比。输出默认位于 `artifacts/comparisons/<时间戳>/`，包含统一指标、验证预测、分组误差、对比图以及输入配置快照。汇总时不重新拟合模型。

汇总会拒绝重复模型、未完成运行、不同原始数据/特征/软件版本/预测规则、不同验证边界或不一致的真实记录。它不会自动挑选“每个模型最好的一次”，输入由实验者明确指定。代码阅读顺序见 [代码导读](code_walkthrough.md)。
