# 项目结构迁移

本次仅整理结构、环境和 A 模块入口，不增加模型训练，不改变特征列表、预处理算法或外层时间划分。

| 旧路径或命令 | 新路径或命令 |
| --- | --- |
| `数据集/共享单车/` | `projects/regression-analysis/v1/data/raw/` |
| `共享单车项目/preprocessing.py` | `src/bike_sharing/data.py` 与 `src/bike_sharing/preprocessing.py` |
| `共享单车项目/prepare_data.py` | `uv run --locked bike-prepare` |
| `共享单车项目/extended_analysis.py` | `uv run --locked bike-eda` |
| `共享单车项目/使用说明.md` | `docs/data_preparation.md` |
| `共享单车项目/数据探索与预处理报告.md` | `reports/data_preparation.md` |
| 原已提交的 `图表/` | `reports/figures/` |
| 新生成的图表 | `artifacts/eda/` |
| 原生成的 `数据/` | `data/processed/` |
| 根目录 `requirements.txt` | 项目 `pyproject.toml` 与 `uv.lock` |

表中未写完整的新路径均相对 `projects/regression-analysis/v1/`。命令也在该项目目录内执行，首次运行前执行 `uv sync --locked`。旧脚本路径不保留重复实现或兼容入口，组员应更新自己的命令与导入。

原 `from preprocessing import ...` 改为：

```python
from bike_sharing.data import FEATURES, TARGET, load_and_split
from bike_sharing.preprocessing import make_pipeline, make_preprocessor
```

数据接口只负责读取与划分；预处理接口负责构建未拟合流水线；`paths.py` 统一管理项目内部路径。分析脚本改成 `main()` 入口，导入不会触发文件写入。补充分析可独立执行，不要求先运行数据准备脚本。

Python 环境从原先的“3.9 或以上”调整为锁定 Python 3.12 系列。安装以 uv 为准，移除手写 requirements，避免两套依赖定义漂移。

原始数据 SHA-256：

- `hour.csv`: `e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f`
- `Readme.txt`: `b92c8628622948bf0828c43a1b315d1517c3d7fd63654730d0dea379b8e88175`
