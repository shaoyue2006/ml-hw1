# 机器学习项目集合

## 项目介绍

本仓库用于组织机器学习课程项目与实验

各项目按照研究任务独立存放，分别管理代码、数据、依赖环境、实验配置和报告。

## 项目索引

| 项目 | 任务与数据 | 主要方法 | 当前进度 |
| --- | --- | --- | --- |
| [回归算法分析](projects/regression-analysis/README.md) | 使用共享单车数据预测每小时租赁量，比较不同回归方法 | Ridge、Poisson 回归、决策树、梯度提升树 | 已完成数据准备、基础模型训练与时间验证；优化及最终测试待完成 |

## 仓库结构

```text
.
├── README.md
├── .gitignore
├── .gitattributes
└── projects/
    └── regression-analysis/
        ├── README.md
        ├── pyproject.toml
        ├── uv.lock
        ├── .python-version
        ├── configs/           # 实验配置
        ├── src/               # 数据处理、模型与训练代码
        ├── data/              # 原始数据及派生数据
        ├── artifacts/         # 本地实验产物
        ├── docs/              # 使用说明与代码导读
        ├── reports/           # 实验报告与交接说明
        └── tests/             # 自动化检查
```

后续项目在 `projects/` 下建立独立目录。项目名称优先描述研究任务。

## 环境管理

各项目使用 uv 独立管理 Python 环境：

- `pyproject.toml` 声明项目依赖、运行入口及开发工具配置。
- `uv.lock` 锁定依赖版本，便于复现实验环境。
- `.python-version` 指定项目使用的 Python 版本。
- `.venv/` 保存各成员本地维护的虚拟环境。

安装 uv 后，进入具体项目目录同步环境：

```shell
cd projects/regression-analysis
uv sync --locked
```

新增依赖时使用 `uv add`，并同时提交更新后的 `pyproject.toml` 和 `uv.lock`。各项目的具体运行方式见对应 README。
