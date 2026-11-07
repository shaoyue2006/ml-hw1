# 共享单车租赁需求预测

本仓库目前包含 A 组员负责的数据准备与探索部分：UCI 共享单车小时数据、按时间划分的方法、可复用的预处理流水线及训练集探索图。模型训练和评估由其他组员接续完成。

## 运行

需要 Python 3.9 或更新版本。安装依赖后在仓库根目录运行：

```bash
python -m pip install -r requirements.txt
python 共享单车项目/prepare_data.py
python 共享单车项目/extended_analysis.py
```

数据划分及运行示例见 [使用说明](共享单车项目/使用说明.md)，数据分析结果见 [数据探索与预处理报告](共享单车项目/数据探索与预处理报告.md)。生成的训练集、测试集和统计文件位于 `共享单车项目/数据/`，可随时通过脚本重建。

## 数据来源

[UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)，原始字段说明及引用要求见 [Readme.txt](数据集/共享单车/Readme.txt)。

