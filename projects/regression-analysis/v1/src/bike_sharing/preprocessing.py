"""共享单车：按时间划分和防止数据泄漏的预处理组件。"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bike_sharing.data import CATEGORICAL, NUMERIC


def make_preprocessor(scale_numeric=True, *, dense=False):
    """只构建未拟合的预处理器，由 Pipeline 在每个训练折内学习参数。"""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    # 未见过的月份、年份或天气类别不从未来数据预先获取，预测时忽略。
    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)),
    ]
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), NUMERIC),
            ("categorical", Pipeline(categorical_steps), CATEGORICAL),
        ]
    )


def make_pipeline(model, scale_numeric=True, *, dense=False):
    # 保存的是“预处理 + 模型”整体，预测时不需要手工重复编码和标准化。
    return Pipeline(
        [
            ("preprocess", make_preprocessor(scale_numeric, dense=dense)),
            ("model", model),
        ]
    )
