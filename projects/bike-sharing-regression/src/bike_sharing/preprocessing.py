"""共享单车：按时间划分和防止数据泄漏的预处理组件。"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bike_sharing.data import CATEGORICAL, NUMERIC


def make_preprocessor(scale_numeric=True):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), NUMERIC),
            ("categorical", Pipeline(categorical_steps), CATEGORICAL),
        ]
    )


def make_pipeline(model, scale_numeric=True):
    return Pipeline([("preprocess", make_preprocessor(scale_numeric)), ("model", model)])
