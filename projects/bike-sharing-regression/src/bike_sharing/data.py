"""A 模块的数据契约：特征白名单和整日时间划分。"""

import pandas as pd

from bike_sharing.paths import SOURCE

TARGET = "cnt"
NUMERIC = ["temp", "atemp", "hum", "windspeed"]
CATEGORICAL = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]
FEATURES = NUMERIC + CATEGORICAL


def load_and_split():
    df = pd.read_csv(SOURCE, parse_dates=["dteday"])
    df = df.sort_values(["dteday", "hr"], kind="stable").reset_index(drop=True)
    # 整天划分：选择最接近 80% 记录数的日期边界。
    daily = df.groupby("dteday", sort=True).size().cumsum()
    boundary = (daily - len(df) * 0.8).abs().idxmin()
    train = df.loc[df.dteday <= boundary].copy()
    test = df.loc[df.dteday > boundary].copy()
    assert train.dteday.max() < test.dteday.min()
    return train, test
