"""按完整日期进行扩展窗口验证，避免同一天跨训练与验证。"""

import numpy as np
from sklearn.model_selection import TimeSeriesSplit


def make_time_folds(frame, n_splits=5):
    if not frame.dteday.is_monotonic_increasing:
        raise ValueError("Input must be sorted by date before time validation.")
    # 先划分唯一日期，再映射回小时行；直接按行切分可能把一天截成两半。
    dates = frame.dteday.drop_duplicates().to_numpy()
    folds = []
    for train_days, valid_days in TimeSeriesSplit(n_splits=n_splits).split(dates):
        # 返回位置索引，供 iloc 和 GridSearchCV 共用；所有算法使用相同折。
        train = np.flatnonzero(frame.dteday.isin(dates[train_days]).to_numpy())
        valid = np.flatnonzero(frame.dteday.isin(dates[valid_days]).to_numpy())
        folds.append((train, valid))
    return folds
