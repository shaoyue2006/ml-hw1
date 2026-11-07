"""在原始租赁量尺度评价，不截断负预测，不取整。"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred):
    truth, pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if truth.ndim != 1 or truth.shape != pred.shape or len(truth) < 2:
        raise ValueError("Metrics require matching 1-D arrays with at least two observations.")
    if not (np.isfinite(truth).all() and np.isfinite(pred).all()):
        raise ValueError("Non-finite target or prediction.")
    # RMSE 作为主选型指标；MAE 解释平均偏差，R² 辅助说明拟合水平。
    # 显式调用函数，不用 model.score()，因为不同模型默认评分可能不同。
    mse = mean_squared_error(truth, pred)
    return {
        "mae": float(mean_absolute_error(truth, pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(truth, pred)),
        "negative_predictions": int((pred < 0).sum()),
    }
