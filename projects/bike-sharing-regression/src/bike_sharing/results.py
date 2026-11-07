"""统一整理逐折指标、验证预测和对比图；不参与模型训练。"""

import pandas as pd

from bike_sharing.metrics import regression_metrics
from bike_sharing.plotting import plot_results


def save_csv(frame, path):
    """UTF-8 BOM 便于组员直接用 Excel 打开中文数据。"""
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def save_results(fold_metrics, predictions, output):
    """训练入口和独立汇总入口使用完全相同的指标汇总规则。"""
    names = list(fold_metrics.model.unique())
    summaries = []
    for name in names:
        part = fold_metrics[fold_metrics.model == name]
        row = {"model": name}
        # 先计算每折指标，再做算术平均；标准差不是置信区间。
        # RMSE 折均值不等于折均 MSE 的平方根，也不是拼接预测后的 RMSE。
        for metric in ("mae", "mse", "rmse", "r2"):
            row[f"{metric}_mean"] = part[metric].mean()
            row[f"{metric}_std"] = part[metric].std(ddof=1)
        row["negative_predictions"] = int(part.negative_predictions.sum())
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    save_csv(fold_metrics, output / "fold_metrics.csv")
    save_csv(summary, output / "summary_metrics.csv")
    save_csv(predictions, output / "validation_predictions.csv")
    # 总体指标之外保留小时/工作日分组结果，供 C 排查高峰和低谷的误差。
    grouped = []
    for column in ("workingday", "hr"):
        for (name, value), part in predictions.groupby(["model", column]):
            grouped.append(
                {
                    "model": name,
                    "group": column,
                    "value": int(value),
                    "rows": len(part),
                    **regression_metrics(part.y_true, part.y_pred),
                }
            )
    save_csv(pd.DataFrame(grouped), output / "subgroup_metrics.csv")
    plot_results(summary, predictions, output / "figures")
    return summary
