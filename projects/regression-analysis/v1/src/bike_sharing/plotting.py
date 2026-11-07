"""仅绘制时间验证预测，不将训练预测或独立测试预测混入对比图。"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_results(summary, predictions, output):
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ("mae", "rmse"), strict=True):
        ax.bar(summary.model, summary[f"{metric}_mean"], yerr=summary[f"{metric}_std"], capsize=3)
        ax.set(title=f"Time-validation {metric.upper()} (fold mean +/- SD)", ylabel="Rentals")
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output / "metrics.png", dpi=150)
    plt.close(fig)
    names = list(predictions.model.unique())
    fig, axes = plt.subplots(len(names), 1, figsize=(10, 3 * len(names)), squeeze=False)
    for ax, name in zip(axes[:, 0], names, strict=True):
        part = predictions[predictions.model == name]
        ax.scatter(part.y_true, part.y_pred, s=3, alpha=0.15)
        lo, hi = (
            min(part.y_true.min(), part.y_pred.min()),
            max(part.y_true.max(), part.y_pred.max()),
        )
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--")
        ax.set(title=f"{name}: validation predictions", xlabel="Actual rentals", ylabel="Predicted")
    fig.tight_layout()
    fig.savefig(output / "actual_vs_predicted.png", dpi=150)
    plt.close(fig)
    # 统一取最后一折的前七个自然日，所有模型对比完全相同的真实记录。
    last = predictions[predictions.fold == predictions.fold.max()].copy()
    last["time"] = pd.to_datetime(last.dteday) + pd.to_timedelta(last.hr, unit="h")
    first_week = pd.date_range(pd.Timestamp(last.dteday.min()), periods=7, freq="D")
    last = last[pd.to_datetime(last.dteday).isin(first_week)]
    fig, ax = plt.subplots(figsize=(12, 4))
    first = last[last.model == names[0]]
    ax.plot(first.time, first.y_true, label="Actual", color="black", linewidth=1.4)
    for name, part in last.groupby("model", sort=False):
        ax.plot(part.time, part.y_pred, label=name, alpha=0.8, linewidth=0.8)
    ax.set(title="First seven days of the last validation fold", ylabel="Rentals")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output / "time_comparison.png", dpi=150)
    plt.close(fig)
