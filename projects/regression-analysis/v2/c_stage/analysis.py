"""复用逐小时预测，生成分组统计、课堂展示图和测试/验证比较。"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from bike_sharing.data import TARGET, load_and_split
from bike_sharing.metrics import regression_metrics
from bike_sharing.results import save_csv

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

LABELS = {
    "ridge_v1": "Ridge（基线）",
    "ridge_hour_workingday": "Ridge（组合特征优化）",
    "boosting_v1": "梯度提升树（基线）",
    "boosting_native": "梯度提升树（原生类别处理）",
    "boosting_tuned": "梯度提升树（原生类别 + 超参数优化）",
}
BLUE, TEAL, GREY = "#395981", "#09776e", "#607180"


def groups(pred):
    rows = []
    for col in ("hr", "workingday"):
        for (case, value), part in pred.groupby(["case", col], sort=False):
            rows.append(
                {
                    "case": case,
                    "group": col,
                    "value": value,
                    "rows": len(part),
                    **regression_metrics(part.y_true, part.y_pred),
                }
            )
    return pd.DataFrame(rows)


def legacy_groups():
    import reproduce

    pred = pd.read_csv(reproduce.V2 / "artifacts/validation_predictions.csv.gz")
    train, _ = load_and_split()
    keys = train[["dteday", "hr", "workingday", TARGET]].rename_axis("row_id").reset_index()
    keys["dteday"] = keys.dteday.dt.strftime("%Y-%m-%d")
    pred = pred.rename(columns={"date": "dteday", "hour": "hr"})
    joined = pred.merge(keys, on=["row_id", "dteday", "hr"], validate="many_to_one", how="left")
    if (
        not np.array_equal(joined.y_true, joined[TARGET])
        or joined.duplicated(["row_id", "case"]).any()
    ):
        raise ValueError("Historical validation predictions do not match the development data")
    return groups(joined)


def style():
    # 优先使用系统中文字体；预先导出的 PNG 可直接跨平台展示。
    available = {f.name for f in font_manager.fontManager.ttflist}
    fonts = ["Microsoft YaHei", "Noto Sans CJK SC", "PingFang SC", "SimHei"]
    plt.rcParams.update(
        {
            "font.family": [f for f in fonts if f in available] + ["DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#b7c1c6",
            "font.size": 11,
            "figure.dpi": 115,
            "savefig.dpi": 170,
        }
    )


def finish(fig, path):
    fig.savefig(path, bbox_inches="tight", facecolor="white", pad_inches=0.22)
    plt.close(fig)


def plot_grid(data, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), layout="constrained")
    for ax, column, title in zip(
        axes,
        ["rmse_mean", "train_rmse_mean"],
        ["验证集 · 五折平均 RMSE", "训练集 · 五折平均 RMSE"],
        strict=True,
    ):
        for leaves, part in data.groupby("max_leaf_nodes"):
            part = part.sort_values("learning_rate")
            color = TEAL if leaves == 15 else BLUE
            ax.plot(
                part.learning_rate,
                part[column],
                marker="o",
                lw=2,
                color=color,
                label=f"最多 {leaves} 个叶节点",
            )
            for row in part.itertuples():
                y = getattr(row, column)
                ax.annotate(
                    f"{y:.2f}",
                    (row.learning_rate, y),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                    color=color,
                )
        ax.set(title=title, xlabel="学习率", xticks=[0.03, 0.05, 0.1], ylabel="每小时租赁量")
        ax.margins(x=0.12, y=0.25)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(loc="lower left", fontsize=9)
    finish(fig, out / "sensitivity.png")


def plot_metrics(comparison, out, filename="model_comparison.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), layout="constrained")
    positions = np.arange(len(comparison))
    for ax, metric in zip(axes, ["rmse", "mae"], strict=True):
        for offset, key, title, color in [
            (-0.18, metric + "_mean", "五折验证均值", BLUE),
            (0.18, metric, "独立测试", TEAL),
        ]:
            bars = ax.barh(positions + offset, comparison[key], 0.31, color=color, label=title)
            ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
        ax.set(
            xlim=(0, comparison[[metric, metric + "_mean"]].max().max() * 1.19),
            xlabel="每小时租赁量",
            title=metric.upper() + "（越低越好）",
            yticks=positions,
        )
        labels = [LABELS[x] for x in comparison.case] if ax is axes[0] else []
        ax.set_yticklabels(labels, fontsize=10)
        ax.invert_yaxis()
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="outside upper center", ncols=2, frameon=False
    )
    finish(fig, out / filename)


def plot_week(pred, winner, out, filename="test_week.png"):
    pred = pred.copy()
    pred["time"] = pd.to_datetime(pred.dteday) + pd.to_timedelta(pred.hr, unit="h")
    start = pred.time.min().normalize()
    dates = pd.date_range(start, periods=7 * 24, freq="h")
    week = pred[pred.time.isin(dates)]
    fig, ax = plt.subplots(figsize=(10.8, 4.1), layout="constrained")
    for case, key, label, color, line in [
        (winner, "y_true", "真实租赁量", "#28343b", "-"),
        ("boosting_v1", "y_pred", LABELS["boosting_v1"], "#ae784f", "--"),
        (winner, "y_pred", LABELS[winner], TEAL, "-"),
    ]:
        values = week[week.case == case].set_index("time")[key].reindex(dates)
        ax.plot(dates, values, label=label, lw=1.7, color=color, ls=line)
    ticks = dates[::24]
    ax.set(
        xticks=ticks,
        xticklabels=ticks.strftime("%m-%d"),
        ylabel="每小时租赁量",
        title="独立测试期的前七个自然日",
    )
    ax.grid(axis="y", alpha=0.2)
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="outside upper center",
        ncols=3,
        fontsize=9,
        frameon=False,
    )
    finish(fig, out / filename)


def analyze(run, reference):
    from c_stage.experiment import TUNED, grid_rows, new_stage, read_json

    test = pd.read_csv(run / "test/predictions.csv.gz")
    selected = read_json(run / "freeze/selection.json")
    grid = pd.read_csv(run / "search/validation_predictions.csv.gz")
    current = grid_rows(grid, selected["search_choice"]).assign(case=TUNED)
    comparison = pd.read_csv(run / "freeze/candidate_cv.csv").merge(
        pd.read_csv(run / "test/metrics.csv"), on="case", validate="one_to_one"
    )
    comparison["test_minus_cv_rmse"] = comparison.rmse - comparison.rmse_mean
    with new_stage(run, "analysis") as out:
        # 新检出仓库可能缺少 Git 忽略的旧逐条预测，可显式使用交付的相同分组表。
        historical = pd.read_csv(reference) if reference else legacy_groups()
        save_csv(historical, out / "legacy_validation_groups.csv")
        cv_groups = pd.concat([historical, groups(current)]).assign(split="validation")
        test_groups = groups(test).assign(split="test")
        save_csv(pd.concat([cv_groups, test_groups]), out / "group_metrics.csv")
        save_csv(comparison, out / "model_comparison.csv")
        style()
        plot_grid(pd.read_csv(run / "search/summary.csv"), out)
        plot_metrics(comparison, out)
        plot_week(test, selected["selected"], out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--reference", type=Path, help="历史四方案的验证分组表")
    args = parser.parse_args()
    analyze(args.run, args.reference)
