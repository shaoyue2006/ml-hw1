"""Build comparisons from recorded predictions without training or changing any model."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from contract import ROOT, TARGET, load_and_split, regression_metrics
from experiment import digest, read_json, stage, write_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import dates as mdates

ARCHIVE = ROOT.parent / "v2/c_stage/runs/pre_1109"
NAMES = {
    "ridge_v1": "Ridge (baseline)",
    "ridge_hour_workingday": "Ridge (combined features)",
    "boosting_v1": "Gradient boosting (baseline)",
    "boosting_native": "Gradient boosting\n(native categorical handling)",
    "boosting_tuned": "Gradient boosting (native categories\n+ optimized hyperparameters)",
    "mlp": "PyTorch MLP",
}
BLUE, TEAL, ORANGE = "#355b85", "#087c70", "#b06537"


def aligned(mlp, tree, keys):
    left = mlp.sort_values("row_id").reset_index(drop=True)
    right = tree.sort_values("row_id").reset_index(drop=True)
    if left.row_id.duplicated().any() or right.row_id.duplicated().any():
        raise ValueError("Repeated row IDs")
    pd.testing.assert_frame_equal(left[keys], right[keys], check_dtype=False)
    return left.assign(tree_pred=right.y_pred)


def curve_figure(run, file):
    fig, axes = plt.subplots(2, 3, figsize=(12.3, 6.4), layout="constrained")
    folders = [(f"Outer fold {i}", run / f"validation/fold_{i}") for i in range(1, 6)]
    folders.append(("Full development window", run / "final_model"))
    for ax, (title, folder) in zip(axes.flat, folders, strict=True):
        curve = pd.read_csv(folder / "early_stopping.csv")
        epoch = read_json(folder / "training.json")["best_epoch"]
        ax.plot(curve.epoch, curve.train_mse_scaled, color=BLUE, label="Inner training")
        ax.plot(curve.epoch, curve.inner_mse_scaled, color=ORANGE, label="Early-stopping segment")
        ax.axvline(epoch, color=TEAL, ls=":", lw=1.5)
        ax.scatter(epoch, curve.inner_mse_scaled.iloc[epoch - 1], s=25, color=TEAL, zorder=3)
        ax.set(title=f"{title}  ·  best epoch {epoch}", xlabel="Epoch", ylabel="Standardized MSE")
    axes.flat[0].legend(fontsize=8)
    fig.savefig(file, dpi=160)
    plt.close(fig)


def prediction_figures(test, out):
    fig, ax = plt.subplots(figsize=(5.6, 5.4), layout="constrained")
    ax.scatter(test.y_true, test.y_pred, s=9, alpha=0.22, color=TEAL, edgecolors="none")
    lo = min(test.y_pred.min(), test.y_true.min()) - 12
    hi = max(test.y_pred.max(), test.y_true.max()) + 12
    ax.plot([lo, hi], [lo, hi], "--", lw=1, color=BLUE, label="Ideal prediction")
    ax.set(
        xlim=(lo, hi),
        ylim=(lo, hi),
        aspect="equal",
        title="MLP predictions · held-out test",
        xlabel="Actual rentals per hour",
        ylabel="Predicted rentals per hour",
    )
    ax.legend()
    fig.savefig(out / "actual_vs_predicted.png", dpi=160)
    plt.close(fig)
    first_day = pd.to_datetime(test.dteday).min()
    hours = pd.date_range(first_day, periods=7 * 24, freq="h")
    series = (
        test.assign(time=pd.to_datetime(test.dteday) + pd.to_timedelta(test.hr, unit="h"))
        .set_index("time")
        .reindex(hours)
    )
    fig, ax = plt.subplots(figsize=(11, 4), layout="constrained")
    for key, label, color, ls in [
        ("y_true", "Actual", "#273138", "-"),
        ("tree_pred", "Tuned boosting", ORANGE, "--"),
        ("y_pred", "MLP", TEAL, "-"),
    ]:
        ax.plot(hours, series[key], label=label, color=color, ls=ls, lw=1.65)
    ax.set(
        title="First seven calendar days of the held-out test",
        ylabel="Rentals per hour",
        xlabel="August 2012",
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend(ncol=3)
    fig.savefig(out / "temporal_comparison.png", dpi=160)
    plt.close(fig)


def comparison_figure(table, path):
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.8), layout="constrained")
    ticks = np.arange(len(NAMES))
    for ax, metric in zip(axes, ["rmse", "mae"], strict=True):
        for split, label, color, shift in [
            ("validation", "5-fold validation mean", BLUE, -0.19),
            ("test", "Held-out test", TEAL, 0.19),
        ]:
            vals = table[table.split == split].set_index("case").loc[list(NAMES), metric]
            bars = ax.barh(ticks + shift, vals, 0.32, label=label, color=color)
            ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
        ax.set(
            yticks=ticks,
            yticklabels=list(NAMES.values()) if ax is axes[0] else [],
            title=metric.upper() + " · lower is better",
            xlabel="Rentals per hour",
        )
        ax.invert_yaxis()
        ax.margins(x=0.14)
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="outside upper center", ncols=2, frameon=False
    )
    fig.savefig(path, dpi=220)
    plt.close(fig)


def analyze(run):
    frozen = read_json(ARCHIVE / "freeze/selection.json")
    dev, _ = load_and_split()
    cfg = read_json(run / "protocol/frozen.json")["config"]
    q1, q3 = dev[TARGET].quantile(cfg["demand_quantiles"]).values
    grid = pd.read_csv(ARCHIVE / "search/validation_predictions.csv.gz")
    chosen = frozen["search_choice"]
    selected = grid[
        (grid.learning_rate == chosen["learning_rate"])
        & (grid.max_leaf_nodes == chosen["max_leaf_nodes"])
        & (grid.max_iter == chosen["max_iter"])
    ]
    base_test = pd.read_csv(ARCHIVE / "test/predictions.csv.gz")
    keys = ["row_id", "dteday", "hr", "workingday", "y_true"]
    cv = aligned(pd.read_csv(run / "validation/predictions.csv.gz"), selected, keys + ["fold"])
    test = aligned(
        pd.read_csv(run / "test/predictions.csv.gz"),
        base_test[base_test.case == frozen["selected"]],
        keys,
    )
    old_cv = pd.read_csv(ARCHIVE / "freeze/candidate_cv.csv").rename(
        columns={k + "_mean": k for k in ["mae", "rmse", "r2"]}
    )
    old_test = pd.read_csv(ARCHIVE / "test/metrics.csv")
    mlp_cv = pd.read_csv(run / "validation/summary.csv").set_index("stat").loc["mean"]
    mlp_test = read_json(run / "test/metrics.json")
    measures = ["mae", "rmse", "r2"]
    blocks = []
    for name, data, mlp in [("validation", old_cv, mlp_cv), ("test", old_test, mlp_test)]:
        blocks.append(
            pd.concat(
                [
                    data[["case", *measures]],
                    pd.DataFrame([{"case": "mlp", **{k: mlp[k] for k in measures}}]),
                ]
            ).assign(split=name)
        )
    table = pd.concat(blocks, ignore_index=True)
    demand = []
    for split, frame in [("validation", cv), ("test", test)]:
        for band, part in frame.groupby(
            pd.cut(frame.y_true, [-np.inf, q1, q3, np.inf], labels=["low", "middle", "high"]),
            observed=True,
        ):
            for case, key in [("mlp", "y_pred"), (frozen["selected"], "tree_pred")]:
                demand.append(
                    {
                        "split": split,
                        "band": band,
                        "case": case,
                        "rows": len(part),
                        "mean_signed_error": (part[key] - part.y_true).mean(),
                        **regression_metrics(part.y_true, part[key]),
                    }
                )
    with stage(run / "analysis") as out:
        table.to_csv(out / "model_comparison.csv", index=False)
        pd.DataFrame(demand).to_csv(out / "demand_bands.csv", index=False)
        write_json(
            out / "comparison_sources.json",
            {
                "files": {
                    str(path.relative_to(ROOT.parent)): digest(path)
                    for path in [
                        ARCHIVE / "freeze/selection.json",
                        ARCHIVE / "freeze/candidate_cv.csv",
                        ARCHIVE / "test/metrics.csv",
                        ARCHIVE / "test/predictions.csv.gz",
                        ARCHIVE / "search/validation_predictions.csv.gz",
                    ]
                },
                "aligned_validation_rows": len(cv),
                "aligned_test_rows": len(test),
                "band_cutoffs": [q1, q3],
                "band_definition": "development target quartiles; low <= q25, middle <= q75, high > q75",
            },
        )
        plt.rcParams.update(
            {
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.edgecolor": "#a5b2b8",
                "font.size": 10,
                "axes.grid": True,
                "grid.alpha": 0.15,
                "axes.axisbelow": True,
            }
        )
        curve_figure(run, out / "early_stopping.png")
        prediction_figures(test, out)
        comparison_figure(table, out / "model_comparison.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    analyze(parser.parse_args().output)
