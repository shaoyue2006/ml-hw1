"""生成训练集探索图、数据检查结果和时间划分文件。"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from bike_sharing.data import FEATURES, TARGET, load_and_split
from bike_sharing.paths import EDA_DIR, PROCESSED_DIR, SOURCE


def main():
    out = EDA_DIR
    data_dir = PROCESSED_DIR
    out.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(SOURCE, parse_dates=["dteday"])
    train, test = load_and_split()
    for name, frame in [("train", train), ("test", test)]:
        frame.to_csv(data_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    summary = {
        "rows": len(raw),
        "columns": len(raw.columns),
        "missing_values": {k: int(v) for k, v in raw.isna().sum().items()},
        "duplicate_full_rows": int(raw.duplicated().sum()),
        "duplicate_date_hour": int(raw.duplicated(["dteday", "hr"]).sum()),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_start": str(train.dteday.min().date()),
        "train_end": str(train.dteday.max().date()),
        "test_start": str(test.dteday.min().date()),
        "test_end": str(test.dteday.max().date()),
        "train_target_mean": float(train[TARGET].mean()),
        "train_target_median": float(train[TARGET].median()),
        "train_target_min": int(train[TARGET].min()),
        "train_target_max": int(train[TARGET].max()),
        "features": FEATURES,
    }
    (data_dir / "data_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.hist(train.cnt, bins=40, color="#2878b5", edgecolor="white")
    ax.set(
        title="Hourly rental distribution (training data)",
        xlabel="Rentals per hour",
        ylabel="Number of records",
    )
    fig.tight_layout()
    fig.savefig(out / "01_rental_distribution.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    for flag, label in [(1, "Working day"), (0, "Non-working day")]:
        s = train[train.workingday == flag].groupby("hr").cnt.mean()
        ax.plot(s.index, s.values, marker="o", markersize=3, label=label)
    ax.set(title="Average rentals by hour (training data)", xlabel="Hour", ylabel="Average rentals")
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "02_hourly_pattern.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.scatter(train.temp * 41, train.cnt, alpha=0.16, s=8, color="#2878b5")
    ax.set(
        title="Temperature and rentals (training data)",
        xlabel="Temperature (°C)",
        ylabel="Rentals per hour",
    )
    fig.tight_layout()
    fig.savefig(out / "03_temperature.png", dpi=160)
    plt.close(fig)

    cols = ["temp", "atemp", "hum", "windspeed", "cnt"]
    corr = train[cols].corr()
    fig, ax = plt.subplots(figsize=(6.5, 5.4))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=35, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(
                j,
                i,
                f"{corr.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if abs(corr.iloc[i, j]) > 0.6 else "black",
            )
    ax.set_title("Numeric correlations (training data)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out / "04_correlations.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
