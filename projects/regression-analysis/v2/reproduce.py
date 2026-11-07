"""复现两组 v2 诊断对照。使用 v1 的锁定环境、数据、参数和时间折。"""

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from bike_sharing.data import CATEGORICAL, FEATURES, NUMERIC, TARGET, load_and_split
from bike_sharing.metrics import regression_metrics
from bike_sharing.models import build_model
from bike_sharing.paths import PROJECT_ROOT, SOURCE
from bike_sharing.train import DEFAULT_CONFIG, load_config
from bike_sharing.validation import make_time_folds
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder
from threadpoolctl import threadpool_limits

V2 = Path(__file__).resolve().parent


def hour_workingday(frame):
    # 小时和工作日的每种组合有自己的类别；不从整份数据预先枚举组合。
    return (frame["hr"] * 2 + frame["workingday"]).to_frame("hour_workingday")


def build_cases(config):
    params, seed = config["models"], config["seed"]
    ridge = build_model("ridge", params["ridge"], seed)
    boosting = build_model("gradient_boosting", params["gradient_boosting"], seed)

    interaction = Pipeline(
        [
            ("combine", FunctionTransformer(hour_workingday)),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    ridge_features = ColumnTransformer(
        [
            *ridge.named_steps["preprocess"].transformers,
            ("hour_workingday", interaction, ["hr", "workingday"]),
        ]
    )
    native_features = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), NUMERIC),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "ordinal",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value", unknown_value=np.nan
                            ),
                        ),
                    ]
                ),
                CATEGORICAL,
            ),
        ]
    )
    return {
        "ridge_v1": ridge,
        "ridge_hour_workingday": clone(ridge).set_params(preprocess=ridge_features),
        "boosting_v1": boosting,
        "boosting_native": clone(boosting).set_params(
            preprocess=native_features,
            model__categorical_features=[False] * len(NUMERIC) + [True] * len(CATEGORICAL),
        ),
    }


def main():
    config = load_config(DEFAULT_CONFIG)
    train, held_out = load_and_split()
    folds = make_time_folds(train, config["n_splits"])
    cases = build_cases(config)
    rows, predictions = [], []
    with threadpool_limits(limits=config["threads"]):
        for name, template in cases.items():
            for number, (fit_index, valid_index) in enumerate(folds, 1):
                fit, valid = train.iloc[fit_index], train.iloc[valid_index]
                model = clone(template).fit(fit[FEATURES], fit[TARGET])
                pred = model.predict(valid[FEATURES])
                rows.append(
                    {
                        "case": name,
                        "fold": number,
                        "train_rows": len(fit),
                        "train_end": fit.dteday.max().date(),
                        "valid_rows": len(valid),
                        "valid_start": valid.dteday.min().date(),
                        "valid_end": valid.dteday.max().date(),
                        **regression_metrics(valid[TARGET], pred),
                    }
                )
                predictions.append(
                    pd.DataFrame(
                        {
                            "case": name,
                            "fold": number,
                            "row_id": valid.index,
                            "date": valid.dteday,
                            "hour": valid.hr,
                            "y_true": valid[TARGET],
                            "y_pred": pred,
                        }
                    )
                )
            print(f"Finished {name}", flush=True)

    metrics = pd.DataFrame(rows)
    predicted = pd.concat(predictions, ignore_index=True)
    grouped = metrics.groupby("case", sort=False)[["mae", "rmse", "r2"]]
    summary = grouped.agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    totals = predicted.assign(abs_error=(predicted.y_pred - predicted.y_true).abs())
    sums = totals.groupby("case")[["abs_error", "y_true"]].sum()
    summary["pooled_abs_error_ratio"] = sums.abs_error / sums.y_true

    results = V2 / "results"
    results.mkdir(exist_ok=True)
    metrics.to_csv(results / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(results / "summary_metrics.csv", encoding="utf-8-sig")
    artifacts = V2 / "artifacts"
    artifacts.mkdir(exist_ok=True)
    predicted.to_csv(artifacts / "validation_predictions.csv.gz", index=False)
    metadata = {
        "python": platform.python_version(),
        "versions": {
            p: version(p) for p in ("numpy", "pandas", "scikit-learn", "scipy", "threadpoolctl")
        },
        "sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (SOURCE, DEFAULT_CONFIG, PROJECT_ROOT / "uv.lock", Path(__file__))
        },
        "config": config,
        "development_rows": len(train),
        "held_out_rows": len(held_out),
        "validation_rows_per_case": sum(len(valid) for _, valid in folds),
        "test_evaluated": False,
    }
    (results / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
