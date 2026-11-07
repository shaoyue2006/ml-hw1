"""从 v2 目录运行 python -m c_stage.experiment {search,freeze,evaluate} RUN。"""

import argparse
import hashlib
import json
import platform
import shutil
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from bike_sharing.data import CATEGORICAL, FEATURES, NUMERIC, TARGET, load_and_split
from bike_sharing.metrics import regression_metrics
from bike_sharing.paths import PROJECT_ROOT, SOURCE
from bike_sharing.results import save_csv
from bike_sharing.train import DEFAULT_CONFIG, git_revision, load_config
from bike_sharing.validation import make_time_folds
from sklearn.base import clone
from threadpoolctl import threadpool_limits

import reproduce

RATES = (0.03, 0.05, 0.10)
LEAVES = (15, 31)
METRICS = ["train_rmse", "mae", "rmse", "r2"]
TUNED = "boosting_tuned"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, obj):
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


@contextmanager
def new_stage(run, name):
    """仅将成功完成的新阶段发布到目标位置；已有实验保留。"""
    run.mkdir(parents=True, exist_ok=True)
    target = run / name
    if target.exists():
        raise FileExistsError(target)
    with TemporaryDirectory(dir=run) as tmp:
        staging = Path(tmp) / name
        staging.mkdir()
        yield staging
        staging.rename(target)


def inputs():
    paths = [
        SOURCE,
        DEFAULT_CONFIG,
        PROJECT_ROOT / "uv.lock",
        *sorted((PROJECT_ROOT / "src/bike_sharing").rglob("*.py")),
        Path(reproduce.__file__),
        Path(__file__),
        *(
            reproduce.V2 / "results" / name
            for name in ("run_metadata.json", "summary_metrics.csv", "fold_metrics.csv")
        ),
    ]
    root = PROJECT_ROOT.parent
    return {path.relative_to(root).as_posix(): digest(path) for path in paths}


def environment():
    return {
        "python": platform.python_version(),
        **{
            p: version(p)
            for p in (
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "threadpoolctl",
                "joblib",
                "matplotlib",
            )
        },
    }


def with_grid(template, rate, leaves):
    return clone(template).set_params(
        model__learning_rate=rate, model__max_leaf_nodes=leaves, model__max_iter=200
    )


def predictions(frame, pred):
    part = frame[["dteday", "hr", "workingday", TARGET]].rename(columns={TARGET: "y_true"}).copy()
    part.insert(0, "row_id", frame.index)
    part["y_pred"] = pred
    part["residual"] = part.y_pred - part.y_true
    return part


def search(run):
    config = load_config(DEFAULT_CONFIG)
    train, _ = load_and_split()
    folds = make_time_folds(train, config["n_splits"])
    template = reproduce.build_cases(config)["boosting_native"]
    rows, oof = [], []
    with new_stage(run, "search") as out, threadpool_limits(limits=config["threads"]):
        shutil.copyfile(DEFAULT_CONFIG, out / "baseline.toml")
        for rate, leaves in product(RATES, LEAVES):
            for fold, (fit_idx, valid_idx) in enumerate(folds, 1):
                fit, valid = train.iloc[fit_idx], train.iloc[valid_idx]
                model = with_grid(template, rate, leaves)
                model.fit(fit[FEATURES], fit[TARGET])
                pred = model.predict(valid[FEATURES])
                train_rmse = regression_metrics(fit[TARGET], model.predict(fit[FEATURES]))["rmse"]
                key = dict(learning_rate=rate, max_leaf_nodes=leaves, max_iter=200, fold=fold)
                rows.append(
                    {
                        **key,
                        "train_start": fit.dteday.min().date(),
                        "train_end": fit.dteday.max().date(),
                        "train_rows": len(fit),
                        "valid_start": valid.dteday.min().date(),
                        "valid_end": valid.dteday.max().date(),
                        "valid_rows": len(valid),
                        "train_rmse": train_rmse,
                        **regression_metrics(valid[TARGET], pred),
                    }
                )
                oof.append(predictions(valid, pred).assign(**key))
            print(
                f"Completed five folds: learning_rate={rate}, max_leaf_nodes={leaves}", flush=True
            )
        frame = pd.DataFrame(rows)
        summary = frame.groupby(["learning_rate", "max_leaf_nodes", "max_iter"])[METRICS].agg(
            ["mean", "std"]
        )
        summary.columns = ["_".join(col) for col in summary.columns]
        save_csv(frame, out / "fold_metrics.csv")
        save_csv(summary.reset_index(), out / "summary.csv")
        pd.concat(oof).to_csv(out / "validation_predictions.csv.gz", index=False)
        write_json(
            out / "protocol.json",
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "git_revision": git_revision(),
                "source_sha256": inputs(),
                "versions": environment(),
                "config": config,
                "grid": {"learning_rate": RATES, "max_leaf_nodes": LEAVES, "max_iter": [200]},
                "selection": "minimum unrounded arithmetic mean of five validation RMSEs; prefer original on exact tie",
                "train_rows": len(train),
                "test_evaluated": False,
            },
        )


def select_grid(summary, original):
    ranked = summary.assign(
        original=(summary.learning_rate == original["learning_rate"])
        & (summary.max_leaf_nodes == original["max_leaf_nodes"])
    )
    best = ranked.sort_values(
        ["rmse_mean", "original", "learning_rate", "max_leaf_nodes"],
        ascending=[True, False, True, True],
    ).iloc[0]
    return {
        "learning_rate": float(best.learning_rate),
        "max_leaf_nodes": int(best.max_leaf_nodes),
        "max_iter": int(best.max_iter),
    }


def grid_rows(frame, params):
    return frame[
        (frame.learning_rate == params["learning_rate"])
        & (frame.max_leaf_nodes == params["max_leaf_nodes"])
        & (frame.max_iter == params["max_iter"])
    ]


def candidates(config, chosen):
    models = reproduce.build_cases(config)
    models[TUNED] = with_grid(
        models["boosting_native"], chosen["learning_rate"], chosen["max_leaf_nodes"]
    )
    return models


def freeze(run):
    folder = run / "search"
    protocol = read_json(folder / "protocol.json")
    config = protocol["config"]
    if inputs() != protocol["source_sha256"] or environment() != protocol["versions"]:
        raise ValueError("Inputs/environment changed after search")
    original = config["models"]["gradient_boosting"]
    summary = pd.read_csv(folder / "summary.csv")
    chosen = select_grid(summary, original)
    grid_folds = pd.read_csv(folder / "fold_metrics.csv")
    old_folds = pd.read_csv(reproduce.V2 / "results/fold_metrics.csv")
    reference = old_folds[old_folds.case == "boosting_native"].sort_values("fold")
    overlap = grid_rows(grid_folds, original).sort_values("fold")
    for col in ["fold", "train_end", "train_rows", "valid_start", "valid_end", "valid_rows"]:
        np.testing.assert_array_equal(reference[col], overlap[col])
    for metric in ["mae", "rmse", "r2"]:
        np.testing.assert_allclose(reference[metric], overlap[metric], rtol=1e-9)
    old = pd.read_csv(reproduce.V2 / "results/summary_metrics.csv")
    cols = [f"{m}_{stat}" for m in ["mae", "rmse", "r2"] for stat in ["mean", "std"]]
    table = pd.concat(
        [old[["case", *cols]], grid_rows(summary, chosen)[cols].assign(case=TUNED)],
        ignore_index=True,
    )
    original_wins = all(original[k] == v for k, v in chosen.items())
    # 原始方案排在调参方案前，配置相同则保留原始名称。
    winner = table.loc[table.rmse_mean.idxmin(), "case"]
    if original_wins and winner == TUNED:
        winner = "boosting_native"
    models = candidates(config, chosen)
    with new_stage(run, "freeze") as out:
        save_csv(table, out / "candidate_cv.csv")
        write_json(
            out / "selection.json",
            {
                "frozen_utc": datetime.now(timezone.utc).isoformat(),
                "selected": winner,
                "rule": protocol["selection"],
                "search_choice": chosen,
                "retained_original": original_wins,
                "candidate_cv": table.to_dict("records"),
                "base_config": config,
                "features": FEATURES,
                "numeric": NUMERIC,
                "categorical": CATEGORICAL,
                "target": TARGET,
                "prediction_policy": "raw; no clipping, no rounding",
                "preprocessing": {
                    "ridge_v1": "median/standard scale numeric, mode/one-hot categorical; unknown ignored",
                    "ridge_hour_workingday": "ridge_v1 plus one-hot(hr * 2 + workingday); unknown ignored",
                    "boosting_v1": "median numeric, mode/dense one-hot categorical; unknown ignored",
                    "boosting_native": "median numeric; mode + fold-fitted ordinal categories; unknown NaN; native category mask",
                    TUNED: "same preprocessing as boosting_native",
                },
                "estimators": {
                    k: {"type": type(v["model"]).__name__, "params": v["model"].get_params()}
                    for k, v in models.items()
                },
                "source_sha256": inputs(),
                "versions": environment(),
                "search_summary_sha256": digest(folder / "summary.csv"),
                "test_evaluated": False,
            },
        )


def evaluate(run):
    path = run / "freeze/selection.json"
    frozen = read_json(path)
    if (
        inputs() != frozen["source_sha256"]
        or environment() != frozen["versions"]
        or digest(run / "search/summary.csv") != frozen["search_summary_sha256"]
    ):
        raise ValueError("Frozen inputs or dependencies changed")
    models = candidates(frozen["base_config"], frozen["search_choice"])
    for name, model in models.items():
        if model["model"].get_params() != frozen["estimators"][name]["params"]:
            raise ValueError(f"Estimator differs from frozen configuration: {name}")
    with new_stage(run, "test") as out, threadpool_limits(limits=frozen["base_config"]["threads"]):
        train, test = load_and_split()
        if (len(train), len(test)) != (13915, 3464) or train.dteday.max() >= test.dteday.min():
            raise ValueError("Unexpected held-out split")
        rows, all_preds = [], []
        for name, model in models.items():
            model.fit(train[FEATURES], train[TARGET])
            pred = model.predict(test[FEATURES])
            rows.append({"case": name, **regression_metrics(test[TARGET], pred)})
            all_preds.append(predictions(test, pred).assign(case=name))
        save_csv(pd.DataFrame(rows), out / "metrics.csv")
        pd.concat(all_preds).to_csv(out / "predictions.csv.gz", index=False)
        write_json(
            out / "evaluation.json",
            {
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "freeze_sha256": digest(path),
                "selected_before_test": frozen["selected"],
                "train_rows": len(train),
                "test_rows": len(test),
                "train_range": [str(train.dteday.min().date()), str(train.dteday.max().date())],
                "test_range": [str(test.dteday.min().date()), str(test.dteday.max().date())],
                "residual": "y_pred - y_true",
                "refits": list(models),
            },
        )


if __name__ == "__main__":
    actions = {"search": search, "freeze": freeze, "evaluate": evaluate}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=actions)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    actions[args.phase](args.run)
