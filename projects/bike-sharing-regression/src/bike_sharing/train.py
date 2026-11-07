"""公共训练流程：读取配置、时间验证、保存模型；不评价独立测试集。"""

import argparse
import hashlib
import json
import platform
import subprocess
import tomllib
import warnings
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from bike_sharing.data import FEATURES, TARGET, load_and_split
from bike_sharing.metrics import regression_metrics
from bike_sharing.models import MODEL_NAMES, build_model
from bike_sharing.paths import PROJECT_ROOT, SOURCE
from bike_sharing.results import save_csv, save_results
from bike_sharing.validation import make_time_folds

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "baseline.toml"


def load_config(path):
    """先检查配置与算法名称，避免训练到一半才发现参数或配置错误。"""
    config = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    if set(config) != {"seed", "n_splits", "threads", "models"}:
        raise ValueError("Config must specify seed, n_splits, threads and models.")
    for key in ("seed", "n_splits", "threads"):
        if type(config[key]) is not int:
            raise ValueError(f"{key} must be an integer.")
    if config["n_splits"] < 2 or config["threads"] < 1 or config["seed"] < 0:
        raise ValueError("Invalid split count, thread limit or seed.")
    if set(config["models"]) != set(MODEL_NAMES):
        raise ValueError(f"Expected model sections: {MODEL_NAMES}")
    for name, params in config["models"].items():
        build_model(name, params, config["seed"])
    return config


def run_experiment(train, config, names, output):
    """只接收开发训练集；刻意不提供 test 参数，避免误用测试集调参。"""
    names = list(names)
    if not names or len(set(names)) != len(names) or not set(names).issubset(MODEL_NAMES):
        raise ValueError("Select distinct supported model names.")
    # 所有算法复用同一组整日时间折，不随机打乱，也不读取独立测试集。
    folds = make_time_folds(train, config["n_splits"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    rows, predictions, boundaries = [], [], []
    for fold, (fit_idx, valid_idx) in enumerate(folds, start=1):
        fit, valid = train.iloc[fit_idx], train.iloc[valid_idx]
        boundaries.append(
            {
                "fold": fold,
                "train_rows": len(fit),
                "validation_rows": len(valid),
                "train_start": str(fit.dteday.min().date()),
                "train_end": str(fit.dteday.max().date()),
                "validation_start": str(valid.dteday.min().date()),
                "validation_end": str(valid.dteday.max().date()),
            }
        )
    save_csv(pd.DataFrame(boundaries), output / "fold_boundaries.csv")
    with threadpool_limits(limits=config["threads"]), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        for name in names:
            base = build_model(name, config["models"][name], config["seed"])
            for fold, (fit_idx, valid_idx) in enumerate(folds, start=1):
                fit, valid = train.iloc[fit_idx], train.iloc[valid_idx]
                # 每折克隆未拟合的完整流水线，避免编码器、标准化参数跨折泄漏。
                model = clone(base)
                start = perf_counter()
                # fit 同时学习本折的预处理参数和模型参数。
                model.fit(fit[FEATURES], fit[TARGET])
                fit_seconds = perf_counter() - start
                pred = model.predict(valid[FEATURES])
                # 统一用原始租赁量尺度计算指标，不截断负数、不取整。
                scores = regression_metrics(valid[TARGET], pred)
                train_scores = regression_metrics(fit[TARGET], model.predict(fit[FEATURES]))
                rows.append(
                    {
                        "model": name,
                        "fold": fold,
                        "fit_seconds": fit_seconds,
                        **scores,
                        **{f"train_{k}": v for k, v in train_scores.items()},
                    }
                )
                part = valid[["dteday", "hr", "workingday"]].copy()
                part.insert(0, "row_id", valid.index)
                part["model"], part["fold"] = name, fold
                part["y_true"], part["y_pred"] = valid[TARGET].to_numpy(), pred
                part["residual"] = part.y_pred - part.y_true
                predictions.append(part)
                print(
                    f"{name} fold {fold}: RMSE={scores['rmse']:.3f}, MAE={scores['mae']:.3f}",
                    flush=True,
                )
            # 验证结束后在全开发集重拟合，保存给 C/D；此模型尚未做独立测试。
            final = clone(base).fit(train[FEATURES], train[TARGET])
            joblib.dump(final, output / "models" / f"{name}.joblib")
    fold_metrics = pd.DataFrame(rows)
    predictions = pd.concat(predictions, ignore_index=True)
    # 统计与可视化交给独立结果模块；这里仅提供逐折指标和验证预测。
    return save_results(fold_metrics, predictions, output)


def git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_dirty():
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def main(model_name=None):
    parser = argparse.ArgumentParser(description=__doc__)
    # 各算法文件传入固定名称；通用入口则允许 --model 选择单个或全部。
    if model_name is None:
        parser.add_argument("--model", choices=["all", *MODEL_NAMES], default="all")
    else:
        parser.set_defaults(model=model_name)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "artifacts" / "runs")
    args = parser.parse_args()
    config = load_config(args.config)
    names = list(MODEL_NAMES) if args.model == "all" else [args.model]
    # held_out 只记录样本量，绝不传入训练/验证流程，留待 C 配置冻结后评价。
    train, held_out = load_and_split()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + args.model
    output = args.output_root / run_id
    summary = run_experiment(train, config, names, output)
    (output / "config.toml").write_bytes(args.config.read_bytes())
    # 递归记录源码哈希，算法拆入子包后也必须包含在复现信息里。
    sources = sorted((PROJECT_ROOT / "src" / "bike_sharing").rglob("*.py"))
    metadata = {
        "run_id": run_id,
        "status": "complete",
        "git_revision": git_revision(),
        "git_dirty": git_dirty(),
        "lock_sha256": hashlib.sha256((PROJECT_ROOT / "uv.lock").read_bytes()).hexdigest(),
        "source_sha256": {
            str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in sources
        },
        "data_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "versions": {
            p: version(p)
            for p in ("numpy", "pandas", "scikit-learn", "joblib", "matplotlib", "threadpoolctl")
        },
        "models": names,
        "features": FEATURES,
        "target": TARGET,
        "train_rows": len(train),
        "held_out_rows": len(held_out),
        "train_start": str(train.dteday.min().date()),
        "train_end": str(train.dteday.max().date()),
        "test_evaluated": False,
        "selection_metric": "mean_fold_rmse",
        "prediction_policy": "raw, no clipping or rounding",
        "saved_models": "refitted on all development rows; not evaluated on held-out test",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
