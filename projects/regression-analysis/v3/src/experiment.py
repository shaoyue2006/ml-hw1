"""Run the predeclared validation, full-development training and held-out scoring."""

import argparse
import hashlib
import json
import platform
import shutil
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import pandas as pd
import tomli
import torch

from contract import (
    FEATURES,
    ROOT,
    SOURCE,
    TARGET,
    load_and_split,
    make_time_folds,
    regression_metrics,
)
from network import RentalMLP, fit_network, fit_transforms, inner_split, predict, tensors

CONFIG = ROOT / "configs/mlp.toml"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


@contextmanager
def stage(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Existing stage: {path}")
    with TemporaryDirectory(dir=path.parent) as tmp:
        out = Path(tmp) / path.name
        out.mkdir()
        yield out
        out.rename(path)


def freeze(run):
    with stage(run / "protocol") as out:
        shutil.copyfile(CONFIG, out / "mlp.toml")
        cfg = tomli.loads(CONFIG.read_text(encoding="utf-8"))
        write_json(
            out / "frozen.json",
            {
                "utc": datetime.now(timezone.utc).isoformat(),
                "config": cfg,
                "features": FEATURES,
                "target": TARGET,
                "data_sha256": digest(SOURCE),
                "design": "Dense train-only imputation, numeric standardization and one-hot; train-only target z-score. ReLU hidden layers; linear output; MSELoss; Adam defaults. No postprocessing.",
                "stopping": "Last ceil(15% of unique training days), strictly lower standardized MSE, first minimum, patience 10. Restart seed 42, fit transforms and network on full window for selected epochs.",
                "evaluation": "Follow-up after v2 test publication; fixed MLP, no model selection across versions.",
                "environment": {
                    p: version(p)
                    for p in (
                        "torch",
                        "scikit-learn",
                        "numpy",
                        "scipy",
                        "pandas",
                        "matplotlib",
                        "joblib",
                        "tomli",
                        "threadpoolctl",
                    )
                },
                "python": sys.version,
                "executable": sys.executable,
                "platform": platform.platform(),
                "device": cfg["device"],
                "cuda_available": torch.cuda.is_available(),
                "cuda_build": torch.version.cuda,
                "cuda_gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "source_sha256": {
                    p.name: digest(p)
                    for p in [
                        ROOT / "src/network.py",
                        ROOT / "src/experiment.py",
                        ROOT / "src/contract.py",
                        *(
                            ROOT.parent / "v1/src/bike_sharing" / name
                            for name in [
                                "data.py",
                                "preprocessing.py",
                                "validation.py",
                                "metrics.py",
                                "paths.py",
                            ]
                        ),
                    ]
                },
            },
        )


def save_model(out, name, model, transforms, cfg, epoch):
    features, _ = transforms
    torch.save(
        {
            "input_dim": len(features.get_feature_names_out()),
            "hidden": cfg["hidden"],
            "epoch": epoch,
            "state_dict": model.state_dict(),
        },
        out / f"{name}.pt",
    )
    joblib.dump(transforms, out / f"{name}_transforms.joblib")


def restore(out, name="refit"):
    state = torch.load(out / f"{name}.pt", map_location="cpu", weights_only=True)
    model = RentalMLP(state["input_dim"], state["hidden"])
    model.load_state_dict(state["state_dict"])
    return model.eval(), joblib.load(out / f"{name}_transforms.joblib")


def train_window(frame, cfg, out):
    out.mkdir(exist_ok=True)
    inner, tail = inner_split(frame, cfg["inner_tail_day_fraction"])
    prep = fit_transforms(inner)
    scout, history, epoch = fit_network(
        tensors(inner, prep), cfg, cfg["max_epochs"], tensors(tail, prep)
    )
    save_model(out, "inner_best", scout, prep, cfg, epoch)
    pd.DataFrame(history).to_csv(out / "early_stopping.csv", index=False)
    full_prep = fit_transforms(frame)
    model, refit_history, _ = fit_network(tensors(frame, full_prep), cfg, epoch)
    save_model(out, "refit", model, full_prep, cfg, epoch)
    pd.DataFrame(refit_history).to_csv(out / "refit_curve.csv", index=False)
    record = {
        "inner_fit_rows": len(inner),
        "inner_tail_rows": len(tail),
        "fit_start": str(inner.dteday.min().date()),
        "inner_fit_end": str(inner.dteday.max().date()),
        "inner_tail_start": str(tail.dteday.min().date()),
        "refit_end": str(frame.dteday.max().date()),
        "refit_rows": len(frame),
        "best_epoch": epoch,
        "epochs_observed": len(history),
        "stop_reason": "patience" if len(history) - epoch >= cfg["patience"] else "epoch_limit",
    }
    write_json(out / "training.json", record)
    return model, full_prep, record


def score(frame, pred):
    part = frame[["dteday", "hr", "workingday", TARGET]].rename(columns={TARGET: "y_true"}).copy()
    part.insert(0, "row_id", frame.index)
    part["y_pred"] = pred
    part["residual"] = pred - part.y_true
    return regression_metrics(part.y_true, pred), part


def run_experiment(run):
    freeze(run)
    frozen = read_json(run / "protocol/frozen.json")
    cfg = frozen["config"]
    dev, held = load_and_split()
    if (len(dev), len(held)) != (13915, 3464) or dev.dteday.max() >= held.dteday.min():
        raise ValueError("Unexpected temporal split")
    with stage(run / "validation") as out:
        rows, predictions = [], []
        for fold, (i, j) in enumerate(make_time_folds(dev, cfg["n_splits"]), 1):
            fit, valid = dev.iloc[i], dev.iloc[j]
            model, prep, record = train_window(fit, cfg, out / f"fold_{fold}")
            metrics, part = score(valid, predict(model, prep, valid))
            rows.append(
                {
                    "fold": fold,
                    **record,
                    "valid_rows": len(valid),
                    "valid_start": str(valid.dteday.min().date()),
                    "valid_end": str(valid.dteday.max().date()),
                    **metrics,
                }
            )
            predictions.append(part.assign(fold=fold))
            print(
                f"Fold {fold}: epoch {record['best_epoch']}, RMSE {metrics['rmse']:.3f}", flush=True
            )
        result = pd.DataFrame(rows)
        result.to_csv(out / "fold_metrics.csv", index=False)
        result[["mae", "mse", "rmse", "r2"]].agg(["mean", "std"]).rename_axis("stat").to_csv(
            out / "summary.csv"
        )
        pd.concat(predictions).to_csv(out / "predictions.csv.gz", index=False)
    with stage(run / "final_model") as out:
        train_window(dev, cfg, out)
        write_json(
            out / "fitted.json",
            {
                "utc": datetime.now(timezone.utc).isoformat(),
                "protocol_sha256": digest(run / "protocol/frozen.json"),
                "refit_weights_sha256": digest(out / "refit.pt"),
                "transforms_sha256": digest(out / "refit_transforms.joblib"),
                "test_scored": False,
            },
        )
    with stage(run / "test") as out:
        model, prep = restore(run / "final_model")
        metrics, part = score(held, predict(model, prep, held))
        part.to_csv(out / "predictions.csv.gz", index=False)
        write_json(
            out / "metrics.json",
            {
                "utc": datetime.now(timezone.utc).isoformat(),
                "final_fit_sha256": digest(run / "final_model/fitted.json"),
                "rows": len(held),
                "start": str(held.dteday.min().date()),
                "end": str(held.dteday.max().date()),
                **metrics,
            },
        )
        print(f"Test RMSE {metrics['rmse']:.3f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    run_experiment(parser.parse_args().output)
