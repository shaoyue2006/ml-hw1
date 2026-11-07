"""Protect time-validation, metric semantics and deployable Pipeline behavior."""

import json
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from bike_sharing.data import FEATURES, TARGET, load_and_split
from bike_sharing.metrics import regression_metrics
from bike_sharing.models import MODEL_NAMES, build_model
from bike_sharing.paths import PROJECT_ROOT
from bike_sharing.train import DEFAULT_CONFIG, HOLDOUT_MODELS, load_config, main, run_experiment
from bike_sharing.validation import make_time_folds


def test_time_folds_preserve_whole_dates_and_expand():
    train, _ = load_and_split()
    folds = make_time_folds(train)
    seen = set()
    previous_train = set()
    for fit_idx, valid_idx in folds:
        fit, valid = train.iloc[fit_idx], train.iloc[valid_idx]
        assert fit.dteday.max() < valid.dteday.min()
        assert set(fit.dteday).isdisjoint(valid.dteday)
        assert previous_train.issubset(set(fit_idx))
        assert seen.isdisjoint(set(valid_idx))
        for day in valid.dteday.unique():
            assert (valid.dteday == day).sum() == (train.dteday == day).sum()
        seen.update(valid_idx)
        previous_train = set(fit_idx)
    with pytest.raises(ValueError, match="sorted"):
        make_time_folds(train.iloc[::-1])


def test_metrics_keep_negative_predictions_and_original_scale():
    scores = regression_metrics([0, 10], [-2, 12])
    assert scores == {"mae": 2.0, "mse": 4.0, "rmse": 2.0, "r2": 0.84, "negative_predictions": 1}
    with pytest.raises(ValueError, match="Non-finite"):
        regression_metrics([0, 1], [np.nan, 1])


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_model_pipeline_and_serialization(name, tmp_path):
    train, _ = load_and_split()
    fit, valid = train.iloc[:400], train.iloc[400:440].copy()
    config = load_config(DEFAULT_CONFIG)
    params = dict(config["models"][name])
    if name == "gradient_boosting":
        params["max_iter"] = 3
    model = build_model(name, params, seed=42)
    with pytest.raises(NotFittedError):
        model.predict(valid[FEATURES])
    model.fit(fit[FEATURES], fit[TARGET])
    if name in ("ridge", "poisson"):
        scaler = (
            model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["scaler"]
        )
        np.testing.assert_allclose(scaler.mean_, fit[["temp", "atemp", "hum", "windspeed"]].mean())
    valid["weathersit"] = 99  # Never observed during fitting.
    prediction = model.predict(valid[FEATURES])
    assert np.isfinite(prediction).all()
    path = tmp_path / f"{name}.joblib"
    joblib.dump(model, path)
    np.testing.assert_array_equal(joblib.load(path).predict(valid[FEATURES]), prediction)
    with pytest.raises(NotFittedError):
        clone(model).predict(valid[FEATURES])


def test_experiment_uses_shared_validation_rows_and_refuses_overwrite(tmp_path):
    train, _ = load_and_split()
    config = load_config(DEFAULT_CONFIG)
    config["n_splits"] = 2
    output = tmp_path / "run"
    summary = run_experiment(train.iloc[:900], config, ["ridge", "decision_tree"], output)
    predictions = pd.read_csv(output / "validation_predictions.csv")
    ridge = predictions[predictions.model == "ridge"]
    tree = predictions[predictions.model == "decision_tree"]
    np.testing.assert_array_equal(ridge[["row_id", "fold"]], tree[["row_id", "fold"]])
    assert len(summary) == 2
    with pytest.raises(FileExistsError):
        run_experiment(train.iloc[:900], config, ["ridge"], output)


def test_gradient_boosting_rejects_random_early_stopping():
    with pytest.raises(ValueError, match="early stopping"):
        build_model("gradient_boosting", {"early_stopping": True})


def test_holdout_entry_uses_development_only(monkeypatch, tmp_path):
    dev, held = load_and_split()

    def fake_fit(self, x, y):
        pd.testing.assert_frame_equal(x, dev[FEATURES])
        pd.testing.assert_series_equal(y, dev[TARGET])
        return self

    def fake_predict(self, x):
        pd.testing.assert_frame_equal(x, held[FEATURES])
        return np.full(len(x), -0.25)

    monkeypatch.setattr(Pipeline, "fit", fake_fit)
    monkeypatch.setattr(Pipeline, "predict", fake_predict)
    monkeypatch.setattr(sys, "argv", ["bike-train", "--holdout", "--output-root", str(tmp_path)])
    main()
    (run,) = tmp_path.iterdir()
    metadata = json.loads((run / "metadata.json").read_text("utf-8"))
    assert metadata["test_evaluated"] and metadata["models"] == list(HOLDOUT_MODELS)
    assert (run / "config.toml").read_bytes() == DEFAULT_CONFIG.read_bytes()
    pred = pd.read_csv(run / "predictions.csv.gz")
    assert pred.groupby("case").size().to_dict() == dict.fromkeys(HOLDOUT_MODELS, len(held))
    assert (pred.y_pred == -0.25).all()
    np.testing.assert_array_equal(pred.residual, pred.y_pred - pred.y_true)


def test_holdout_published_predictions_match_test_set_and_metrics():
    _, held = load_and_split()
    reports = PROJECT_ROOT / "reports"
    pred = pd.read_csv(reports / "holdout_predictions.csv.gz", float_precision="round_trip")
    metrics = pd.read_csv(reports / "holdout_metrics.csv", float_precision="round_trip")
    assert set(pred.case) == set(metrics.case) == set(HOLDOUT_MODELS)
    keys = held[["dteday", "hr", "workingday", TARGET]].rename(columns={TARGET: "y_true"})
    keys = keys.rename_axis("row_id").reset_index()
    for name, part in pred.groupby("case"):
        part = part.reset_index(drop=True)
        part["dteday"] = pd.to_datetime(part.dteday)
        pd.testing.assert_frame_equal(part[keys.columns], keys)
        np.testing.assert_array_equal(part.residual, part.y_pred - part.y_true)
        scores = regression_metrics(keys.y_true, part.y_pred)
        np.testing.assert_allclose(
            metrics.set_index("case").loc[name, list(scores)], list(scores.values())
        )
