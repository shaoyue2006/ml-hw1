"""Protect time-validation, metric semantics and deployable Pipeline behavior."""

import joblib
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from bike_sharing.data import FEATURES, TARGET, load_and_split
from bike_sharing.metrics import regression_metrics
from bike_sharing.models import MODEL_NAMES, build_model
from bike_sharing.train import DEFAULT_CONFIG, load_config, run_experiment
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
    import pandas as pd

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
