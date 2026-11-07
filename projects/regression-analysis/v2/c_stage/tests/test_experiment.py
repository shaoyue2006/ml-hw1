"""独立复算交付数据，验证时间、编码、冻结边界与阶段产物。"""

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from bike_sharing.data import CATEGORICAL, FEATURES, NUMERIC, TARGET, load_and_split
from bike_sharing.train import DEFAULT_CONFIG, load_config
from bike_sharing.validation import make_time_folds

import reproduce
from c_stage.experiment import LEAVES, RATES, digest, new_stage, read_json, select_grid, with_grid

RUN = Path(__file__).resolve().parents[1] / "runs/pre_1109"


def check_metrics(row, frame):
    diff = frame.y_pred - frame.y_true
    mse = (diff**2).mean()
    expected = [
        diff.abs().mean(),
        mse,
        np.sqrt(mse),
        1 - (diff**2).sum() / ((frame.y_true - frame.y_true.mean()) ** 2).sum(),
    ]
    np.testing.assert_allclose([row[k] for k in ("mae", "mse", "rmse", "r2")], expected)
    np.testing.assert_allclose(frame.residual, diff, atol=1e-12)


def test_native_preprocessor_only_learns_from_training_days():
    dev, _ = load_and_split()
    fit_idx, valid_idx = make_time_folds(dev)[0]
    train, valid = dev.iloc[fit_idx], dev.iloc[valid_idx]
    config = load_config(DEFAULT_CONFIG)
    template = reproduce.build_cases(config)["boosting_native"]
    pipeline = with_grid(template, 0.10, 15)
    encoder = pipeline["preprocess"].fit(train[FEATURES])
    assert set(FEATURES).isdisjoint({"instant", "casual", "registered", "cnt", "dteday"})
    np.testing.assert_allclose(
        encoder.named_transformers_["numeric"].statistics_, train[NUMERIC].median()
    )
    categories = encoder.named_transformers_["categorical"]["ordinal"].categories_
    for col, learned in zip(CATEGORICAL, categories, strict=True):
        np.testing.assert_array_equal(learned, sorted(train[col].unique()))
    unseen_month = valid[~valid.mnth.isin(train.mnth)]
    assert len(unseen_month) > 0
    assert np.isnan(encoder.transform(unseen_month[FEATURES])[:, FEATURES.index("mnth")]).all()
    assert pipeline["model"].categorical_features == [False] * 4 + [True] * 8
    assert not hasattr(template["preprocess"], "transformers_")


def test_grid_predictions_summary_and_date_boundaries():
    dev, held = load_and_split()
    assert (len(dev), len(held)) == (13915, 3464)
    folds = make_time_folds(dev)
    metrics = pd.read_csv(RUN / "search/fold_metrics.csv")
    summary = pd.read_csv(RUN / "search/summary.csv")
    pred = pd.read_csv(RUN / "search/validation_predictions.csv.gz")
    assert len(metrics) == 30 and len(summary) == 6 and len(pred) == 6 * 11608
    assert set(metrics.max_iter) == {200} and set(
        zip(summary.learning_rate, summary.max_leaf_nodes)
    ) == set(product(RATES, LEAVES))
    for _, row in summary.iterrows():
        part = metrics[
            (metrics.learning_rate == row.learning_rate)
            & (metrics.max_leaf_nodes == row.max_leaf_nodes)
        ]
        assert sorted(part.fold) == list(range(1, 6))
        for key in ("train_rmse", "rmse", "mae", "r2"):
            np.testing.assert_allclose(
                [part[key].mean(), part[key].std(ddof=1)], [row[key + "_mean"], row[key + "_std"]]
            )
        for _, rec in part.iterrows():
            fit, valid = folds[int(rec.fold) - 1]
            frame = pred[
                (pred.learning_rate == row.learning_rate)
                & (pred.max_leaf_nodes == row.max_leaf_nodes)
                & (pred.fold == rec.fold)
            ]
            np.testing.assert_array_equal(frame.row_id, dev.iloc[valid].index)
            np.testing.assert_array_equal(frame.y_true, dev.iloc[valid][TARGET])
            assert rec.train_rows == len(fit) and rec.valid_rows == len(valid)
            assert dev.iloc[fit].dteday.max() < dev.iloc[valid].dteday.min() < held.dteday.min()
            check_metrics(rec, frame)


def test_freeze_and_independent_test_metrics():
    frozen = read_json(RUN / "freeze/selection.json")
    audit = read_json(RUN / "test/evaluation.json")
    table = pd.read_csv(RUN / "freeze/candidate_cv.csv")
    assert (
        frozen["selected"]
        == table.loc[table.rmse_mean.idxmin(), "case"]
        == audit["selected_before_test"]
    )
    assert not frozen["test_evaluated"] and frozen["frozen_utc"] < audit["completed_utc"]
    assert digest(RUN / "freeze/selection.json") == audit["freeze_sha256"]
    test = pd.read_csv(RUN / "test/predictions.csv.gz")
    scores = pd.read_csv(RUN / "test/metrics.csv")
    assert (
        len(test) == 5 * 3464
        and len(scores) == 5
        and set(scores.case) == set(table.case) == set(test.case)
    )
    _, held = load_and_split()
    for _, row in scores.iterrows():
        part = test[test.case == row.case]
        np.testing.assert_array_equal(part.row_id, held.index)
        np.testing.assert_array_equal(part.y_true, held[TARGET])
        np.testing.assert_array_equal(part.dteday, held.dteday.dt.strftime("%Y-%m-%d"))
        np.testing.assert_array_equal(part.hr, held.hr)
        check_metrics(row, part)


def test_original_config_wins_exact_tie():
    data = pd.DataFrame(
        {
            "learning_rate": [0.03, 0.05, 0.1],
            "max_leaf_nodes": [15, 31, 15],
            "max_iter": [200] * 3,
            "rmse_mean": [5.0, 5.0, 6.0],
        }
    )
    params = {"learning_rate": 0.05, "max_leaf_nodes": 31, "max_iter": 200}
    assert select_grid(data, params) == params


def test_completed_outputs_protected_and_failed_stage_retryable(tmp_path):
    with pytest.raises(ValueError), new_stage(tmp_path, "trial") as output:
        (output / "partial").touch()
        raise ValueError("incomplete")
    assert not (tmp_path / "trial").exists()
    with new_stage(tmp_path, "trial") as output:
        (output / "complete").touch()
    with pytest.raises(FileExistsError), new_stage(tmp_path, "trial"):
        pass
    assert (tmp_path / "trial/complete").exists()


def test_group_statistics_match_predictions():
    grouped = pd.read_csv(RUN / "analysis/group_metrics.csv")
    assert len(grouped) == 2 * 5 * (24 + 2)
    for (_, split_group), part in grouped.groupby(["case", "split"]):
        expected = 11608 if split_group == "validation" else 3464
        assert part.groupby("group").rows.sum().to_dict() == {
            "hr": expected,
            "workingday": expected,
        }
    pred = pd.read_csv(RUN / "test/predictions.csv.gz")
    for col in ("hr", "workingday"):
        for (name, value), frame in pred.groupby(["case", col]):
            rec = grouped[
                (grouped.split == "test")
                & (grouped.case == name)
                & (grouped.group == col)
                & (grouped.value == value)
            ].iloc[0]
            check_metrics(rec, frame)
