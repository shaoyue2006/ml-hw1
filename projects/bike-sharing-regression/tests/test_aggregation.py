"""交接时不能把不同数据、不同验证记录或重复模型的结果混在一起。"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bike_sharing.aggregate import aggregate_runs
from bike_sharing.data import load_and_split
from bike_sharing.train import DEFAULT_CONFIG, load_config, run_experiment


@pytest.fixture(scope="module")
def independent_runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("independent-runs")
    train, _ = load_and_split()
    config_text = DEFAULT_CONFIG.read_text(encoding="utf-8").replace("n_splits = 5", "n_splits = 2")
    config_path = root / "config.toml"
    config_path.write_text(config_text, encoding="utf-8")
    config = load_config(config_path)
    runs = []
    for name in ("ridge", "decision_tree"):
        output = root / name
        run_experiment(train.iloc[:900], config, [name], output)
        (output / "config.toml").write_text(config_text, encoding="utf-8")
        info = {
            "status": "complete",
            "test_evaluated": False,
            "models": [name],
            "run_id": name,
            "data_sha256": "same-data",
            "features": ["same-features"],
            "target": "cnt",
            "versions": {},
            "prediction_policy": "raw",
            "selection_metric": "mean_fold_rmse",
        }
        (output / "metadata.json").write_text(json.dumps(info), encoding="utf-8")
        runs.append(output)
    return runs


def test_aggregate_preserves_single_model_metrics(independent_runs, tmp_path):
    summary = aggregate_runs(independent_runs, tmp_path / "comparison")
    expected = pd.concat([pd.read_csv(p / "summary_metrics.csv") for p in independent_runs])
    np.testing.assert_allclose(summary.rmse_mean, expected.rmse_mean)
    assert set(summary.model) == {"ridge", "decision_tree"}
    assert (tmp_path / "comparison" / "figures" / "metrics.png").exists()
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_runs([independent_runs[0], independent_runs[0]], tmp_path / "duplicate")


@pytest.mark.parametrize("tamper", ["data", "record", "fold"])
def test_aggregate_rejects_incompatible_results(independent_runs, tmp_path, tamper):
    modified = Path(shutil.copytree(independent_runs[1], tmp_path / "modified"))
    if tamper == "data":
        path = modified / "metadata.json"
        info = json.loads(path.read_text())
        info["data_sha256"] = "different-data"
        path.write_text(json.dumps(info))
    elif tamper == "record":
        path = modified / "validation_predictions.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "y_true"] += 1
        frame.to_csv(path, index=False)
    else:
        path = modified / "fold_boundaries.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "validation_start"] = "2099-01-01"
        frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        aggregate_runs([independent_runs[0], modified], tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
