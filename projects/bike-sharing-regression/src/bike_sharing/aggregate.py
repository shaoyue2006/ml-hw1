"""读取已经完成的独立训练结果，生成统一对比表和图，不重新训练模型。"""

import argparse
import json
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from bike_sharing.models import MODEL_NAMES
from bike_sharing.paths import PROJECT_ROOT
from bike_sharing.results import save_csv, save_results


def aggregate_runs(inputs, output):
    """允许输入四个单模型目录，或一个包含四模型的完成目录。"""
    if not inputs:
        raise ValueError("At least one completed run is required.")
    metadata, configs, metrics, predictions = [], [], [], []
    seen = set()
    reference_folds, reference_records = None, None
    # 先检查再创建输出目录，避免把不可比较的结果拼接成貌似有效的报告。
    for directory in map(Path, inputs):
        info = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if info.get("status") != "complete" or info.get("test_evaluated") is not False:
            raise ValueError("Only completed development-validation runs can be aggregated.")
        config_text = (directory / "config.toml").read_text(encoding="utf-8")
        config = tomllib.loads(config_text)
        if metadata:
            for key in (
                "data_sha256",
                "features",
                "target",
                "versions",
                "prediction_policy",
                "selection_metric",
            ):
                if info[key] != metadata[0][key]:
                    raise ValueError(f"Incompatible runs: {key}")
            first_config = tomllib.loads(configs[0])
            if any(config[k] != first_config[k] for k in ("seed", "n_splits")):
                raise ValueError("Incompatible validation configuration.")
        folds = pd.read_csv(directory / "fold_boundaries.csv")
        current_metrics = pd.read_csv(directory / "fold_metrics.csv")
        current_predictions = pd.read_csv(directory / "validation_predictions.csv")
        names = set(current_metrics.model)
        if names != set(info["models"]) or names != set(current_predictions.model):
            raise ValueError("Model names disagree between metadata and result files.")
        if not names.issubset(MODEL_NAMES) or seen.intersection(names):
            raise ValueError("Duplicate or unsupported models; select one run per model.")
        if reference_folds is None:
            reference_folds = folds
        elif not reference_folds.equals(folds):
            raise ValueError("Validation fold boundaries differ.")
        # 不只比日期范围：真实值、行号和折号必须逐条一致，避免错行比较。
        columns = ["fold", "row_id", "dteday", "hr", "workingday", "y_true"]
        for name in names:
            part = current_predictions[current_predictions.model == name]
            records = part[columns].sort_values(["fold", "row_id"]).reset_index(drop=True)
            model_metrics = current_metrics[current_metrics.model == name]
            if (
                records.duplicated(["fold", "row_id"]).any()
                or model_metrics.fold.duplicated().any()
            ):
                raise ValueError("Duplicate validation records or fold metrics.")
            if set(model_metrics.fold) != set(folds.fold):
                raise ValueError("Missing fold metrics.")
            expected = folds.set_index("fold").validation_rows.sort_index()
            actual = part.groupby("fold").size().sort_index()
            if not expected.equals(actual):
                raise ValueError("Incomplete validation predictions.")
            if reference_records is None:
                reference_records = records
            elif not reference_records.equals(records):
                raise ValueError("Validation records differ.")
        seen.update(names)
        metadata.append(info)
        configs.append(config_text)
        metrics.append(current_metrics)
        predictions.append(current_predictions)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save_csv(reference_folds, output / "fold_boundaries.csv")
    summary = save_results(pd.concat(metrics), pd.concat(predictions), output)
    for index, config_text in enumerate(configs, start=1):
        (output / f"input_config_{index}.toml").write_text(config_text, encoding="utf-8")
    (output / "aggregation.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "input_runs": [m["run_id"] for m in metadata],
                "models": list(summary.model),
                "test_evaluated": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or PROJECT_ROOT / "artifacts" / "comparisons" / stamp
    summary = aggregate_runs(args.inputs, output)
    print(summary.to_string(index=False))
    print(f"Comparison: {output}")


if __name__ == "__main__":
    main()
