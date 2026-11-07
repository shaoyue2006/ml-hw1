import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn

from contract import FEATURES, ROOT, TARGET, load_and_split, make_time_folds
from experiment import digest, read_json, restore
from network import RentalMLP, inner_split, predict, tensors

RUN = ROOT / "results"


@pytest.fixture(scope="module")
def split():
    torch.set_num_threads(1)
    return load_and_split()


def test_protocol_and_linear_output():
    frozen = read_json(RUN / "protocol/frozen.json")
    fitted = read_json(RUN / "final_model/fitted.json")
    scored = read_json(RUN / "test/metrics.json")
    assert frozen["utc"] < fitted["utc"] < scored["utc"]
    assert digest(RUN / "protocol/frozen.json") == fitted["protocol_sha256"]
    assert digest(RUN / "final_model/refit.pt") == fitted["refit_weights_sha256"]
    assert digest(RUN / "final_model/refit_transforms.joblib") == fitted["transforms_sha256"]
    assert digest(RUN / "final_model/fitted.json") == scored["final_fit_sha256"]
    assert FEATURES == frozen["features"] and len(FEATURES) == 12
    assert set(FEATURES).isdisjoint({TARGET, "casual", "registered", "dteday", "instant"})
    layers = RentalMLP(60, [64, 32]).layers
    assert [type(x) for x in layers] == [nn.Linear, nn.ReLU, nn.Linear, nn.ReLU, nn.Linear]
    assert [x.out_features for x in layers if isinstance(x, nn.Linear)] == [64, 32, 1]


def assert_scores(part, row):
    d = part.y_pred - part.y_true
    np.testing.assert_allclose(part.residual, d, atol=1e-10)
    expected = [
        d.abs().mean(),
        (d * d).mean(),
        np.sqrt((d * d).mean()),
        1 - (d * d).sum() / ((part.y_true - part.y_true.mean()) ** 2).sum(),
    ]
    np.testing.assert_allclose([row[k] for k in ("mae", "mse", "rmse", "r2")], expected)


def check_fit(folder, frame):
    inner, tail = inner_split(frame, 0.15)
    meta = read_json(folder / "training.json")
    assert inner.dteday.max() < tail.dteday.min() and tail.index.intersection(inner.index).empty
    assert (len(inner), len(tail), len(frame)) == (
        meta["inner_fit_rows"],
        meta["inner_tail_rows"],
        meta["refit_rows"],
    )
    curve = pd.read_csv(folder / "early_stopping.csv")
    epoch = meta["best_epoch"]
    assert epoch == int(curve.loc[curve.inner_mse_scaled.idxmin(), "epoch"])
    assert len(curve) == meta["epochs_observed"] and len(curve) <= 100
    assert len(curve) == min(epoch + 10, 100)
    assert len(pd.read_csv(folder / "refit_curve.csv")) == epoch
    for stage, data in [("inner_best", inner), ("refit", frame)]:
        model, prep = restore(folder, stage)
        feat, target = prep
        np.testing.assert_allclose(target.mean_[0], data[TARGET].mean())
        numeric = feat.named_transformers_["numeric"]
        cols = feat.transformers_[0][2]
        np.testing.assert_allclose(numeric["imputer"].statistics_, data[cols].median())
        np.testing.assert_allclose(numeric["scaler"].mean_, data[cols].mean())
        for col, values in zip(
            feat.transformers_[1][2],
            feat.named_transformers_["categorical"]["onehot"].categories_,
            strict=True,
        ):
            np.testing.assert_array_equal(values, sorted(data[col].unique()))
        if stage == "inner_best":
            x, y = tensors(tail, prep)
            with torch.no_grad():
                mse = nn.functional.mse_loss(model(x), y).item()
            np.testing.assert_allclose(mse, curve.inner_mse_scaled.min(), rtol=2e-6)
    return model, prep


def test_outer_predictions_and_original_units(split):
    dev, held = split
    assert len(dev) == 13915 and len(held) == 3464
    pred = pd.read_csv(RUN / "validation/predictions.csv.gz")
    scores = pd.read_csv(RUN / "validation/fold_metrics.csv").set_index("fold")
    assert len(pred) == 11608 and pred.row_id.is_unique
    for fold, (i, j) in enumerate(make_time_folds(dev), 1):
        train, valid = dev.iloc[i], dev.iloc[j]
        assert train.dteday.max() < valid.dteday.min() <= valid.dteday.max() < held.dteday.min()
        model, prep = check_fit(RUN / f"validation/fold_{fold}", train)
        part = pred[pred.fold == fold]
        np.testing.assert_array_equal(part.row_id, valid.index)
        np.testing.assert_allclose(part.y_true, valid[TARGET])
        np.testing.assert_allclose(part.y_pred, predict(model, prep, valid), rtol=1e-6, atol=1e-5)
        assert_scores(part, scores.loc[fold])
    cols = ["mae", "mse", "rmse", "r2"]
    summary = pd.read_csv(RUN / "validation/summary.csv").set_index("stat")
    np.testing.assert_allclose(summary.loc["mean", cols], scores[cols].mean())
    np.testing.assert_allclose(summary.loc["std", cols], scores[cols].std(ddof=1))


def test_final_checkpoint_and_held_out_errors(split):
    dev, held = split
    model, prep = check_fit(RUN / "final_model", dev)
    pred = pd.read_csv(RUN / "test/predictions.csv.gz")
    np.testing.assert_array_equal(pred.row_id, held.index)
    np.testing.assert_array_equal(pred.dteday, held.dteday.dt.strftime("%Y-%m-%d"))
    np.testing.assert_array_equal(pred.hr, held.hr)
    np.testing.assert_array_equal(pred.y_true, held[TARGET])
    np.testing.assert_allclose(pred.y_pred, predict(model, prep, held), rtol=1e-6, atol=1e-5)
    assert_scores(pred, read_json(RUN / "test/metrics.json"))


def test_comparison_and_demand_groups(split):
    dev, _ = split
    folder = RUN / "analysis"
    archive = ROOT.parent / "v2/c_stage/runs/pre_1109"
    sources = read_json(folder / "comparison_sources.json")
    assert (sources["aligned_validation_rows"], sources["aligned_test_rows"]) == (11608, 3464)
    for path, expected in sources["files"].items():
        assert digest(ROOT.parent / path.replace("\\", "/")) == expected
    cuts = dev[TARGET].quantile([0.25, 0.75]).tolist()
    assert cuts == sources["band_cutoffs"]
    table = pd.read_csv(folder / "model_comparison.csv")
    for name, path, suffix in [
        ("validation", archive / "freeze/candidate_cv.csv", "_mean"),
        ("test", archive / "test/metrics.csv", ""),
    ]:
        baseline = pd.read_csv(path).set_index("case")
        actual = table[table.split == name].set_index("case")
        for m in ["rmse", "mae", "r2"]:
            np.testing.assert_allclose(actual.loc[baseline.index, m], baseline[m + suffix])
    group = pd.read_csv(folder / "demand_bands.csv")
    for name in ["validation", "test"]:
        pred = pd.read_csv(RUN / name / "predictions.csv.gz")
        band = pd.cut(pred.y_true, [-np.inf, *cuts, np.inf], labels=["low", "middle", "high"])
        for label, rows in pred.groupby(band, observed=True):
            entry = group[
                (group.split == name) & (group.band == label) & (group.case == "mlp")
            ].iloc[0]
            assert len(rows) == entry.rows
            assert_scores(rows, entry)
    for image in (
        "early_stopping",
        "actual_vs_predicted",
        "model_comparison",
        "temporal_comparison",
    ):
        with Image.open(folder / (image + ".png")) as im:
            assert im.width >= 800 and im.height >= 600
            im.verify()
