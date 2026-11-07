"""Protect the source data and A's split during the directory migration."""

import hashlib

import pytest

from bike_sharing.data import FEATURES, TARGET, load_and_split
from bike_sharing.paths import SOURCE


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("hour.csv", "e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f"),
        ("Readme.txt", "b92c8628622948bf0828c43a1b315d1517c3d7fd63654730d0dea379b8e88175"),
    ],
)
def test_raw_data_is_unchanged(filename, expected):
    assert hashlib.sha256((SOURCE.parent / filename).read_bytes()).hexdigest() == expected


def test_original_time_split_and_feature_contract():
    train, test = load_and_split()
    assert (len(train), len(test)) == (13915, 3464)
    assert str(train.dteday.min().date()) == "2011-01-01"
    assert str(train.dteday.max().date()) == "2012-08-07"
    assert str(test.dteday.min().date()) == "2012-08-08"
    assert str(test.dteday.max().date()) == "2012-12-31"
    assert set(train.dteday).isdisjoint(test.dteday)
    assert len(FEATURES) == 12
    assert not {TARGET, "casual", "registered", "instant", "dteday"}.intersection(FEATURES)
    assert round(train[TARGET].mean(), 2) == 174.85
