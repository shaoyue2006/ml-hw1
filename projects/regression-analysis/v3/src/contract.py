"""Import the original data contract in place; paths.py locates its original hour.csv."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "v1" / "src"))

from bike_sharing.data import FEATURES, TARGET, load_and_split  # noqa: E402,F401
from bike_sharing.metrics import regression_metrics  # noqa: E402,F401
from bike_sharing.paths import SOURCE  # noqa: E402,F401
from bike_sharing.preprocessing import make_preprocessor  # noqa: E402,F401
from bike_sharing.validation import make_time_folds  # noqa: E402,F401
