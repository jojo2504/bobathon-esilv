"""Data loading for the Parkinson's true-OFF competition.

All CSVs live under ``data/`` at the repo root. Call ``load_data()`` once
and pass the returned ``Dataset`` around — no path strings scattered
across experiment scripts.

Usage
-----
    from parkinson.data import load_data, FEATURE_COLS

    ds = load_data()
    print(ds.X.shape, ds.y.shape, ds.X_test.shape)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Numeric-only feature columns (safe for Ridge / imputers that need no NaN handling).
# Extend to FEATURE_COLS_FULL when using HistGBT or tabular_pipeline.
FEATURE_COLS = [
    "sexM",
    "age_at_diagnosis",
    "age",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
]

# All non-identifier columns — includes categoricals (cohort, gene, rater_id).
# Use with TableVectorizer / tabular_pipeline which handle strings and NaNs.
FEATURE_COLS_FULL = [
    "cohort",
    "sexM",
    "gene",
    "age_at_diagnosis",
    "age",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "rater_id",
    "on",
    "off",
]


@dataclass
class Dataset:
    """Holds every split needed for training and submission.

    Attributes
    ----------
    visits:
        X_train merged with y_train on ``Index`` — the full training table.
    X:
        Feature matrix built from ``visits`` (excludes Index, patient_id, target).
    y:
        Target series (true OFF score).
    groups:
        ``patient_id`` aligned with ``X`` — for GroupKFold.
    X_test:
        Test feature matrix, same columns as ``X``.
    X_test_raw:
        Full X_test CSV (includes Index) — needed to build the submission CSV.
    feature_cols:
        Column names used to slice ``X`` and ``X_test``.
    """

    visits: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series
    X_test: pd.DataFrame
    X_test_raw: pd.DataFrame
    feature_cols: list[str]


def load_data(
    data_dir: str | Path = "data",
    feature_cols: list[str] | None = None,
) -> Dataset:
    """Load all CSVs and return a ready-to-use ``Dataset``.

    Parameters
    ----------
    data_dir:
        Directory that contains ``X_train.csv``, ``y_train.csv``, ``X_test.csv``.
    feature_cols:
        Columns to use as features. Defaults to ``FEATURE_COLS`` (numeric only).
        Pass ``FEATURE_COLS_FULL`` to include categoricals for tree-based models.
    """
    data_dir = Path(data_dir)
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    X_train = pd.read_csv(data_dir / "X_train.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv")
    X_test_raw = pd.read_csv(data_dir / "X_test.csv")

    visits = X_train.merge(y_train, on="Index")
    y = visits["target"]
    groups = visits["patient_id"]

    X = visits[feature_cols]
    X_test = X_test_raw[feature_cols]

    return Dataset(
        visits=visits,
        X=X,
        y=y,
        groups=groups,
        X_test=X_test,
        X_test_raw=X_test_raw,
        feature_cols=feature_cols,
    )
