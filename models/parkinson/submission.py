"""Submission CSV generation.

Fits the chosen model on all training data, predicts on X_test, and writes
``submissions/<key>.csv`` ready to upload to Kaggle.

Usage
-----
    from parkinson.data import load_data
    from parkinson.submission import make_submission

    ds = load_data()
    make_submission("02_ridge", fitted_model, ds)
    # writes submissions/02_ridge.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import clone


def make_submission(
    key: str,
    model: Any,
    ds: Any,
    *,
    submissions_dir: str | Path = "submissions",
    fit: bool = True,
) -> Path:
    """Fit ``model`` on all training data and write the Kaggle submission CSV.

    Parameters
    ----------
    key:
        Report key (e.g. ``"02_ridge"``). The CSV is written to
        ``submissions/<key>.csv``.
    model:
        Sklearn-compatible estimator or pipeline. Cloned and re-fitted on
        ``ds.X`` / ``ds.y`` unless ``fit=False``.
    ds:
        ``Dataset`` from ``load_data()``.
    submissions_dir:
        Directory to write the CSV into (created if it doesn't exist).
    fit:
        When ``True`` (default), clone ``model`` and fit on ``ds.X``, ``ds.y``.
        Pass ``False`` and a pre-fitted model to skip re-fitting.

    Returns
    -------
    Path
        Path to the written CSV file.
    """
    submissions_dir = Path(submissions_dir)
    submissions_dir.mkdir(exist_ok=True)

    if fit:
        final = clone(model).fit(ds.X, ds.y)
    else:
        final = model

    predictions = final.predict(ds.X_test)

    submission = pd.DataFrame(
        {"Index": ds.X_test_raw["Index"], "target": predictions}
    )

    out_path = submissions_dir / f"{key}.csv"
    submission.to_csv(out_path, index=False)
    print(f"Wrote {out_path}  ({len(submission)} rows)")
    return out_path
