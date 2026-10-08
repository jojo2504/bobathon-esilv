"""Experiment runner — evaluation, RMSE reporting, and Hub push.

The ``Experiment`` class is the single entry point for running a model:
evaluate it with patient-grouped CV, print the RMSE, and push the report
to Skore Hub in one call.

Usage
-----
    from sklearn.dummy import DummyRegressor
    from parkinson.data import load_data
    from parkinson.evaluate import Experiment

    ds = load_data()
    exp = Experiment("01_dummy", DummyRegressor(strategy="mean"))
    report = exp.run(ds)          # evaluate + print RMSE
    exp.push(report)              # push to Hub, prints URL
    # or combined:
    report = exp.run_and_push(ds)

Comparing two models
--------------------
    from sklearn.linear_model import Ridge
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from parkinson.evaluate import compare

    ridge = make_pipeline(SimpleImputer(strategy="median"), Ridge())
    report = compare(
        {"01_dummy": DummyRegressor(), "02_ridge": ridge},
        ds,
        push_key="02_ridge",   # which key to put on Hub
    )
"""

from __future__ import annotations

from typing import Any

from sklearn.model_selection import GroupKFold
from skore import evaluate as skore_evaluate

from src.parkinson.data import Dataset
from src.parkinson.hub import get_project


def _extract_rmse(report: Any, **kwargs: Any) -> float:
    """Extract the mean RMSE scalar from a skore metrics DataFrame."""
    result = report.metrics.rmse(**kwargs)
    # CrossValidationReport returns a DataFrame; EstimatorReport may return a scalar.
    if hasattr(result, "iloc"):
        return float(result.iloc[0, 0])
    return float(result)


def _default_cv_splits(ds: Dataset, n_splits: int = 5) -> list:
    """Precompute patient-grouped CV index pairs from ``ds``."""
    return list(GroupKFold(n_splits=n_splits).split(ds.X, ds.y, groups=ds.groups))


class Experiment:
    """Evaluate one sklearn estimator, report RMSE, push to Skore Hub.

    Parameters
    ----------
    key:
        Report key used for ``project.put(key, report)`` and the submission
        CSV filename (``submissions/<key>.csv``).
    model:
        Any sklearn-compatible estimator or pipeline.
    n_splits:
        Number of GroupKFold folds. Defaults to 5.
    """

    def __init__(self, key: str, model: Any, n_splits: int = 5) -> None:
        self.key = key
        self.model = model
        self.n_splits = n_splits
        self._project = None

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def run(self, ds: Dataset) -> Any:
        """Run grouped cross-validation and print RMSE.

        Parameters
        ----------
        ds:
            Dataset returned by ``load_data()``.

        Returns
        -------
        skore EstimatorReport or CrossValidationReport
        """
        cv_splits = _default_cv_splits(ds, self.n_splits)
        report = skore_evaluate(self.model, ds.X, ds.y, splitter=cv_splits)
        rmse = _extract_rmse(report)
        print(f"[{self.key}] RMSE (mean) = {rmse:.4f}")
        return report

    def rmse(self, ds: Dataset) -> float:
        """Evaluate and return the mean grouped-CV RMSE as a float."""
        cv_splits = _default_cv_splits(ds, self.n_splits)
        report = skore_evaluate(self.model, ds.X, ds.y, splitter=cv_splits)
        return _extract_rmse(report)

    # ------------------------------------------------------------------
    # Hub
    # ------------------------------------------------------------------

    def push(self, report: Any) -> str:
        """Push ``report`` to Skore Hub under ``self.key``.

        Returns
        -------
        str
            The Hub URL printed by skore.
        """
        if self._project is None:
            self._project = get_project()
        self._project.put(self.key, report)
        return self.key

    # ------------------------------------------------------------------
    # Combined helper
    # ------------------------------------------------------------------

    def run_and_push(self, ds: Dataset) -> Any:
        """Evaluate, print RMSE, push to Hub, and return the report."""
        report = self.run(ds)
        self.push(report)
        return report


# ------------------------------------------------------------------
# Multi-model comparison
# ------------------------------------------------------------------

def compare(
    models: dict[str, Any],
    ds: Dataset,
    push_key: str | None = None,
    n_splits: int = 5,
) -> Any:
    """Evaluate multiple models on the same grouped-CV folds.

    Parameters
    ----------
    models:
        ``{"key": estimator, ...}`` — the keys label each model in the report.
    ds:
        Dataset returned by ``load_data()``.
    push_key:
        If given, push the comparison report to Hub under this key.
    n_splits:
        Number of GroupKFold folds.

    Returns
    -------
    skore ComparisonReport
    """
    cv_splits = _default_cv_splits(ds, n_splits)
    report = skore_evaluate(models, ds.X, ds.y, splitter=cv_splits)

    # Print RMSE per model
    for name in models:
        try:
            rmse = _extract_rmse(report, estimator_name=name)
            print(f"[{name}] RMSE (mean) = {rmse:.4f}")
        except Exception:
            pass

    if push_key is not None:
        project = get_project()
        project.put(push_key, report)

    return report
