# %% [markdown]
# # EDA: Parkinson's True-OFF Score
#
# Exploratory data analysis of the synthetic multi-cohort Parkinson's dataset,
# run before designing any model.
#
# - **Raw data** is read-only — X_train, y_train, X_test live in `data/`.
#   This file never cleans or modifies those files.
# - **Outputs** go under `EDA_DIR` (the repo's `data/`): an
#   `eda_<table>.html` report per table, summarized in `eda.md`.

# %%
import json
from pathlib import Path

import pandas as pd
import skrub

# This file lives in data/, so parents[1] is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# EDA outputs always land here (created if missing); the raw data may
# live elsewhere.
EDA_DIR = PROJECT_ROOT / "data"
EDA_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load the raw data
#
# Merge X_train with y_train on Index to get the full training table
# (features + target together). Also load X_test for a shape check.

# %%
X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")
X_test = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv")

# Full training table with target
RAW = X_train.merge(y_train, on="Index")

{"train_shape": RAW.shape, "test_shape": X_test.shape, "n_patients_train": RAW["patient_id"].nunique(), "n_patients_test": X_test["patient_id"].nunique()}  # noqa: B018

# %% [markdown]
# ## Table overview — training visits
#
# Per-column summary: dtype, fraction missing, and number of unique values.

# %%
report_train = skrub.TableReport(RAW, title="Training visits (X_train + y_train)", verbose=0)
report_train.write_html(EDA_DIR / "eda_train.html")

summary_train = json.loads(report_train.json())
n_rows = summary_train.get("n_rows")
overview_train = [
    {
        "column": col.get("name"),
        "dtype": col.get("dtype"),
        "null_pct": col.get("null_proportion"),
        "n_unique": col.get("n_unique"),
    }
    for col in summary_train.get("columns", [])
]
{"n_rows": n_rows, "n_columns": len(overview_train), "columns": overview_train}  # noqa: B018

# %% [markdown]
# ## Table overview — test visits
#
# Check test shape, columns, and missingness.

# %%
report_test = skrub.TableReport(X_test, title="Test visits (X_test)", verbose=0)
report_test.write_html(EDA_DIR / "eda_test.html")

summary_test = json.loads(report_test.json())
overview_test = [
    {
        "column": col.get("name"),
        "dtype": col.get("dtype"),
        "null_pct": col.get("null_proportion"),
        "n_unique": col.get("n_unique"),
    }
    for col in summary_test.get("columns", [])
]
{"n_rows_test": summary_test.get("n_rows"), "n_columns_test": len(overview_test), "columns_test": overview_test}  # noqa: B018

# %% [markdown]
# ## Patient visit distribution
#
# Each patient appears multiple times. How many visits per patient?
# This determines whether a row holdout leaks patients across folds.

# %%
visits_per_patient = RAW["patient_id"].value_counts()
{
    "min_visits": int(visits_per_patient.min()),
    "max_visits": int(visits_per_patient.max()),
    "median_visits": float(visits_per_patient.median()),
    "mean_visits": round(float(visits_per_patient.mean()), 2),
}  # noqa: B018

# %% [markdown]
# ## Target distribution
#
# The target is the true (unbiased) OFF MDS-UPDRS motor score (range 0–132).
# Distribution shape tells us whether RMSE is appropriate and whether a
# row holdout will draw representative validation splits.

# %%
TARGET = "target"
target_col = next((col for col in summary_train.get("columns", []) if col.get("name") == TARGET), None)
target_stats = {
    "mean": round(float(RAW[TARGET].mean()), 3),
    "std": round(float(RAW[TARGET].std()), 3),
    "min": float(RAW[TARGET].min()),
    "max": float(RAW[TARGET].max()),
    "skrub_summary": target_col,
}
target_stats  # noqa: B018

# %% [markdown]
# ## Missingness deep-dive
#
# Which columns are most missing? Which visits have no off score?
# Missingness in `off`, `on`, `ledd`, and intake delays is clinically meaningful.

# %%
missing = {
    col: round(RAW[col].isna().mean(), 4)
    for col in RAW.columns
    if RAW[col].isna().any()
}
missing_sorted = dict(sorted(missing.items(), key=lambda x: -x[1]))
both_missing = (RAW["on"].isna() & RAW["off"].isna()).sum()
{
    "missing_fractions": missing_sorted,
    "visits_no_off": int(RAW["off"].isna().sum()),
    "visits_no_on": int(RAW["on"].isna().sum()),
    "visits_both_missing": int(both_missing),
}  # noqa: B018

# %% [markdown]
# ## Structure signals
#
# High-cardinality identifiers (patient_id, rater_id) that should not be
# fed raw to a model, and any datetime columns.

# %%
datetime_cols = [
    col.get("name")
    for col in summary_train.get("columns", [])
    if "date" in str(col.get("dtype", "")).lower()
]
n_rows_train = summary_train.get("n_rows") or 1
unique_ratio = sorted(
    (
        {
            "column": col.get("name"),
            "unique_ratio": (col.get("n_unique") or 0) / n_rows_train,
        }
        for col in summary_train.get("columns", [])
    ),
    key=lambda r: r["unique_ratio"],
    reverse=True,
)
{"datetime_cols": datetime_cols, "top_unique_ratio": unique_ratio[:10]}  # noqa: B018

# %% [markdown]
# ## Associations with the target
#
# Strongest pairwise column associations. `off` being close to `target`
# would be useful signal, but also a leakage risk if `off` is too close
# (it's a biased version of the same quantity). `on` should be weaker.

# %%
assoc = skrub.column_associations(RAW)
rows = assoc.to_dicts() if hasattr(assoc, "to_dicts") else assoc.to_dict(orient="records")
target_links = [
    row
    for row in rows
    if row["left_column_name"] == TARGET or row["right_column_name"] == TARGET
]
{"with_target": target_links[:15], "strongest": rows[:10]}  # noqa: B018

# %% [markdown]
# ## Summary
#
# The findings and their modelling implications are written up in
# `data/eda.md`.
