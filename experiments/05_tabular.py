# %% [markdown]
# # Experiment 05 — skrub tabular_pipeline: mixed types without hand-encoding
#
# Step 5 of the Day 2 guide:
#   - Include categoricals: cohort, gene, rater_id (excluded from Ridge/HGBR above)
#   - skrub TableVectorizer auto-encodes: one-hot for low-cardinality, StringEncoder for high
#   - HGBR still handles NaN natively on numeric columns
#   - Drop only Index and patient_id (identifiers); pass everything else
#   - Grouped CV on same folds as prior steps
#   - Push "05_tabular"

# %%
from pathlib import Path
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from skrub import TableVectorizer
from skore import evaluate

from parkinson.data import load_data, FEATURE_COLS_FULL
from parkinson.hub import get_project

# ── Data ──────────────────────────────────────────────────────────────────────
print("Loading data...")
ds = load_data(feature_cols=FEATURE_COLS_FULL)
X_full = ds.X          # includes cohort, gene, rater_id (strings)
y = ds.y
groups = ds.groups
print(f"  train: {X_full.shape}  (includes categoricals)")
print(f"  columns: {list(X_full.columns)}")

# ── Grouped CV splits ─────────────────────────────────────────────────────────
cv_splits = list(GroupKFold(n_splits=5).split(X_full, y, groups=groups))

# ── tabular_pipeline = TableVectorizer + HistGradientBoostingRegressor ────────
print("\n[Step 5] tabular_pipeline (TableVectorizer + HGBR)")
model = make_pipeline(
    TableVectorizer(),
    HistGradientBoostingRegressor(random_state=0),
)

rep = evaluate(model, X_full, y, splitter=cv_splits)
rmse = float(rep.metrics.rmse().iloc[:, 0].mean())
print(f"  tabular_pipeline RMSE (grouped CV) = {rmse:.4f}")
print(f"  numeric-only HGBR was              = 7.7556")
print(f"  delta                              = {rmse - 7.7556:+.4f}")
print(f"  improvement over dummy             = {(16.4992 - rmse) / 16.4992 * 100:.1f}%")

# ── Push to Hub ───────────────────────────────────────────────────────────────
print("\n[Push] tabular_pipeline report → Hub key '05_tabular'")
project = get_project()
project.put("05_tabular", rep)

# ── Submission CSV ─────────────────────────────────────────────────────────────
print("\n[Submission] Fitting on all training data...")
final = clone(model).fit(X_full, y)
X_test_full = ds.X_test  # already sliced to FEATURE_COLS_FULL by load_data
submission = ds.X_test_raw[["Index"]].copy()
submission["target"] = final.predict(X_test_full)
Path("submissions").mkdir(exist_ok=True)
submission.to_csv("submissions/05_tabular.csv", index=False)
print(f"Wrote submissions/05_tabular.csv  ({len(submission)} rows)")
print("\nDone. Upload submissions/05_tabular.csv to Kaggle with the Hub URL above.")
