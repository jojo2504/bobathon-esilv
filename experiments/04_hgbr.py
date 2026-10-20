# %% [markdown]
# # Experiment 04 — HistGradientBoosting: missingness as signal
#
# Step 4 of the Day 2 guide:
#   - Numeric-only features, no imputation — NaNs are left in place
#   - HGBR natively routes missing values at each split (they are signal, not noise)
#   - Grouped CV, compare vs Ridge on same folds
#   - Push "04_hgbr"

# %%
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from skore import evaluate

from parkinson.data import load_data
from parkinson.hub import get_project

# ── Data ──────────────────────────────────────────────────────────────────────
print("Loading data...")
ds = load_data()
X, y = ds.X, ds.y
groups = ds.groups
print(f"  train: {X.shape}  groups (patients): {groups.nunique()}")

# ── Grouped CV splits ─────────────────────────────────────────────────────────
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))

# ── HGBR — no imputation, NaNs stay ──────────────────────────────────────────
print("\n[Step 4] HistGradientBoostingRegressor — NaNs kept as signal")
hgbr = HistGradientBoostingRegressor(random_state=0)
ridge = make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=0.1))

rep_hgbr = evaluate(hgbr, X, y, splitter=cv_splits)
rep_ridge = evaluate(ridge, X, y, splitter=cv_splits)

rmse_ridge = float(rep_ridge.metrics.rmse().iloc[:, 0].mean())
rmse_hgbr = float(rep_hgbr.metrics.rmse().iloc[:, 0].mean())
print(f"  ridge RMSE (grouped CV) = {rmse_ridge:.4f}")
print(f"  hgbr  RMSE (grouped CV) = {rmse_hgbr:.4f}")
print(f"  improvement over ridge  = {(rmse_ridge - rmse_hgbr) / rmse_ridge * 100:.1f}%")
print(f"  improvement over dummy  = {(16.4992 - rmse_hgbr) / 16.4992 * 100:.1f}%")

# ── Push to Hub ───────────────────────────────────────────────────────────────
print("\n[Push] HGBR report → Hub key '04_hgbr'")
project = get_project()
project.put("04_hgbr", rep_hgbr)
