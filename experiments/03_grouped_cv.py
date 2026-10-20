# %% [markdown]
# # Experiment 03 — Patient-grouped CV comparison (dummy vs Ridge)
#
# Step 3 of the Day 2 guide:
#   - GroupKFold(n_splits=5) on patient_id — honest evaluation, mirrors Kaggle holdout
#   - Compare dummy vs Ridge side-by-side on the same folds
#   - Push each individual report to Hub: "03_dummy_grouped", "03_ridge_grouped"

# %%
from sklearn.dummy import DummyRegressor
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

# ── Precompute GroupKFold splits ───────────────────────────────────────────────
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))
print(f"  Grouped CV: {len(cv_splits)} folds")

# ── Evaluate both models on grouped CV ────────────────────────────────────────
print("\n[Step 3] Patient-grouped CV (GroupKFold 5-fold)")
dummy = DummyRegressor(strategy="mean")
ridge = make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=0.1))

rep_dummy = evaluate(dummy, X, y, splitter=cv_splits)
rep_ridge = evaluate(ridge, X, y, splitter=cv_splits)

rmse_dummy = float(rep_dummy.metrics.rmse().iloc[:, 0].mean())
rmse_ridge = float(rep_ridge.metrics.rmse().iloc[:, 0].mean())
print(f"  dummy RMSE (grouped CV) = {rmse_dummy:.4f}")
print(f"  ridge RMSE (grouped CV) = {rmse_ridge:.4f}")
print(f"  improvement             = {(rmse_dummy - rmse_ridge) / rmse_dummy * 100:.1f}%")
print()
print("  Row-holdout ridge was 10.49 — grouped CV shows the honest score.")

# ── Push both reports to Hub ──────────────────────────────────────────────────
print("\n[Push] Reports → Hub")
project = get_project()
project.put("03_dummy_grouped", rep_dummy)
project.put("03_ridge_grouped", rep_ridge)
print("  Pushed 03_dummy_grouped and 03_ridge_grouped")
