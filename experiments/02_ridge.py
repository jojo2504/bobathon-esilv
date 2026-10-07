# %% [markdown]
# # Experiment 02 — Ridge with row holdout
#
# Steps 1–2 of the Day 2 guide:
#   - Row holdout (splitter=0.2) comparison: dummy RMSE 16.48 vs Ridge RMSE ~10.49
#   - Alpha scan (0.1 → 100) shows negligible movement — Ridge is not regularization-sensitive here
#   - Push Ridge's own CrossValidationReport as "02_ridge"
#   - Write submissions/02_ridge.csv

# %%
from pathlib import Path
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from skore import evaluate

from parkinson.data import load_data
from parkinson.hub import get_project

# ── Data ──────────────────────────────────────────────────────────────────────
print("Loading data...")
ds = load_data()
X, y = ds.X, ds.y
print(f"  train: {X.shape}  test: {ds.X_test.shape}")

# ── Step 1: Row holdout comparison ────────────────────────────────────────────
print("\n[Step 1] Row-holdout comparison (splitter=0.2)")
dummy = DummyRegressor(strategy="mean")
ridge_1 = make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=1.0))

rep_dummy = evaluate(dummy, X, y, splitter=0.2)
rep_ridge1 = evaluate(ridge_1, X, y, splitter=0.2)

rmse_dummy = float(rep_dummy.metrics.rmse())
rmse_ridge1 = float(rep_ridge1.metrics.rmse())
print(f"  dummy RMSE       = {rmse_dummy:.4f}")
print(f"  ridge(a=1) RMSE  = {rmse_ridge1:.4f}")
print(f"  improvement      = {(rmse_dummy - rmse_ridge1) / rmse_dummy * 100:.1f}%")

# ── Step 2: Alpha scan ────────────────────────────────────────────────────────
print("\n[Step 2] Alpha scan (row holdout splitter=0.2)")
best_alpha, best_rmse = 1.0, rmse_ridge1
for alpha in [0.1, 1.0, 10.0, 100.0]:
    pipe = make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=alpha))
    rep = evaluate(pipe, X, y, splitter=0.2)
    rmse = float(rep.metrics.rmse())
    print(f"  alpha={alpha:6.1f}  RMSE={rmse:.4f}")
    if rmse < best_rmse:
        best_rmse, best_alpha = rmse, alpha

print(f"\n  Best alpha: {best_alpha}  RMSE: {best_rmse:.4f}")

# ── Evaluate best Ridge and push to Hub ───────────────────────────────────────
best_ridge = make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=best_alpha))
report_ridge = evaluate(best_ridge, X, y, splitter=0.2)

print("\n[Push] Ridge report → Hub key '02_ridge'")
project = get_project()
project.put("02_ridge", report_ridge)

# ── Submission CSV ─────────────────────────────────────────────────────────────
print("\n[Submission] Fitting on all training data...")
final = clone(best_ridge).fit(X, y)
submission = ds.X_test_raw[["Index"]].copy()
submission["target"] = final.predict(ds.X_test)
Path("submissions").mkdir(exist_ok=True)
submission.to_csv("submissions/02_ridge.csv", index=False)
print(f"Wrote submissions/02_ridge.csv  ({len(submission)} rows)")
print("\nDone. Upload submissions/02_ridge.csv to Kaggle with the Hub URL above.")
