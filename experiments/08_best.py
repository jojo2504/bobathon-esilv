# %% [markdown]
# # Experiment 08 — LightGBM v2: extended features + tuned hyperparams
#
# Builds on 07 (RMSE 5.17). Additions:
#   - More patient-level aggregates (min/max off/on, ledd std)
#   - PK timing for ON as well as OFF
#   - Ratio features (ledd_per_year, off/on ratio)
#   - Deviation from patient median (not just mean)
#   - More estimators + lower learning rate for better convergence
#   - XGBoost comparison on same features
#   - Blend LightGBM + XGBoost predictions (simple average)

# %%
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
import xgboost as xgb
from skore import evaluate

from parkinson.hub import get_project

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data...")
X_train_raw = pd.read_csv("data/X_train.csv")
y_train_df = pd.read_csv("data/y_train.csv")
X_test_raw = pd.read_csv("data/X_test.csv")
visits = X_train_raw.merge(y_train_df, on="Index")
print(f"  train: {visits.shape}  test: {X_test_raw.shape}")


# ── Feature engineering v2 ────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extended feature set. Preserves original row order."""
    out = df.copy()

    for col in ["cohort", "gene", "rater_id"]:
        if col in out.columns:
            out[col] = out[col].astype("category")

    # Disease duration
    out["years_since_dx"] = out["age"] - out["age_at_diagnosis"]

    # Per-patient aggregates (X-only, no target)
    pat = (
        out.groupby("patient_id", sort=False)
        .agg(
            pat_mean_off=("off", "mean"),
            pat_mean_on=("on", "mean"),
            pat_std_off=("off", "std"),
            pat_std_on=("on", "std"),
            pat_mean_ledd=("ledd", "mean"),
            pat_std_ledd=("ledd", "std"),
            pat_median_off=("off", "median"),
            pat_median_on=("on", "median"),
            pat_min_off=("off", "min"),
            pat_max_off=("off", "max"),
            pat_min_on=("on", "min"),
            pat_max_on=("on", "max"),
            pat_n_visits=("age", "count"),
            pat_age_range=("age", lambda x: x.max() - x.min()),
            pat_mean_age=("age", "mean"),
        )
        .reset_index()
    )
    out = out.merge(pat, on="patient_id", how="left")

    # Motor gap features
    out["off_minus_on"] = out["off"] - out["on"]
    out["mean_motor"] = out[["off", "on"]].mean(axis=1)
    out["best_motor"] = out[["off", "on"]].max(axis=1)
    out["worst_motor"] = out[["off", "on"]].min(axis=1)

    # Deviation from patient mean/median
    out["off_vs_pat_mean"] = out["off"] - out["pat_mean_off"]
    out["on_vs_pat_mean"] = out["on"] - out["pat_mean_on"]
    out["off_vs_pat_median"] = out["off"] - out["pat_median_off"]
    out["on_vs_pat_median"] = out["on"] - out["pat_median_on"]
    out["off_range_pct"] = (out["off"] - out["pat_min_off"]) / (
        out["pat_max_off"] - out["pat_min_off"] + 1e-6
    )

    # PK timing — OFF
    toff = out["time_since_intake_off"].fillna(
        out["time_since_intake_off"].median()
    )
    out["off_pk_weight"] = 1.0 / (1.0 + np.exp(-0.3 * (toff - 12.0)))
    out["off_debiased"] = out["off"] * out["off_pk_weight"]

    # PK timing — ON (short time_since_intake_on → still under drug effect)
    ton = out["time_since_intake_on"].fillna(
        out["time_since_intake_on"].median()
    )
    out["on_pk_weight"] = 1.0 / (1.0 + np.exp(0.5 * (ton - 2.0)))
    out["on_adjusted"] = out["on"] * out["on_pk_weight"]

    # Ratio features
    out["ledd_per_year"] = out["ledd"] / (out["years_since_dx"].clip(lower=0.1))
    out["off_on_ratio"] = out["off"] / (out["on"] + 1.0)

    # Drop identifiers
    drop_cols = ["Index", "patient_id"]
    if "target" in out.columns:
        drop_cols.append("target")
    return out.drop(columns=drop_cols)


# ── Build features ────────────────────────────────────────────────────────────
print("\nEngineering features v2...")
X_eng = engineer_features(visits)
y = visits["target"].values
groups = visits["patient_id"]
assert len(X_eng) == len(y)
print(f"  Feature matrix: {X_eng.shape}")

cv_splits = list(GroupKFold(n_splits=5).split(X_eng, y, groups=groups))

# ── LightGBM tuned ────────────────────────────────────────────────────────────
print("\n[08a] LightGBM tuned (GroupKFold-5)")
lgbm = lgb.LGBMRegressor(
    n_estimators=3000,
    num_leaves=255,
    learning_rate=0.015,
    min_child_samples=15,
    colsample_bytree=0.7,
    subsample=0.8,
    subsample_freq=1,
    reg_alpha=0.05,
    reg_lambda=0.3,
    max_depth=-1,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
rep_lgbm = evaluate(lgbm, X_eng, y, splitter=cv_splits)
rmse_lgbm = float(rep_lgbm.metrics.rmse().iloc[:, 0].mean())
print(f"  LightGBM RMSE = {rmse_lgbm:.4f}  (07 was 5.1736)")

# ── XGBoost comparison ────────────────────────────────────────────────────────
print("\n[08b] XGBoost (GroupKFold-5)")
# XGBoost needs numeric — encode categoricals as int codes
X_xgb = X_eng.copy()
for col in ["cohort", "gene", "rater_id"]:
    if col in X_xgb.columns:
        X_xgb[col] = X_xgb[col].cat.codes.astype("float32")

xgb_model = xgb.XGBRegressor(
    n_estimators=2000,
    max_depth=7,
    learning_rate=0.02,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.05,
    reg_lambda=0.5,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
    tree_method="hist",
)
rep_xgb = evaluate(xgb_model, X_xgb, y, splitter=cv_splits)
rmse_xgb = float(rep_xgb.metrics.rmse().iloc[:, 0].mean())
print(f"  XGBoost RMSE  = {rmse_xgb:.4f}")

# ── Push best to Hub ──────────────────────────────────────────────────────────
print("\n[Push] Reports → Hub")
project = get_project()
project.put("08_lgbm_v2", rep_lgbm)
project.put("08_xgb", rep_xgb)
print(f"  Best single model: {'LightGBM' if rmse_lgbm < rmse_xgb else 'XGBoost'} RMSE {min(rmse_lgbm, rmse_xgb):.4f}")

# ── Submission: blend LightGBM + XGBoost ─────────────────────────────────────
print("\n[Submission] Fitting on ALL training data + blending...")
X_test_eng = engineer_features(X_test_raw)
X_test_xgb = X_test_eng.copy()
for col in ["cohort", "gene", "rater_id"]:
    if col in X_test_xgb.columns:
        X_test_xgb[col] = X_test_xgb[col].cat.set_categories(
            X_eng[col].cat.categories
        ).cat.codes.astype("float32")

# Align columns
for col in X_eng.columns:
    if col not in X_test_eng.columns:
        X_test_eng[col] = np.nan
X_test_eng = X_test_eng[X_eng.columns]
X_test_xgb = X_test_xgb.reindex(columns=X_xgb.columns, fill_value=np.nan)

# Fit final models on all training data
final_lgbm = lgb.LGBMRegressor(
    n_estimators=3000, num_leaves=255, learning_rate=0.015,
    min_child_samples=15, colsample_bytree=0.7, subsample=0.8,
    subsample_freq=1, reg_alpha=0.05, reg_lambda=0.3,
    random_state=42, n_jobs=-1, verbose=-1,
)
final_lgbm.fit(X_eng, y)

final_xgb = xgb.XGBRegressor(
    n_estimators=2000, max_depth=7, learning_rate=0.02,
    subsample=0.8, colsample_bytree=0.7, reg_alpha=0.05,
    reg_lambda=0.5, random_state=42, n_jobs=-1,
    verbosity=0, tree_method="hist",
)
final_xgb.fit(X_xgb, y)

# Blend: equal weight average
pred_lgbm = final_lgbm.predict(X_test_eng)
pred_xgb = final_xgb.predict(X_test_xgb)
pred_blend = 0.5 * pred_lgbm + 0.5 * pred_xgb

Path("submissions").mkdir(exist_ok=True)

# Save individual + blend
submission_lgbm = X_test_raw[["Index"]].copy()
submission_lgbm["target"] = pred_lgbm
submission_lgbm.to_csv("submissions/08_lgbm_v2.csv", index=False)
print(f"  Wrote submissions/08_lgbm_v2.csv")

submission_blend = X_test_raw[["Index"]].copy()
submission_blend["target"] = pred_blend
submission_blend.to_csv("submissions/08_blend.csv", index=False)
print(f"  Wrote submissions/08_blend.csv")

print(f"\n{'='*60}")
print(f"RESULTS SUMMARY")
print(f"  07_lgbm_engineered (baseline):  5.1736")
print(f"  08_lgbm_v2:                     {rmse_lgbm:.4f}")
print(f"  08_xgb:                         {rmse_xgb:.4f}")
print(f"  05_tabular (Day 2 best):        7.5238")
print(f"  01_dummy floor:                 16.4992")
print(f"{'='*60}")
print(f"\nBest submission: submissions/08_lgbm_v2.csv")
print(f"Upload to Kaggle with the '08_lgbm_v2' Hub report URL above.")
