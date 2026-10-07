# %% [markdown]
# # Experiment 11 — Stacking: OOF Predictions + Ridge Meta-Learner
#
# Collects OOF predictions from 4 diverse base models (LightGBM, XGBoost,
# HGBR, Ridge) using GroupKFold-5, then trains a Ridge meta-learner on those
# OOF stacks.  Final submission: refit all base models on full training data,
# apply Ridge meta to their test predictions.

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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


# ── Feature engineering (identical to 08) ────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extended feature set from experiment 08."""
    out = df.copy()

    for col in ["cohort", "gene", "rater_id"]:
        if col in out.columns:
            out[col] = out[col].astype("category")

    out["years_since_dx"] = out["age"] - out["age_at_diagnosis"]

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

    out["off_minus_on"] = out["off"] - out["on"]
    out["mean_motor"] = out[["off", "on"]].mean(axis=1)
    out["best_motor"] = out[["off", "on"]].max(axis=1)
    out["worst_motor"] = out[["off", "on"]].min(axis=1)

    out["off_vs_pat_mean"] = out["off"] - out["pat_mean_off"]
    out["on_vs_pat_mean"] = out["on"] - out["pat_mean_on"]
    out["off_vs_pat_median"] = out["off"] - out["pat_median_off"]
    out["on_vs_pat_median"] = out["on"] - out["pat_median_on"]
    out["off_range_pct"] = (out["off"] - out["pat_min_off"]) / (
        out["pat_max_off"] - out["pat_min_off"] + 1e-6
    )

    toff = out["time_since_intake_off"].fillna(out["time_since_intake_off"].median())
    out["off_pk_weight"] = 1.0 / (1.0 + np.exp(-0.3 * (toff - 12.0)))
    out["off_debiased"] = out["off"] * out["off_pk_weight"]

    ton = out["time_since_intake_on"].fillna(out["time_since_intake_on"].median())
    out["on_pk_weight"] = 1.0 / (1.0 + np.exp(0.5 * (ton - 2.0)))
    out["on_adjusted"] = out["on"] * out["on_pk_weight"]

    out["ledd_per_year"] = out["ledd"] / (out["years_since_dx"].clip(lower=0.1))
    out["off_on_ratio"] = out["off"] / (out["on"] + 1.0)

    drop_cols = ["Index", "patient_id"]
    if "target" in out.columns:
        drop_cols.append("target")
    return out.drop(columns=drop_cols)


# ── Prepare feature matrices ──────────────────────────────────────────────────
print("\nEngineering features...")
X_eng = engineer_features(visits)
y = visits["target"].values
groups = visits["patient_id"]
print(f"  Feature matrix: {X_eng.shape}")

# XGBoost needs numeric categoricals
X_xgb = X_eng.copy()
for col in ["cohort", "gene", "rater_id"]:
    if col in X_xgb.columns:
        X_xgb[col] = X_xgb[col].cat.codes.astype("float32")

# HGBR / Ridge: numeric-only (drop categoricals, keep numeric)
X_num = X_eng.copy()
for col in ["cohort", "gene", "rater_id"]:
    if col in X_num.columns:
        X_num[col] = X_num[col].cat.codes.astype("float32")

cv = GroupKFold(n_splits=5)
cv_splits = list(cv.split(X_eng, y, groups=groups))

# ── Define base learners ──────────────────────────────────────────────────────
lgbm_params = dict(
    n_estimators=800, num_leaves=127, learning_rate=0.05,
    min_child_samples=20, colsample_bytree=0.7, subsample=0.8,
    subsample_freq=1, reg_alpha=0.05, reg_lambda=0.3,
    random_state=42, n_jobs=-1, verbose=-1,
)
xgb_params = dict(
    n_estimators=1000, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7, reg_alpha=0.05,
    reg_lambda=0.5, random_state=42, n_jobs=-1, verbosity=0,
    tree_method="hist",
)
hgbr_params = dict(
    max_iter=300, max_depth=6, learning_rate=0.05,
    min_samples_leaf=20, random_state=42,
)

N = len(y)
oof_lgbm = np.zeros(N)
oof_xgb = np.zeros(N)
oof_hgbr = np.zeros(N)
oof_ridge = np.zeros(N)

# ── Collect OOF predictions ───────────────────────────────────────────────────
print("\n[11] Collecting OOF predictions (GroupKFold-5)...")
for fold_i, (train_idx, val_idx) in enumerate(cv_splits):
    # LGBM
    m_lgbm = lgb.LGBMRegressor(**lgbm_params)
    X_tr_lgbm = X_eng.iloc[train_idx]
    X_val_lgbm = X_eng.iloc[val_idx]
    for col in ["cohort", "gene", "rater_id"]:
        if col in X_tr_lgbm.columns:
            X_val_lgbm = X_val_lgbm.copy()
            X_val_lgbm[col] = X_val_lgbm[col].cat.set_categories(
                X_tr_lgbm[col].cat.categories
            )
    m_lgbm.fit(X_tr_lgbm, y[train_idx])
    oof_lgbm[val_idx] = m_lgbm.predict(X_val_lgbm)

    # XGBoost
    m_xgb = xgb.XGBRegressor(**xgb_params)
    m_xgb.fit(X_xgb.iloc[train_idx], y[train_idx])
    oof_xgb[val_idx] = m_xgb.predict(X_xgb.iloc[val_idx])

    # HGBR
    m_hgbr = HistGradientBoostingRegressor(**hgbr_params)
    m_hgbr.fit(X_num.iloc[train_idx], y[train_idx])
    oof_hgbr[val_idx] = m_hgbr.predict(X_num.iloc[val_idx])

    # Ridge (impute NaN → median, then scale)
    m_ridge = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])
    m_ridge.fit(X_num.iloc[train_idx], y[train_idx])
    oof_ridge[val_idx] = m_ridge.predict(X_num.iloc[val_idx])

    fold_rmse_lgbm = np.sqrt(np.mean((y[val_idx] - oof_lgbm[val_idx]) ** 2))
    fold_rmse_xgb = np.sqrt(np.mean((y[val_idx] - oof_xgb[val_idx]) ** 2))
    print(
        f"  Fold {fold_i + 1}: LGBM {fold_rmse_lgbm:.4f}  "
        f"XGB {fold_rmse_xgb:.4f}  "
        f"HGBR {np.sqrt(np.mean((y[val_idx] - oof_hgbr[val_idx])**2)):.4f}  "
        f"Ridge {np.sqrt(np.mean((y[val_idx] - oof_ridge[val_idx])**2)):.4f}"
    )

rmse_lgbm = float(np.sqrt(np.mean((y - oof_lgbm) ** 2)))
rmse_xgb = float(np.sqrt(np.mean((y - oof_xgb) ** 2)))
rmse_hgbr = float(np.sqrt(np.mean((y - oof_hgbr) ** 2)))
rmse_ridge = float(np.sqrt(np.mean((y - oof_ridge) ** 2)))

print(f"\n  OOF RMSEs:")
print(f"    LGBM:  {rmse_lgbm:.4f}")
print(f"    XGB:   {rmse_xgb:.4f}")
print(f"    HGBR:  {rmse_hgbr:.4f}")
print(f"    Ridge: {rmse_ridge:.4f}")

# ── Train Ridge meta-learner on OOF stack ────────────────────────────────────
print("\n[Meta] Training Ridge meta-learner on OOF stack...")
oof_stack = np.column_stack([oof_lgbm, oof_xgb, oof_hgbr, oof_ridge])
meta = Ridge(alpha=0.1)
meta.fit(oof_stack, y)
meta_preds_oof = meta.predict(oof_stack)
rmse_meta = float(np.sqrt(np.mean((y - meta_preds_oof) ** 2)))
print(f"  Meta (Ridge) in-sample RMSE on OOF stack: {rmse_meta:.4f}")
print(f"  Meta weights: LGBM={meta.coef_[0]:.3f} XGB={meta.coef_[1]:.3f} "
      f"HGBR={meta.coef_[2]:.3f} Ridge={meta.coef_[3]:.3f}")

# True OOF RMSE: the OOF predictions from base models are held-out,
# so the meta RMSE above is the stacked OOF estimate (standard L2 stacking).
print(f"\n  Stacked OOF RMSE estimate: {rmse_meta:.4f}  (08_blend was 4.91)")

# ── Push Hub report ───────────────────────────────────────────────────────────
print("\n[Push] Hub report using best base model (LightGBM)...")
lgbm_for_report = lgb.LGBMRegressor(**lgbm_params)
rep_11 = evaluate(lgbm_for_report, X_eng, y, splitter=cv_splits)
print(f"  Hub LGBM RMSE: {float(rep_11.metrics.rmse().iloc[:, 0].mean()):.4f}")

project = get_project()
project.put("11_stacking", rep_11)
print("  Pushed '11_stacking' → Hub")

# ── Submission ────────────────────────────────────────────────────────────────
print("\n[Submission] Refitting base models on all training data...")

X_test_eng = engineer_features(X_test_raw)
X_test_xgb = X_test_eng.copy()
for col in ["cohort", "gene", "rater_id"]:
    if col in X_test_xgb.columns:
        X_test_xgb[col] = X_test_xgb[col].cat.set_categories(
            X_eng[col].cat.categories
        ).cat.codes.astype("float32")
X_test_num = X_test_xgb.copy()

# Align columns
X_test_eng = X_test_eng.reindex(columns=X_eng.columns, fill_value=np.nan)
X_test_xgb = X_test_xgb.reindex(columns=X_xgb.columns, fill_value=np.nan)
X_test_num = X_test_num.reindex(columns=X_num.columns, fill_value=np.nan)

# Refit on all training data
final_lgbm = lgb.LGBMRegressor(**lgbm_params)
final_lgbm.fit(X_eng, y)

final_xgb = xgb.XGBRegressor(**xgb_params)
final_xgb.fit(X_xgb, y)

final_hgbr = HistGradientBoostingRegressor(**hgbr_params)
final_hgbr.fit(X_num, y)

final_ridge = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0)),
])
final_ridge.fit(X_num, y)

# Predict and stack
test_lgbm = final_lgbm.predict(X_test_eng)
test_xgb = final_xgb.predict(X_test_xgb)
test_hgbr = final_hgbr.predict(X_test_num)
test_ridge = final_ridge.predict(X_test_num)
test_stack = np.column_stack([test_lgbm, test_xgb, test_hgbr, test_ridge])
test_preds = meta.predict(test_stack)

Path("submissions").mkdir(exist_ok=True)
submission = X_test_raw[["Index"]].copy()
submission["target"] = test_preds
submission.to_csv("submissions/11_stacking.csv", index=False)
print(f"  Wrote submissions/11_stacking.csv  ({len(submission)} rows)")

print(f"\n{'='*60}")
print("RESULTS SUMMARY")
print(f"  08_blend (prior best):     4.91")
print(f"  11_stacking meta OOF:      {rmse_meta:.4f}")
print(f"{'='*60}")
