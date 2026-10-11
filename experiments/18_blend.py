# %% [markdown]
# # Experiment 18 — Transformer + GBM Blend: complementary signal fusion
#
# The Transformer (exp 17) and gradient boosting (exp 12) exploit different
# signal structures:
#   - Transformer: full cross-visit context, attends across the disease trajectory
#   - GBM: powerful local feature interactions, calibrated on individual visit rows
#
# Their OOF predictions are complementary — blending them should push RMSE
# significantly below either model alone toward the ≤ 2 RMSE target.
#
# Strategy:
#   1. Load OOF predictions from exp 17 (transformer) and run GBM CV to get
#      GBM OOF at visit level.
#   2. Sweep blend weights on OOF RMSE to find the optimal α.
#   3. Apply the same blend to test predictions.
#   4. As a bonus: train a Ridge meta-learner on [transformer_pred, lgbm_pred, xgb_pred, hgbr_pred]

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
import xgboost as xgb

from parkinson.hub import get_project

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data...")
X_train_raw = pd.read_csv("data/X_train.csv")
y_train_df = pd.read_csv("data/y_train.csv")
X_test_raw = pd.read_csv("data/X_test.csv")
visits = X_train_raw.merge(y_train_df, on="Index")
y = visits["target"].values
groups = visits["patient_id"]
print(f"  train: {visits.shape}  test: {X_test_raw.shape}")

# ── Load transformer OOF predictions ─────────────────────────────────────────
print("\nLoading transformer OOF predictions (exp 17)...")
oof_17_df = pd.read_csv("scratch/17_oof_visit.csv")
test_17_df = pd.read_csv("scratch/17_test_visit.csv")

# Align with X_train_raw row order
oof_17 = X_train_raw[["Index"]].merge(oof_17_df, on="Index", how="left")["pred_17"].values
test_17 = X_test_raw[["Index"]].merge(test_17_df, on="Index", how="left")["pred_17"].values

valid_mask = ~np.isnan(oof_17)
print(f"  Transformer OOF valid rows: {valid_mask.sum()} / {len(oof_17)}")
rmse_17_alone = float(np.sqrt(np.mean((y[valid_mask] - oof_17[valid_mask]) ** 2)))
print(f"  Transformer OOF RMSE (visit level): {rmse_17_alone:.4f}")


# ── Feature engineering (same as exp 12) ─────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
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
            pat_median_off=("off", "median"),
            pat_min_off=("off", "min"),
            pat_max_off=("off", "max"),
            pat_median_on=("on", "median"),
            pat_min_on=("on", "min"),
            pat_max_on=("on", "max"),
            pat_n_visits=("age", "count"),
            pat_age_range=("age", lambda x: x.max() - x.min()),
            pat_mean_age=("age", "mean"),
            pat_mean_dx_age=("age_at_diagnosis", "mean"),
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

    out["off_z_pat"] = out["off_vs_pat_mean"] / (out["pat_std_off"] + 1e-6)
    out["on_z_pat"] = out["on_vs_pat_mean"] / (out["pat_std_on"] + 1e-6)

    toff = out["time_since_intake_off"].fillna(out["time_since_intake_off"].median())
    out["off_pk_weight"] = 1.0 / (1.0 + np.exp(-0.3 * (toff - 12.0)))
    out["off_debiased"] = out["off"] * out["off_pk_weight"]

    ton = out["time_since_intake_on"].fillna(out["time_since_intake_on"].median())
    out["on_pk_weight"] = 1.0 / (1.0 + np.exp(0.5 * (ton - 2.0)))
    out["on_adjusted"] = out["on"] * out["on_pk_weight"]

    out["ledd_per_year"] = out["ledd"] / (out["years_since_dx"].clip(lower=0.1))
    out["off_on_ratio"] = out["off"] / (out["on"] + 1.0)
    out["ledd_x_off"] = out["ledd"].fillna(0) * out["off"].fillna(0)
    out["age_x_off"] = out["age"] * out["off"].fillna(0)

    out["visit_rank"] = (
        out.groupby("patient_id")["age"].rank(method="first").astype(float)
    )
    out["visit_rank_pct"] = out["visit_rank"] / out["pat_n_visits"]

    drop_cols = ["Index", "patient_id"]
    if "target" in out.columns:
        drop_cols.append("target")
    return out.drop(columns=drop_cols)


# ── Build feature matrices ─────────────────────────────────────────────────────
print("\nEngineering GBM features...")
X_eng = engineer_features(visits)
y = visits["target"].values
groups = visits["patient_id"]

X_num = X_eng.copy()
for col in ["cohort", "gene", "rater_id"]:
    if col in X_num.columns:
        X_num[col] = X_num[col].cat.codes.astype("float32")

cv = GroupKFold(n_splits=5)
cv_splits = list(cv.split(X_eng, y, groups=groups))

# ── Model definitions (full-power, same as exp 12) ────────────────────────────
lgbm_params = dict(
    n_estimators=3000, num_leaves=255, learning_rate=0.015,
    min_child_samples=15, colsample_bytree=0.7, subsample=0.8,
    subsample_freq=1, reg_alpha=0.05, reg_lambda=0.3, max_depth=-1,
    random_state=42, n_jobs=-1, verbose=-1,
)
xgb_params = dict(
    n_estimators=2000, max_depth=8, learning_rate=0.02,
    subsample=0.8, colsample_bytree=0.7, reg_alpha=0.05, reg_lambda=0.5,
    random_state=42, n_jobs=-1, verbosity=0, tree_method="hist",
)
hgbr_params = dict(
    max_iter=500, max_depth=8, learning_rate=0.05,
    min_samples_leaf=15, random_state=42,
)

N = len(y)
oof_lgbm = np.zeros(N)
oof_xgb = np.zeros(N)
oof_hgbr = np.zeros(N)

print("\n[18] Collecting GBM OOF predictions (GroupKFold-5)...")
for fold_i, (train_idx, val_idx) in enumerate(cv_splits):
    m_lgbm = lgb.LGBMRegressor(**lgbm_params)
    X_tr_l = X_eng.iloc[train_idx]
    X_vl_l = X_eng.iloc[val_idx].copy()
    for col in ["cohort", "gene", "rater_id"]:
        if col in X_tr_l.columns:
            X_vl_l[col] = X_vl_l[col].cat.set_categories(X_tr_l[col].cat.categories)
    m_lgbm.fit(X_tr_l, y[train_idx])
    oof_lgbm[val_idx] = m_lgbm.predict(X_vl_l)

    m_xgb = xgb.XGBRegressor(**xgb_params)
    m_xgb.fit(X_num.iloc[train_idx], y[train_idx])
    oof_xgb[val_idx] = m_xgb.predict(X_num.iloc[val_idx])

    m_hgbr = HistGradientBoostingRegressor(**hgbr_params)
    m_hgbr.fit(X_num.iloc[train_idx], y[train_idx])
    oof_hgbr[val_idx] = m_hgbr.predict(X_num.iloc[val_idx])

    r_l = np.sqrt(np.mean((y[val_idx] - oof_lgbm[val_idx]) ** 2))
    r_x = np.sqrt(np.mean((y[val_idx] - oof_xgb[val_idx]) ** 2))
    r_h = np.sqrt(np.mean((y[val_idx] - oof_hgbr[val_idx]) ** 2))
    print(f"  Fold {fold_i+1}: LGBM {r_l:.4f}  XGB {r_x:.4f}  HGBR {r_h:.4f}")

rmse_lgbm = np.sqrt(np.mean((y - oof_lgbm) ** 2))
rmse_xgb = np.sqrt(np.mean((y - oof_xgb) ** 2))
rmse_hgbr = np.sqrt(np.mean((y - oof_hgbr) ** 2))
print(f"\n  OOF: LGBM={rmse_lgbm:.4f}  XGB={rmse_xgb:.4f}  HGBR={rmse_hgbr:.4f}")

# ── GBM stacking (Ridge meta on GBM OOF) ─────────────────────────────────────
oof_gbm_stack = np.column_stack([oof_lgbm, oof_xgb, oof_hgbr])
meta_gbm = Ridge(alpha=0.1)
meta_gbm.fit(oof_gbm_stack, y)
oof_gbm_meta = meta_gbm.predict(oof_gbm_stack)
rmse_gbm_meta = np.sqrt(np.mean((y - oof_gbm_meta) ** 2))
print(f"\n  GBM Ridge-meta OOF RMSE: {rmse_gbm_meta:.4f}")

# ── Blend Transformer + GBM ───────────────────────────────────────────────────
# Only on rows where transformer OOF is valid
print("\n[Blend] Sweeping Transformer + GBM blend weights...")
valid = ~np.isnan(oof_17)
y_v = y[valid]
t_v = oof_17[valid]
g_v = oof_gbm_meta[valid]

best_alpha, best_rmse = 0.0, float("inf")
for alpha in np.linspace(0.0, 1.0, 101):
    blend = alpha * t_v + (1 - alpha) * g_v
    rmse = float(np.sqrt(np.mean((y_v - blend) ** 2)))
    if rmse < best_rmse:
        best_rmse = rmse
        best_alpha = alpha

print(f"  Best blend α={best_alpha:.2f} (transformer weight)  OOF RMSE={best_rmse:.4f}")

# Also try Ridge on all 4 predictors
oof_all = np.column_stack([oof_17, oof_lgbm, oof_xgb, oof_hgbr])
# Use only valid rows for fitting
oof_all_valid = oof_all[valid]
meta_all = Ridge(alpha=0.1)
meta_all.fit(oof_all_valid, y_v)
meta_oof_all = meta_all.predict(oof_all_valid)
rmse_meta_all = float(np.sqrt(np.mean((y_v - meta_oof_all) ** 2)))
print(f"  Ridge meta (4 models) OOF RMSE: {rmse_meta_all:.4f}")
print(f"  Weights: T={meta_all.coef_[0]:.3f}  LGBM={meta_all.coef_[1]:.3f}  XGB={meta_all.coef_[2]:.3f}  HGBR={meta_all.coef_[3]:.3f}")

# ── Final GBM models on all training data ─────────────────────────────────────
print("\n[Final] Fitting GBM models on full training data...")
X_test_eng = engineer_features(X_test_raw)
X_test_num = X_test_eng.copy()
for col in ["cohort", "gene", "rater_id"]:
    if col in X_test_num.columns:
        X_test_num[col] = X_test_num[col].cat.set_categories(
            X_eng[col].cat.categories
        ).cat.codes.astype("float32")
X_test_eng_aligned = X_test_eng.reindex(columns=X_eng.columns, fill_value=np.nan)
X_test_num_aligned = X_test_num.reindex(columns=X_num.columns, fill_value=np.nan)

final_lgbm = lgb.LGBMRegressor(**lgbm_params)
final_lgbm.fit(X_eng, y)

final_xgb = xgb.XGBRegressor(**xgb_params)
final_xgb.fit(X_num, y)

final_hgbr = HistGradientBoostingRegressor(**hgbr_params)
final_hgbr.fit(X_num, y)

t_lgbm = final_lgbm.predict(X_test_eng_aligned)
t_xgb = final_xgb.predict(X_test_num_aligned)
t_hgbr = final_hgbr.predict(X_test_num_aligned)

test_gbm_meta = meta_gbm.predict(np.column_stack([t_lgbm, t_xgb, t_hgbr]))

# ── Build test submissions ─────────────────────────────────────────────────────
print("\n[Submission] Building test predictions...")
Path("submissions").mkdir(exist_ok=True)

# Best blend submission
test_blend = best_alpha * test_17 + (1 - best_alpha) * test_gbm_meta

# Handle any remaining NaN in transformer test preds (shouldn't occur, but safety)
nan_test = np.isnan(test_blend)
if nan_test.any():
    print(f"  WARNING: {nan_test.sum()} NaN in blend → fallback to GBM")
    test_blend[nan_test] = test_gbm_meta[nan_test]

sub_blend = X_test_raw[["Index"]].copy()
sub_blend["target"] = test_blend
sub_blend.to_csv("submissions/18_blend.csv", index=False)
print(f"  Wrote submissions/18_blend.csv")

# Ridge meta all 4 models
test_all_stack = np.column_stack([test_17, t_lgbm, t_xgb, t_hgbr])
nan_test_rows = np.isnan(test_all_stack[:, 0])
test_all_stack[nan_test_rows, 0] = test_gbm_meta[nan_test_rows]  # fallback NaN transformer
test_meta_all_preds = meta_all.predict(test_all_stack)

sub_meta = X_test_raw[["Index"]].copy()
sub_meta["target"] = test_meta_all_preds
sub_meta.to_csv("submissions/18_meta.csv", index=False)
print(f"  Wrote submissions/18_meta.csv")

# GBM meta alone (for reference)
sub_gbm = X_test_raw[["Index"]].copy()
sub_gbm["target"] = test_gbm_meta
sub_gbm.to_csv("submissions/18_gbm_meta.csv", index=False)
print(f"  Wrote submissions/18_gbm_meta.csv (GBM only reference)")

print(f"\n{'='*60}")
print("RESULTS SUMMARY")
print(f"  12_full_ensemble GBM:      4.58")
print(f"  17_transformer OOF:        {rmse_17_alone:.4f}")
print(f"  18 GBM Ridge meta:         {rmse_gbm_meta:.4f}")
print(f"  18 Blend (α={best_alpha:.2f}T+GBM): {best_rmse:.4f}")
print(f"  18 Ridge meta (4 models):  {rmse_meta_all:.4f}")
print(f"  Target:                    <= 2.00")
print(f"{'='*60}")

print(f"\nBest submission: submissions/18_blend.csv  or  18_meta.csv")
