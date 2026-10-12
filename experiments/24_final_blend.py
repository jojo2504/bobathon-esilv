# %% [markdown]
# # Experiment 24 — Final blend: Transformer (exp 23) + full-power GBM
#
# Strategy: the Transformer and GBM exploit different signal structures.
# A Ridge meta-learner on OOF stacks the two signal types optimally.
#
# This is the experiment that should push below RMSE 2.0.
# Pre-requisite: experiments/23_fast_transformer.py must have run and produced
#   scratch/23_oof_visit.csv and scratch/23_test_visit.csv.

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
import xgboost as xgb

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data...")
X_train_raw = pd.read_csv("data/X_train.csv")
y_train_df = pd.read_csv("data/y_train.csv")
X_test_raw = pd.read_csv("data/X_test.csv")
visits = X_train_raw.merge(y_train_df, on="Index")
y = visits["target"].values
groups = visits["patient_id"]

# Load transformer OOF from exp 23
oof_23_df = pd.read_csv("scratch/23_oof_visit.csv")
test_23_df = pd.read_csv("scratch/23_test_visit.csv")
oof_23 = X_train_raw[["Index"]].merge(oof_23_df, on="Index", how="left")["pred_23"].values
test_23 = X_test_raw[["Index"]].merge(test_23_df, on="Index", how="left")["pred_23"].values

valid_mask = ~np.isnan(oof_23)
rmse_23_alone = float(np.sqrt(np.mean((y[valid_mask] - oof_23[valid_mask]) ** 2)))
print(f"  Transformer 23 OOF RMSE: {rmse_23_alone:.4f}")


# ── Feature engineering (same as exp 12/18) ───────────────────────────────────
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
        )
        .reset_index()
    )
    out = out.merge(pat, on="patient_id", how="left")
    out["off_minus_on"] = out["off"] - out["on"]
    out["off_vs_pat_mean"] = out["off"] - out["pat_mean_off"]
    out["on_vs_pat_mean"] = out["on"] - out["pat_mean_on"]
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
    out["off_sq"] = out["off"].fillna(0) ** 2
    out["off_x_on"] = out["off"].fillna(0) * out["on"].fillna(0)
    out["visit_rank"] = (
        out.groupby("patient_id")["age"].rank(method="first").astype(float)
    )
    out["visit_rank_pct"] = out["visit_rank"] / out["pat_n_visits"]
    drop_cols = ["Index", "patient_id"]
    if "target" in out.columns:
        drop_cols.append("target")
    return out.drop(columns=drop_cols)


print("\nEngineering GBM features...")
X_eng = engineer_features(visits)
X_num = X_eng.copy()
for col in ["cohort", "gene", "rater_id"]:
    if col in X_num.columns:
        X_num[col] = X_num[col].cat.codes.astype("float32")

cv = GroupKFold(n_splits=5)
cv_splits = list(cv.split(X_eng, y, groups=groups))

# Full-power GBM (same params as exp 12 — proven best)
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

print("\n[24] Collecting GBM OOF predictions (GroupKFold-5)...")
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

# ── Ridge meta on all 4 predictors ────────────────────────────────────────────
# Only on rows where transformer OOF is valid
valid = ~np.isnan(oof_23)
y_v = y[valid]
oof_all_valid = np.column_stack([oof_23[valid], oof_lgbm[valid], oof_xgb[valid], oof_hgbr[valid]])

meta = Ridge(alpha=0.1)
meta.fit(oof_all_valid, y_v)
meta_oof = meta.predict(oof_all_valid)
rmse_meta = float(np.sqrt(np.mean((y_v - meta_oof) ** 2)))
print(f"\n  Ridge meta (Transformer + LGBM + XGB + HGBR) OOF RMSE: {rmse_meta:.4f}")
print(f"  Weights: T={meta.coef_[0]:.3f}  LGBM={meta.coef_[1]:.3f}  "
      f"XGB={meta.coef_[2]:.3f}  HGBR={meta.coef_[3]:.3f}")

# Simple sweep for optimal binary blend (transformer vs GBM meta)
oof_gbm_avg = (oof_lgbm[valid] + oof_xgb[valid]) / 2
best_alpha, best_blend_rmse = 0.0, float("inf")
for alpha in np.linspace(0.0, 1.0, 101):
    blend = alpha * oof_23[valid] + (1 - alpha) * oof_gbm_avg
    rmse = float(np.sqrt(np.mean((y_v - blend) ** 2)))
    if rmse < best_blend_rmse:
        best_blend_rmse = rmse
        best_alpha = alpha
print(f"  Best simple blend α={best_alpha:.2f} (T weight)  RMSE={best_blend_rmse:.4f}")

# ── Fit final GBM models on all training data ─────────────────────────────────
print("\n[Final] Fitting GBM models on full training data...")
X_test_eng = engineer_features(X_test_raw)
X_test_num = X_test_eng.copy()
for col in ["cohort", "gene", "rater_id"]:
    if col in X_test_num.columns:
        X_test_num[col] = (
            X_test_num[col]
            .cat.set_categories(X_eng[col].cat.categories)
            .cat.codes.astype("float32")
        )
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

# ── Build test submissions ─────────────────────────────────────────────────────
print("\n[Submission] Building test predictions...")
Path("submissions").mkdir(exist_ok=True)

# Handle NaN in transformer test preds (safety)
test_23_safe = test_23.copy()
nan_test = np.isnan(test_23_safe)
if nan_test.any():
    fallback = (t_lgbm + t_xgb) / 2
    test_23_safe[nan_test] = fallback[nan_test]
    print(f"  WARNING: {nan_test.sum()} NaN in transformer test → fallback to GBM avg")

# Ridge meta submission (best method)
test_stack = np.column_stack([test_23_safe, t_lgbm, t_xgb, t_hgbr])
test_meta_preds = meta.predict(test_stack)
sub_meta = X_test_raw[["Index"]].copy()
sub_meta["target"] = test_meta_preds
sub_meta.to_csv("submissions/24_final_blend.csv", index=False)
print("  Wrote submissions/24_final_blend.csv  ← BEST SUBMISSION")

# Simple blend submission as backup
test_gbm_avg = (t_lgbm + t_xgb) / 2
test_blend = best_alpha * test_23_safe + (1 - best_alpha) * test_gbm_avg
sub_blend = X_test_raw[["Index"]].copy()
sub_blend["target"] = test_blend
sub_blend.to_csv("submissions/24_blend_simple.csv", index=False)
print("  Wrote submissions/24_blend_simple.csv  (backup)")

print(f"\n{'='*60}")
print("RESULTS SUMMARY")
print(f"  12 GBM full ensemble:          4.58")
print(f"  23 Transformer alone:          {rmse_23_alone:.4f}")
print(f"  24 LGBM OOF:                   {rmse_lgbm:.4f}")
print(f"  24 XGB OOF:                    {rmse_xgb:.4f}")
print(f"  24 Ridge meta (T+L+X+H):       {rmse_meta:.4f}  ← TARGET: <= 2.00")
print(f"  24 Simple blend (α={best_alpha:.2f}):     {best_blend_rmse:.4f}")
print(f"{'='*60}")
print("\nBest submission: submissions/24_final_blend.csv")
