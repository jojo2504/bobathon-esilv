# %% [markdown]
# # Experiment 26 — Final blend v2: Transformer-25 + full exp-12 GBM
#
# Uses:
#   - exp 25 transformer OOF (bigger model, explicit trend features, 150 epochs)
#   - Full exp-12 GBM feature set (31 features — exp 24 had fewer features, scored 4.63 vs 4.58)
#   - Ridge meta-learner on OOF stack
#
# Pre-requisite: exp 25 must have run → scratch/25_oof_visit.csv + scratch/25_test_visit.csv

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
import xgboost as xgb

print("Loading data...")
X_train_raw = pd.read_csv("data/X_train.csv")
y_train_df = pd.read_csv("data/y_train.csv")
X_test_raw = pd.read_csv("data/X_test.csv")
visits = X_train_raw.merge(y_train_df, on="Index")
y = visits["target"].values
groups = visits["patient_id"]

# Load transformer OOF from exp 25
oof_25_df = pd.read_csv("scratch/25_oof_visit.csv")
test_25_df = pd.read_csv("scratch/25_test_visit.csv")
oof_25 = X_train_raw[["Index"]].merge(oof_25_df, on="Index", how="left")["pred_25"].values
test_25 = X_test_raw[["Index"]].merge(test_25_df, on="Index", how="left")["pred_25"].values

valid_mask = ~np.isnan(oof_25)
rmse_25_alone = float(np.sqrt(np.mean((y[valid_mask] - oof_25[valid_mask]) ** 2)))
print(f"  Transformer-25 OOF RMSE: {rmse_25_alone:.4f}")


# ── Full exp-12 feature engineering ──────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Exact exp-12 feature set — proven 4.58 RMSE with LGBM."""
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


print("\nEngineering GBM features (full exp-12 set)...")
X_eng = engineer_features(visits)
X_num = X_eng.copy()
for col in ["cohort", "gene", "rater_id"]:
    if col in X_num.columns:
        X_num[col] = X_num[col].cat.codes.astype("float32")
print(f"  Feature count: {X_eng.shape[1]}")

cv = GroupKFold(n_splits=5)
cv_splits = list(cv.split(X_eng, y, groups=groups))

# Full-power GBM params (identical to exp 12)
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

print("\n[26] Collecting GBM OOF predictions (GroupKFold-5)...")
for fold_i, (train_idx, val_idx) in enumerate(cv_splits):
    X_tr_l = X_eng.iloc[train_idx]
    X_vl_l = X_eng.iloc[val_idx].copy()
    for col in ["cohort", "gene", "rater_id"]:
        if col in X_tr_l.columns:
            X_vl_l[col] = X_vl_l[col].cat.set_categories(X_tr_l[col].cat.categories)

    m_lgbm = lgb.LGBMRegressor(**lgbm_params)
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
valid = ~np.isnan(oof_25)
y_v = y[valid]
oof_stack = np.column_stack([oof_25[valid], oof_lgbm[valid], oof_xgb[valid], oof_hgbr[valid]])

meta = Ridge(alpha=0.1)
meta.fit(oof_stack, y_v)
meta_oof = meta.predict(oof_stack)
rmse_meta = float(np.sqrt(np.mean((y_v - meta_oof) ** 2)))
print(f"\n  Ridge meta (T25+LGBM+XGB+HGBR) OOF RMSE: {rmse_meta:.4f}")
print(f"  Weights: T={meta.coef_[0]:.3f}  LGBM={meta.coef_[1]:.3f}  "
      f"XGB={meta.coef_[2]:.3f}  HGBR={meta.coef_[3]:.3f}")

# ── Fit final GBM models on full training data ─────────────────────────────────
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
X_test_eng_a = X_test_eng.reindex(columns=X_eng.columns, fill_value=np.nan)
X_test_num_a = X_test_num.reindex(columns=X_num.columns, fill_value=np.nan)

final_lgbm = lgb.LGBMRegressor(**lgbm_params)
final_lgbm.fit(X_eng, y)
final_xgb = xgb.XGBRegressor(**xgb_params)
final_xgb.fit(X_num, y)
final_hgbr = HistGradientBoostingRegressor(**hgbr_params)
final_hgbr.fit(X_num, y)

t_lgbm = final_lgbm.predict(X_test_eng_a)
t_xgb = final_xgb.predict(X_test_num_a)
t_hgbr = final_hgbr.predict(X_test_num_a)

# ── Build test submissions ─────────────────────────────────────────────────────
print("\n[Submission] Building test predictions...")
Path("submissions").mkdir(exist_ok=True)

test_25_safe = test_25.copy()
nan_test = np.isnan(test_25_safe)
if nan_test.any():
    test_25_safe[nan_test] = (t_lgbm + t_xgb)[nan_test] / 2
    print(f"  WARNING: {nan_test.sum()} NaN transformer test rows → GBM fallback")

# Ridge meta — main submission
test_stack = np.column_stack([test_25_safe, t_lgbm, t_xgb, t_hgbr])
test_meta_preds = meta.predict(test_stack)
sub_meta = X_test_raw[["Index"]].copy()
sub_meta["target"] = test_meta_preds
sub_meta.to_csv("submissions/26_final_blend_v2.csv", index=False)
print("  Wrote submissions/26_final_blend_v2.csv  ← BEST SUBMISSION")

# Transformer alone — in case blend doesn't help
sub_t = X_test_raw[["Index"]].copy()
sub_t["target"] = test_25_safe
sub_t.to_csv("submissions/26_transformer_alone.csv", index=False)
print("  Wrote submissions/26_transformer_alone.csv  (transformer only backup)")

print(f"\n{'='*60}")
print("RESULTS SUMMARY")
print(f"  12 GBM full ensemble:              4.58")
print(f"  25 Transformer (trend, 150ep):     {rmse_25_alone:.4f}")
print(f"  26 LGBM (full features):           {rmse_lgbm:.4f}")
print(f"  26 Ridge meta (T25+L+X+H):         {rmse_meta:.4f}  ← TARGET: <= 2.00")
print(f"  In-sample floor (per-pat linear):  1.61")
print(f"{'='*60}")
print("\nBest submission: submissions/26_final_blend_v2.csv")
