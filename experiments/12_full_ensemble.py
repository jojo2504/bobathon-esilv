# %% [markdown]
# # Experiment 12 — Full-Power Ensemble: max-depth LGBM + XGB + stacking
#
# Learnings from 09/10: per-patient target leakage is impossible at test time
# since test patients are fully held out.  The signal ceiling from X-only
# features (off/on aggregates) is well-captured by gradient boosting.
#
# This experiment pushes quality by:
#   1. Full-power LightGBM (3000 trees, num_leaves=255, lr=0.015)
#   2. Full-power XGBoost  (2000 trees, max_depth=8, lr=0.02)
#   3. Deeper HGBR         (500 trees, max_depth=8)
#   4. Ridge meta-learner  trained on OOF stack of the three above
#   5. Additional features: visit rank, z-score within patient

# %%
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import xgboost as xgb
from skore import evaluate

from parkinson.hub import get_project

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data...")
X_train_raw = pd.read_csv("data/X_train.csv")
y_train_df  = pd.read_csv("data/y_train.csv")
X_test_raw  = pd.read_csv("data/X_test.csv")
visits = X_train_raw.merge(y_train_df, on="Index")
print(f"  train: {visits.shape}  test: {X_test_raw.shape}")


# ── Feature engineering ───────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extended feature set (08) plus visit rank and within-patient z-scores."""
    out = df.copy()

    for col in ["cohort", "gene", "rater_id"]:
        if col in out.columns:
            out[col] = out[col].astype("category")

    out["years_since_dx"] = out["age"] - out["age_at_diagnosis"]

    pat = (
        out.groupby("patient_id", sort=False)
        .agg(
            pat_mean_off   =("off",  "mean"),
            pat_mean_on    =("on",   "mean"),
            pat_std_off    =("off",  "std"),
            pat_std_on     =("on",   "std"),
            pat_mean_ledd  =("ledd", "mean"),
            pat_std_ledd   =("ledd", "std"),
            pat_median_off =("off",  "median"),
            pat_median_on  =("on",   "median"),
            pat_min_off    =("off",  "min"),
            pat_max_off    =("off",  "max"),
            pat_min_on     =("on",   "min"),
            pat_max_on     =("on",   "max"),
            pat_n_visits   =("age",  "count"),
            pat_age_range  =("age",  lambda x: x.max() - x.min()),
            pat_mean_age   =("age",  "mean"),
            pat_mean_dx_age=("age_at_diagnosis", "mean"),
        )
        .reset_index()
    )
    out = out.merge(pat, on="patient_id", how="left")

    # Motor gap
    out["off_minus_on"]  = out["off"] - out["on"]
    out["mean_motor"]    = out[["off", "on"]].mean(axis=1)
    out["best_motor"]    = out[["off", "on"]].max(axis=1)
    out["worst_motor"]   = out[["off", "on"]].min(axis=1)

    # Deviation from patient mean/median
    out["off_vs_pat_mean"]   = out["off"] - out["pat_mean_off"]
    out["on_vs_pat_mean"]    = out["on"]  - out["pat_mean_on"]
    out["off_vs_pat_median"] = out["off"] - out["pat_median_off"]
    out["on_vs_pat_median"]  = out["on"]  - out["pat_median_on"]
    out["off_range_pct"]     = (out["off"] - out["pat_min_off"]) / (
        out["pat_max_off"] - out["pat_min_off"] + 1e-6
    )

    # Within-patient z-score of off and on
    out["off_z_pat"] = out["off_vs_pat_mean"] / (out["pat_std_off"] + 1e-6)
    out["on_z_pat"]  = out["on_vs_pat_mean"]  / (out["pat_std_on"]  + 1e-6)

    # PK timing — OFF
    toff = out["time_since_intake_off"].fillna(out["time_since_intake_off"].median())
    out["off_pk_weight"]  = 1.0 / (1.0 + np.exp(-0.3 * (toff - 12.0)))
    out["off_debiased"]   = out["off"] * out["off_pk_weight"]

    # PK timing — ON
    ton = out["time_since_intake_on"].fillna(out["time_since_intake_on"].median())
    out["on_pk_weight"]   = 1.0 / (1.0 + np.exp(0.5 * (ton - 2.0)))
    out["on_adjusted"]    = out["on"] * out["on_pk_weight"]

    # Ratio / interaction features
    out["ledd_per_year"] = out["ledd"] / (out["years_since_dx"].clip(lower=0.1))
    out["off_on_ratio"]  = out["off"] / (out["on"] + 1.0)
    out["ledd_x_off"]    = out["ledd"].fillna(0) * out["off"].fillna(0)
    out["age_x_off"]     = out["age"] * out["off"].fillna(0)

    # Visit rank within patient (sorted by age)
    out["visit_rank"] = (
        out.groupby("patient_id")["age"]
        .rank(method="first")
        .astype(float)
    )
    out["visit_rank_pct"] = out["visit_rank"] / out["pat_n_visits"]

    drop_cols = ["Index", "patient_id"]
    if "target" in out.columns:
        drop_cols.append("target")
    return out.drop(columns=drop_cols)


# ── Build feature matrices ─────────────────────────────────────────────────────
print("\nEngineering features...")
X_eng = engineer_features(visits)
y     = visits["target"].values
groups = visits["patient_id"]
print(f"  Feature matrix: {X_eng.shape}")

# Numeric copy for XGB / HGBR / Ridge
X_num = X_eng.copy()
for col in ["cohort", "gene", "rater_id"]:
    if col in X_num.columns:
        X_num[col] = X_num[col].cat.codes.astype("float32")

cv = GroupKFold(n_splits=5)
cv_splits = list(cv.split(X_eng, y, groups=groups))

# ── Model definitions ─────────────────────────────────────────────────────────
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
oof_lgbm  = np.zeros(N)
oof_xgb   = np.zeros(N)
oof_hgbr  = np.zeros(N)

# ── Collect OOF predictions ───────────────────────────────────────────────────
print("\n[12] Collecting OOF predictions (GroupKFold-5)...")
for fold_i, (train_idx, val_idx) in enumerate(cv_splits):
    # LightGBM
    m_lgbm = lgb.LGBMRegressor(**lgbm_params)
    X_tr_l = X_eng.iloc[train_idx]
    X_vl_l = X_eng.iloc[val_idx].copy()
    for col in ["cohort", "gene", "rater_id"]:
        if col in X_tr_l.columns:
            X_vl_l[col] = X_vl_l[col].cat.set_categories(X_tr_l[col].cat.categories)
    m_lgbm.fit(X_tr_l, y[train_idx])
    oof_lgbm[val_idx] = m_lgbm.predict(X_vl_l)

    # XGBoost
    m_xgb = xgb.XGBRegressor(**xgb_params)
    m_xgb.fit(X_num.iloc[train_idx], y[train_idx])
    oof_xgb[val_idx] = m_xgb.predict(X_num.iloc[val_idx])

    # HGBR
    m_hgbr = HistGradientBoostingRegressor(**hgbr_params)
    m_hgbr.fit(X_num.iloc[train_idx], y[train_idx])
    oof_hgbr[val_idx] = m_hgbr.predict(X_num.iloc[val_idx])

    r_l = np.sqrt(np.mean((y[val_idx] - oof_lgbm[val_idx])**2))
    r_x = np.sqrt(np.mean((y[val_idx] - oof_xgb[val_idx])**2))
    r_h = np.sqrt(np.mean((y[val_idx] - oof_hgbr[val_idx])**2))
    print(f"  Fold {fold_i+1}: LGBM {r_l:.4f}  XGB {r_x:.4f}  HGBR {r_h:.4f}")

rmse_lgbm = np.sqrt(np.mean((y - oof_lgbm)**2))
rmse_xgb  = np.sqrt(np.mean((y - oof_xgb)**2))
rmse_hgbr = np.sqrt(np.mean((y - oof_hgbr)**2))
print(f"\n  OOF: LGBM={rmse_lgbm:.4f}  XGB={rmse_xgb:.4f}  HGBR={rmse_hgbr:.4f}")

# ── Ridge meta-learner ────────────────────────────────────────────────────────
print("\n[Meta] Ridge stacking on OOF...")
oof_stack = np.column_stack([oof_lgbm, oof_xgb, oof_hgbr])
meta = Ridge(alpha=0.1, fit_intercept=True)
meta.fit(oof_stack, y)
meta_oof = meta.predict(oof_stack)
rmse_meta = np.sqrt(np.mean((y - meta_oof)**2))
print(f"  Meta OOF RMSE: {rmse_meta:.4f}")
print(f"  Weights: LGBM={meta.coef_[0]:.3f}  XGB={meta.coef_[1]:.3f}  HGBR={meta.coef_[2]:.3f}  intercept={meta.intercept_:.3f}")

# Simple blends for comparison
blend_lx    = 0.4*oof_lgbm + 0.6*oof_xgb
blend_equal = (oof_lgbm + oof_xgb + oof_hgbr) / 3
print(f"  40/60 LGBM+XGB blend: {np.sqrt(np.mean((y-blend_lx)**2)):.4f}")
print(f"  Equal 3-way blend:    {np.sqrt(np.mean((y-blend_equal)**2)):.4f}")

# ── Push Hub report ───────────────────────────────────────────────────────────
print("\n[Push] Hub report (LightGBM)...")
lgbm_report = lgb.LGBMRegressor(**lgbm_params)
rep_12 = evaluate(lgbm_report, X_eng, y, splitter=cv_splits)
hub_rmse = float(rep_12.metrics.rmse().iloc[:, 0].mean())
print(f"  Hub LGBM RMSE: {hub_rmse:.4f}")

project = get_project()
project.put("12_full_ensemble", rep_12)
print("  Pushed '12_full_ensemble' → Hub")

# ── Submission: Ridge-stacked ensemble ────────────────────────────────────────
print("\n[Submission] Fitting final models on all training data...")
X_test_eng = engineer_features(X_test_raw)
X_test_num = X_test_eng.copy()
for col in ["cohort", "gene", "rater_id"]:
    if col in X_test_num.columns:
        X_test_num[col] = X_test_num[col].cat.set_categories(
            X_eng[col].cat.categories
        ).cat.codes.astype("float32")
X_test_eng = X_test_eng.reindex(columns=X_eng.columns, fill_value=np.nan)
X_test_num = X_test_num.reindex(columns=X_num.columns, fill_value=np.nan)

final_lgbm = lgb.LGBMRegressor(**lgbm_params)
final_lgbm.fit(X_eng, y)

final_xgb = xgb.XGBRegressor(**xgb_params)
final_xgb.fit(X_num, y)

final_hgbr = HistGradientBoostingRegressor(**hgbr_params)
final_hgbr.fit(X_num, y)

t_lgbm = final_lgbm.predict(X_test_eng)
t_xgb  = final_xgb.predict(X_test_num)
t_hgbr = final_hgbr.predict(X_test_num)

test_stack  = np.column_stack([t_lgbm, t_xgb, t_hgbr])
test_meta   = meta.predict(test_stack)
test_blend  = 0.4 * t_lgbm + 0.6 * t_xgb

Path("submissions").mkdir(exist_ok=True)

# Meta-stacked submission
sub_meta = X_test_raw[["Index"]].copy()
sub_meta["target"] = test_meta
sub_meta.to_csv("submissions/12_stacked.csv", index=False)
print(f"  Wrote submissions/12_stacked.csv")

# LGBM-only for reference
sub_lgbm = X_test_raw[["Index"]].copy()
sub_lgbm["target"] = t_lgbm
sub_lgbm.to_csv("submissions/12_lgbm.csv", index=False)
print(f"  Wrote submissions/12_lgbm.csv")

print(f"\n{'='*60}")
print("RESULTS SUMMARY")
print(f"  08_blend (prior best):      4.91")
print(f"  12 LightGBM OOF:            {rmse_lgbm:.4f}")
print(f"  12 XGBoost OOF:             {rmse_xgb:.4f}")
print(f"  12 HGBR OOF:                {rmse_hgbr:.4f}")
print(f"  12 Ridge meta (stacked):    {rmse_meta:.4f}")
print(f"{'='*60}")
print(f"\nBest submission: submissions/12_stacked.csv")
print(f"Hub URL: https://skore.probabl.ai/degensunite/bobathon-esilv/cross-validations/...")
