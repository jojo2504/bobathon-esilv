# %% [markdown]
# # Experiment 21 — Rater-bias correction + cohort×gene interactions + polynomial off
#
# Three new signal sources not yet exploited:
#   1. **Rater bias**: each of 50 raters has a systematic offset (some score high,
#      some low). A per-rater OOF mean residual (computed from train fold only)
#      can be added as a feature to debias.
#   2. **Cohort × gene interaction**: the two cohorts (A/B) may have different
#      gene-expression relationships with true OFF. One-hot encode all combinations.
#   3. **Polynomial off**: off² and off×on capture non-linear disease severity.
#      The relationship between biased OFF and true OFF is non-linear near extremes.
#
# Base model: full-power LGBM (same params as exp 12).

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

# ── Load ──────────────────────────────────────────────────────────────────────
X_train_raw = pd.read_csv("data/X_train.csv")
y_train_df  = pd.read_csv("data/y_train.csv")
X_test_raw  = pd.read_csv("data/X_test.csv")
visits = X_train_raw.merge(y_train_df, on="Index")
y = visits["target"].values
groups = visits["patient_id"]


def engineer_features(df: pd.DataFrame, rater_bias: dict | None = None,
                       fit_rater: bool = False, y_resid: np.ndarray | None = None
                       ) -> tuple[pd.DataFrame, dict]:
    """Extended feature set with rater bias, cohort×gene, and polynomial off."""
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
            pat_median_off =("off",  "median"),
            pat_median_on  =("on",   "median"),
            pat_min_off    =("off",  "min"),
            pat_max_off    =("off",  "max"),
            pat_min_on     =("on",   "min"),
            pat_max_on     =("on",   "max"),
            pat_n_visits   =("age",  "count"),
            pat_age_range  =("age",  lambda x: x.max() - x.min()),
            pat_mean_age   =("age",  "mean"),
        )
        .reset_index()
    )
    out = out.merge(pat, on="patient_id", how="left")

    # Standard engineered features
    out["off_minus_on"]      = out["off"]  - out["on"]
    out["off_on_ratio"]      = out["off"]  / (out["on"] + 1.0)
    out["off_vs_pat_mean"]   = out["off"]  - out["pat_mean_off"]
    out["on_vs_pat_mean"]    = out["on"]   - out["pat_mean_on"]
    out["off_z_pat"]         = out["off_vs_pat_mean"] / (out["pat_std_off"] + 1e-6)
    out["on_z_pat"]          = out["on_vs_pat_mean"]  / (out["pat_std_on"]  + 1e-6)
    toff = out["time_since_intake_off"].fillna(out["time_since_intake_off"].median())
    out["off_pk_weight"]     = 1.0 / (1.0 + np.exp(-0.3 * (toff - 12.0)))
    out["off_debiased"]      = out["off"] * out["off_pk_weight"]
    ton  = out["time_since_intake_on"].fillna(out["time_since_intake_on"].median())
    out["on_pk_weight"]      = 1.0 / (1.0 + np.exp(0.5 * (ton - 2.0)))
    out["on_adjusted"]       = out["on"]  * out["on_pk_weight"]
    out["ledd_per_year"]     = out["ledd"] / (out["years_since_dx"].clip(lower=0.1))
    out["ledd_x_off"]        = out["ledd"].fillna(0) * out["off"].fillna(0)
    out["visit_rank"]        = out.groupby("patient_id")["age"].rank(method="first").astype(float)
    out["visit_rank_pct"]    = out["visit_rank"] / out["pat_n_visits"]

    # ── NEW: polynomial off features ──────────────────────────────────────────
    out["off_sq"]            = out["off"].fillna(0) ** 2
    out["on_sq"]             = out["on"].fillna(0)  ** 2
    out["off_x_on"]          = out["off"].fillna(0) * out["on"].fillna(0)
    out["off_cb"]            = out["off"].fillna(0) ** 3  # captures extreme values

    # ── NEW: cohort × gene interaction dummies ─────────────────────────────────
    out["cohort_str"] = out["cohort"].astype(str)
    out["gene_str"]   = out["gene"].astype(str).replace("nan", "unknown")
    for coh in ["A", "B"]:
        for gene in ["GBA+", "LRRK2+", "OTHER+", "No Mutation", "unknown"]:
            col_name = f"coh{coh}_gene_{gene.replace('+','p').replace(' ','_')}"
            out[col_name] = ((out["cohort_str"] == coh) & (out["gene_str"] == gene)).astype("int8")
    out.drop(columns=["cohort_str", "gene_str"], inplace=True)

    # ── NEW: rater bias correction ────────────────────────────────────────────
    if fit_rater and y_resid is not None:
        # Compute per-rater mean residual from training data
        tmp = out[["rater_id"]].copy()
        tmp["resid"] = y_resid
        rater_mean = tmp.groupby("rater_id")["resid"].mean().to_dict()
        rater_bias = rater_mean
    # Apply rater bias (0 for unknown raters)
    if rater_bias is not None:
        out["rater_bias"] = out["rater_id"].map(
            lambda r: rater_bias.get(str(r), 0.0) if rater_bias else 0.0
        )
    else:
        out["rater_bias"] = 0.0

    drop = ["Index", "patient_id"] + (["target"] if "target" in out.columns else [])
    return out.drop(columns=drop), rater_bias or {}


# ── GroupKFold CV ─────────────────────────────────────────────────────────────
cv = GroupKFold(n_splits=5)
cv_splits = list(cv.split(visits, y, groups=groups))

lgbm_params = dict(
    n_estimators=3000, num_leaves=255, learning_rate=0.015,
    min_child_samples=15, colsample_bytree=0.7, subsample=0.8,
    subsample_freq=1, reg_alpha=0.05, reg_lambda=0.3, max_depth=-1,
    random_state=42, n_jobs=-1, verbose=-1,
)

oof_21 = np.zeros(len(y))
print("[21] Rater-bias + cohort×gene + polynomial off  GroupKFold-5")

for fold_i, (train_idx, val_idx) in enumerate(cv_splits):
    tr_raw = visits.iloc[train_idx].copy()
    vl_raw = visits.iloc[val_idx].copy()
    y_tr   = y[train_idx]

    # First pass: fit model without rater bias to get residuals
    X_tr_pass1, _ = engineer_features(tr_raw, rater_bias=None, fit_rater=False)
    X_vl_pass1, _ = engineer_features(vl_raw, rater_bias=None, fit_rater=False)
    for col in ["cohort", "gene", "rater_id"]:
        if col in X_tr_pass1.columns:
            X_vl_pass1[col] = X_vl_pass1[col].cat.set_categories(X_tr_pass1[col].cat.categories)

    m1 = lgb.LGBMRegressor(**lgbm_params)
    m1.fit(X_tr_pass1, y_tr)
    resid_tr = y_tr - m1.predict(X_tr_pass1)

    # Second pass: add rater bias computed from train residuals
    X_tr, rb = engineer_features(tr_raw, rater_bias=None, fit_rater=True, y_resid=resid_tr)
    X_vl, _  = engineer_features(vl_raw, rater_bias=rb,   fit_rater=False)
    for col in ["cohort", "gene", "rater_id"]:
        if col in X_tr.columns:
            X_vl[col] = X_vl[col].cat.set_categories(X_tr[col].cat.categories)

    m2 = lgb.LGBMRegressor(**lgbm_params)
    m2.fit(X_tr, y_tr)
    oof_21[val_idx] = m2.predict(X_vl)

    r = np.sqrt(np.mean((y[val_idx] - oof_21[val_idx]) ** 2))
    print(f"  Fold {fold_i+1}: {r:.4f}")

rmse_21 = np.sqrt(np.mean((y - oof_21) ** 2))
print(f"\n  21 OOF RMSE: {rmse_21:.4f}  (exp 12 was 4.59)")

Path("scratch").mkdir(exist_ok=True)
oof_df = visits[["Index"]].copy()
oof_df["pred_21"] = oof_21
oof_df.to_csv("scratch/21_oof.csv", index=False)

# ── Final model ───────────────────────────────────────────────────────────────
print("\n[Final] Fitting on all training data...")
X_full_pass1, _ = engineer_features(visits, rater_bias=None, fit_rater=False)
m_full1 = lgb.LGBMRegressor(**lgbm_params)
m_full1.fit(X_full_pass1, y)
resid_full = y - m_full1.predict(X_full_pass1)

X_full, rb_full = engineer_features(visits, rater_bias=None, fit_rater=True, y_resid=resid_full)
m_final = lgb.LGBMRegressor(**lgbm_params)
m_final.fit(X_full, y)

X_test_eng, _ = engineer_features(X_test_raw, rater_bias=rb_full, fit_rater=False)
X_test_eng = X_test_eng.reindex(columns=X_full.columns, fill_value=0)
test_preds = m_final.predict(X_test_eng)

Path("submissions").mkdir(exist_ok=True)
sub = X_test_raw[["Index"]].copy()
sub["target"] = test_preds
sub.to_csv("submissions/21_rater_poly.csv", index=False)

test_df = X_test_raw[["Index"]].copy()
test_df["pred_21"] = test_preds
test_df.to_csv("scratch/21_test.csv", index=False)

print(f"  Wrote submissions/21_rater_poly.csv")
print(f"\n{'='*60}")
print(f"  21 OOF RMSE: {rmse_21:.4f}")
print(f"{'='*60}")
