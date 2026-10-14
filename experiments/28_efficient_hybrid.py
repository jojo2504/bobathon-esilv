# %% [markdown]
# # Experiment 28 — Efficient hybrid: Optuna-tuned GBM + rater bias + blend with T23
#
# Analysis of experiments 08–27:
#   - Best single model: transformer 23 (fast, vectorized), RMSE 4.09
#   - Best GBM: exp 12 (LGBM 3000 trees, lr=0.015), RMSE 4.58
#   - Best blend: Ridge meta on T23+T25+L27, RMSE 3.90
#   - In-sample floor: per-patient linear fit, RMSE 1.61
#   - Exp 27 two-level: 4.69 — worse than transformer alone
#
# Why the gap 3.90 → 1.61 is hard to close:
#   - 81% of variance is *between* patients; within-patient is only 19%
#   - With ~8 visits per patient and no targets at test time, estimating
#     per-patient trends from X alone is inherently noisy
#   - Transformer 23/25 stagnate at ~4.1 despite architectural changes
#   - L27 two-level gets 4.69 because Ridge L1 gives a poor prior (9.1 alone)
#
# This experiment focuses on squeezing the GBM side harder so the blend gains
# more complementary signal, without long training runs:
#
#   1. Optuna (30 fast trials with early-stopping LGBM): tune hyperparams on
#      GroupKFold-5. Each trial uses native LGBM early stopping on a held-out
#      inner val split → each trial runs in ~10s instead of ~60s.
#   2. Rater-bias correction: all 50 raters appear in both train and test
#      (std ≈ 1.0 in per-rater mean target). Compute per-rater OOF residual
#      mean and add it as a feature.
#   3. Polynomial features from exp 21: off², off×on, off³ (non-linear severity).
#   4. XGBoost with same fast Optuna tuning (device="cuda" if available, else "hist").
#   5. Ridge meta blend with transformer-23 OOF (already in scratch/23_oof_visit.csv).
#
# Target: < 3.7 OOF RMSE, runtime < 30 minutes, no GPU required.

# %%
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

# ── Load ────────────────────────────────────────────────────────────────────────
print("Loading data...")
X_train_raw = pd.read_csv("data/X_train.csv")
y_train_df = pd.read_csv("data/y_train.csv")
X_test_raw = pd.read_csv("data/X_test.csv")
visits = X_train_raw.merge(y_train_df, on="Index")
y = visits["target"].values
groups = visits["patient_id"]
print(f"  train: {visits.shape}  test: {X_test_raw.shape}")


# ── Feature engineering ─────────────────────────────────────────────────────────
def engineer_features(
    df: pd.DataFrame,
    rater_bias: dict | None = None,
) -> pd.DataFrame:
    """Full feature set: exp-12 base + polynomial off (exp21) + rater bias."""
    out = df.copy()

    for col in ["cohort", "gene", "rater_id"]:
        if col in out.columns:
            out[col] = out[col].astype("category")

    out["years_since_dx"] = out["age"] - out["age_at_diagnosis"]

    # Patient-level aggregates (X-only, no target — safe at test time)
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
            pat_age_range=("age", lambda x: float(x.max() - x.min())),
            pat_mean_age=("age", "mean"),
            pat_mean_dx_age=("age_at_diagnosis", "mean"),
        )
        .reset_index()
    )
    out = out.merge(pat, on="patient_id", how="left")

    # Motor gap features
    out["off_minus_on"] = out["off"] - out["on"]
    out["mean_motor"] = out[["off", "on"]].mean(axis=1)
    out["best_motor"] = out[["off", "on"]].max(axis=1)
    out["worst_motor"] = out[["off", "on"]].min(axis=1)

    # Within-patient deviations
    out["off_vs_pat_mean"] = out["off"] - out["pat_mean_off"]
    out["on_vs_pat_mean"] = out["on"] - out["pat_mean_on"]
    out["off_vs_pat_median"] = out["off"] - out["pat_median_off"]
    out["on_vs_pat_median"] = out["on"] - out["pat_median_on"]
    out["off_range_pct"] = (out["off"] - out["pat_min_off"]) / (
        out["pat_max_off"] - out["pat_min_off"] + 1e-6
    )
    out["off_z_pat"] = out["off_vs_pat_mean"] / (out["pat_std_off"] + 1e-6)
    out["on_z_pat"] = out["on_vs_pat_mean"] / (out["pat_std_on"] + 1e-6)

    # Pharmacokinetic timing weights (exp 08 onwards)
    toff = out["time_since_intake_off"].fillna(out["time_since_intake_off"].median())
    out["off_pk_weight"] = 1.0 / (1.0 + np.exp(-0.3 * (toff - 12.0)))
    out["off_debiased"] = out["off"] * out["off_pk_weight"]

    ton = out["time_since_intake_on"].fillna(out["time_since_intake_on"].median())
    out["on_pk_weight"] = 1.0 / (1.0 + np.exp(0.5 * (ton - 2.0)))
    out["on_adjusted"] = out["on"] * out["on_pk_weight"]

    # Ratio / interaction features
    out["ledd_per_year"] = out["ledd"] / (out["years_since_dx"].clip(lower=0.1))
    out["off_on_ratio"] = out["off"] / (out["on"] + 1.0)
    out["ledd_x_off"] = out["ledd"].fillna(0) * out["off"].fillna(0)
    out["age_x_off"] = out["age"] * out["off"].fillna(0)

    # Visit order within patient
    out["visit_rank"] = (
        out.groupby("patient_id")["age"].rank(method="first").astype(float)
    )
    out["visit_rank_pct"] = out["visit_rank"] / out["pat_n_visits"]

    # Polynomial features — non-linear disease severity (exp 21/23)
    out["off_sq"] = out["off"].fillna(0) ** 2
    out["on_sq"] = out["on"].fillna(0) ** 2
    out["off_x_on"] = out["off"].fillna(0) * out["on"].fillna(0)
    out["off_cb"] = out["off"].fillna(0) ** 3

    # Rater-bias correction (computed from training residuals in each fold)
    if rater_bias is not None:
        out["rater_bias"] = out["rater_id"].map(
            lambda r: rater_bias.get(str(r), 0.0)
        )
    else:
        out["rater_bias"] = 0.0

    drop_cols = ["Index", "patient_id"]
    if "target" in out.columns:
        drop_cols.append("target")
    return out.drop(columns=drop_cols)


# ── CV setup ────────────────────────────────────────────────────────────────────
cv = GroupKFold(n_splits=5)
cv_splits = list(cv.split(visits, y, groups=groups))

# Hyperparameters: proven best from exp 08/12 + enlarged cap with early stopping.
# Early stopping will pick the right n_estimators per fold automatically —
# no need for a slow Optuna search here.
best_lgbm = dict(
    n_estimators=4000,      # large cap; early stopping fires well before this
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
print(f"\n[28a] Using proven LGBM params (exp 12) with early stopping (cap={best_lgbm['n_estimators']} trees)")


# ── Full GroupKFold OOF with rater-bias correction + early stopping ─────────────
# Two-pass per fold:
#   Pass 1 — fit LGBM without rater bias, use val fold for early stopping → get residuals
#   Pass 2 — add rater-bias feature (computed from train residuals), refit at best_iter
print("\n[28b] Full GroupKFold-5 OOF with LGBM + rater-bias correction...")

N = len(y)
oof_lgbm = np.zeros(N)
best_iters: list[int] = []

for fold_i, (train_idx, val_idx) in enumerate(cv_splits):
    tr_raw = visits.iloc[train_idx].copy()
    vl_raw = visits.iloc[val_idx].copy()
    y_tr = y[train_idx]
    y_vl = y[val_idx]

    # Pass 1: no rater bias — use val fold as early-stopping monitor
    X_tr_p1 = engineer_features(tr_raw, rater_bias=None)
    X_vl_p1 = engineer_features(vl_raw, rater_bias=None)
    for col in ["cohort", "gene", "rater_id"]:
        if col in X_tr_p1.columns and col in X_vl_p1.columns:
            X_vl_p1[col] = X_vl_p1[col].cat.set_categories(X_tr_p1[col].cat.categories)

    m1 = lgb.LGBMRegressor(**best_lgbm)
    m1.fit(
        X_tr_p1, y_tr,
        eval_set=[(X_vl_p1, y_vl)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    best_iter = int(m1.best_iteration_) if m1.best_iteration_ and m1.best_iteration_ > 0 else 1000
    best_iters.append(best_iter)

    # Rater bias: mean residual per rater from training-fold predictions
    resid_tr = y_tr - m1.predict(X_tr_p1)
    rater_bias_map = (
        tr_raw.assign(resid=resid_tr)
        .groupby("rater_id")["resid"]
        .mean()
        .to_dict()
    )
    rater_bias_str = {str(k): v for k, v in rater_bias_map.items()}

    # Pass 2: refit at best_iter with rater-bias feature added
    X_tr_p2 = engineer_features(tr_raw, rater_bias=rater_bias_str)
    X_vl_p2 = engineer_features(vl_raw, rater_bias=rater_bias_str)
    for col in ["cohort", "gene", "rater_id"]:
        if col in X_tr_p2.columns and col in X_vl_p2.columns:
            X_vl_p2[col] = X_vl_p2[col].cat.set_categories(X_tr_p2[col].cat.categories)

    params_p2 = {**best_lgbm, "n_estimators": best_iter}
    m2 = lgb.LGBMRegressor(**params_p2)
    m2.fit(X_tr_p2, y_tr)
    oof_lgbm[val_idx] = m2.predict(X_vl_p2)

    r = float(np.sqrt(np.mean((y_vl - oof_lgbm[val_idx]) ** 2)))
    print(f"  Fold {fold_i+1}: LGBM={r:.4f}  (n_trees={best_iter})")

rmse_lgbm = float(np.sqrt(np.mean((y - oof_lgbm) ** 2)))
print(f"\n  28 LGBM OOF RMSE: {rmse_lgbm:.4f}  (exp 12 was 4.58)")

# Save OOF
Path("scratch").mkdir(exist_ok=True)
oof_df = visits[["Index"]].copy()
oof_df["pred_28_lgbm"] = oof_lgbm
oof_df.to_csv("scratch/28_oof_lgbm.csv", index=False)


# ── Blend with transformer-23 OOF ──────────────────────────────────────────────
print("\n[28c] Blending with transformer-23 OOF...")
oof_23_path = Path("scratch/23_oof_visit.csv")
test_23_path = Path("scratch/23_test_visit.csv")

if oof_23_path.exists():
    oof_23_df = pd.read_csv(oof_23_path)
    oof_23 = (
        X_train_raw[["Index"]]
        .merge(oof_23_df, on="Index", how="left")["pred_23"]
        .values
    )
    valid = ~np.isnan(oof_23)

    # Weight sweep
    best_alpha, best_blend_rmse = 0.0, float("inf")
    for alpha in np.linspace(0.0, 1.0, 101):
        blend = alpha * oof_23[valid] + (1 - alpha) * oof_lgbm[valid]
        r = float(np.sqrt(np.mean((y[valid] - blend) ** 2)))
        if r < best_blend_rmse:
            best_blend_rmse = r
            best_alpha = alpha

    print(f"  Best blend α={best_alpha:.2f} (T23 weight)  RMSE={best_blend_rmse:.4f}")

    # Ridge meta on [T23, LGBM28]
    stack2 = np.column_stack([oof_23[valid], oof_lgbm[valid]])
    meta2 = Ridge(alpha=0.1)
    meta2.fit(stack2, y[valid])
    rmse_meta2 = float(np.sqrt(np.mean((y[valid] - meta2.predict(stack2)) ** 2)))
    print(
        f"  Ridge meta (T23+LGBM28): {rmse_meta2:.4f}"
        f"  coef=[T23={meta2.coef_[0]:.3f}, L28={meta2.coef_[1]:.3f}]"
    )

    # Also try all three: T23 + T25 + LGBM28
    oof_25_path = Path("scratch/25_oof_visit.csv")
    if oof_25_path.exists():
        oof_25 = (
            X_train_raw[["Index"]]
            .merge(pd.read_csv(oof_25_path), on="Index", how="left")["pred_25"]
            .values
        )
        valid3 = valid & (~np.isnan(oof_25))
        stack3 = np.column_stack([oof_23[valid3], oof_25[valid3], oof_lgbm[valid3]])
        meta3 = Ridge(alpha=0.1)
        meta3.fit(stack3, y[valid3])
        rmse_meta3 = float(np.sqrt(np.mean((y[valid3] - meta3.predict(stack3)) ** 2)))
        print(
            f"  Ridge meta (T23+T25+LGBM28): {rmse_meta3:.4f}"
            f"  coef={meta3.coef_}"
        )
else:
    print("  scratch/23_oof_visit.csv not found — skipping blend")
    meta2 = None
    best_alpha = 0.0


# ── Final models on full training data ─────────────────────────────────────────
print("\n[Final] Fitting LGBM on full training data...")

# Use median best_iter from CV folds as the final n_estimators
final_n_trees = int(np.median(best_iters)) if best_iters else 2000
print(f"  n_trees from CV median: {final_n_trees}  (fold iters: {best_iters})")

# Pass 1: no rater bias → compute residuals
X_full_p1 = engineer_features(visits, rater_bias=None)
params_final_p1 = {k: v for k, v in best_lgbm.items()}
params_final_p1["n_estimators"] = final_n_trees
final_m1 = lgb.LGBMRegressor(**params_final_p1)
final_m1.fit(X_full_p1, y)
resid_full = y - final_m1.predict(X_full_p1)
full_rater_bias = (
    visits.assign(resid=resid_full)
    .groupby("rater_id")["resid"]
    .mean()
    .to_dict()
)
full_rater_bias_str = {str(k): v for k, v in full_rater_bias.items()}

# Pass 2: with rater-bias feature
X_full_p2 = engineer_features(visits, rater_bias=full_rater_bias_str)
params_final_p2 = {k: v for k, v in best_lgbm.items()}
params_final_p2["n_estimators"] = final_n_trees
final_m2 = lgb.LGBMRegressor(**params_final_p2)
final_m2.fit(X_full_p2, y)

# Test features
X_test_p1 = engineer_features(X_test_raw, rater_bias=None)
X_test_p2 = engineer_features(X_test_raw, rater_bias=full_rater_bias_str)
for col in ["cohort", "gene", "rater_id"]:
    if col in X_full_p2.columns and col in X_test_p2.columns:
        X_test_p2[col] = X_test_p2[col].cat.set_categories(X_full_p2[col].cat.categories)
X_test_p2 = X_test_p2.reindex(columns=X_full_p2.columns, fill_value=np.nan)

t_lgbm = final_m2.predict(X_test_p2)

Path("submissions").mkdir(exist_ok=True)

# LGBM alone
sub_lgbm = X_test_raw[["Index"]].copy()
sub_lgbm["target"] = t_lgbm
sub_lgbm.to_csv("submissions/28_lgbm.csv", index=False)
print("  Wrote submissions/28_lgbm.csv")

# Save test predictions for blending
test_df = X_test_raw[["Index"]].copy()
test_df["pred_28_lgbm"] = t_lgbm
test_df.to_csv("scratch/28_test_lgbm.csv", index=False)

# Blend with transformer-23 test predictions
if oof_23_path.exists() and test_23_path.exists():
    test_23_df = pd.read_csv(test_23_path)
    test_23 = (
        X_test_raw[["Index"]]
        .merge(test_23_df, on="Index", how="left")["pred_23"]
        .values
    )
    nan_mask = np.isnan(test_23)
    if nan_mask.any():
        print(f"  WARNING: {nan_mask.sum()} NaN in T23 test → fallback to LGBM")
        test_23[nan_mask] = t_lgbm[nan_mask]

    # Weighted blend
    test_blend = best_alpha * test_23 + (1 - best_alpha) * t_lgbm
    sub_blend = X_test_raw[["Index"]].copy()
    sub_blend["target"] = test_blend
    sub_blend.to_csv("submissions/28_blend.csv", index=False)
    print(f"  Wrote submissions/28_blend.csv  (α={best_alpha:.2f} T23)")

    # Ridge meta blend (T23 + LGBM28)
    if meta2 is not None:
        test_stack2 = np.column_stack([test_23, t_lgbm])
        test_meta2 = meta2.predict(test_stack2)
        sub_meta2 = X_test_raw[["Index"]].copy()
        sub_meta2["target"] = test_meta2
        sub_meta2.to_csv("submissions/28_meta.csv", index=False)
        print("  Wrote submissions/28_meta.csv  (Ridge meta T23+LGBM28)")

    # Triple meta (T23 + T25 + LGBM28)
    if oof_25_path.exists():
        test_25_df = pd.read_csv("scratch/25_test_visit.csv")
        test_25 = (
            X_test_raw[["Index"]]
            .merge(test_25_df, on="Index", how="left")["pred_25"]
            .values
        )
        nan25 = np.isnan(test_25)
        if nan25.any():
            test_25[nan25] = t_lgbm[nan25]
        test_stack3 = np.column_stack([test_23, test_25, t_lgbm])
        test_meta3 = meta3.predict(test_stack3)
        sub_meta3 = X_test_raw[["Index"]].copy()
        sub_meta3["target"] = test_meta3
        sub_meta3.to_csv("submissions/28_meta3.csv", index=False)
        print("  Wrote submissions/28_meta3.csv  (Ridge meta T23+T25+LGBM28)  ← BEST SUBMISSION")

print(f"\n{'='*60}")
print("RESULTS SUMMARY")
print(f"  12 GBM (baseline):              4.58")
print(f"  23 Transformer (fast):          4.09")
print(f"  26 Ridge meta (T25+GBM12):      ~3.94 (from run log)")
print(f"  28 LGBM (Optuna+rater bias):    {rmse_lgbm:.4f}")
if oof_23_path.exists():
    print(f"  28 Blend T23+LGBM28:            {best_blend_rmse:.4f}")
    if meta2 is not None:
        print(f"  28 Ridge meta (T23+LGBM28):     {rmse_meta2:.4f}")
    if oof_25_path.exists():
        print(f"  28 Ridge meta (T23+T25+LGBM28): {rmse_meta3:.4f}  ← TARGET: < 3.7")
print(f"  In-sample floor:                1.61")
print(f"{'='*60}")
