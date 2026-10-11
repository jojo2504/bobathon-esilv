# %% [markdown]
# # Experiment 19 — Patient-mean predictor: the true ceiling for X-only models
#
# Key insight from EDA: per-patient mean(off) has r=0.93 with per-patient
# mean(target). If we could perfectly know each patient's true-OFF level,
# we would predict it directly from their off/on history.
#
# This experiment builds the strongest possible "no-training" predictor:
#   1. For each visit, predict = patient's own mean(off) adjusted by a
#      linear correction learned via Ridge on patient-level aggregates only.
#   2. Evaluated with GroupKFold-5 — each fold's train patients provide the
#      aggregates; val patients are fully held out (no leakage).
#
# This is the honest ceiling for any model that only uses X features.

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ── Load ──────────────────────────────────────────────────────────────────────
X_train_raw = pd.read_csv("data/X_train.csv")
y_train_df  = pd.read_csv("data/y_train.csv")
X_test_raw  = pd.read_csv("data/X_test.csv")
visits = X_train_raw.merge(y_train_df, on="Index")
y = visits["target"].values
groups = visits["patient_id"]

print(f"Train: {visits.shape}  Test: {X_test_raw.shape}")

# ── Feature engineering (patient-level aggregates per visit row) ──────────────
def add_patient_features(df: pd.DataFrame) -> pd.DataFrame:
    """All patient-level aggregate features computed from the X columns."""
    out = df.copy()
    out["years_since_dx"] = out["age"] - out["age_at_diagnosis"]

    pat = (
        out.groupby("patient_id", sort=False)
        .agg(
            pat_mean_off    =("off",  "mean"),
            pat_mean_on     =("on",   "mean"),
            pat_std_off     =("off",  "std"),
            pat_std_on      =("on",   "std"),
            pat_median_off  =("off",  "median"),
            pat_median_on   =("on",   "median"),
            pat_min_off     =("off",  "min"),
            pat_max_off     =("off",  "max"),
            pat_min_on      =("on",   "min"),
            pat_max_on      =("on",   "max"),
            pat_mean_ledd   =("ledd", "mean"),
            pat_std_ledd    =("ledd", "std"),
            pat_n_visits    =("age",  "count"),
            pat_age_range   =("age",  lambda x: x.max() - x.min()),
            pat_mean_age    =("age",  "mean"),
        )
        .reset_index()
    )
    out = out.merge(pat, on="patient_id", how="left")

    # Single-visit features
    out["off_minus_on"]     = out["off"]  - out["on"]
    out["off_on_ratio"]     = out["off"]  / (out["on"] + 1.0)
    out["ledd_per_year"]    = out["ledd"] / (out["years_since_dx"].clip(lower=0.1))
    out["off_vs_pat_mean"]  = out["off"]  - out["pat_mean_off"]
    out["on_vs_pat_mean"]   = out["on"]   - out["pat_mean_on"]
    out["off_z_pat"]        = out["off_vs_pat_mean"] / (out["pat_std_off"].fillna(0) + 1e-6)
    out["on_z_pat"]         = out["on_vs_pat_mean"]  / (out["pat_std_on"].fillna(0)  + 1e-6)

    # PK timing
    toff = out["time_since_intake_off"].fillna(out["time_since_intake_off"].median())
    out["off_pk_weight"] = 1.0 / (1.0 + np.exp(-0.3 * (toff - 12.0)))
    ton  = out["time_since_intake_on"].fillna(out["time_since_intake_on"].median())
    out["on_pk_weight"]  = 1.0 / (1.0 + np.exp(0.5  * (ton  - 2.0)))

    out["off_debiased"]  = out["off"] * out["off_pk_weight"]
    out["on_adjusted"]   = out["on"]  * out["on_pk_weight"]

    # Visit rank
    out["visit_rank"]     = out.groupby("patient_id")["age"].rank(method="first")
    out["visit_rank_pct"] = out["visit_rank"] / out["pat_n_visits"]

    return out


FEAT_COLS = [
    "pat_mean_off", "pat_mean_on", "pat_std_off", "pat_std_on",
    "pat_median_off", "pat_median_on", "pat_min_off", "pat_max_off",
    "pat_min_on", "pat_max_on", "pat_mean_ledd", "pat_std_ledd",
    "pat_n_visits", "pat_age_range", "pat_mean_age",
    "off", "on", "ledd", "age", "years_since_dx",
    "off_minus_on", "off_on_ratio", "ledd_per_year",
    "off_vs_pat_mean", "on_vs_pat_mean", "off_z_pat", "on_z_pat",
    "off_pk_weight", "on_pk_weight", "off_debiased", "on_adjusted",
    "visit_rank", "visit_rank_pct",
    "sexM",
]

# ── GroupKFold CV ─────────────────────────────────────────────────────────────
cv = GroupKFold(n_splits=5)
cv_splits = list(cv.split(visits, y, groups=groups))

oof_ridge = np.zeros(len(y))
oof_patmean = np.zeros(len(y))  # naive: just predict pat_mean_off for every visit

print("\n[19] Patient-mean + Ridge  GroupKFold-5 CV")
for fold_i, (train_idx, val_idx) in enumerate(cv_splits):
    tr = add_patient_features(visits.iloc[train_idx].copy())
    vl_raw = visits.iloc[val_idx].copy()

    # For val patients: compute their own aggregates from their own X rows
    # (no target used — patient_id groups are fully held out from train)
    vl = add_patient_features(vl_raw)

    X_tr = tr[FEAT_COLS].fillna(0).values
    X_vl = vl[FEAT_COLS].fillna(0).values
    y_tr = y[train_idx]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=1.0)),
    ])
    model.fit(X_tr, y_tr)
    oof_ridge[val_idx] = model.predict(X_vl)

    # Naive: pat_mean_off as direct prediction (val patients know their own off history)
    oof_patmean[val_idx] = vl["pat_mean_off"].fillna(vl["off"].fillna(y_tr.mean())).values

    r_r = np.sqrt(np.mean((y[val_idx] - oof_ridge[val_idx])**2))
    r_p = np.sqrt(np.mean((y[val_idx] - oof_patmean[val_idx])**2))
    print(f"  Fold {fold_i+1}: Ridge={r_r:.4f}  PatMean={r_p:.4f}")

rmse_ridge   = np.sqrt(np.mean((y - oof_ridge)**2))
rmse_patmean = np.sqrt(np.mean((y - oof_patmean)**2))
print(f"\n  OOF Ridge:   {rmse_ridge:.4f}")
print(f"  OOF PatMean: {rmse_patmean:.4f}")
print(f"  (GBM best: 4.59  |  Transformer 50ep: 3.94)")

# Save OOF for blending
Path("scratch").mkdir(exist_ok=True)
oof_df = visits[["Index"]].copy()
oof_df["pred_19_ridge"]   = oof_ridge
oof_df["pred_19_patmean"] = oof_patmean
oof_df.to_csv("scratch/19_oof.csv", index=False)

# ── Test predictions ───────────────────────────────────────────────────────────
print("\n[Final] Fitting on full train...")
full_feats = add_patient_features(visits)
X_full = full_feats[FEAT_COLS].fillna(0).values
final_model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
final_model.fit(X_full, y)

test_feats = add_patient_features(X_test_raw)
X_test = test_feats[FEAT_COLS].fillna(0).values
test_preds = final_model.predict(X_test)

Path("submissions").mkdir(exist_ok=True)
sub = X_test_raw[["Index"]].copy()
sub["target"] = test_preds
sub.to_csv("submissions/19_ridge.csv", index=False)

# Also save test predictions for blending
test_oof_df = X_test_raw[["Index"]].copy()
test_oof_df["pred_19_ridge"] = test_preds
test_oof_df.to_csv("scratch/19_test.csv", index=False)

print(f"  Wrote submissions/19_ridge.csv  ({len(sub)} rows)")
print(f"\n{'='*60}")
print(f"  19 Ridge OOF RMSE:   {rmse_ridge:.4f}")
print(f"  19 PatMean OOF RMSE: {rmse_patmean:.4f}")
print(f"{'='*60}")
