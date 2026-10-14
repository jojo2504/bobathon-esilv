# %% [markdown]
# # Experiment 30 — Per-patient linear fit: le vrai plancher
#
# Insight clé: target = per_patient_linear(age, off) donne RMSE 1.00 in-sample.
# Les leaders Kaggle à 2.05 font exactement ça, honnêtement pour les patients test.
#
# Pourquoi ça marche pour les patients test:
#   - Chaque patient test a 4-12 visites dans X_test avec leurs valeurs off et age
#   - On peut fitter les paramètres (slope_age, slope_off, intercept) de chaque
#     patient test DEPUIS leurs propres visites — sans voir leurs targets
#   - On apprend la relation (slope_age, slope_off, intercept) -> mean_target
#     depuis les patients train, puis on prédit les paramètres des patients test
#
# Architecture:
#
#   Étape 1 — Global learner (Ridge/LGBM):
#     Pour chaque patient, extraire ses paramètres per-patient:
#       - pat_age_slope: coef de age dans régression off~age (progression maladie)
#       - pat_off_slope: coef de off dans régression target~off (estimé depuis X)
#       - pat_intercept: intercept estimé
#     Ces paramètres sont appris depuis X seulement (pas de target) => valables test.
#
#   Étape 2 — Per-visit correction:
#     target_i = L1_pred(patient_i) + L2_correction(off_z_i, visit_rank_i, ...)
#     Le L2 apprend les déviations intra-patient.
#
#   Étape 3 — Blend avec transformers 23+25 (OOF déjà disponibles).

# %%
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

print("Loading data...")
X_train_raw = pd.read_csv("data/X_train.csv")
y_train_df  = pd.read_csv("data/y_train.csv")
X_test_raw  = pd.read_csv("data/X_test.csv")
visits = X_train_raw.merge(y_train_df, on="Index")
y = visits["target"].values
groups = visits["patient_id"]
print(f"  train: {visits.shape}  test: {X_test_raw.shape}")

cv = GroupKFold(n_splits=5)
cv_splits = list(cv.split(visits, y, groups=groups))


# ── Extraction des paramètres per-patient depuis X seulement ────────────────────
def per_patient_params(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque patient, calcule des paramètres de sa trajectoire
    estimables depuis X uniquement (pas de target).

    Retourne un DataFrame avec une ligne par patient.
    """
    rows = []
    for pid, grp in df.groupby("patient_id"):
        ages  = grp["age"].values
        off   = grp["off"].values
        on    = grp["on"].values
        ledd  = grp["ledd"].values
        m_age = float(ages.mean())
        n     = len(grp)

        def fit_slope(y_vals: np.ndarray, x_vals: np.ndarray) -> tuple[float, float]:
            """Retourne (slope, intercept) d'une régression linéaire simple."""
            valid = ~np.isnan(y_vals)
            if valid.sum() < 2:
                return 0.0, float(np.nanmean(y_vals))
            xv = x_vals[valid] - x_vals[valid].mean()
            yv = y_vals[valid]
            slope = float(np.dot(xv, yv) / (np.dot(xv, xv) + 1e-8))
            intercept = float(yv.mean() - slope * x_vals[valid].mean())
            return slope, intercept

        # Slope de off ~ age (progression de la maladie visible)
        off_age_slope, off_age_int = fit_slope(off, ages)
        # Slope de on ~ age
        on_age_slope, on_age_int = fit_slope(on, ages)
        # Valeur prédite de off à l'age moyen du patient
        off_at_mean_age = off_age_int + off_age_slope * m_age

        # Coef de variation de off (signal de variabilité intra-patient)
        off_cv = float(np.nanstd(off) / (np.nanmean(off) + 1e-6))
        on_cv  = float(np.nanstd(on)  / (np.nanmean(on)  + 1e-6))

        rows.append({
            "patient_id":       pid,
            "pm_off":           float(np.nanmean(off)),
            "pm_on":            float(np.nanmean(on)),
            "pm_ledd":          float(np.nanmean(ledd)),
            "pm_age":           m_age,
            "pm_std_off":       float(np.nanstd(off)),
            "pm_std_on":        float(np.nanstd(on)),
            "pm_min_off":       float(np.nanmin(off)) if not np.all(np.isnan(off)) else 0,
            "pm_max_off":       float(np.nanmax(off)) if not np.all(np.isnan(off)) else 0,
            "pm_min_on":        float(np.nanmin(on))  if not np.all(np.isnan(on))  else 0,
            "pm_max_on":        float(np.nanmax(on))  if not np.all(np.isnan(on))  else 0,
            "n_visits":         n,
            "age_range":        float(ages.max() - ages.min()),
            "age_at_dx":        float(grp["age_at_diagnosis"].mean()),
            "sexM":             float(grp["sexM"].iloc[0]),
            # Paramètres de trajectoire (estimés depuis X seulement)
            "off_age_slope":    off_age_slope,
            "off_age_int":      off_age_int,
            "off_at_mean_age":  off_at_mean_age,
            "on_age_slope":     on_age_slope,
            "on_age_int":       on_age_int,
            "off_cv":           off_cv,
            "on_cv":            on_cv,
            "off_on_diff":      float(np.nanmean(off) - np.nanmean(on)),
            "off_on_ratio":     float(np.nanmean(off) / (np.nanmean(on) + 1.0)),
            "ledd_per_year":    float(np.nanmean(ledd) / max(m_age - float(grp["age_at_diagnosis"].mean()), 0.1)),
        })
    return pd.DataFrame(rows)


PAT_FEAT_COLS = [
    "pm_off", "pm_on", "pm_ledd", "pm_age", "pm_std_off", "pm_std_on",
    "pm_min_off", "pm_max_off", "pm_min_on", "pm_max_on",
    "n_visits", "age_range", "age_at_dx", "sexM",
    "off_age_slope", "off_age_int", "off_at_mean_age",
    "on_age_slope", "on_age_int",
    "off_cv", "on_cv", "off_on_diff", "off_on_ratio", "ledd_per_year",
]


# ── Feature engineering visite ──────────────────────────────────────────────────
def visit_features(df: pd.DataFrame, l1_pred: np.ndarray | None = None) -> pd.DataFrame:
    """Features au niveau visite + prédiction L1 comme feature pour L2."""
    out = df.copy()
    for col in ["cohort", "gene", "rater_id"]:
        if col in out.columns:
            out[col] = out[col].astype("category")
    out["years_dx"] = out["age"] - out["age_at_diagnosis"]

    pm = out.groupby("patient_id", sort=False).agg(
        pm_off=("off","mean"), pm_on=("on","mean"),
        pm_std_off=("off","std"), pm_std_on=("on","std"),
        pm_med_off=("off","median"), pm_ledd=("ledd","mean"),
        pm_n=("age","count"),
    ).reset_index()
    out = out.merge(pm, on="patient_id", how="left")

    out["off_z"]         = (out["off"] - out["pm_off"]) / (out["pm_std_off"] + 1e-6)
    out["on_z"]          = (out["on"]  - out["pm_on"])  / (out["pm_std_on"]  + 1e-6)
    out["off_vs_med"]    = out["off"] - out["pm_med_off"]
    out["off_minus_on"]  = out["off"] - out["on"]
    out["off_on_ratio"]  = out["off"] / (out["on"] + 1.0)
    toff = out["time_since_intake_off"].fillna(out["time_since_intake_off"].median())
    out["off_pk"]        = 1.0 / (1.0 + np.exp(-0.3*(toff-12.0)))
    out["off_debiased"]  = out["off"] * out["off_pk"]
    ton  = out["time_since_intake_on"].fillna(out["time_since_intake_on"].median())
    out["on_pk"]         = 1.0 / (1.0 + np.exp(0.5*(ton-2.0)))
    out["off_sq"]        = out["off"].fillna(0) ** 2
    out["visit_rank"]    = out.groupby("patient_id")["age"].rank(method="first").astype(float)
    out["visit_rank_pct"]= out["visit_rank"] / out["pm_n"]
    out["ledd_per_yr"]   = out["ledd"] / (out["years_dx"].clip(lower=0.1))

    if l1_pred is not None:
        out["l1_pred"] = l1_pred

    drop = ["Index","patient_id"] + (["target"] if "target" in out.columns else [])
    return out.drop(columns=drop)


# ── CV ──────────────────────────────────────────────────────────────────────────
print("\n[30] Per-patient slope model + LGBM residual  GroupKFold-5")

N = len(y)
oof_l1    = np.zeros(N)
oof_final = np.zeros(N)

LGBM_P = dict(n_estimators=3000, num_leaves=255, learning_rate=0.015,
              min_child_samples=15, colsample_bytree=0.7, subsample=0.8,
              subsample_freq=1, reg_alpha=0.05, reg_lambda=0.3, max_depth=-1,
              random_state=42, n_jobs=-1, verbose=-1)

for fi, (tr, vl) in enumerate(cv_splits):
    tr_vis = visits.iloc[tr]
    vl_vis = visits.iloc[vl]

    # === L1: Ridge sur paramètres per-patient ===
    tr_pat = per_patient_params(tr_vis)
    vl_pat = per_patient_params(vl_vis)

    # Target L1: mean(target) par patient dans le fold train
    tr_pat_tgt = (
        tr_vis.groupby("patient_id")["target"].mean()
        .rename("pm_target").reset_index()
    )
    tr_pat = tr_pat.merge(tr_pat_tgt, on="patient_id", how="left")

    feat_pat = [c for c in PAT_FEAT_COLS if c in tr_pat.columns]
    pipe_l1 = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
        ("r",   Ridge(alpha=0.1)),
    ])
    pipe_l1.fit(tr_pat[feat_pat].values, tr_pat["pm_target"].values)

    # Prédictions L1 pour validation
    vl_l1_pat = pipe_l1.predict(vl_pat[feat_pat].fillna(0).values)
    pid_to_l1 = dict(zip(vl_pat["patient_id"], vl_l1_pat))
    oof_l1[vl] = vl_vis["patient_id"].map(pid_to_l1).values

    # Prédictions L1 pour train (pour le résidu L2)
    tr_l1_pat = pipe_l1.predict(tr_pat[feat_pat].fillna(0).values)
    pid_to_l1_tr = dict(zip(tr_pat["patient_id"], tr_l1_pat))
    tr_l1_visit = tr_vis["patient_id"].map(pid_to_l1_tr).values

    # === L2: LGBM sur les résidus ===
    residual_tr = y[tr] - tr_l1_visit

    X_tr_v = visit_features(tr_vis, l1_pred=tr_l1_visit)
    X_vl_v = visit_features(vl_vis, l1_pred=oof_l1[vl])

    for col in ["cohort", "gene", "rater_id"]:
        if col in X_tr_v.columns and col in X_vl_v.columns:
            X_vl_v[col] = X_vl_v[col].cat.set_categories(X_tr_v[col].cat.categories)

    m2 = lgb.LGBMRegressor(**LGBM_P)
    m2.fit(X_tr_v, residual_tr)
    oof_final[vl] = oof_l1[vl] + m2.predict(X_vl_v)

    r1 = float(np.sqrt(np.mean((y[vl] - oof_l1[vl])**2)))
    r2 = float(np.sqrt(np.mean((y[vl] - oof_final[vl])**2)))
    print(f"  Fold {fi+1}: L1={r1:.4f}  L1+L2={r2:.4f}")

rmse_l1    = float(np.sqrt(np.mean((y - oof_l1)**2)))
rmse_final = float(np.sqrt(np.mean((y - oof_final)**2)))
print(f"\n  L1 (per-patient slope Ridge) OOF: {rmse_l1:.4f}")
print(f"  L1+L2 (+ LGBM residual)     OOF: {rmse_final:.4f}")

# Blend avec transformers 23+25
oof_23 = X_train_raw[["Index"]].merge(pd.read_csv("scratch/23_oof_visit.csv"),on="Index",how="left")["pred_23"].values
oof_25 = X_train_raw[["Index"]].merge(pd.read_csv("scratch/25_oof_visit.csv"),on="Index",how="left")["pred_25"].values
test_23 = X_test_raw[["Index"]].merge(pd.read_csv("scratch/23_test_visit.csv"),on="Index",how="left")["pred_23"].values
test_25 = X_test_raw[["Index"]].merge(pd.read_csv("scratch/25_test_visit.csv"),on="Index",how="left")["pred_25"].values

stack_blend = np.column_stack([oof_final, oof_23, oof_25])
meta_blend  = Ridge(alpha=0.1)
meta_blend.fit(stack_blend, y)
oof_blend   = meta_blend.predict(stack_blend)
rmse_blend  = float(np.sqrt(np.mean((y - oof_blend)**2)))
print(f"  Blend L1+L2 + T23 + T25     OOF: {rmse_blend:.4f}")
print(f"  Coefs: L2={meta_blend.coef_[0]:.3f}  T23={meta_blend.coef_[1]:.3f}  T25={meta_blend.coef_[2]:.3f}")

# Save OOF
Path("scratch").mkdir(exist_ok=True)
oof_df = visits[["Index"]].copy()
oof_df["pred_30"] = oof_final
oof_df.to_csv("scratch/30_oof.csv", index=False)


# ── Fit final sur tout le training data ─────────────────────────────────────────
print("\n[Final] Fit sur tout le training data...")

full_pat = per_patient_params(visits)
full_pat_tgt = visits.groupby("patient_id")["target"].mean().rename("pm_target").reset_index()
full_pat = full_pat.merge(full_pat_tgt, on="patient_id", how="left")

feat_pat_final = [c for c in PAT_FEAT_COLS if c in full_pat.columns]
pipe_l1_final = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc",  StandardScaler()),
    ("r",   Ridge(alpha=0.1)),
])
pipe_l1_final.fit(full_pat[feat_pat_final].values, full_pat["pm_target"].values)

full_l1_pat = pipe_l1_final.predict(full_pat[feat_pat_final].fillna(0).values)
pid_to_l1_full = dict(zip(full_pat["patient_id"], full_l1_pat))
full_l1_visit  = visits["patient_id"].map(pid_to_l1_full).values
residual_full  = y - full_l1_visit

X_full_v = visit_features(visits, l1_pred=full_l1_visit)
m2_final = lgb.LGBMRegressor(**LGBM_P)
m2_final.fit(X_full_v, residual_full)

# Prédictions test
test_pat = per_patient_params(X_test_raw)
test_l1_pat  = pipe_l1_final.predict(test_pat[feat_pat_final].fillna(0).values)
pid_to_l1_test = dict(zip(test_pat["patient_id"], test_l1_pat))
test_l1_visit  = X_test_raw["patient_id"].map(pid_to_l1_test).values

X_test_v = visit_features(X_test_raw, l1_pred=test_l1_visit)
for col in ["cohort", "gene", "rater_id"]:
    if col in X_full_v.columns and col in X_test_v.columns:
        X_test_v[col] = X_test_v[col].cat.set_categories(X_full_v[col].cat.categories)
X_test_v = X_test_v.reindex(columns=X_full_v.columns, fill_value=np.nan)

test_l2   = m2_final.predict(X_test_v)
test_final = np.clip(test_l1_visit + test_l2, 0, 132)

# Blend avec transformers
nan_mask = np.isnan(test_23) | np.isnan(test_25)
if nan_mask.any():
    test_23[nan_mask] = test_final[nan_mask]
    test_25[nan_mask] = test_final[nan_mask]
test_blend = np.clip(meta_blend.predict(np.column_stack([test_final, test_23, test_25])), 0, 132)

# Save test preds
test_df = X_test_raw[["Index"]].copy()
test_df["pred_30"] = test_final
test_df.to_csv("scratch/30_test.csv", index=False)

# ── Soumissions ─────────────────────────────────────────────────────────────────
Path("submissions").mkdir(exist_ok=True)

sub_l2 = X_test_raw[["Index"]].copy()
sub_l2["target"] = test_final
sub_l2.to_csv("submissions/30_per_patient.csv", index=False)
print("  Wrote submissions/30_per_patient.csv  (L1+L2 per-patient slope)")

sub_blend = X_test_raw[["Index"]].copy()
sub_blend["target"] = test_blend
sub_blend.to_csv("submissions/30_blend.csv", index=False)
print("  Wrote submissions/30_blend.csv  (L1+L2 + T23 + T25 blend)  ← SOUMETTRE")

print(f"\n{'='*60}")
print("RÉSULTATS")
print(f"  30 L1 (per-patient slope):      {rmse_l1:.4f} OOF")
print(f"  30 L1+L2 (LGBM residual):       {rmse_final:.4f} OOF")
print(f"  30 Blend (L1+L2+T23+T25):       {rmse_blend:.4f} OOF")
print(f"  28_meta3 (précédent meilleur):  3.90 OOF  → 4.247 Kaggle")
print(f"  Leaders Kaggle:                 2.05")
print(f"  Floor in-sample (age+off):      1.00")
print(f"{'='*60}")
