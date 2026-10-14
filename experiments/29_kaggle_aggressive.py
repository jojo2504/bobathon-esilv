# %% [markdown]
# # Experiment 29 — Stratégie Kaggle aggressive
#
# Objectif: meilleur score public Kaggle possible.
#
# Leviers:
#   1. pat_mean_target comme feature — pour les patients train on injecte
#      leurs vrais targets (GroupKFold: train fold → autres train folds, pas val).
#      Pour les patients test: pseudo-labels du meilleur modèle disponible.
#   2. Pseudo-labeling: on utilise les OOF transformers (23+25) déjà calculés
#      + GBM exp12 pour bootstrapper les pseudo-labels dès le round 0.
#   3. Stack final: LGBM + XGB + HGBR + Ridge meta + blend avec T23/T25.

# %%
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

# ── Load ────────────────────────────────────────────────────────────────────────
print("Loading data...")
X_train_raw = pd.read_csv("data/X_train.csv")
y_train_df  = pd.read_csv("data/y_train.csv")
X_test_raw  = pd.read_csv("data/X_test.csv")
visits = X_train_raw.merge(y_train_df, on="Index")
y = visits["target"].values
groups = visits["patient_id"]
print(f"  train: {visits.shape}  test: {X_test_raw.shape}")

# Charge les meilleures prédictions test déjà disponibles
# pour bootstrapper les pseudo-labels (T23+T25 blend = 3.90 OOF)
print("  Chargement pseudo-labels bootstrap depuis exp 28 (T23+T25+GBM blend)...")
best_test_csv = Path("submissions/28_meta3.csv")
if best_test_csv.exists():
    pseudo_bootstrap = pd.read_csv(best_test_csv)
    pseudo_map_boot = (
        X_test_raw[["patient_id","Index"]]
        .merge(pseudo_bootstrap, on="Index")
        .groupby("patient_id")["target"].mean().to_dict()
    )
    print(f"  Bootstrap: {len(pseudo_map_boot)} patients test, mean={np.mean(list(pseudo_map_boot.values())):.2f}")
else:
    pseudo_map_boot = {}
    print("  Pas de bootstrap disponible — pseudo-labels = pm_off")

cv = GroupKFold(n_splits=5)
cv_splits = list(cv.split(visits, y, groups=groups))


# ── Feature engineering ─────────────────────────────────────────────────────────
def engineer(df: pd.DataFrame, pat_target_map: dict | None = None) -> pd.DataFrame:
    """Features complètes + pat_mean_target (clé Kaggle)."""
    out = df.copy()
    for col in ["cohort","gene","rater_id"]:
        if col in out.columns:
            out[col] = out[col].astype("category")
    out["years_dx"] = out["age"] - out["age_at_diagnosis"]

    pat = out.groupby("patient_id", sort=False).agg(
        pm_off=("off","mean"), pm_on=("on","mean"),
        pm_std_off=("off","std"), pm_std_on=("on","std"),
        pm_ledd=("ledd","mean"), pm_std_ledd=("ledd","std"),
        pm_med_off=("off","median"), pm_med_on=("on","median"),
        pm_min_off=("off","min"), pm_max_off=("off","max"),
        pm_min_on=("on","min"), pm_max_on=("on","max"),
        pm_n=("age","count"),
        pm_age=("age","mean"),
        pm_age_range=("age", lambda x: float(x.max()-x.min())),
        pm_age_dx=("age_at_diagnosis","mean"),
    ).reset_index()
    out = out.merge(pat, on="patient_id", how="left")

    out["off_minus_on"]  = out["off"]  - out["on"]
    out["mean_motor"]    = out[["off","on"]].mean(axis=1)
    out["off_vs_mean"]   = out["off"]  - out["pm_off"]
    out["on_vs_mean"]    = out["on"]   - out["pm_on"]
    out["off_vs_med"]    = out["off"]  - out["pm_med_off"]
    out["off_range_pct"] = (out["off"] - out["pm_min_off"]) / (out["pm_max_off"] - out["pm_min_off"] + 1e-6)
    out["off_z"]         = out["off_vs_mean"] / (out["pm_std_off"] + 1e-6)
    out["on_z"]          = out["on_vs_mean"]  / (out["pm_std_on"]  + 1e-6)

    toff = out["time_since_intake_off"].fillna(out["time_since_intake_off"].median())
    out["off_pk"]       = 1.0 / (1.0 + np.exp(-0.3*(toff-12.0)))
    out["off_debiased"] = out["off"] * out["off_pk"]
    ton = out["time_since_intake_on"].fillna(out["time_since_intake_on"].median())
    out["on_pk"]        = 1.0 / (1.0 + np.exp(0.5*(ton-2.0)))
    out["on_adjusted"]  = out["on"] * out["on_pk"]

    out["off_on_ratio"] = out["off"] / (out["on"] + 1.0)
    out["ledd_per_yr"]  = out["ledd"] / (out["years_dx"].clip(lower=0.1))
    out["ledd_x_off"]   = out["ledd"].fillna(0) * out["off"].fillna(0)
    out["age_x_off"]    = out["age"] * out["off"].fillna(0)
    out["off_sq"]       = out["off"].fillna(0) ** 2
    out["on_sq"]        = out["on"].fillna(0)  ** 2
    out["off_x_on"]     = out["off"].fillna(0) * out["on"].fillna(0)
    out["off_cb"]       = out["off"].fillna(0) ** 3
    out["visit_rank"]   = out.groupby("patient_id")["age"].rank(method="first").astype(float)
    out["visit_rank_pct"] = out["visit_rank"] / out["pm_n"]

    # ★ Feature clé: pat_mean_target
    # Patients train  → vrais targets (depuis pat_target_map)
    # Patients test   → pseudo-labels (depuis pat_target_map)
    # Fallback        → pm_off (r=0.93 avec pat_mean_target)
    if pat_target_map is not None:
        out["pat_mean_target"] = out["patient_id"].map(pat_target_map).fillna(out["pm_off"])
    else:
        out["pat_mean_target"] = out["pm_off"]

    drop = ["Index","patient_id"] + (["target"] if "target" in out.columns else [])
    return out.drop(columns=drop)


def align_cats(src: pd.DataFrame, dst: pd.DataFrame) -> pd.DataFrame:
    for col in ["cohort","gene","rater_id"]:
        if col in src.columns and col in dst.columns:
            dst[col] = dst[col].cat.set_categories(src[col].cat.categories)
    return dst


def to_num(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["cohort","gene","rater_id"]:
        if col in out.columns:
            out[col] = out[col].cat.codes.astype("float32")
    return out


LGBM_P = dict(n_estimators=3000, num_leaves=255, learning_rate=0.015,
              min_child_samples=15, colsample_bytree=0.7, subsample=0.8,
              subsample_freq=1, reg_alpha=0.05, reg_lambda=0.3, max_depth=-1,
              random_state=42, n_jobs=-1, verbose=-1)
XGB_P  = dict(n_estimators=2000, max_depth=8, learning_rate=0.02,
              subsample=0.8, colsample_bytree=0.7, reg_alpha=0.05, reg_lambda=0.5,
              random_state=42, n_jobs=-1, verbosity=0, tree_method="hist")
HGBR_P = dict(max_iter=500, max_depth=8, learning_rate=0.05,
              min_samples_leaf=15, random_state=42)


# ══════════════════════════════════════════════════════════════════════════════
# CV GroupKFold — avec pseudo-labels bootstrap dans pat_mean_target
# ══════════════════════════════════════════════════════════════════════════════
print("\n[29] CV GroupKFold-5 avec pat_mean_target + pseudo-labels bootstrap...")
N = len(y)
oof_lgbm = np.zeros(N)
oof_xgb  = np.zeros(N)
oof_hgbr = np.zeros(N)

for fi, (tr, vl) in enumerate(cv_splits):
    tr_vis = visits.iloc[tr]
    vl_vis = visits.iloc[vl]

    # pat_mean_target pour ce fold:
    #   - patients train du fold: leurs vrais targets
    #   - patients test: pseudo-labels bootstrap
    #   - patients val: absents => fallback pm_off (honnête)
    pat_tgt = tr_vis.groupby("patient_id")["target"].mean().to_dict()
    pat_tgt.update(pseudo_map_boot)  # pseudo-labels test patients

    X_tr = engineer(tr_vis, pat_target_map=pat_tgt)
    X_vl = engineer(vl_vis, pat_target_map=pat_tgt)  # val patients → pm_off fallback
    X_vl = align_cats(X_tr, X_vl)
    X_tr_n = to_num(X_tr); X_vl_n = to_num(X_vl)

    m_l = lgb.LGBMRegressor(**LGBM_P)
    m_l.fit(X_tr, y[tr])
    oof_lgbm[vl] = m_l.predict(X_vl)

    m_x = xgb.XGBRegressor(**XGB_P)
    m_x.fit(X_tr_n, y[tr])
    oof_xgb[vl] = m_x.predict(X_vl_n)

    m_h = HistGradientBoostingRegressor(**HGBR_P)
    m_h.fit(X_tr_n, y[tr])
    oof_hgbr[vl] = m_h.predict(X_vl_n)

    rl = float(np.sqrt(np.mean((y[vl]-oof_lgbm[vl])**2)))
    rx = float(np.sqrt(np.mean((y[vl]-oof_xgb[vl])**2)))
    rh = float(np.sqrt(np.mean((y[vl]-oof_hgbr[vl])**2)))
    print(f"  Fold {fi+1}: LGBM={rl:.4f}  XGB={rx:.4f}  HGBR={rh:.4f}")

rmse_lgbm = float(np.sqrt(np.mean((y-oof_lgbm)**2)))
rmse_xgb  = float(np.sqrt(np.mean((y-oof_xgb)**2)))
rmse_hgbr = float(np.sqrt(np.mean((y-oof_hgbr)**2)))
print(f"\n  OOF: LGBM={rmse_lgbm:.4f}  XGB={rmse_xgb:.4f}  HGBR={rmse_hgbr:.4f}")

# Ridge meta sur les 3 GBM
meta_gbm = Ridge(alpha=0.1)
meta_gbm.fit(np.column_stack([oof_lgbm, oof_xgb, oof_hgbr]), y)
oof_gbm = meta_gbm.predict(np.column_stack([oof_lgbm, oof_xgb, oof_hgbr]))
rmse_gbm = float(np.sqrt(np.mean((y-oof_gbm)**2)))
print(f"  GBM Ridge meta OOF: {rmse_gbm:.4f}")

# Blend avec transformers 23 + 25
oof_23 = X_train_raw[["Index"]].merge(pd.read_csv("scratch/23_oof_visit.csv"),on="Index",how="left")["pred_23"].values
oof_25 = X_train_raw[["Index"]].merge(pd.read_csv("scratch/25_oof_visit.csv"),on="Index",how="left")["pred_25"].values
test_23 = X_test_raw[["Index"]].merge(pd.read_csv("scratch/23_test_visit.csv"),on="Index",how="left")["pred_23"].values
test_25 = X_test_raw[["Index"]].merge(pd.read_csv("scratch/25_test_visit.csv"),on="Index",how="left")["pred_25"].values

stack_all = np.column_stack([oof_23, oof_25, oof_gbm])
meta_all = Ridge(alpha=0.1)
meta_all.fit(stack_all, y)
oof_final = meta_all.predict(stack_all)
rmse_final = float(np.sqrt(np.mean((y-oof_final)**2)))
print(f"  Final blend (T23+T25+GBM29) OOF: {rmse_final:.4f}")
print(f"  Coefs: T23={meta_all.coef_[0]:.3f}  T25={meta_all.coef_[1]:.3f}  GBM={meta_all.coef_[2]:.3f}")


# ── Fit final sur tout le training data ─────────────────────────────────────────
print("\n[Final] Fit sur tout le training data...")
pat_tgt_final = visits.groupby("patient_id")["target"].mean().to_dict()
pat_tgt_final.update(pseudo_map_boot)  # pseudo-labels test

X_full = engineer(visits, pat_target_map=pat_tgt_final)
X_full_n = to_num(X_full)

X_test_final = engineer(X_test_raw, pat_target_map=pseudo_map_boot)
X_test_final = align_cats(X_full, X_test_final)
X_test_final = X_test_final.reindex(columns=X_full.columns, fill_value=np.nan)
X_test_final_n = to_num(X_test_final)

fl = lgb.LGBMRegressor(**LGBM_P); fl.fit(X_full, y)
fx = xgb.XGBRegressor(**XGB_P);  fx.fit(X_full_n, y)
fh = HistGradientBoostingRegressor(**HGBR_P); fh.fit(X_full_n, y)

t_l = fl.predict(X_test_final)
t_x = fx.predict(X_test_final_n)
t_h = fh.predict(X_test_final_n)
t_gbm = meta_gbm.predict(np.column_stack([t_l, t_x, t_h]))

# Blend final avec transformers
test_nan_mask = np.isnan(test_23) | np.isnan(test_25)
if test_nan_mask.any():
    test_23[test_nan_mask] = t_gbm[test_nan_mask]
    test_25[test_nan_mask] = t_gbm[test_nan_mask]
t_final = meta_all.predict(np.column_stack([test_23, test_25, t_gbm]))

# ── Soumissions ─────────────────────────────────────────────────────────────────
Path("submissions").mkdir(exist_ok=True)

sub_gbm = X_test_raw[["Index"]].copy()
sub_gbm["target"] = t_gbm
sub_gbm.to_csv("submissions/29_gbm_pseudolabel.csv", index=False)
print("  Wrote submissions/29_gbm_pseudolabel.csv  (GBM+pseudo-labels seul)")

sub_final = X_test_raw[["Index"]].copy()
sub_final["target"] = t_final
sub_final.to_csv("submissions/29_kaggle.csv", index=False)
print("  Wrote submissions/29_kaggle.csv  ← SOUMETTRE SUR KAGGLE")

print(f"\n{'='*60}")
print("RÉSULTATS")
print(f"  Exp 28 (meilleur précédent):       3.90 OOF")
print(f"  29 GBM seul (avec pseudo-labels):  {rmse_gbm:.4f} OOF")
print(f"  29 Blend T23+T25+GBM29:            {rmse_final:.4f} OOF  ← MEILLEUR")
print(f"  Floor théorique:                   1.61")
print(f"{'='*60}")
print("\nNote: OOF optimiste (pat_mean_target partiellement leaky pour train).")
print("Le score public Kaggle sera la vraie mesure.")
