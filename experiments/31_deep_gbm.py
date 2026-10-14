# %% [markdown]
# # Experiment 31 — Formule synthétique: pm_off + on
#
# Insight du reverse-engineering:
#   - RF in-sample avec 28 features: RMSE 1.52
#   - Top features: pm_off (55%), on (21%), years_dx (7%)
#   - La formule cible = f(pm_off, on, years_dx) + petit bruit
#
# Les leaders à 2.05 ont probablement trouvé cette structure simple.
# Un LGBM profond sur (pm_off, on, years_dx + quelques features) devrait
# généraliser bien sur les patients test car ces features sont calculables
# directement depuis X_test.
#
# Stratégie:
#   1. LGBM focalisé sur pm_off + on + years_dx (features robustes)
#   2. XGBoost idem
#   3. Blend Ridge
#   4. Blend final avec T23+T25

# %%
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

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


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Features focalisées sur pm_off + on + years_dx — les 3 qui dominent."""
    out = df.copy()
    for col in ["cohort", "gene", "rater_id"]:
        if col in out.columns:
            out[col] = out[col].astype("category")
    out["years_dx"] = out["age"] - out["age_at_diagnosis"]

    pm = out.groupby("patient_id", sort=False).agg(
        pm_off=("off","mean"), pm_on=("on","mean"),
        pm_std_off=("off","std"), pm_std_on=("on","std"),
        pm_ledd=("ledd","mean"),
        pm_med_off=("off","median"), pm_med_on=("on","median"),
        pm_min_off=("off","min"), pm_max_off=("off","max"),
        pm_min_on=("on","min"),  pm_max_on=("on","max"),
        pm_n=("age","count"),
        pm_age=("age","mean"),
        pm_age_range=("age", lambda x: float(x.max()-x.min())),
        pm_age_dx=("age_at_diagnosis","mean"),
    ).reset_index()
    out = out.merge(pm, on="patient_id", how="left")

    # Déviations intra-patient
    out["off_z"]         = (out["off"] - out["pm_off"]) / (out["pm_std_off"] + 1e-6)
    out["on_z"]          = (out["on"]  - out["pm_on"])  / (out["pm_std_on"]  + 1e-6)
    out["off_vs_med"]    = out["off"]  - out["pm_med_off"]
    out["off_minus_on"]  = out["off"]  - out["on"]
    out["off_on_ratio"]  = out["off"]  / (out["on"] + 1.0)

    # PK timing
    toff = out["time_since_intake_off"].fillna(out["time_since_intake_off"].median())
    out["off_pk"]       = 1.0 / (1.0 + np.exp(-0.3*(toff-12.0)))
    out["off_debiased"] = out["off"] * out["off_pk"]
    ton  = out["time_since_intake_on"].fillna(out["time_since_intake_on"].median())
    out["on_pk"]        = 1.0 / (1.0 + np.exp(0.5*(ton-2.0)))
    out["on_adjusted"]  = out["on"] * out["on_pk"]

    # Interactions pm_off × on (le signal dominant selon RF)
    out["pm_off_x_on"]   = out["pm_off"].fillna(0) * out["on"].fillna(0)
    out["pm_off_minus_on"]= out["pm_off"] - out["on"].fillna(out["pm_off"])
    out["on_sq"]          = out["on"].fillna(0) ** 2
    out["off_sq"]         = out["off"].fillna(0) ** 2
    out["off_x_on"]       = out["off"].fillna(0) * out["on"].fillna(0)

    out["ledd_per_yr"]   = out["ledd"] / (out["years_dx"].clip(lower=0.1))
    out["visit_rank"]    = out.groupby("patient_id")["age"].rank(method="first").astype(float)
    out["visit_rank_pct"]= out["visit_rank"] / out["pm_n"]

    drop = ["Index","patient_id"] + (["target"] if "target" in out.columns else [])
    return out.drop(columns=drop)


def align(src, dst):
    for col in ["cohort","gene","rater_id"]:
        if col in src.columns and col in dst.columns:
            dst[col] = dst[col].cat.set_categories(src[col].cat.categories)
    return dst


def to_num(df):
    out = df.copy()
    for col in ["cohort","gene","rater_id"]:
        if col in out.columns:
            out[col] = out[col].cat.codes.astype("float32")
    return out


# Hyperparamètres: plus de profondeur pour capturer la structure non-linéaire
LGBM_P = dict(n_estimators=5000, num_leaves=511, learning_rate=0.008,
              min_child_samples=10, colsample_bytree=0.8, subsample=0.8,
              subsample_freq=1, reg_alpha=0.01, reg_lambda=0.1, max_depth=-1,
              random_state=42, n_jobs=-1, verbose=-1)
XGB_P  = dict(n_estimators=3000, max_depth=10, learning_rate=0.01,
              subsample=0.8, colsample_bytree=0.8, reg_alpha=0.01, reg_lambda=0.1,
              min_child_weight=5, random_state=42, n_jobs=-1, verbosity=0,
              tree_method="hist")


N = len(y)
oof_lgbm = np.zeros(N)
oof_xgb  = np.zeros(N)

print("\n[31] LGBM + XGB deep (pm_off × on focus)  GroupKFold-5")
for fi, (tr, vl) in enumerate(cv_splits):
    tr_vis = visits.iloc[tr]; vl_vis = visits.iloc[vl]
    X_tr = engineer(tr_vis)
    X_vl = engineer(vl_vis)
    X_vl = align(X_tr, X_vl)
    X_tr_n = to_num(X_tr); X_vl_n = to_num(X_vl)

    m_l = lgb.LGBMRegressor(**LGBM_P)
    m_l.fit(X_tr, y[tr])
    oof_lgbm[vl] = m_l.predict(X_vl)

    m_x = xgb.XGBRegressor(**XGB_P)
    m_x.fit(X_tr_n, y[tr])
    oof_xgb[vl] = m_x.predict(X_vl_n)

    rl = float(np.sqrt(np.mean((y[vl]-oof_lgbm[vl])**2)))
    rx = float(np.sqrt(np.mean((y[vl]-oof_xgb[vl])**2)))
    print(f"  Fold {fi+1}: LGBM={rl:.4f}  XGB={rx:.4f}")

rmse_lgbm = float(np.sqrt(np.mean((y-oof_lgbm)**2)))
rmse_xgb  = float(np.sqrt(np.mean((y-oof_xgb)**2)))
print(f"\n  OOF LGBM={rmse_lgbm:.4f}  XGB={rmse_xgb:.4f}")

# Ridge meta GBM
meta_gbm = Ridge(alpha=0.1)
meta_gbm.fit(np.column_stack([oof_lgbm, oof_xgb]), y)
oof_gbm = meta_gbm.predict(np.column_stack([oof_lgbm, oof_xgb]))
rmse_gbm = float(np.sqrt(np.mean((y-oof_gbm)**2)))
print(f"  Ridge meta GBM: {rmse_gbm:.4f}")

# Blend avec T23 + T25
oof_23 = X_train_raw[["Index"]].merge(pd.read_csv("scratch/23_oof_visit.csv"),on="Index",how="left")["pred_23"].values
oof_25 = X_train_raw[["Index"]].merge(pd.read_csv("scratch/25_oof_visit.csv"),on="Index",how="left")["pred_25"].values
test_23 = X_test_raw[["Index"]].merge(pd.read_csv("scratch/23_test_visit.csv"),on="Index",how="left")["pred_23"].values
test_25 = X_test_raw[["Index"]].merge(pd.read_csv("scratch/25_test_visit.csv"),on="Index",how="left")["pred_25"].values

stack_all = np.column_stack([oof_lgbm, oof_xgb, oof_23, oof_25])
meta_all = Ridge(alpha=0.1)
meta_all.fit(stack_all, y)
oof_blend = meta_all.predict(stack_all)
rmse_blend = float(np.sqrt(np.mean((y-oof_blend)**2)))
print(f"  Blend LGBM+XGB+T23+T25: {rmse_blend:.4f}")
print(f"  Coefs: L={meta_all.coef_[0]:.3f}  X={meta_all.coef_[1]:.3f}  T23={meta_all.coef_[2]:.3f}  T25={meta_all.coef_[3]:.3f}")

# ── Final sur tout le training ──────────────────────────────────────────────────
print("\n[Final] Fit sur tout le training data...")
X_full = engineer(visits)
X_full_n = to_num(X_full)

X_test_e = engineer(X_test_raw)
X_test_e = align(X_full, X_test_e)
X_test_e = X_test_e.reindex(columns=X_full.columns, fill_value=np.nan)
X_test_n = to_num(X_test_e)

fl = lgb.LGBMRegressor(**LGBM_P); fl.fit(X_full, y)
fx = xgb.XGBRegressor(**XGB_P);  fx.fit(X_full_n, y)

t_l = fl.predict(X_test_e)
t_x = fx.predict(X_test_n)
t_gbm = meta_gbm.predict(np.column_stack([t_l, t_x]))

nan_mask = np.isnan(test_23) | np.isnan(test_25)
if nan_mask.any():
    test_23[nan_mask] = t_gbm[nan_mask]
    test_25[nan_mask] = t_gbm[nan_mask]

t_blend = np.clip(meta_all.predict(np.column_stack([t_l, t_x, test_23, test_25])), 0, 132)
t_gbm_clipped = np.clip(t_gbm, 0, 132)

Path("submissions").mkdir(exist_ok=True)

sub_gbm = X_test_raw[["Index"]].copy()
sub_gbm["target"] = t_gbm_clipped
sub_gbm.to_csv("submissions/31_gbm.csv", index=False)
print("  Wrote submissions/31_gbm.csv")

sub_blend = X_test_raw[["Index"]].copy()
sub_blend["target"] = t_blend
sub_blend.to_csv("submissions/31_blend.csv", index=False)
print("  Wrote submissions/31_blend.csv  ← SOUMETTRE")

print(f"\n{'='*60}")
print("RÉSULTATS")
print(f"  30_blend (précédent):    en cours (attendu ~3.9)")
print(f"  31 LGBM:                 {rmse_lgbm:.4f} OOF")
print(f"  31 XGB:                  {rmse_xgb:.4f} OOF")
print(f"  31 Ridge GBM:            {rmse_gbm:.4f} OOF")
print(f"  31 Blend LGBM+XGB+T23+T25: {rmse_blend:.4f} OOF")
print(f"  28_meta3:                3.90 OOF → 4.247 Kaggle")
print(f"  Leaders:                 2.05 Kaggle")
print(f"  RF in-sample floor:      1.52")
print(f"{'='*60}")
print("\nSubmissions: 31_gbm.csv (GBM seul) et 31_blend.csv (blend+transformers)")
