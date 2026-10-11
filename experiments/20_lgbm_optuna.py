# %% [markdown]
# # Experiment 20 — LGBM with Optuna hyperparameter search (GroupKFold-aware)
#
# Exp 12 used hand-picked hyperparameters (3000 trees, num_leaves=255, lr=0.015).
# Optuna with 60 trials and GroupKFold-5 inner CV can find a better combination.
# Budget: 60 trials × ~1 min each = ~1 hour. Run with n_jobs=1 to avoid
# multiprocessing conflicts inside LightGBM.

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import GroupKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Load & engineer features ──────────────────────────────────────────────────
X_train_raw = pd.read_csv("data/X_train.csv")
y_train_df  = pd.read_csv("data/y_train.csv")
X_test_raw  = pd.read_csv("data/X_test.csv")
visits = X_train_raw.merge(y_train_df, on="Index")
y = visits["target"].values
groups = visits["patient_id"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
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
            pat_mean_dx_age=("age_at_diagnosis", "mean"),
        )
        .reset_index()
    )
    out = out.merge(pat, on="patient_id", how="left")
    out["off_minus_on"]      = out["off"]  - out["on"]
    out["mean_motor"]        = out[["off", "on"]].mean(axis=1)
    out["off_vs_pat_mean"]   = out["off"]  - out["pat_mean_off"]
    out["on_vs_pat_mean"]    = out["on"]   - out["pat_mean_on"]
    out["off_vs_pat_median"] = out["off"]  - out["pat_median_off"]
    out["on_vs_pat_median"]  = out["on"]   - out["pat_median_on"]
    out["off_range_pct"]     = (out["off"] - out["pat_min_off"]) / (
        out["pat_max_off"] - out["pat_min_off"] + 1e-6
    )
    out["off_z_pat"]    = out["off_vs_pat_mean"] / (out["pat_std_off"] + 1e-6)
    out["on_z_pat"]     = out["on_vs_pat_mean"]  / (out["pat_std_on"]  + 1e-6)
    toff = out["time_since_intake_off"].fillna(out["time_since_intake_off"].median())
    out["off_pk_weight"] = 1.0 / (1.0 + np.exp(-0.3 * (toff - 12.0)))
    out["off_debiased"]  = out["off"] * out["off_pk_weight"]
    ton  = out["time_since_intake_on"].fillna(out["time_since_intake_on"].median())
    out["on_pk_weight"]  = 1.0 / (1.0 + np.exp(0.5 * (ton - 2.0)))
    out["on_adjusted"]   = out["on"]  * out["on_pk_weight"]
    out["ledd_per_year"] = out["ledd"] / (out["years_since_dx"].clip(lower=0.1))
    out["off_on_ratio"]  = out["off"]  / (out["on"] + 1.0)
    out["ledd_x_off"]    = out["ledd"].fillna(0) * out["off"].fillna(0)
    out["age_x_off"]     = out["age"]  * out["off"].fillna(0)
    out["visit_rank"]    = out.groupby("patient_id")["age"].rank(method="first").astype(float)
    out["visit_rank_pct"]= out["visit_rank"] / out["pat_n_visits"]
    drop = ["Index", "patient_id"] + (["target"] if "target" in out.columns else [])
    return out.drop(columns=drop)


print("Engineering features...")
X_eng = engineer_features(visits)
cv = GroupKFold(n_splits=5)
cv_splits = list(cv.split(X_eng, y, groups=groups))


# ── Optuna objective ──────────────────────────────────────────────────────────
def objective(trial: optuna.Trial) -> float:
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 500, 5000),
        "num_leaves":        trial.suggest_int("num_leaves", 31, 511),
        "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq":    1,
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 2.0, log=True),
        "max_depth":         trial.suggest_int("max_depth", -1, 12),
        "random_state": 42, "n_jobs": -1, "verbose": -1,
    }
    fold_rmses = []
    for train_idx, val_idx in cv_splits:
        X_tr = X_eng.iloc[train_idx]
        X_vl = X_eng.iloc[val_idx].copy()
        for col in ["cohort", "gene", "rater_id"]:
            if col in X_tr.columns:
                X_vl[col] = X_vl[col].cat.set_categories(X_tr[col].cat.categories)
        m = lgb.LGBMRegressor(**params)
        m.fit(X_tr, y[train_idx])
        preds = m.predict(X_vl)
        fold_rmses.append(np.sqrt(np.mean((y[val_idx] - preds) ** 2)))
    return float(np.mean(fold_rmses))


print("\n[20] Optuna search: 60 trials, GroupKFold-5")
study = optuna.create_study(direction="minimize",
                            sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=60, show_progress_bar=False)

best = study.best_params
best_rmse = study.best_value
print(f"\n  Best OOF RMSE: {best_rmse:.4f}")
print(f"  Best params:   {best}")

# ── Retrain final model with best params + OOF ────────────────────────────────
print("\n[Final OOF] Re-running with best params for submission...")
oof_20 = np.zeros(len(y))
for fold_i, (train_idx, val_idx) in enumerate(cv_splits):
    X_tr = X_eng.iloc[train_idx]
    X_vl = X_eng.iloc[val_idx].copy()
    for col in ["cohort", "gene", "rater_id"]:
        if col in X_tr.columns:
            X_vl[col] = X_vl[col].cat.set_categories(X_tr[col].cat.categories)
    m = lgb.LGBMRegressor(**best, random_state=42, n_jobs=-1, verbose=-1)
    m.fit(X_tr, y[train_idx])
    oof_20[val_idx] = m.predict(X_vl)
    print(f"  Fold {fold_i+1}: {np.sqrt(np.mean((y[val_idx]-oof_20[val_idx])**2)):.4f}")

rmse_20 = np.sqrt(np.mean((y - oof_20) ** 2))
print(f"\n  20 OOF RMSE: {rmse_20:.4f}  (exp 12 was 4.59)")

Path("scratch").mkdir(exist_ok=True)
oof_df = visits[["Index"]].copy()
oof_df["pred_20"] = oof_20
oof_df.to_csv("scratch/20_oof.csv", index=False)

# Test predictions
X_test_eng = engineer_features(X_test_raw).reindex(columns=X_eng.columns, fill_value=np.nan)
final_m = lgb.LGBMRegressor(**best, random_state=42, n_jobs=-1, verbose=-1)
final_m.fit(X_eng, y)
test_preds = final_m.predict(X_test_eng)

Path("submissions").mkdir(exist_ok=True)
sub = X_test_raw[["Index"]].copy()
sub["target"] = test_preds
sub.to_csv("submissions/20_lgbm_optuna.csv", index=False)

test_df = X_test_raw[["Index"]].copy()
test_df["pred_20"] = test_preds
test_df.to_csv("scratch/20_test.csv", index=False)

print(f"  Wrote submissions/20_lgbm_optuna.csv")
print(f"\n{'='*60}")
print(f"  20 Optuna LGBM OOF RMSE: {rmse_20:.4f}")
print(f"{'='*60}")
