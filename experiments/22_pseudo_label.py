# %% [markdown]
# # Experiment 22 — Pseudo-labeling: use test predictions as soft labels
#
# After exp 17/18 we have test predictions. The test patients are fully held out
# but their X features are known. If we add test rows (with predicted targets as
# soft labels) to the training set and retrain, the model learns the test
# distribution's feature space too.
#
# Semi-supervised trick: weight pseudo-labeled rows lower than real labels (0.3)
# to avoid overconfident extrapolation. Use the exp 18 meta predictions as the
# pseudo labels (best available).

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

# Load best available test predictions as pseudo-labels
# Prefer exp 18 meta (blend of transformer + GBM), fall back to 17
pseudo_path = Path("submissions/18_meta.csv")
if not pseudo_path.exists():
    pseudo_path = Path("submissions/17_transformer_v4.csv")
pseudo_df = pd.read_csv(pseudo_path)
print(f"Pseudo-labels from: {pseudo_path.name}")


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
        )
        .reset_index()
    )
    out = out.merge(pat, on="patient_id", how="left")
    out["off_minus_on"]      = out["off"]  - out["on"]
    out["off_on_ratio"]      = out["off"]  / (out["on"] + 1.0)
    out["off_vs_pat_mean"]   = out["off"]  - out["pat_mean_off"]
    out["on_vs_pat_mean"]    = out["on"]   - out["pat_mean_on"]
    out["off_vs_pat_median"] = out["off"]  - out["pat_median_off"]
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
    out["age_x_off"]         = out["age"]  * out["off"].fillna(0)
    out["visit_rank"]        = out.groupby("patient_id")["age"].rank(method="first").astype(float)
    out["visit_rank_pct"]    = out["visit_rank"] / out["pat_n_visits"]
    drop = ["Index", "patient_id"] + (["target"] if "target" in out.columns else [])
    return out.drop(columns=drop)


# ── Build augmented training set ──────────────────────────────────────────────
print("Engineering features...")
X_eng_train = engineer_features(visits)
X_eng_test  = engineer_features(
    X_test_raw.merge(pseudo_df.rename(columns={"target": "target"}), on="Index")
)
X_eng_test_pred = engineer_features(X_test_raw)  # for final submission (no pseudo y)

# Align categories
for col in ["cohort", "gene", "rater_id"]:
    if col in X_eng_train.columns:
        X_eng_test[col] = X_eng_test[col].cat.set_categories(X_eng_train[col].cat.categories)
        X_eng_test_pred[col] = X_eng_test_pred[col].cat.set_categories(X_eng_train[col].cat.categories)

y_test_pseudo = pseudo_df.set_index("Index").loc[X_test_raw["Index"], "target"].values

# ── Pseudo-label weight ───────────────────────────────────────────────────────
PSEUDO_WEIGHT = 0.3  # down-weight pseudo-labeled rows vs real labels

# ── GroupKFold CV on real train (pseudo rows NOT in val — honest eval) ────────
cv = GroupKFold(n_splits=5)
cv_splits = list(cv.split(X_eng_train, y, groups=groups))

lgbm_params = dict(
    n_estimators=3000, num_leaves=255, learning_rate=0.015,
    min_child_samples=15, colsample_bytree=0.7, subsample=0.8,
    subsample_freq=1, reg_alpha=0.05, reg_lambda=0.3, max_depth=-1,
    random_state=42, n_jobs=-1, verbose=-1,
)

oof_22 = np.zeros(len(y))
print("\n[22] Pseudo-labeling  GroupKFold-5 CV")

for fold_i, (train_idx, val_idx) in enumerate(cv_splits):
    X_tr = X_eng_train.iloc[train_idx]
    X_vl = X_eng_train.iloc[val_idx].copy()
    y_tr = y[train_idx]

    # Align val categories
    for col in ["cohort", "gene", "rater_id"]:
        if col in X_tr.columns:
            X_vl[col] = X_vl[col].cat.set_categories(X_tr[col].cat.categories)

    # Augment train fold with pseudo-labeled test rows
    X_aug = pd.concat([X_tr, X_eng_test], ignore_index=True)
    y_aug = np.concatenate([y_tr, y_test_pseudo])
    w_aug = np.concatenate([
        np.ones(len(y_tr)),
        np.full(len(y_test_pseudo), PSEUDO_WEIGHT),
    ])

    # Align augmented categories
    for col in ["cohort", "gene", "rater_id"]:
        if col in X_aug.columns:
            X_aug[col] = X_aug[col].cat.set_categories(X_tr[col].cat.categories)

    m = lgb.LGBMRegressor(**lgbm_params)
    m.fit(X_aug, y_aug, sample_weight=w_aug)
    oof_22[val_idx] = m.predict(X_vl)

    r = np.sqrt(np.mean((y[val_idx] - oof_22[val_idx]) ** 2))
    print(f"  Fold {fold_i+1}: {r:.4f}")

rmse_22 = np.sqrt(np.mean((y - oof_22) ** 2))
print(f"\n  22 Pseudo-label OOF RMSE: {rmse_22:.4f}  (exp 12 was 4.59)")

Path("scratch").mkdir(exist_ok=True)
oof_df = visits[["Index"]].copy()
oof_df["pred_22"] = oof_22
oof_df.to_csv("scratch/22_oof.csv", index=False)

# ── Final model (all train + all pseudo-labeled test) ─────────────────────────
print("\n[Final] Fitting on all train + pseudo-labeled test...")
X_final_aug = pd.concat([X_eng_train, X_eng_test], ignore_index=True)
y_final_aug  = np.concatenate([y, y_test_pseudo])
w_final_aug  = np.concatenate([np.ones(len(y)), np.full(len(y_test_pseudo), PSEUDO_WEIGHT)])
for col in ["cohort", "gene", "rater_id"]:
    if col in X_final_aug.columns:
        X_final_aug[col] = X_final_aug[col].cat.set_categories(X_eng_train[col].cat.categories)

m_final = lgb.LGBMRegressor(**lgbm_params)
m_final.fit(X_final_aug, y_final_aug, sample_weight=w_final_aug)

X_test_aligned = X_eng_test_pred.reindex(columns=X_eng_train.columns, fill_value=np.nan)
test_preds = m_final.predict(X_test_aligned)

Path("submissions").mkdir(exist_ok=True)
sub = X_test_raw[["Index"]].copy()
sub["target"] = test_preds
sub.to_csv("submissions/22_pseudo_label.csv", index=False)

test_df = X_test_raw[["Index"]].copy()
test_df["pred_22"] = test_preds
test_df.to_csv("scratch/22_test.csv", index=False)

print(f"  Wrote submissions/22_pseudo_label.csv")
print(f"\n{'='*60}")
print(f"  22 Pseudo-label OOF RMSE: {rmse_22:.4f}")
print(f"{'='*60}")
