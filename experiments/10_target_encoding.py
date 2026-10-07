# %% [markdown]
# # Experiment 10 — Leave-One-Out Target Encoding of patient_id
#
# For each training visit, the "LOO target enc" feature is the mean target of
# that patient's OTHER visits in the same fold (not the current one).
# For unseen test patients: fall back to cohort×gene → cohort → global mean.
#
# This captures individual patient baseline severity directly from the target,
# without using the current row's target (no leakage).

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
from skore import evaluate

from parkinson.hub import get_project

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data...")
X_train_raw = pd.read_csv("data/X_train.csv")
y_train_df = pd.read_csv("data/y_train.csv")
X_test_raw = pd.read_csv("data/X_test.csv")
visits = X_train_raw.merge(y_train_df, on="Index")
print(f"  train: {visits.shape}  test: {X_test_raw.shape}")


# ── Base feature engineering (identical to 08) ────────────────────────────────
def engineer_features_base(df: pd.DataFrame, cat_columns=None) -> pd.DataFrame:
    """Feature set from experiment 08. Returns df with patient_id kept."""
    out = df.copy()

    for col in ["cohort", "gene", "rater_id"]:
        if col in out.columns:
            if cat_columns and col in cat_columns:
                out[col] = pd.Categorical(
                    out[col], categories=cat_columns[col]
                )
            else:
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
            pat_std_ledd=("ledd", "std"),
            pat_median_off=("off", "median"),
            pat_median_on=("on", "median"),
            pat_min_off=("off", "min"),
            pat_max_off=("off", "max"),
            pat_min_on=("on", "min"),
            pat_max_on=("on", "max"),
            pat_n_visits=("age", "count"),
            pat_age_range=("age", lambda x: x.max() - x.min()),
            pat_mean_age=("age", "mean"),
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

    toff = out["time_since_intake_off"].fillna(out["time_since_intake_off"].median())
    out["off_pk_weight"] = 1.0 / (1.0 + np.exp(-0.3 * (toff - 12.0)))
    out["off_debiased"] = out["off"] * out["off_pk_weight"]

    ton = out["time_since_intake_on"].fillna(out["time_since_intake_on"].median())
    out["on_pk_weight"] = 1.0 / (1.0 + np.exp(0.5 * (ton - 2.0)))
    out["on_adjusted"] = out["on"] * out["on_pk_weight"]

    out["ledd_per_year"] = out["ledd"] / (out["years_since_dx"].clip(lower=0.1))
    out["off_on_ratio"] = out["off"] / (out["on"] + 1.0)

    return out  # keeps patient_id + cohort + gene


def add_loo_target_encoding(
    df_train: pd.DataFrame,
    y_train: np.ndarray,
    df_test: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Compute leave-one-out target encoding of patient_id.

    For each training row i belonging to patient p:
        enc[i] = (sum(y[p]) - y[i]) / (n[p] - 1)
    For patients with only 1 visit: enc[i] = mean(y[p]) (no leave-out possible).

    For test rows (unseen patients): fall back to cohort×gene mean, then cohort
    mean, then global mean — all computed from the training fold.

    Parameters
    ----------
    df_train, y_train: training fold with reset index.
    df_test: optional test / val fold (reset index).

    Returns
    -------
    df_train with 'patient_target_enc' column,
    df_test  with 'patient_target_enc' column (or None).
    """
    df_tr = df_train.copy()
    df_tr["_y"] = y_train

    # Per-patient sum and count
    pat_stats = df_tr.groupby("patient_id")["_y"].agg(["sum", "count"])
    pat_stats.columns = ["sum_y", "count_y"]

    pat_sum = pat_stats["sum_y"].to_dict()
    pat_cnt = pat_stats["count_y"].to_dict()

    # LOO encoding for train
    pid = df_tr["patient_id"].values
    ys = df_tr["_y"].values
    enc_train = np.empty(len(df_tr))
    for i, (p, yi) in enumerate(zip(pid, ys)):
        n = pat_cnt[p]
        s = pat_sum[p]
        enc_train[i] = (s - yi) / (n - 1) if n > 1 else s  # single visit: use own mean
    df_tr["patient_target_enc"] = enc_train
    df_tr.drop(columns=["_y"], inplace=True)

    if df_test is None:
        return df_tr, None

    # Build fallback maps from training fold
    global_mean = float(np.mean(y_train))

    # cohort×gene mean (using pat mean per patient, then group)
    pat_mean_map = {p: pat_stats.loc[p, "sum_y"] / pat_stats.loc[p, "count_y"]
                    for p in pat_stats.index}

    # Per-cohort×gene mean
    df_tr_copy = df_train.copy()
    df_tr_copy["_pat_mean"] = df_tr_copy["patient_id"].map(pat_mean_map)
    cg_mean = (
        df_tr_copy.groupby(["cohort", "gene"], observed=False)["_pat_mean"].mean().to_dict()
    )
    c_mean = (
        df_tr_copy.groupby("cohort", observed=False)["_pat_mean"].mean().to_dict()
    )

    df_te = df_test.copy()
    enc_test = np.empty(len(df_te))
    for i, row in enumerate(df_te.itertuples(index=False)):
        p = row.patient_id
        if p in pat_mean_map:
            enc_test[i] = pat_mean_map[p]
        else:
            cohort = getattr(row, "cohort", None)
            gene = getattr(row, "gene", None)
            key = (cohort, gene)
            if key in cg_mean and not np.isnan(cg_mean[key]):
                enc_test[i] = cg_mean[key]
            elif cohort in c_mean and not np.isnan(c_mean[cohort]):
                enc_test[i] = c_mean[cohort]
            else:
                enc_test[i] = global_mean
    df_te["patient_target_enc"] = enc_test

    return df_tr, df_te


# ── Manual GroupKFold CV with LOO encoding ────────────────────────────────────
print("\nBuilding base features...")
df_base = engineer_features_base(visits)
y = visits["target"].values
groups = visits["patient_id"]

print("\n[10] LOO Target Encoding of patient_id (GroupKFold-5)")

cv = GroupKFold(n_splits=5)
cv_splits = list(cv.split(df_base, y, groups=groups))

lgbm_params = dict(
    n_estimators=800,
    num_leaves=127,
    learning_rate=0.05,
    min_child_samples=20,
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

oof_preds = np.zeros(len(y))

for fold_i, (train_idx, val_idx) in enumerate(cv_splits):
    df_tr_raw = df_base.iloc[train_idx].reset_index(drop=True)
    df_val_raw = df_base.iloc[val_idx].reset_index(drop=True)
    y_tr = y[train_idx]

    # LOO encoding
    df_tr_enc, df_val_enc = add_loo_target_encoding(df_tr_raw, y_tr, df_val_raw)

    # Drop identifiers
    drop_cols = ["Index", "patient_id"]
    if "target" in df_tr_enc.columns:
        drop_cols.append("target")

    X_tr = df_tr_enc.drop(columns=drop_cols, errors="ignore")
    X_val = df_val_enc.drop(columns=drop_cols, errors="ignore")

    # Align categories
    for col in ["cohort", "gene", "rater_id"]:
        if col in X_tr.columns:
            X_val[col] = X_val[col].cat.set_categories(X_tr[col].cat.categories)

    model = lgb.LGBMRegressor(**lgbm_params)
    model.fit(X_tr, y_tr)
    oof_preds[val_idx] = model.predict(X_val)

    fold_rmse = np.sqrt(np.mean((y[val_idx] - oof_preds[val_idx]) ** 2))
    print(f"  Fold {fold_i + 1}: RMSE = {fold_rmse:.4f}")

rmse_10 = float(np.sqrt(np.mean((y - oof_preds) ** 2)))
print(f"\n  10 OOF RMSE = {rmse_10:.4f}  (08_blend was 4.91)")

# ── Push Hub report ───────────────────────────────────────────────────────────
print("\n[Push] Hub report...")

# Compute full-data LOO encoding for the Hub report
df_full_enc, _ = add_loo_target_encoding(df_base, y)
drop_full = ["Index", "patient_id"]
if "target" in df_full_enc.columns:
    drop_full.append("target")
X_full = df_full_enc.drop(columns=drop_full, errors="ignore")

lgbm_for_report = lgb.LGBMRegressor(**lgbm_params)
rep_10 = evaluate(lgbm_for_report, X_full, y, splitter=cv_splits)
print(f"  Hub report RMSE (full-data LOO enc): {float(rep_10.metrics.rmse().iloc[:, 0].mean()):.4f}")

project = get_project()
project.put("10_target_encoding", rep_10)
print("  Pushed '10_target_encoding' → Hub")

# ── Submission ────────────────────────────────────────────────────────────────
print("\n[Submission] Building test predictions...")

# Store category maps from full training for test alignment
cat_cols_map = {
    col: df_base[col].cat.categories.tolist()
    for col in ["cohort", "gene", "rater_id"]
    if col in df_base.columns
}
df_test_base = engineer_features_base(X_test_raw, cat_columns=cat_cols_map)

# LOO enc for test (unseen patients → fallback)
_, df_test_enc = add_loo_target_encoding(df_base, y, df_test_base)

drop_test = ["Index", "patient_id"]
X_test_final = df_test_enc.drop(columns=drop_test, errors="ignore")
X_test_final = X_test_final.reindex(columns=X_full.columns, fill_value=np.nan)

final_model = lgb.LGBMRegressor(**lgbm_params)
final_model.fit(X_full, y)

Path("submissions").mkdir(exist_ok=True)
submission = X_test_raw[["Index"]].copy()
submission["target"] = final_model.predict(X_test_final)
submission.to_csv("submissions/10_target_encoding.csv", index=False)
print(f"  Wrote submissions/10_target_encoding.csv  ({len(submission)} rows)")

print(f"\n{'='*60}")
print("RESULTS SUMMARY")
print(f"  08_blend (best prior):     4.91")
print(f"  10_target_encoding OOF:    {rmse_10:.4f}")
print(f"{'='*60}")
