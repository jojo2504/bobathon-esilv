# %% [markdown]
# # Experiment 09 — Within-Patient Lagged Target Features (corrected)
#
# Key idea: for each patient visit, use the expanding mean of that patient's
# own prior target values (sorted by age) as a feature.  At val/test time,
# we predict patient visits sequentially: predict the earliest visit first
# (fallback proxy), then use that prediction as the lag for the next visit.
#
# GroupKFold ensures all visits of a patient land in the same fold, so there
# is no cross-patient leakage.  Within-patient lags on training rows are exact
# (true targets used).  At val/test time, lags are filled autoregressively
# using the model's own predictions for earlier visits of the same patient.

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


# ── Base feature engineering ──────────────────────────────────────────────────
def engineer_features_base(df: pd.DataFrame) -> pd.DataFrame:
    """Build the 08 feature set. Keeps patient_id + age for lag step."""
    out = df.copy()

    for col in ["cohort", "gene", "rater_id"]:
        if col in out.columns:
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

    return out  # keeps patient_id + age


def compute_lag_features_train(df: pd.DataFrame, target: np.ndarray) -> pd.DataFrame:
    """Compute within-patient expanding lag features for training rows.

    Uses true target values.  Each row's lag = expanding stats of all
    strictly prior visits (sorted by age) of the same patient.
    First visit of each patient: NaN → filled by fallback (pat_mean_off).
    """
    out = df.copy()
    n = len(out)
    lag_mean = np.full(n, np.nan)
    lag_std = np.full(n, np.nan)
    lag_last = np.full(n, np.nan)
    lag_count = np.zeros(n, dtype=float)

    out["_target_tmp"] = target
    sorted_df = out[["patient_id", "age", "_target_tmp"]].sort_values(
        ["patient_id", "age"], kind="stable"
    )
    orig_positions = sorted_df.index.values
    pids = sorted_df["patient_id"].values
    tgts = sorted_df["_target_tmp"].values
    pid_codes, _ = pd.factorize(pids, sort=False)

    n_sorted = len(pids)
    cum_sum = np.zeros(n_sorted)
    cum_sum2 = np.zeros(n_sorted)
    cnt = np.zeros(n_sorted, dtype=float)
    lasts = np.full(n_sorted, np.nan)

    prev_pid = -1
    rs, rs2, rc, rl = 0.0, 0.0, 0, np.nan

    for i in range(n_sorted):
        pid = pid_codes[i]
        if pid != prev_pid:
            rs, rs2, rc, rl = 0.0, 0.0, 0, np.nan
            prev_pid = pid
        cum_sum[i] = rs
        cum_sum2[i] = rs2
        cnt[i] = rc
        lasts[i] = rl
        rs += tgts[i]
        rs2 += tgts[i] ** 2
        rc += 1
        rl = tgts[i]

    with np.errstate(invalid="ignore"):
        means = np.where(cnt > 0, cum_sum / cnt, np.nan)
        vars_ = np.where(
            cnt > 1,
            np.maximum(0.0, cum_sum2 / cnt - (cum_sum / cnt) ** 2),
            0.0,
        )
        stds = np.sqrt(vars_)

    lag_mean[orig_positions] = means
    lag_std[orig_positions] = stds
    lag_last[orig_positions] = lasts
    lag_count[orig_positions] = cnt
    out.drop(columns=["_target_tmp"], inplace=True)

    # Fallback for first visit (NaN lag): use pat_mean_off
    fallback = out["pat_mean_off"].fillna(out["off"].fillna(0))
    out["lag_target_mean"] = np.where(np.isnan(lag_mean), fallback.values, lag_mean)
    out["lag_target_std"] = np.where(np.isnan(lag_std), 0.0, lag_std)
    out["lag_target_last"] = np.where(np.isnan(lag_last), fallback.values, lag_last)
    out["lag_target_count"] = lag_count
    return out


def predict_autoregressive(model, df_val: pd.DataFrame, fallback: float,
                            feat_cols: list[str]) -> np.ndarray:
    """Predict val/test rows autoregressively within each patient.

    For each patient (sorted by age):
    - Visit 1: lag = fallback (pat_mean_off or global proxy)
    - Visit k>1: lag = expanding mean of own predictions from visits 1..k-1

    Returns predictions in the original row order of df_val.
    """
    n = len(df_val)
    preds = np.zeros(n)

    for pid, grp in df_val.groupby("patient_id", sort=False):
        sorted_grp = grp.sort_values("age", kind="stable")
        idxs = sorted_grp.index.tolist()  # original iloc positions (reset_index)

        running_sum = 0.0
        running_sum2 = 0.0
        running_cnt = 0
        running_last = np.nan

        for pos, orig_idx in enumerate(idxs):
            row_df = df_val.iloc[[orig_idx]].copy()
            # Set lag features from prior predictions
            if running_cnt == 0:
                row_df["lag_target_mean"] = fallback
                row_df["lag_target_std"] = 0.0
                row_df["lag_target_last"] = fallback
                row_df["lag_target_count"] = 0.0
            else:
                row_df["lag_target_mean"] = running_sum / running_cnt
                row_df["lag_target_std"] = np.sqrt(
                    max(0.0, running_sum2 / running_cnt - (running_sum / running_cnt) ** 2)
                )
                row_df["lag_target_last"] = running_last
                row_df["lag_target_count"] = float(running_cnt)

            X_row = row_df[feat_cols]
            pred = float(model.predict(X_row)[0])
            preds[orig_idx] = pred

            running_sum += pred
            running_sum2 += pred ** 2
            running_cnt += 1
            running_last = pred

    return preds


# ── Build base features ───────────────────────────────────────────────────────
print("\nBuilding base features...")
df_base = engineer_features_base(visits)
y = visits["target"].values
groups = visits["patient_id"]

print("\n[09] Within-Patient Lagged Target Features (corrected, GroupKFold-5)")

cv = GroupKFold(n_splits=5)
cv_splits = list(cv.split(df_base, y, groups=groups))

lgbm_params = dict(
    n_estimators=500,
    num_leaves=63,
    learning_rate=0.1,
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

    # Training: exact lag from true targets
    df_tr_lag = compute_lag_features_train(df_tr_raw, y_tr)

    # Determine feature columns (everything except identifiers)
    drop_cols = ["Index", "patient_id"]
    if "target" in df_tr_lag.columns:
        drop_cols.append("target")
    feat_cols = [c for c in df_tr_lag.columns if c not in drop_cols]

    X_tr = df_tr_lag[feat_cols]
    # Align val categories
    df_val_ready = df_val_raw.copy()
    for col in ["cohort", "gene", "rater_id"]:
        if col in df_tr_lag.columns:
            df_val_ready[col] = df_val_ready[col].cat.set_categories(
                df_tr_lag[col].cat.categories
            )
    # Add placeholder lag columns
    for lag_col in ["lag_target_mean", "lag_target_std", "lag_target_last", "lag_target_count"]:
        df_val_ready[lag_col] = 0.0

    # Fit model on training fold
    model = lgb.LGBMRegressor(**lgbm_params)
    model.fit(X_tr, y_tr)

    # Val: autoregressive prediction within each patient
    global_fallback = float(np.nanmean(y_tr))
    fold_preds = predict_autoregressive(model, df_val_ready, global_fallback, feat_cols)
    oof_preds[val_idx] = fold_preds

    fold_rmse = np.sqrt(np.mean((y[val_idx] - fold_preds) ** 2))
    print(f"  Fold {fold_i + 1}: RMSE = {fold_rmse:.4f}")

rmse_09 = float(np.sqrt(np.mean((y - oof_preds) ** 2)))
print(f"\n  09 OOF RMSE = {rmse_09:.4f}  (08_blend was 4.91)")

# ── Push Hub report ───────────────────────────────────────────────────────────
print("\n[Push] Hub report...")
df_all_lag = compute_lag_features_train(df_base, y)
drop_full = ["Index", "patient_id"]
if "target" in df_all_lag.columns:
    drop_full.append("target")
X_all = df_all_lag.drop(columns=drop_full, errors="ignore")

lgbm_for_report = lgb.LGBMRegressor(**lgbm_params)
rep_09 = evaluate(lgbm_for_report, X_all, y, splitter=cv_splits)
print(f"  Hub report RMSE (reference, leaky full-data lag): {float(rep_09.metrics.rmse().iloc[:, 0].mean()):.4f}")
print(f"  09 true OOF RMSE (autoregressive, no leakage): {rmse_09:.4f}")

project = get_project()
project.put("09_lagged_target", rep_09)
print("  Pushed '09_lagged_target' → Hub")

# ── Submission ────────────────────────────────────────────────────────────────
print("\n[Submission] Building test predictions...")

# Fit final model on all training data with exact lag features
final_model = lgb.LGBMRegressor(**lgbm_params)
final_model.fit(X_all, y)

# For test: autoregressive within each patient
df_test_base = engineer_features_base(X_test_raw)
# Align categories
for col in ["cohort", "gene", "rater_id"]:
    if col in df_all_lag.columns:
        df_test_base[col] = df_test_base[col].cat.set_categories(
            df_all_lag[col].cat.categories
        )
for lag_col in ["lag_target_mean", "lag_target_std", "lag_target_last", "lag_target_count"]:
    df_test_base[lag_col] = 0.0

feat_cols_final = [c for c in X_all.columns]
global_fallback_final = float(np.nanmean(y))
test_preds = predict_autoregressive(
    final_model, df_test_base, global_fallback_final, feat_cols_final
)

Path("submissions").mkdir(exist_ok=True)
submission = X_test_raw[["Index"]].copy()
submission["target"] = test_preds
submission.to_csv("submissions/09_lagged_target.csv", index=False)
print(f"  Wrote submissions/09_lagged_target.csv  ({len(submission)} rows)")

print(f"\n{'='*60}")
print("RESULTS SUMMARY")
print(f"  08_blend (best prior):   4.91")
print(f"  09_lagged_target OOF:    {rmse_09:.4f}")
print(f"{'='*60}")
