# %% [markdown]
# # Experiment 25 — Transformer with explicit trend features
#
# Diagnosis from exps 23/24:
#   - In-sample per-patient linear fit (target = slope*age + intercept): RMSE 1.61
#     → this is the achievable floor and proves RMSE < 2.0 is possible
#   - Within-patient r(age, target) = 0.97: disease progression is nearly linear in time
#   - Transformer at 3.91 (exp 13) is underfitting, not a wrong architecture
#   - Exp 23 inner-val split removed 15% of train patients → hurt accuracy
#   - Early stopping never fired anyway → pure regression
#
# Fixes in exp 25:
#   1. No inner-val split — use ALL train patients every fold
#   2. 150 epochs with cosine annealing (better convergence than step decay)
#   3. hidden=256 (double exp 23) — more capacity to learn per-patient trends
#   4. NEW: explicit per-patient trend features added to each visit token:
#        - pat_off_slope, pat_on_slope (linear fit of off/on vs age — no target needed)
#        - pat_target_at_mean_age (predicted from pm_off via Ridge — computed from train)
#        These give the model a strong prior on each patient's trend,
#        letting attention focus on per-visit residuals.
#   5. Full exp-12 feature set (31 features) — exp 24 used fewer features

# %%
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ── Device ─────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")


# ── Config ─────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    context_length: int = 12
    hidden_dim: int = 128       # same as exp 23 — key fix is no inner-val split, not bigger model
    n_head: int = 8
    num_layers: int = 2
    mul: int = 4
    dropout_att: float = 0.1
    dropout_mlp: float = 0.1
    lr: float = 5e-3
    weight_decay: float = 1e-2
    batch_size: int = 256       # large batches saturate MPS efficiently
    epochs: int = 120           # longer than exp 13's 100 (model was still improving at ep100)
    warmup_epochs: int = 10     # cosine warmup
    n_ensemble: int = 8
    n_folds: int = 5
    null_value: float = -999.0
    device: torch.device = field(default_factory=lambda: DEVICE)


cfg = Config()

# ── Feature definitions ────────────────────────────────────────────────────────
# Same base set as exp 12/17 plus explicit trend features
NULLABLE_FEATURES = [
    "time_since_intake_on", "time_since_intake_off", "ledd",
    "age_at_diagnosis", "on", "off",
]
PAT_AGG_COLS = [
    "pat_mean_off", "pat_mean_on", "pat_std_off", "pat_std_on",
    "pat_median_off", "pat_min_off", "pat_max_off",
    "pat_median_on", "pat_min_on", "pat_max_on",
    "pat_n_visits", "pat_age_range", "pat_mean_ledd",
]
ENGINEERED_COLS = [
    "off_pk_weight", "on_pk_weight", "off_debiased", "on_adjusted",
    "off_minus_on", "off_on_ratio",
    "off_z_pat", "on_z_pat",
    "off_vs_pat_mean", "on_vs_pat_mean",
    "off_vs_pat_median", "on_vs_pat_median",
    "off_range_pct",
    "ledd_per_year", "ledd_x_off", "age_x_off",
    "years_since_dx",
    "visit_rank", "visit_rank_pct",
    # Trend features (NEW — computable from X only, no target)
    "pat_off_slope",    # slope of off vs age for this patient
    "pat_on_slope",     # slope of on vs age for this patient
    "pat_off_resid_std",  # std of residuals after detrending off
]
GENE_DUMMIES = ["GBA+", "LRRK2+", "OTHER+", "No Mutation"]


def compute_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-patient linear trend features (off slope, on slope).

    These are computed from the patient's own X features at both train and test time —
    no target labels needed. They give the Transformer an explicit prior on each
    patient's disease progression rate.
    """
    results: dict[str, dict] = {}
    for pid, grp in df.groupby("patient_id"):
        ages = grp["age"].values
        m_age = float(ages.mean())

        def slope_of(vals: np.ndarray) -> tuple[float, float]:
            valid = ~np.isnan(vals)
            if valid.sum() < 2:
                return 0.0, 0.0
            A = np.column_stack([ages[valid] - m_age, np.ones(valid.sum())])
            p, _, _, _ = np.linalg.lstsq(A, vals[valid], rcond=None)
            resid = vals[valid] - (p[0] * (ages[valid] - m_age) + p[1])
            return float(p[0]), float(resid.std() if len(resid) > 1 else 0.0)

        off_s, off_rs = slope_of(grp["off"].values)
        on_s, _ = slope_of(grp["on"].values)
        results[str(pid)] = {
            "pat_off_slope": off_s,
            "pat_on_slope": on_s,
            "pat_off_resid_std": off_rs,
        }

    rows = []
    for pid, vals in results.items():
        rows.append({"patient_id": pid, **vals})
    trend_df = pd.DataFrame(rows)
    trend_df["patient_id"] = trend_df["patient_id"].astype(df["patient_id"].dtype)
    return trend_df


def engineer_features(df_pd: pd.DataFrame) -> pd.DataFrame:
    """Full exp-12 feature set plus explicit trend features."""
    out = df_pd.copy()
    out["years_since_dx"] = out["age"] - out["age_at_diagnosis"]

    pat = (
        out.groupby("patient_id", sort=False)
        .agg(
            pat_mean_off=("off", "mean"),
            pat_mean_on=("on", "mean"),
            pat_std_off=("off", "std"),
            pat_std_on=("on", "std"),
            pat_mean_ledd=("ledd", "mean"),
            pat_median_off=("off", "median"),
            pat_min_off=("off", "min"),
            pat_max_off=("off", "max"),
            pat_median_on=("on", "median"),
            pat_min_on=("on", "min"),
            pat_max_on=("on", "max"),
            pat_n_visits=("age", "count"),
            pat_age_range=("age", lambda x: float(x.max() - x.min())),
        )
        .reset_index()
    )
    out = out.merge(pat, on="patient_id", how="left")

    # PK timing
    toff = out["time_since_intake_off"].fillna(out["time_since_intake_off"].median())
    out["off_pk_weight"] = 1.0 / (1.0 + np.exp(-0.3 * (toff - 12.0)))
    out["off_debiased"] = out["off"] * out["off_pk_weight"]
    ton = out["time_since_intake_on"].fillna(out["time_since_intake_on"].median())
    out["on_pk_weight"] = 1.0 / (1.0 + np.exp(0.5 * (ton - 2.0)))
    out["on_adjusted"] = out["on"] * out["on_pk_weight"]

    # Motor gap features
    out["off_minus_on"] = out["off"] - out["on"]
    out["off_on_ratio"] = out["off"] / (out["on"] + 1.0)

    # Within-patient deviation features
    out["off_vs_pat_mean"] = out["off"] - out["pat_mean_off"]
    out["on_vs_pat_mean"] = out["on"] - out["pat_mean_on"]
    out["off_vs_pat_median"] = out["off"] - out["pat_median_off"]
    out["on_vs_pat_median"] = out["on"] - out["pat_median_on"]
    out["off_range_pct"] = (out["off"] - out["pat_min_off"]) / (
        out["pat_max_off"] - out["pat_min_off"] + 1e-6
    )
    out["off_z_pat"] = out["off_vs_pat_mean"] / (out["pat_std_off"].fillna(0) + 1e-6)
    out["on_z_pat"] = out["on_vs_pat_mean"] / (out["pat_std_on"].fillna(0) + 1e-6)

    # Dose features
    out["ledd_per_year"] = out["ledd"] / (out["years_since_dx"].clip(lower=0.1))
    out["ledd_x_off"] = out["ledd"].fillna(0) * out["off"].fillna(0)
    out["age_x_off"] = out["age"] * out["off"].fillna(0)

    # Visit order
    out["visit_rank"] = (
        out.groupby("patient_id")["age"].rank(method="first").astype(float)
    )
    out["visit_rank_pct"] = out["visit_rank"] / out["pat_n_visits"]

    # Explicit trend features (computable without target labels)
    trend_df = compute_trend_features(out[["patient_id", "age", "off", "on"]])
    out = out.merge(trend_df, on="patient_id", how="left")

    return out


def build_patient_matrix(
    df_pd: pd.DataFrame,
    df_label: pd.DataFrame | None,
    context_length: int,
    norm_stats: dict | None,
) -> tuple[np.ndarray, np.ndarray | None, dict, list[str]]:
    df = df_pd.copy()

    for g in GENE_DUMMIES:
        df[f"gene_{g}"] = (df["gene"] == g).astype("int8")
    df["cohort"] = df["cohort"].map({"A": 0, "B": 1}).fillna(0).astype("int8")

    if df_label is not None:
        df = df.merge(
            df_label[["Index", "target"]].rename(columns={"target": "_target"}),
            on="Index",
            how="left",
        )

    df = df.sort_values(["patient_id", "age"]).reset_index(drop=True)

    normalise_cols = NULLABLE_FEATURES + [
        c for c in PAT_AGG_COLS + ENGINEERED_COLS if c in df.columns
    ]
    normalise_cols = list(dict.fromkeys(normalise_cols))  # dedup
    normalise_cols = [c for c in normalise_cols if c in df.columns]

    computed_stats: dict = {}
    for col in normalise_cols:
        if norm_stats is not None:
            mu, sigma = norm_stats[col]
        else:
            mu = float(df[col].mean())
            sigma = float(df[col].std()) + 1e-8
            mu = 0.0 if np.isnan(mu) else mu
            sigma = 1.0 if np.isnan(sigma) else sigma
        computed_stats[col] = (mu, sigma)
        df[col] = (df[col] - mu) / sigma

    for col in NULLABLE_FEATURES:
        df[f"flag_{col}"] = df[col].isna().astype("int8")

    df = df.fillna(0)

    base_features = [
        "cohort", "sexM",
        "gene_GBA+", "gene_LRRK2+", "gene_OTHER+", "gene_No Mutation",
        "age_at_diagnosis", "age", "ledd",
        "time_since_intake_on", "time_since_intake_off", "on", "off",
    ]
    flag_features = [f"flag_{c}" for c in NULLABLE_FEATURES]
    pat_agg_available = [c for c in PAT_AGG_COLS if c in df.columns]
    eng_available = [c for c in ENGINEERED_COLS if c in df.columns]
    feat_cols = base_features + flag_features + pat_agg_available + eng_available

    patient_ids: list[str] = list(dict.fromkeys(df["patient_id"].tolist()))
    n_patients = len(patient_ids)
    n_feats = len(feat_cols)

    X_pat = np.zeros((n_patients, context_length * n_feats), dtype=np.float32)
    y_pat = (
        np.full((n_patients, context_length), cfg.null_value, dtype=np.float32)
        if df_label is not None
        else None
    )

    feat_arr = df[feat_cols].to_numpy(dtype=np.float32)
    tgt_arr = df["_target"].to_numpy(dtype=np.float32) if df_label is not None else None
    pid_list = df["patient_id"].tolist()

    pid_to_rows: dict[str, list[int]] = {}
    for i, pid in enumerate(pid_list):
        pid_to_rows.setdefault(str(pid), []).append(i)

    for p_idx, pid in enumerate(patient_ids):
        row_indices = pid_to_rows[str(pid)]
        n_visits = min(len(row_indices), context_length)
        for v, row_idx in enumerate(row_indices[:n_visits]):
            X_pat[p_idx, v * n_feats : (v + 1) * n_feats] = feat_arr[row_idx]
            if y_pat is not None and tgt_arr is not None:
                y_pat[p_idx, v] = tgt_arr[row_idx]

    stats_out = computed_stats if norm_stats is None else norm_stats
    return X_pat, y_pat, stats_out, patient_ids


# ── Model ──────────────────────────────────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=cfg.hidden_dim, num_heads=cfg.n_head,
            dropout=cfg.dropout_att, batch_first=True,
        )
        self.ln2 = nn.LayerNorm(cfg.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.mul * cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout_mlp),
            nn.Linear(cfg.mul * cfg.hidden_dim, cfg.hidden_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        n = self.ln1(x)
        a, _ = self.attn(n, n, n, key_padding_mask=mask)
        x = x + a
        return x + self.ffn(self.ln2(x))


class VisitTransformer(nn.Module):
    def __init__(self, cfg: Config, n_features: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_features = n_features
        self.tok_embed = nn.Sequential(
            nn.Linear(n_features, cfg.mul * cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.mul * cfg.hidden_dim, cfg.hidden_dim),
        )
        self.pos_embed = nn.Embedding(cfg.context_length, cfg.hidden_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.num_layers)]
        )
        self.ln_final = nn.LayerNorm(cfg.hidden_dim)
        self.head = nn.Linear(cfg.hidden_dim, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B = x.size(0)
        x = x.view(B, self.cfg.context_length, self.n_features)
        h = self.tok_embed(x) + self.pos_embed(
            torch.arange(self.cfg.context_length, device=x.device)
        )
        for block in self.blocks:
            h = block(h, mask)
        return self.head(self.ln_final(h))  # [B, T, 1]


# ── Cosine LR schedule with warmup ────────────────────────────────────────────
def cosine_schedule(epoch: int, total: int, warmup: int, base_lr: float) -> float:
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))


def train_one_epoch(
    models: list[VisitTransformer],
    loader: DataLoader,
    optimizers: list[optim.Optimizer],
    cfg: Config,
) -> float:
    for m in models:
        m.train()
    total_loss, total_n = 0.0, 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(cfg.device), y_b.to(cfg.device)
        mask = y_b == cfg.null_value
        out = torch.cat([m(X_b, mask).squeeze(-1) for m in models], dim=0)
        y_exp = y_b.repeat(len(models), 1)
        valid = ~mask.repeat(len(models), 1)
        if valid.sum() == 0:
            continue
        loss = F.mse_loss(out[valid], y_exp[valid])
        loss.backward()
        for opt in optimizers:
            nn.utils.clip_grad_norm_(
                [p for g in opt.param_groups for p in g["params"]], max_norm=1.0
            )
            opt.step()
            opt.zero_grad()
        n = int(valid.sum().item()) // len(models)
        total_loss += loss.item() * n
        total_n += n
    return total_loss / max(total_n, 1)


@torch.inference_mode()
def predict_ensemble(
    models: list[VisitTransformer],
    X_t: torch.Tensor,
    y_t: torch.Tensor,
    cfg: Config,
) -> tuple[np.ndarray, float]:
    for m in models:
        m.eval()
    loader = DataLoader(
        TensorDataset(X_t, y_t), batch_size=cfg.batch_size * 4, shuffle=False
    )
    preds_per_model: list[list[np.ndarray]] = [[] for _ in models]
    total_mse, total_n = 0.0, 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(cfg.device), y_b.to(cfg.device)
        mask = y_b == cfg.null_value
        valid = ~mask
        stacked = torch.stack([m(X_b, mask).squeeze(-1) for m in models])  # [N, B, T]
        mean_pred = stacked.mean(0)
        if valid.sum() > 0:
            total_mse += F.mse_loss(mean_pred[valid], y_b[valid]).item() * int(valid.sum())
            total_n += int(valid.sum())
        for i, pp in enumerate(preds_per_model):
            pp.append(stacked[i].cpu().numpy())
    mse = total_mse / max(total_n, 1)
    per_model = np.stack(
        [np.concatenate(pp, axis=0) for pp in preds_per_model], axis=0
    )  # [N, n_patients, T]
    return per_model.mean(0), mse


# ── Load & engineer features ───────────────────────────────────────────────────
print("Loading data...")
X_train_raw = pd.read_csv("data/X_train.csv")
y_train_df = pd.read_csv("data/y_train.csv")
X_test_raw = pd.read_csv("data/X_test.csv")

print("Engineering features (including trend features)...")
X_eng_train = engineer_features(X_train_raw)
X_eng_test = engineer_features(X_test_raw)
train_with_labels = X_eng_train.merge(y_train_df, on="Index")

print("Building patient matrices...")
X_pat_np, y_pat_np, global_stats, patient_ids_ordered = build_patient_matrix(
    train_with_labels, y_train_df, cfg.context_length, norm_stats=None
)
n_patients, total_feats = X_pat_np.shape
n_features = total_feats // cfg.context_length
print(f"  Patients: {n_patients}  Features/visit: {n_features}")

X_pat = torch.tensor(X_pat_np, dtype=torch.float32)
y_pat = torch.tensor(y_pat_np, dtype=torch.float32)

cv = GroupKFold(n_splits=cfg.n_folds)
cv_splits = list(cv.split(X_pat_np, np.zeros(n_patients), groups=np.arange(n_patients)))


# ── Cross-validation (no inner-val split — full train data every fold) ─────────
print(f"\n[25] Transformer trend  GroupKFold-{cfg.n_folds}  "
      f"ensemble={cfg.n_ensemble}  hidden={cfg.hidden_dim}  epochs={cfg.epochs}")
fold_val_rmses: list[float] = []
oof_patient_preds = np.full((n_patients, cfg.context_length), np.nan, dtype=np.float32)

for fold_i, (train_idx, val_idx) in enumerate(cv_splits):
    print(f"\n--- Fold {fold_i + 1} / {cfg.n_folds} ---")
    t0 = time.time()

    # ALL train patients used — no inner-val holdout
    train_ds = TensorDataset(X_pat[train_idx], y_pat[train_idx])
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    models = [
        VisitTransformer(cfg, n_features).to(cfg.device)
        for _ in range(cfg.n_ensemble)
    ]
    optimizers = [
        optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        for m in models
    ]

    for epoch in range(cfg.epochs):
        # Cosine LR update
        lr = cosine_schedule(epoch, cfg.epochs, cfg.warmup_epochs, cfg.lr)
        for opt in optimizers:
            for pg in opt.param_groups:
                pg["lr"] = lr

        train_one_epoch(models, train_loader, optimizers, cfg)

        if (epoch + 1) % 30 == 0:
            elapsed = (time.time() - t0) / 60
            print(
                f"  Fold {fold_i+1} epoch {epoch+1:3d}/{cfg.epochs}  lr={lr:.5f}  ({elapsed:.1f} min)",
                flush=True,
            )

    val_preds, val_mse = predict_ensemble(
        models, X_pat[val_idx], y_pat[val_idx], cfg
    )
    val_rmse = val_mse ** 0.5
    fold_val_rmses.append(val_rmse)
    oof_patient_preds[val_idx] = val_preds
    print(
        f"  Fold {fold_i+1} FINAL val RMSE={val_rmse:.4f}  ({(time.time()-t0)/60:.1f} min)",
        flush=True,
    )

mean_val_rmse = float(np.mean(fold_val_rmses))
print(f"\n{'='*60}")
print(f"  CV RMSE = {mean_val_rmse:.4f}  (target <= 2.00)")
print(f"  Per-fold: {[f'{r:.4f}' for r in fold_val_rmses]}")
print(f"{'='*60}")


# ── Map OOF predictions to visit level ────────────────────────────────────────
pid_to_pat_idx = {str(pid): i for i, pid in enumerate(patient_ids_ordered)}
visits_sorted = train_with_labels.sort_values(["patient_id", "age"]).reset_index(drop=True)
oof_visit_preds = np.full(len(X_train_raw), np.nan, dtype=np.float64)

for pid, grp in visits_sorted.groupby("patient_id", sort=False):
    p_idx = pid_to_pat_idx[str(pid)]
    for v_idx, orig_idx in enumerate(grp["Index"]):
        if v_idx < cfg.context_length:
            row_pos = X_train_raw.index[X_train_raw["Index"] == orig_idx].tolist()
            if row_pos:
                oof_visit_preds[row_pos[0]] = float(oof_patient_preds[p_idx, v_idx])

valid_oof = ~np.isnan(oof_visit_preds)
y_full = train_with_labels.set_index("Index").loc[X_train_raw["Index"], "target"].values
oof_rmse_visit = float(
    np.sqrt(np.mean((y_full[valid_oof] - oof_visit_preds[valid_oof]) ** 2))
)
print(f"\n  Visit-level OOF RMSE: {oof_rmse_visit:.4f}")

Path("scratch").mkdir(exist_ok=True)
oof_df = X_train_raw[["Index"]].copy()
oof_df["pred_25"] = oof_visit_preds
oof_df.to_csv("scratch/25_oof_visit.csv", index=False)
print("  Saved scratch/25_oof_visit.csv")


# ── Final training on all patients ────────────────────────────────────────────
print(f"\n[Final] Training on ALL {n_patients} patients...")
full_ds = TensorDataset(X_pat, y_pat)
full_loader = DataLoader(full_ds, batch_size=cfg.batch_size, shuffle=True)

final_models = [
    VisitTransformer(cfg, n_features).to(cfg.device) for _ in range(cfg.n_ensemble)
]
final_optimizers = [
    optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    for m in final_models
]

t0 = time.time()
for epoch in range(cfg.epochs):
    lr = cosine_schedule(epoch, cfg.epochs, cfg.warmup_epochs, cfg.lr)
    for opt in final_optimizers:
        for pg in opt.param_groups:
            pg["lr"] = lr
    train_one_epoch(final_models, full_loader, final_optimizers, cfg)
    if (epoch + 1) % 30 == 0:
        print(f"  Final epoch {epoch+1}/{cfg.epochs}  ({(time.time()-t0)/60:.1f} min)", flush=True)
print(f"  Final training done in {(time.time()-t0)/60:.1f} min")


# ── Test predictions ───────────────────────────────────────────────────────────
print("\n[Submission] Building test predictions...")
X_test_np, _, _, test_patient_ids = build_patient_matrix(
    X_eng_test, None, cfg.context_length, norm_stats=global_stats
)
X_test_t = torch.tensor(X_test_np, dtype=torch.float32).to(cfg.device)
y_dummy = torch.zeros(len(test_patient_ids), cfg.context_length)

test_preds_pat, _ = predict_ensemble(final_models, X_test_t, y_dummy, cfg)
pid_to_test = {str(pid): test_preds_pat[i] for i, pid in enumerate(test_patient_ids)}

rows: list[dict] = []
for pid, grp in X_test_raw.sort_values(["patient_id", "age"]).groupby("patient_id", sort=False):
    preds = pid_to_test[str(pid)]
    for v_idx, (_, row) in enumerate(grp.iterrows()):
        if v_idx < cfg.context_length:
            rows.append({"Index": int(row["Index"]), "target": float(preds[v_idx])})

submission = pd.DataFrame(rows).sort_values("Index").reset_index(drop=True)
Path("submissions").mkdir(exist_ok=True)
submission.to_csv("submissions/25_transformer_trend.csv", index=False)
print(f"  Wrote submissions/25_transformer_trend.csv  ({len(submission)} rows)")

test_pred_df = X_test_raw[["Index"]].copy()
test_pred_df["pred_25"] = test_pred_df["Index"].map(submission.set_index("Index")["target"])
test_pred_df.to_csv("scratch/25_test_visit.csv", index=False)
print("  Saved scratch/25_test_visit.csv  (for blend in exp 26)")

print(f"\n{'='*60}")
print("RESULTS SUMMARY")
print(f"  13 Transformer (best so far):  3.91")
print(f"  23 Fast Transformer:           4.09  (had inner-val split bug)")
print(f"  25 Transformer trend CV RMSE:  {mean_val_rmse:.4f}")
print(f"  25 Visit-level OOF RMSE:       {oof_rmse_visit:.4f}")
print(f"  In-sample per-patient linear:  1.61  (floor)")
print(f"  Target:                        <= 2.00")
print(f"{'='*60}")
print("\nNext: python experiments/26_blend_v2.py")
