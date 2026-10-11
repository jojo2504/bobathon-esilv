# %% [markdown]
# # Experiment 17 — Transformer v4: enriched features + ensemble=30 + correct alignment
#
# Three key improvements over exp 14/15/16:
#   1. Correct patient-index alignment: `build_patient_matrix` returns the
#      ordered patient list; GroupKFold splits are derived from that exact list.
#   2. Full enriched feature set from exp 12/14 (~47 features/visit):
#      raw + flags + patient aggregates + PK timing + motor gap + z-scores + visit rank.
#   3. Peer-inspired architecture: num_layers=2, n_head=8, ensemble=30.
#      Shallow+wide + massive ensembling is the reference 2nd-place recipe.
#   4. Fold-local normalisation: norm stats computed from train patients only.
#   5. OOF predictions saved for blending with GBM in exp 18.

# %%
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupKFold

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")


# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    context_length: int = 12
    hidden_dim: int = 64
    n_head: int = 8              # peer: wide attention, fewer layers
    num_layers: int = 2          # shallow to avoid overfitting
    mul: int = 4
    dropout_att: float = 0.1
    dropout_mlp: float = 0.1
    lr: float = 1e-2             # peer's higher lr
    weight_decay: float = 1e-2
    batch_size: int = 64
    epochs: int = 100            # full run — peer recipe
    step_size: int = 20
    gamma: float = 0.5
    n_ensemble: int = 30         # peer's massive ensembling
    n_folds: int = 5
    null_value: float = -999.0   # sentinel for padded positions
    deviation: float = 0.02
    device: torch.device = field(default_factory=lambda: DEVICE)


cfg = Config()

# ── Feature definitions ────────────────────────────────────────────────────────
NULLABLE_FEATURES = [
    "time_since_intake_on", "time_since_intake_off", "ledd",
    "age_at_diagnosis", "time_since_diagnosis", "on", "off",
]
PAT_AGG_COLS = [
    "pat_mean_off", "pat_mean_on", "pat_std_off", "pat_std_on",
    "pat_median_off", "pat_min_off", "pat_max_off",
    "pat_min_on", "pat_max_on",
    "pat_n_visits", "pat_age_range", "pat_mean_ledd",
]
ENGINEERED_COLS = [
    "off_pk_weight", "on_pk_weight", "off_debiased", "on_adjusted",
    "off_minus_on", "off_on_ratio",
    "off_z_pat", "on_z_pat",
    "off_vs_pat_mean", "on_vs_pat_mean",
    "ledd_per_year", "years_since_dx",
    "visit_rank", "visit_rank_pct",
]
GENE_DUMMIES = ["GBA+", "LRRK2+", "OTHER+", "No Mutation"]


# ── Feature engineering (pandas) ──────────────────────────────────────────────
def engineer_features(df_pd: pd.DataFrame) -> pd.DataFrame:
    """Add patient aggregates and engineered features (exp 12 feature set)."""
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
            pat_min_on=("on", "min"),
            pat_max_on=("on", "max"),
            pat_n_visits=("age", "count"),
            pat_age_range=("age", lambda x: float(x.max() - x.min())),
        )
        .reset_index()
    )
    out = out.merge(pat, on="patient_id", how="left")

    toff = out["time_since_intake_off"].fillna(out["time_since_intake_off"].median())
    out["off_pk_weight"] = 1.0 / (1.0 + np.exp(-0.3 * (toff - 12.0)))
    out["off_debiased"] = out["off"] * out["off_pk_weight"]

    ton = out["time_since_intake_on"].fillna(out["time_since_intake_on"].median())
    out["on_pk_weight"] = 1.0 / (1.0 + np.exp(0.5 * (ton - 2.0)))
    out["on_adjusted"] = out["on"] * out["on_pk_weight"]

    out["off_minus_on"] = out["off"] - out["on"]
    out["off_on_ratio"] = out["off"] / (out["on"] + 1.0)

    out["off_vs_pat_mean"] = out["off"] - out["pat_mean_off"]
    out["on_vs_pat_mean"] = out["on"] - out["pat_mean_on"]
    out["off_z_pat"] = out["off_vs_pat_mean"] / (out["pat_std_off"].fillna(0) + 1e-6)
    out["on_z_pat"] = out["on_vs_pat_mean"] / (out["pat_std_on"].fillna(0) + 1e-6)

    out["ledd_per_year"] = out["ledd"] / (out["years_since_dx"].clip(lower=0.1))

    out["visit_rank"] = (
        out.groupby("patient_id")["age"].rank(method="first").astype(float)
    )
    out["visit_rank_pct"] = out["visit_rank"] / out["pat_n_visits"]

    return out


# ── Patient matrix builder ─────────────────────────────────────────────────────
def build_patient_matrix(
    df_pd: pd.DataFrame,
    df_label: pd.DataFrame | None,
    context_length: int,
    norm_stats: dict | None,
) -> tuple[np.ndarray, np.ndarray | None, dict, list[str]]:
    """Convert per-visit rows into per-patient matrices (pure pandas/numpy).

    Returns
    -------
    X_pat      : [n_patients, context_length * n_features]
    y_pat      : [n_patients, context_length] padded with null_value, or None
    stats      : normalisation stats dict
    patient_ids: list of patient_id strings in the same row order as X_pat/y_pat
    """
    df = df_pd.copy()

    # Gene one-hot
    for g in GENE_DUMMIES:
        df[f"gene_{g}"] = (df["gene"] == g).astype("int8")
    # Cohort binary
    df["cohort"] = df["cohort"].map({"A": 0, "B": 1}).astype("int8")
    # Time since diagnosis
    df["time_since_diagnosis"] = df["age"] - df["age_at_diagnosis"]

    if df_label is not None:
        df = df.merge(df_label[["Index", "target"]].rename(columns={"target": "_target"}),
                      on="Index", how="left")

    # Sort chronologically within each patient
    df = df.sort_values(["patient_id", "age"]).reset_index(drop=True)

    # Normalisation — using train-fold stats if provided
    normalise_cols = NULLABLE_FEATURES + [
        "pat_mean_off", "pat_mean_on", "pat_std_off", "pat_std_on",
        "pat_median_off", "pat_min_off", "pat_max_off",
        "pat_min_on", "pat_max_on", "pat_mean_ledd", "pat_age_range",
        "off_pk_weight", "on_pk_weight", "off_debiased", "on_adjusted",
        "off_minus_on", "off_on_ratio", "off_z_pat", "on_z_pat",
        "off_vs_pat_mean", "on_vs_pat_mean",
        "ledd_per_year", "years_since_dx", "visit_rank", "visit_rank_pct",
        "time_since_diagnosis",
    ]
    normalise_cols = [c for c in normalise_cols if c in df.columns]

    computed_stats: dict = {}
    for col in normalise_cols:
        if norm_stats is not None:
            mu, sigma = norm_stats[col]
        else:
            mu = float(df[col].mean())
            sigma = float(df[col].std()) + 1e-8
            if np.isnan(mu):
                mu = 0.0
            if np.isnan(sigma):
                sigma = 1.0
        computed_stats[col] = (mu, sigma)
        df[col] = (df[col] - mu) / sigma

    # Flag columns (1 = was null before normalisation)
    for col in NULLABLE_FEATURES:
        df[f"flag_{col}"] = df[col].isna().astype("int8")

    df = df.fillna(0)

    base_features = [
        "cohort", "sexM",
        "gene_GBA+", "gene_LRRK2+", "gene_OTHER+", "gene_No Mutation",
        "age_at_diagnosis", "age", "ledd",
        "time_since_intake_on", "time_since_intake_off",
        "on", "off", "time_since_diagnosis",
    ]
    flag_features = [f"flag_{c}" for c in NULLABLE_FEATURES]
    pat_agg_available = [c for c in PAT_AGG_COLS if c in df.columns]
    eng_available = [c for c in ENGINEERED_COLS if c in df.columns]
    feat_cols = base_features + flag_features + pat_agg_available + eng_available

    # Canonical patient order (first appearance after sort by age)
    patient_ids: list[str] = list(dict.fromkeys(df["patient_id"].tolist()))
    n_patients = len(patient_ids)
    n_feats = len(feat_cols)

    X_pat = np.zeros((n_patients, context_length * n_feats), dtype=np.float32)
    y_pat = (
        np.full((n_patients, context_length), cfg.null_value, dtype=np.float32)
        if df_label is not None else None
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
            X_pat[p_idx, v * n_feats:(v + 1) * n_feats] = feat_arr[row_idx]
            if y_pat is not None and tgt_arr is not None:
                y_pat[p_idx, v] = tgt_arr[row_idx]

    stats_out = computed_stats if norm_stats is None else norm_stats
    return X_pat, y_pat, stats_out, patient_ids


# ── Model ─────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, hidden_dim: int, mul: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mul * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mul * hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=cfg.hidden_dim, num_heads=cfg.n_head,
            dropout=cfg.dropout_att, batch_first=True,
        )
        self.ln2 = nn.LayerNorm(cfg.hidden_dim)
        self.mlp = MLP(cfg.hidden_dim, cfg.mul, cfg.dropout_mlp)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        normed = self.ln1(x)
        attn_out, _ = self.attn(normed, normed, normed, key_padding_mask=key_padding_mask)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class VisitTransformer(nn.Module):
    def __init__(self, cfg: Config, n_features: int):
        super().__init__()
        self.cfg = cfg
        self.n_features = n_features
        self.tok_embed = nn.Sequential(
            nn.Linear(n_features, cfg.mul * cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.mul * cfg.hidden_dim, cfg.hidden_dim),
        )
        self.pos_embed = nn.Embedding(cfg.context_length, cfg.hidden_dim)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.ln_final = nn.LayerNorm(cfg.hidden_dim)
        self.head = nn.Linear(cfg.hidden_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=self.cfg.deviation)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=self.cfg.deviation)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        B = x.size(0)
        x = x.view(B, self.cfg.context_length, self.n_features)
        tok = self.tok_embed(x)
        pos = self.pos_embed(torch.arange(self.cfg.context_length, device=x.device))
        h = tok + pos
        for block in self.blocks:
            h = block(h, key_padding_mask=padding_mask)
        h = self.ln_final(h)
        return self.head(h)  # [B, T, 1]


# ── Training helpers ──────────────────────────────────────────────────────────
def make_padding_mask(y_batch: torch.Tensor, null_value: float) -> torch.Tensor:
    return y_batch == null_value


def train_one_epoch(
    model: VisitTransformer,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    cfg: Config,
) -> float:
    model.train()
    total_loss, total_n = 0.0, 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(cfg.device), y_b.to(cfg.device)
        padding_mask = make_padding_mask(y_b, cfg.null_value)
        optimizer.zero_grad()
        pred = model(X_b, padding_mask).squeeze(-1)
        valid = ~padding_mask
        if valid.sum() == 0:
            continue
        loss = F.mse_loss(pred[valid], y_b[valid])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        n = int(valid.sum().item())
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
    """Predict with all ensemble members, average, return predictions + MSE."""
    for m in models:
        m.eval()
    loader = DataLoader(
        TensorDataset(X_t, y_t), batch_size=cfg.batch_size * 4, shuffle=False
    )
    all_preds: list[np.ndarray] = []
    all_valid_y: list[np.ndarray] = []
    total_mse = 0.0
    total_n = 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(cfg.device), y_b.to(cfg.device)
        padding_mask = make_padding_mask(y_b, cfg.null_value)
        member_preds = torch.stack([m(X_b, padding_mask).squeeze(-1) for m in models])
        pred_mean = member_preds.mean(0)
        valid = ~padding_mask
        if valid.sum() > 0:
            total_mse += F.mse_loss(pred_mean[valid], y_b[valid]).item() * int(valid.sum().item())
            total_n += int(valid.sum().item())
        # Collect per-position predictions [B, T] to CPU
        all_preds.append(pred_mean.cpu().numpy())
        all_valid_y.append(y_b.cpu().numpy())
    mse = total_mse / max(total_n, 1)
    return np.concatenate(all_preds, axis=0), mse  # [n_patients, context_length]


def train_fold_models(
    X_pat: torch.Tensor,
    y_pat: torch.Tensor,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    cfg: Config,
    n_features: int,
    fold_id: int,
) -> tuple[list[VisitTransformer], np.ndarray, float]:
    """Train ensemble on one fold. Returns models, val predictions, val MSE."""
    train_ds = TensorDataset(X_pat[train_idx], y_pat[train_idx])
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    models = [VisitTransformer(cfg, n_features).to(cfg.device) for _ in range(cfg.n_ensemble)]
    optimizers = [optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay) for m in models]
    schedulers = [optim.lr_scheduler.StepLR(opt, step_size=cfg.step_size, gamma=cfg.gamma) for opt in optimizers]

    for epoch in range(cfg.epochs):
        for m, opt, sched in zip(models, optimizers, schedulers):
            train_one_epoch(m, train_loader, opt, cfg)
            sched.step()
        if (epoch + 1) % 20 == 0:
            print(f"  Fold {fold_id} epoch {epoch+1:3d}/{cfg.epochs}", flush=True)

    val_preds, val_mse = predict_ensemble(models, X_pat[val_idx], y_pat[val_idx], cfg)
    print(f"  Fold {fold_id} FINAL val MSE={val_mse:.4f}  RMSE={val_mse**0.5:.4f}", flush=True)
    return models, val_preds, val_mse


# ── Load & engineer features ──────────────────────────────────────────────────
print("Loading data...")
X_train_raw = pd.read_csv("data/X_train.csv")
y_train_df = pd.read_csv("data/y_train.csv")
X_test_raw = pd.read_csv("data/X_test.csv")

print("Engineering features...")
X_eng_train = engineer_features(X_train_raw)
X_eng_test = engineer_features(X_test_raw)
train_with_labels = X_eng_train.merge(y_train_df, on="Index")

print("Building patient-level matrices (global norm stats)...")
X_pat_np, y_pat_np, global_stats, patient_ids_ordered = build_patient_matrix(
    train_with_labels, y_train_df, cfg.context_length, norm_stats=None
)
n_patients, total_feats = X_pat_np.shape
n_features = total_feats // cfg.context_length
print(f"  Patients: {n_patients}  Features/visit: {n_features}  Total: {total_feats}")

X_pat = torch.tensor(X_pat_np, dtype=torch.float32)
y_pat = torch.tensor(y_pat_np, dtype=torch.float32)

# Build GroupKFold splits based on patient_id groups (each patient = 1 row in X_pat)
# patient_ids_ordered is the exact list in the same order as X_pat rows
pid_array = np.array(patient_ids_ordered)

cv = GroupKFold(n_splits=cfg.n_folds)
# X_pat has 1 row per patient; groups are just the patient indices (each patient is its own group)
# This correctly splits patients into train/val folds without any within-patient leakage
cv_splits = list(cv.split(X_pat_np, np.zeros(n_patients), groups=np.arange(n_patients)))

# ── Cross-validation ──────────────────────────────────────────────────────────
print(f"\n[17] Transformer v4 GroupKFold-{cfg.n_folds} CV  (ensemble={cfg.n_ensemble})")
fold_val_mses: list[float] = []

# OOF predictions at patient level [n_patients, context_length]
oof_patient_preds = np.full((n_patients, cfg.context_length), np.nan, dtype=np.float32)

for fold_i, (train_idx, val_idx) in enumerate(cv_splits):
    print(f"\n--- Fold {fold_i + 1} / {cfg.n_folds} ---")
    t0 = time.time()
    fold_models, val_preds, val_mse = train_fold_models(
        X_pat, y_pat, train_idx, val_idx, cfg, n_features, fold_i + 1
    )
    fold_val_mses.append(val_mse)
    oof_patient_preds[val_idx] = val_preds
    print(f"  Fold {fold_i+1} done in {(time.time()-t0)/60:.1f} min")

mean_val_mse = float(np.mean(fold_val_mses))
mean_val_rmse = mean_val_mse ** 0.5
print(f"\n{'='*60}")
print(f"  CV MSE  = {mean_val_mse:.4f}")
print(f"  CV RMSE = {mean_val_rmse:.4f}  (target <= 2.00)")
print(f"{'='*60}")

# ── Map OOF patient-level predictions back to visit-level ─────────────────────
# Reconstruct visit-level OOF predictions for blending in exp 18
pid_to_pat_idx = {str(pid): i for i, pid in enumerate(patient_ids_ordered)}

# Build visit-level OOF: for each row in X_train, find its patient + visit rank
# (visits are sorted by age within patient in the patient matrix)
visits_sorted = train_with_labels.sort_values(["patient_id", "age"]).reset_index(drop=True)
oof_visit_preds = np.full(len(X_train_raw), np.nan, dtype=np.float64)

for pid, grp in visits_sorted.groupby("patient_id", sort=False):
    p_idx = pid_to_pat_idx[str(pid)]
    for v_idx, orig_idx in enumerate(grp["Index"]):
        if v_idx < cfg.context_length:
            # orig_idx is the Index column value; need to map to position in X_train_raw
            row_pos = X_train_raw.index[X_train_raw["Index"] == orig_idx].tolist()
            if row_pos:
                oof_visit_preds[row_pos[0]] = float(oof_patient_preds[p_idx, v_idx])

valid_oof = ~np.isnan(oof_visit_preds)
y_full = train_with_labels.set_index("Index").loc[X_train_raw["Index"], "target"].values
oof_rmse_visit = float(np.sqrt(np.mean((y_full[valid_oof] - oof_visit_preds[valid_oof]) ** 2)))
print(f"\n  Visit-level OOF RMSE: {oof_rmse_visit:.4f}")

# Save OOF for blending
Path("scratch").mkdir(exist_ok=True)
oof_df = X_train_raw[["Index"]].copy()
oof_df["pred_17"] = oof_visit_preds
oof_df.to_csv("scratch/17_oof_visit.csv", index=False)
print(f"  Saved scratch/17_oof_visit.csv")

# ── Final training on all patients ────────────────────────────────────────────
print(f"\n[Final] Training on ALL {n_patients} patients for submission...")
full_ds = TensorDataset(X_pat, y_pat)
full_loader = DataLoader(full_ds, batch_size=cfg.batch_size, shuffle=True)

final_models = [VisitTransformer(cfg, n_features).to(cfg.device) for _ in range(cfg.n_ensemble)]
final_optimizers = [optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay) for m in final_models]
final_schedulers = [optim.lr_scheduler.StepLR(opt, step_size=cfg.step_size, gamma=cfg.gamma) for opt in final_optimizers]

t0 = time.time()
for epoch in range(cfg.epochs):
    for m, opt, sched in zip(final_models, final_optimizers, final_schedulers):
        train_one_epoch(m, full_loader, opt, cfg)
        sched.step()
    if (epoch + 1) % 20 == 0:
        print(f"  Final epoch {epoch+1}/{cfg.epochs} ({(time.time()-t0)/60:.1f} min)")
print(f"  Final training done in {(time.time()-t0)/60:.1f} min")

# ── Test predictions ──────────────────────────────────────────────────────────
print("\n[Submission] Building test predictions...")
X_test_np, _, _, test_patient_ids = build_patient_matrix(
    X_eng_test, None, cfg.context_length, norm_stats=global_stats
)
X_test_t = torch.tensor(X_test_np, dtype=torch.float32).to(cfg.device)
y_dummy_test = torch.zeros(len(test_patient_ids), cfg.context_length)

test_preds_pat, _ = predict_ensemble(
    final_models, X_test_t, y_dummy_test, cfg
)  # [n_test_patients, context_length] — no valid y so MSE is meaningless

pid_to_test_preds = {str(pid): test_preds_pat[i] for i, pid in enumerate(test_patient_ids)}

rows: list[dict] = []
test_sorted = X_test_raw.sort_values(["patient_id", "age"]).reset_index(drop=True)
for pid, grp in test_sorted.groupby("patient_id", sort=False):
    preds = pid_to_test_preds[str(pid)]
    for v_idx, (_, row) in enumerate(grp.iterrows()):
        if v_idx < cfg.context_length:
            rows.append({"Index": int(row["Index"]), "target": float(preds[v_idx])})

submission = pd.DataFrame(rows).sort_values("Index").reset_index(drop=True)
Path("submissions").mkdir(exist_ok=True)
submission.to_csv("submissions/17_transformer_v4.csv", index=False)
print(f"  Wrote submissions/17_transformer_v4.csv  ({len(submission)} rows)")

# Save test preds per patient for blending
test_visit_preds = submission.set_index("Index")["target"]
test_pred_df = X_test_raw[["Index"]].copy()
test_pred_df["pred_17"] = test_pred_df["Index"].map(test_visit_preds)
test_pred_df.to_csv("scratch/17_test_visit.csv", index=False)
print(f"  Saved scratch/17_test_visit.csv")

print(f"\n{'='*60}")
print("RESULTS SUMMARY")
print(f"  14_transformer_v2 CV:     RMSE ~3.90")
print(f"  16_transformer_peer CV:   RMSE ~unknown")
print(f"  17_transformer_v4 CV MSE: {mean_val_mse:.4f}")
print(f"  17_transformer_v4 CV RMSE:{mean_val_rmse:.4f}")
print(f"  Visit OOF RMSE:           {oof_rmse_visit:.4f}")
print(f"  Target:                   <= 2.00")
print(f"{'='*60}")
