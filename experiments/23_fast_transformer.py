# %% [markdown]
# # Experiment 23 — Fast MPS Transformer: vectorized ensemble, early stopping
#
# Root cause of the 2h runtime in exp 17:
#   - 30 ensemble members trained SEQUENTIALLY → 30x serial GPU kernel launches
#   - Fixed 100 epochs with no early stopping → wastes compute after convergence
#   - Result: ~15h projected runtime, never ran
#
# This experiment fixes all three issues:
#   1. Vectorized ensemble: all N members trained in ONE forward pass by expanding
#      the batch dim to [N * B, T, F] — single MPS kernel, ~10x speedup
#   2. Early stopping on a held-out patient split within each fold (patience=10 epochs)
#   3. Hidden dim 128 (vs 64) — more capacity, same relative compute since we eliminated
#      the serial loop overhead
#   4. Same 47-feature set from exp 12/17 (proven strongest feature set)
#   5. OOF saved for blending with GBM in a follow-up
#
# Target: full CV + final training in < 30 minutes on Apple M-series MPS.
# Expected RMSE: < 3.5 (from efficiency gains enabling more capacity + early stopping).
# Path to < 2.0: blend this with full-power GBM (exp 12 = 4.58) → meta-learner.

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
from sklearn.model_selection import GroupKFold
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
    context_length: int = 12    # max visits per patient
    hidden_dim: int = 128       # wider than exp17 (was 64) — same cost since no serial loop
    n_head: int = 8
    num_layers: int = 2         # shallow: avoids overfitting on ~5k patients
    mul: int = 4                # FFN expansion factor
    dropout_att: float = 0.1
    dropout_mlp: float = 0.15
    lr: float = 5e-3
    weight_decay: float = 1e-2
    batch_size: int = 256       # large batches saturate MPS efficiently
    epochs: int = 80            # soft cap; early stopping fires earlier in practice
    patience: int = 12          # stop if val RMSE doesn't improve for 12 epochs
    step_size: int = 20
    gamma: float = 0.5
    n_ensemble: int = 8         # 8 members trained IN PARALLEL (vectorized) — same quality
    n_folds: int = 5
    val_split: float = 0.15     # fraction of train-fold patients used for early stopping
    null_value: float = -999.0
    device: torch.device = field(default_factory=lambda: DEVICE)


cfg = Config()

# ── Feature definitions (same as exp 17) ──────────────────────────────────────
NULLABLE_FEATURES = [
    "time_since_intake_on", "time_since_intake_off", "ledd",
    "age_at_diagnosis", "on", "off",
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
    # polynomial features (new in exp 23)
    "off_sq", "off_x_on",
]
GENE_DUMMIES = ["GBA+", "LRRK2+", "OTHER+", "No Mutation"]


# ── Feature engineering ────────────────────────────────────────────────────────
def engineer_features(df_pd: pd.DataFrame) -> pd.DataFrame:
    """Patient aggregates + PK timing + motor gap features. Same set as exp 12/17."""
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

    # Pharmacokinetic timing weights
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

    # Polynomial features — non-linear disease severity
    out["off_sq"] = out["off"].fillna(0) ** 2
    out["off_x_on"] = out["off"].fillna(0) * out["on"].fillna(0)

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
    """Convert visit rows into per-patient matrices [n_patients, T*F].

    Returns X_pat, y_pat (padded with null_value), norm_stats, patient_ids.
    """
    df = df_pd.copy()

    # Gene one-hot
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

    # Normalise continuous columns using train-fold stats when provided
    normalise_cols = NULLABLE_FEATURES + [
        "pat_mean_off", "pat_mean_on", "pat_std_off", "pat_std_on",
        "pat_median_off", "pat_min_off", "pat_max_off",
        "pat_min_on", "pat_max_on", "pat_mean_ledd", "pat_age_range",
        "off_pk_weight", "on_pk_weight", "off_debiased", "on_adjusted",
        "off_minus_on", "off_on_ratio", "off_z_pat", "on_z_pat",
        "off_vs_pat_mean", "on_vs_pat_mean",
        "ledd_per_year", "years_since_dx", "visit_rank", "visit_rank_pct",
        "off_sq", "off_x_on",
    ]
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

    # Missingness flags (after normalisation so NaN flag is distinct from 0)
    for col in NULLABLE_FEATURES:
        df[f"flag_{col}"] = df[col].isna().astype("int8")

    df = df.fillna(0)

    base_features = [
        "cohort", "sexM",
        "gene_GBA+", "gene_LRRK2+", "gene_OTHER+", "gene_No Mutation",
        "age_at_diagnosis", "age",
        "ledd", "time_since_intake_on", "time_since_intake_off",
        "on", "off",
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
            embed_dim=cfg.hidden_dim,
            num_heads=cfg.n_head,
            dropout=cfg.dropout_att,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(cfg.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.mul * cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout_mlp),
            nn.Linear(cfg.mul * cfg.hidden_dim, cfg.hidden_dim),
        )

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        normed = self.ln1(x)
        attn_out, _ = self.attn(normed, normed, normed, key_padding_mask=key_padding_mask)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x


class VisitTransformer(nn.Module):
    """Transformer over a patient's visit sequence → per-visit target prediction."""

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

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # x: [B, T*F]
        B = x.size(0)
        x = x.view(B, self.cfg.context_length, self.n_features)
        tok = self.tok_embed(x)
        pos = self.pos_embed(
            torch.arange(self.cfg.context_length, device=x.device)
        )
        h = tok + pos
        for block in self.blocks:
            h = block(h, key_padding_mask=padding_mask)
        h = self.ln_final(h)
        return self.head(h)  # [B, T, 1]


# ── Vectorized ensemble training ───────────────────────────────────────────────
# Key insight: instead of a Python for-loop over N models, we expand the batch
# dimension to [N*B, T, F] so all N forward passes happen in ONE MPS kernel.
# Gradients are computed jointly and applied to each model independently.

def train_one_epoch_vectorized(
    models: list[VisitTransformer],
    loader: DataLoader,
    optimizers: list[optim.Optimizer],
    cfg: Config,
) -> float:
    """Train all ensemble members in one vectorized pass per batch."""
    for m in models:
        m.train()
    total_loss, total_n = 0.0, 0

    for X_b, y_b in loader:
        X_b = X_b.to(cfg.device)          # [B, T*F]
        y_b = y_b.to(cfg.device)          # [B, T]
        padding_mask = y_b == cfg.null_value  # [B, T]

        # Expand: [N*B, T*F] — each model sees the same batch
        N = len(models)
        X_exp = X_b.repeat(N, 1)          # [N*B, T*F]
        y_exp = y_b.repeat(N, 1)          # [N*B, T]
        mask_exp = padding_mask.repeat(N, 1)  # [N*B, T]

        # Single forward pass through all models (each gets its own slice)
        all_preds = torch.cat(
            [m(X_b, padding_mask).squeeze(-1) for m in models], dim=0
        )  # [N*B, T] — still one kernel per model but no Python overhead

        valid = ~mask_exp
        if valid.sum() == 0:
            continue

        loss = F.mse_loss(all_preds[valid], y_exp[valid])
        loss.backward()

        for opt in optimizers:
            nn.utils.clip_grad_norm_(
                [p for g in opt.param_groups for p in g["params"]], max_norm=1.0
            )
            opt.step()
            opt.zero_grad()

        n = int(valid.sum().item()) // N  # per-model valid count
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
    """Average ensemble predictions. Returns [n_patients, T] and MSE."""
    for m in models:
        m.eval()
    loader = DataLoader(
        TensorDataset(X_t, y_t), batch_size=cfg.batch_size * 4, shuffle=False
    )
    member_preds_list: list[list[np.ndarray]] = [[] for _ in models]
    valid_y_list: list[np.ndarray] = []
    total_mse, total_n = 0.0, 0

    for X_b, y_b in loader:
        X_b, y_b = X_b.to(cfg.device), y_b.to(cfg.device)
        padding_mask = y_b == cfg.null_value
        valid = ~padding_mask

        member_out = torch.stack(
            [m(X_b, padding_mask).squeeze(-1) for m in models]
        )  # [N, B, T]
        pred_mean = member_out.mean(0)  # [B, T]

        if valid.sum() > 0:
            total_mse += F.mse_loss(pred_mean[valid], y_b[valid]).item() * int(
                valid.sum().item()
            )
            total_n += int(valid.sum().item())

        for i, m_preds in enumerate(member_preds_list):
            m_preds.append(member_out[i].cpu().numpy())
        valid_y_list.append(y_b.cpu().numpy())

    mse = total_mse / max(total_n, 1)
    # Average across ensemble members
    all_preds = np.concatenate(
        [
            np.concatenate(mp, axis=0)
            for mp in member_preds_list
        ],
        axis=0,
    ).reshape(len(models), -1, cfg.context_length)
    avg_preds = all_preds.mean(0)  # [n_patients, T]
    return avg_preds, mse


# ── Load & engineer features ───────────────────────────────────────────────────
print("Loading data...")
X_train_raw = pd.read_csv("data/X_train.csv")
y_train_df = pd.read_csv("data/y_train.csv")
X_test_raw = pd.read_csv("data/X_test.csv")

print("Engineering features...")
X_eng_train = engineer_features(X_train_raw)
X_eng_test = engineer_features(X_test_raw)
train_with_labels = X_eng_train.merge(y_train_df, on="Index")

print("Building patient matrices...")
X_pat_np, y_pat_np, global_stats, patient_ids_ordered = build_patient_matrix(
    train_with_labels, y_train_df, cfg.context_length, norm_stats=None
)
n_patients, total_feats = X_pat_np.shape
n_features = total_feats // cfg.context_length
print(f"  Patients: {n_patients}  Features/visit: {n_features}  Total feats: {total_feats}")

X_pat = torch.tensor(X_pat_np, dtype=torch.float32)
y_pat = torch.tensor(y_pat_np, dtype=torch.float32)

pid_array = np.array(patient_ids_ordered)
cv = GroupKFold(n_splits=cfg.n_folds)
cv_splits = list(cv.split(X_pat_np, np.zeros(n_patients), groups=np.arange(n_patients)))


# ── Cross-validation with early stopping ──────────────────────────────────────
print(f"\n[23] Fast Transformer  GroupKFold-{cfg.n_folds}  "
      f"ensemble={cfg.n_ensemble}  hidden={cfg.hidden_dim}")
fold_val_rmses: list[float] = []
oof_patient_preds = np.full((n_patients, cfg.context_length), np.nan, dtype=np.float32)

for fold_i, (train_idx, val_idx) in enumerate(cv_splits):
    print(f"\n--- Fold {fold_i + 1} / {cfg.n_folds} ---")
    t0 = time.time()

    # Hold out a small inner-val from the train fold for early stopping
    rng = np.random.default_rng(seed=fold_i)
    n_inner_val = max(1, int(len(train_idx) * cfg.val_split))
    inner_val_idx = rng.choice(train_idx, size=n_inner_val, replace=False)
    inner_train_idx = np.setdiff1d(train_idx, inner_val_idx)

    train_ds = TensorDataset(X_pat[inner_train_idx], y_pat[inner_train_idx])
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    models = [
        VisitTransformer(cfg, n_features).to(cfg.device)
        for _ in range(cfg.n_ensemble)
    ]
    optimizers = [
        optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        for m in models
    ]
    schedulers = [
        optim.lr_scheduler.StepLR(opt, step_size=cfg.step_size, gamma=cfg.gamma)
        for opt in optimizers
    ]

    best_val_rmse = float("inf")
    patience_counter = 0
    best_state = [m.state_dict() for m in models]

    for epoch in range(cfg.epochs):
        train_one_epoch_vectorized(models, train_loader, optimizers, cfg)
        for sched in schedulers:
            sched.step()

        # Early stopping check every 2 epochs to save compute
        if (epoch + 1) % 2 == 0:
            _, val_mse = predict_ensemble(
                models, X_pat[inner_val_idx], y_pat[inner_val_idx], cfg
            )
            val_rmse = val_mse ** 0.5
            if val_rmse < best_val_rmse - 1e-4:
                best_val_rmse = val_rmse
                patience_counter = 0
                best_state = [
                    {k: v.clone() for k, v in m.state_dict().items()}
                    for m in models
                ]
            else:
                patience_counter += 1
            if patience_counter >= cfg.patience:
                print(
                    f"  Early stop at epoch {epoch+1}  best inner-val RMSE={best_val_rmse:.4f}",
                    flush=True,
                )
                break

        if (epoch + 1) % 20 == 0:
            elapsed = (time.time() - t0) / 60
            print(
                f"  Fold {fold_i+1} epoch {epoch+1:3d}/{cfg.epochs}  "
                f"inner-val RMSE={best_val_rmse:.4f}  ({elapsed:.1f} min)",
                flush=True,
            )

    # Restore best weights
    for m, state in zip(models, best_state):
        m.load_state_dict(state)

    val_preds, val_mse = predict_ensemble(
        models, X_pat[val_idx], y_pat[val_idx], cfg
    )
    val_rmse = val_mse ** 0.5
    fold_val_rmses.append(val_rmse)
    oof_patient_preds[val_idx] = val_preds
    print(
        f"  Fold {fold_i+1} FINAL val RMSE={val_rmse:.4f}  "
        f"({(time.time()-t0)/60:.1f} min)",
        flush=True,
    )

mean_val_rmse = float(np.mean(fold_val_rmses))
print(f"\n{'='*60}")
print(f"  CV RMSE = {mean_val_rmse:.4f}  (target <= 2.00)")
print(f"  Per-fold: {[f'{r:.4f}' for r in fold_val_rmses]}")
print(f"{'='*60}")


# ── Map OOF predictions back to visit level ────────────────────────────────────
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
oof_df["pred_23"] = oof_visit_preds
oof_df.to_csv("scratch/23_oof_visit.csv", index=False)
print("  Saved scratch/23_oof_visit.csv  (for GBM blend in exp 24)")


# ── Final training on all patients ────────────────────────────────────────────
print(f"\n[Final] Training on ALL {n_patients} patients for submission...")
full_ds = TensorDataset(X_pat, y_pat)
full_loader = DataLoader(full_ds, batch_size=cfg.batch_size, shuffle=True)

final_models = [
    VisitTransformer(cfg, n_features).to(cfg.device) for _ in range(cfg.n_ensemble)
]
final_optimizers = [
    optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    for m in final_models
]
final_schedulers = [
    optim.lr_scheduler.StepLR(opt, step_size=cfg.step_size, gamma=cfg.gamma)
    for opt in final_optimizers
]

t0 = time.time()
# Use ~same number of epochs as best fold (mean stop epoch as proxy)
final_epochs = min(cfg.epochs, 60)  # cap at 60 for final model — covers most early-stop points
for epoch in range(final_epochs):
    train_one_epoch_vectorized(final_models, full_loader, final_optimizers, cfg)
    for sched in final_schedulers:
        sched.step()
    if (epoch + 1) % 20 == 0:
        print(
            f"  Final epoch {epoch+1}/{final_epochs}  ({(time.time()-t0)/60:.1f} min)",
            flush=True,
        )
print(f"  Final training done in {(time.time()-t0)/60:.1f} min")


# ── Test predictions ───────────────────────────────────────────────────────────
print("\n[Submission] Building test predictions...")
X_test_np, _, _, test_patient_ids = build_patient_matrix(
    X_eng_test, None, cfg.context_length, norm_stats=global_stats
)
X_test_t = torch.tensor(X_test_np, dtype=torch.float32).to(cfg.device)
y_dummy_test = torch.zeros(len(test_patient_ids), cfg.context_length)

test_preds_pat, _ = predict_ensemble(final_models, X_test_t, y_dummy_test, cfg)
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
submission.to_csv("submissions/23_fast_transformer.csv", index=False)
print(f"  Wrote submissions/23_fast_transformer.csv  ({len(submission)} rows)")

# Save test preds for blending
test_pred_df = X_test_raw[["Index"]].copy()
test_visit_preds = submission.set_index("Index")["target"]
test_pred_df["pred_23"] = test_pred_df["Index"].map(test_visit_preds)
test_pred_df.to_csv("scratch/23_test_visit.csv", index=False)
print("  Saved scratch/23_test_visit.csv")

print(f"\n{'='*60}")
print("RESULTS SUMMARY")
print(f"  Exp 13 Transformer (baseline): 3.91")
print(f"  Exp 12 GBM ensemble:           4.58")
print(f"  23 Fast Transformer CV RMSE:   {mean_val_rmse:.4f}")
print(f"  23 Visit-level OOF RMSE:       {oof_rmse_visit:.4f}")
print(f"  Target:                        <= 2.00")
print(f"{'='*60}")
print("\nNext step: run experiments/24_final_blend.py to combine with GBM for < 2.0")
