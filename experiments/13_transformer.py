# %% [markdown]
# # Experiment 13 — Patient-level Transformer with Visit Tokens
#
# Architecture:  each patient's visits are concatenated into a single row
# (padded to context_length=12).  A Transformer attends across visits,
# predicting the true OFF score for every visit simultaneously.
#
# Inspired by a 2nd-place solution (MSE ~7.35) to this competition.
# Key differences vs. gradient boosting: the model sees the full disease
# trajectory at once — early visits inform later ones and vice versa.

# %%
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import time

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.model_selection import GroupKFold

from parkinson.hub import get_project

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
    context_length: int = 12        # max visits per patient
    hidden_dim: int = 64
    n_head: int = 4
    num_layers: int = 4
    mul: int = 4                    # MLP expansion factor
    dropout_att: float = 0.1
    dropout_mlp: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-2
    batch_size: int = 64
    epochs: int = 100
    step_size: int = 20             # LR halved every step_size epochs
    gamma: float = 0.5
    n_ensemble: int = 5             # independent models for deep ensemble
    n_folds: int = 5
    null_value: float = -999.0      # sentinel for padded / missing targets
    deviation: float = 0.02         # weight init std
    device: torch.device = field(default_factory=lambda: DEVICE)

cfg = Config()

# ── Data pipeline ─────────────────────────────────────────────────────────────
NULLABLE_FEATURES = [
    "time_since_intake_on",
    "time_since_intake_off",
    "ledd",
    "age_at_diagnosis",
    "time_since_diagnosis",
    "on",
    "off",
]
NORMALIZE_FEATURES = NULLABLE_FEATURES  # same set

GENE_DUMMIES = ["GBA+", "LRRK2+", "OTHER+", "No Mutation"]


def build_patient_matrix(
    df_input: pl.DataFrame,
    df_label: pl.DataFrame | None,
    context_length: int,
    norm_stats: dict | None,
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """Convert per-visit rows into per-patient matrices.

    Parameters
    ----------
    df_input  : Polars DataFrame of X (one row per visit)
    df_label  : Polars DataFrame of y (one row per visit) or None for test
    norm_stats: dict {col: (mean, std)} from the train fold.  If None, compute
                from df_input and return in the result dict.

    Returns
    -------
    X_pat : float32 array [n_patients, context_length * n_features]
    y_pat : float32 array [n_patients, context_length] or None
    stats : norm_stats dict (for reuse on val/test folds)
    """
    # ── 1. Categorical encoding ───────────────────────────────────────────────
    df = df_input.clone()

    # gene → one-hot
    for g in GENE_DUMMIES:
        df = df.with_columns(
            (pl.col("gene") == g).cast(pl.Int8).alias(f"gene_{g}")
        )
    # cohort → 0/1
    df = df.with_columns(
        pl.col("cohort").replace({"A": "0", "B": "1"}).cast(pl.Int8)
    )
    # time_since_diagnosis
    df = df.with_columns(
        (pl.col("age") - pl.col("age_at_diagnosis")).alias("time_since_diagnosis")
    )

    # ── 2. Attach labels if provided ──────────────────────────────────────────
    if df_label is not None:
        lbl = df_label.rename({"target": "_target"}).select(["Index", "_target"])
        df = df.join(lbl, on="Index", how="left")

    # ── 3. Sort visits by age within patient ──────────────────────────────────
    df = df.sort(["patient_id", "age"])

    # ── 4. Compute / apply normalization stats ────────────────────────────────
    computed_stats: dict = {}
    for col in NORMALIZE_FEATURES:
        if norm_stats is not None:
            mu, sigma = norm_stats[col]
        else:
            mu = float(df[col].mean())
            sigma = float(df[col].std()) + 1e-8
        computed_stats[col] = (mu, sigma)
        df = df.with_columns(
            ((pl.col(col) - mu) / sigma).alias(col)
        )

    # ── 5. Flag columns (1 = was null) ────────────────────────────────────────
    for col in NULLABLE_FEATURES:
        df = df.with_columns(
            pl.col(col).is_null().cast(pl.Int8).alias(f"flag_{col}")
        )

    # ── 6. Fill remaining nulls with 0 (= column mean after normalizing) ──────
    df = df.fill_null(0)

    # ── 7. Choose feature columns (fixed order, no Index / patient_id) ────────
    base_features = [
        "cohort", "sexM",
        "gene_GBA+", "gene_LRRK2+", "gene_OTHER+", "gene_No Mutation",
        "age_at_diagnosis", "age", "ledd",
        "time_since_intake_on", "time_since_intake_off",
        "on", "off", "time_since_diagnosis",
    ]
    flag_features = [f"flag_{c}" for c in NULLABLE_FEATURES]
    feat_cols = base_features + flag_features   # 21 features per visit

    # ── 8. Group by patient, pad to context_length ────────────────────────────
    patients = df["patient_id"].unique(maintain_order=True).to_list()
    n_patients = len(patients)
    n_feats = len(feat_cols)

    X_pat = np.full(
        (n_patients, context_length * n_feats), 0.0, dtype=np.float32
    )
    y_pat = (
        np.full((n_patients, context_length), cfg.null_value, dtype=np.float32)
        if df_label is not None else None
    )

    pid_col = df["patient_id"].to_list()
    feat_arr = df.select(feat_cols).to_numpy().astype(np.float32)
    tgt_arr = df["_target"].to_numpy().astype(np.float32) if df_label is not None else None

    # Build index map for efficiency
    pid_to_rows: dict[str, list[int]] = {}
    for i, pid in enumerate(pid_col):
        pid_to_rows.setdefault(pid, []).append(i)

    for p_idx, pid in enumerate(patients):
        rows = pid_to_rows[pid]
        n_visits = min(len(rows), context_length)
        for v, row_idx in enumerate(rows[:n_visits]):
            X_pat[p_idx, v * n_feats:(v + 1) * n_feats] = feat_arr[row_idx]
            if y_pat is not None:
                y_pat[p_idx, v] = tgt_arr[row_idx]

    return X_pat, y_pat, computed_stats if norm_stats is None else norm_stats


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
            embed_dim=cfg.hidden_dim,
            num_heads=cfg.n_head,
            dropout=cfg.dropout_att,
            batch_first=True,
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
    """Transformer that predicts OFF score for every visit of a patient."""

    def __init__(self, cfg: Config, n_features: int):
        super().__init__()
        self.cfg = cfg
        self.n_features = n_features

        # Token embedding: per-visit features → hidden_dim
        self.tok_embed = nn.Sequential(
            nn.Linear(n_features, cfg.mul * cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.mul * cfg.hidden_dim, cfg.hidden_dim),
        )
        # Learned positional embedding
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
        """
        x: [B, context_length * n_features]
        returns: [B, context_length, 1]
        """
        B = x.size(0)
        x = x.view(B, self.cfg.context_length, self.n_features)

        tok = self.tok_embed(x)
        pos = self.pos_embed(
            torch.arange(self.cfg.context_length, device=x.device)
        )
        h = tok + pos  # [B, T, hidden_dim]

        for block in self.blocks:
            h = block(h, key_padding_mask=padding_mask)

        h = self.ln_final(h)
        return self.head(h)  # [B, T, 1]


# ── Training helpers ──────────────────────────────────────────────────────────
def make_padding_mask(y_batch: torch.Tensor, null_value: float) -> torch.Tensor:
    """True where the position is padding (no real visit). Shape [B, T]."""
    return y_batch == null_value


def train_one_epoch(
    model: VisitTransformer,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler,
    cfg: Config,
) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0
    for X_b, y_b in loader:
        X_b = X_b.to(cfg.device)
        y_b = y_b.to(cfg.device)
        padding_mask = make_padding_mask(y_b, cfg.null_value)

        optimizer.zero_grad()
        pred = model(X_b, padding_mask).squeeze(-1)  # [B, T]

        valid = ~padding_mask
        loss = F.mse_loss(pred[valid], y_b[valid])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        n = valid.sum().item()
        total_loss += loss.item() * n
        total_n += n
    scheduler.step()
    return total_loss / max(total_n, 1)


@torch.inference_mode()
def eval_ensemble(
    models: list[VisitTransformer],
    loader: DataLoader,
    cfg: Config,
) -> float:
    for m in models:
        m.eval()
    total_loss = 0.0
    total_n = 0
    for X_b, y_b in loader:
        X_b = X_b.to(cfg.device)
        y_b = y_b.to(cfg.device)
        padding_mask = make_padding_mask(y_b, cfg.null_value)

        preds = torch.stack([m(X_b, padding_mask).squeeze(-1) for m in models])
        pred_mean = preds.mean(0)

        valid = ~padding_mask
        loss = F.mse_loss(pred_mean[valid], y_b[valid])
        n = valid.sum().item()
        total_loss += loss.item() * n
        total_n += n
    return total_loss / max(total_n, 1)


def train_fold(
    X_pat: torch.Tensor,
    y_pat: torch.Tensor,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    cfg: Config,
    n_features: int,
    fold_id: int,
) -> tuple[list[VisitTransformer], float]:
    """Train an ensemble on one GroupKFold split, return models + val MSE."""
    train_ds = TensorDataset(X_pat[train_idx], y_pat[train_idx])
    val_ds   = TensorDataset(X_pat[val_idx],   y_pat[val_idx])
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False)

    models = [
        VisitTransformer(cfg, n_features).to(cfg.device)
        for _ in range(cfg.n_ensemble)
    ]
    optimizers = [optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay) for m in models]
    schedulers = [optim.lr_scheduler.StepLR(opt, step_size=cfg.step_size, gamma=cfg.gamma) for opt in optimizers]

    best_val = float("inf")
    for epoch in range(cfg.epochs):
        train_losses = []
        for m, opt, sched in zip(models, optimizers, schedulers):
            loss = train_one_epoch(m, train_loader, opt, sched, cfg)
            train_losses.append(loss)

        if (epoch + 1) % 10 == 0:
            val_mse = eval_ensemble(models, val_loader, cfg)
            best_val = min(best_val, val_mse)
            print(f"  Fold {fold_id} epoch {epoch+1:3d}: "
                  f"train_avg={np.mean(train_losses):.4f}  val_mse={val_mse:.4f}")

    val_mse = eval_ensemble(models, val_loader, cfg)
    print(f"  Fold {fold_id} FINAL val MSE={val_mse:.4f}  RMSE={val_mse**0.5:.4f}")
    return models, val_mse


# ── Main ──────────────────────────────────────────────────────────────────────
print("Loading data...")
df_X  = pl.read_csv("data/X_train.csv")
df_y  = pl.read_csv("data/y_train.csv")
df_Xt = pl.read_csv("data/X_test.csv")
# Keep pandas copy of X_test for submission row alignment
X_test_raw = pd.read_csv("data/X_test.csv")

print("Building patient-level matrices (full train for norm stats)...")
X_pat_np, y_pat_np, train_stats = build_patient_matrix(df_X, df_y, cfg.context_length, norm_stats=None)
n_patients, total_feats = X_pat_np.shape
n_features = total_feats // cfg.context_length
print(f"  Patients: {n_patients}  Features/visit: {n_features}  Total: {total_feats}")

X_pat = torch.tensor(X_pat_np, dtype=torch.float32)
y_pat = torch.tensor(y_pat_np, dtype=torch.float32)

# Patient-level groups for GroupKFold
patient_ids = (
    df_X.sort(["patient_id", "age"])
    ["patient_id"].unique(maintain_order=True).to_list()
)
assert len(patient_ids) == n_patients

# ── Cross-validation ──────────────────────────────────────────────────────────
print(f"\n[13] Transformer GroupKFold-{cfg.n_folds} CV")
cv = GroupKFold(n_splits=cfg.n_folds)
# GroupKFold on patient-level (each row = one patient)
dummy_X = np.zeros((n_patients, 1))
dummy_groups = np.arange(n_patients)  # each patient is its own group
cv_splits = list(cv.split(dummy_X, dummy_groups, groups=dummy_groups))

fold_val_mses: list[float] = []
fold_models_list: list[list[VisitTransformer]] = []

for fold_i, (train_idx, val_idx) in enumerate(cv_splits):
    print(f"\n--- Fold {fold_i + 1} / {cfg.n_folds} ---")
    t0 = time.time()
    fold_models, val_mse = train_fold(
        X_pat, y_pat, train_idx, val_idx, cfg, n_features, fold_i + 1
    )
    fold_val_mses.append(val_mse)
    fold_models_list.append(fold_models)
    print(f"  Fold {fold_i+1} done in {(time.time()-t0)/60:.1f} min")

mean_val_mse  = float(np.mean(fold_val_mses))
mean_val_rmse = mean_val_mse ** 0.5
print(f"\n{'='*60}")
print(f"  CV MSE  = {mean_val_mse:.4f}")
print(f"  CV RMSE = {mean_val_rmse:.4f}  (GBM best was 4.58)")
print(f"{'='*60}")

# ── Push a lightweight report to Hub ─────────────────────────────────────────
# We can't use skore.evaluate on a PyTorch model directly, so we log the OOF
# MSE as a skore item.
print("\n[Push] → Hub")
project = get_project()
result_dict = {
    "experiment": "13_transformer",
    "cv_mse":     mean_val_mse,
    "cv_rmse":    mean_val_rmse,
    "fold_mses":  fold_val_mses,
    "config": {
        "hidden_dim": cfg.hidden_dim,
        "n_head":     cfg.n_head,
        "num_layers": cfg.num_layers,
        "n_ensemble": cfg.n_ensemble,
        "epochs":     cfg.epochs,
        "n_features_per_visit": n_features,
    },
}
project.put("13_transformer_results", result_dict)
print(f"  Pushed '13_transformer_results' → Hub")

# ── Final training on all patients ────────────────────────────────────────────
print("\n[Final] Training on ALL patients for submission...")
full_ds = TensorDataset(X_pat, y_pat)
full_loader = DataLoader(full_ds, batch_size=cfg.batch_size, shuffle=True)

final_models = [
    VisitTransformer(cfg, n_features).to(cfg.device)
    for _ in range(cfg.n_ensemble)
]
final_optimizers = [optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay) for m in final_models]
final_schedulers = [optim.lr_scheduler.StepLR(opt, step_size=cfg.step_size, gamma=cfg.gamma) for opt in final_optimizers]

t0 = time.time()
for epoch in range(cfg.epochs):
    for m, opt, sched in zip(final_models, final_optimizers, final_schedulers):
        train_one_epoch(m, full_loader, opt, sched, cfg)
    if (epoch + 1) % 20 == 0:
        print(f"  Final epoch {epoch+1}/{cfg.epochs} ({(time.time()-t0)/60:.1f} min)")
print(f"  Final training done in {(time.time()-t0)/60:.1f} min")

# ── Build test predictions ─────────────────────────────────────────────────────
print("\n[Submission] Building test predictions...")
X_test_np, _, _ = build_patient_matrix(df_Xt, None, cfg.context_length, norm_stats=train_stats)
X_test_t = torch.tensor(X_test_np, dtype=torch.float32).to(cfg.device)

# Test patient order (same as build_patient_matrix output)
test_patient_ids = (
    df_Xt.sort(["patient_id", "age"])
    ["patient_id"].unique(maintain_order=True).to_list()
)
test_visits_sorted = (
    df_Xt.sort(["patient_id", "age"])
    .select(["Index", "patient_id", "age"])
)

# Predict for each test patient: [n_test_patients, context_length]
for m in final_models:
    m.eval()

with torch.inference_mode():
    # Dummy padding mask: all positions real (no padded targets at test time)
    dummy_y = torch.zeros(len(test_patient_ids), cfg.context_length, device=cfg.device)
    pad_mask = dummy_y == cfg.null_value  # all False

    preds_stack = torch.stack([
        m(X_test_t, pad_mask).squeeze(-1)
        for m in final_models
    ])
    pred_mean = preds_stack.mean(0).cpu().numpy()  # [n_test_patients, 12]

# Map predictions back to (Index, target) rows
# pred_mean[p_idx, v_idx] → test_patient_ids[p_idx], v_idx-th visit (sorted by age)
rows: list[dict] = []
pid_to_preds: dict[str, np.ndarray] = {
    pid: pred_mean[i] for i, pid in enumerate(test_patient_ids)
}

# Re-sort X_test_raw by patient+age to align with model output
test_sorted = X_test_raw.sort_values(["patient_id", "age"]).reset_index(drop=True)
for pid, grp in test_sorted.groupby("patient_id", sort=False):
    preds = pid_to_preds[pid]
    for v_idx, (_, row) in enumerate(grp.iterrows()):
        if v_idx < cfg.context_length:
            rows.append({"Index": int(row["Index"]), "target": float(preds[v_idx])})

submission = pd.DataFrame(rows).sort_values("Index").reset_index(drop=True)
Path("submissions").mkdir(exist_ok=True)
submission.to_csv("submissions/13_transformer.csv", index=False)
print(f"  Wrote submissions/13_transformer.csv  ({len(submission)} rows)")

print(f"\n{'='*60}")
print(f"FINAL RESULTS")
print(f"  GBM best (12_stacked):   RMSE 4.58")
print(f"  13_transformer CV:       MSE {mean_val_mse:.4f}  RMSE {mean_val_rmse:.4f}")
print(f"{'='*60}")
