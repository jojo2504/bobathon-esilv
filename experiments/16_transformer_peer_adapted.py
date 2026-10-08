# %% [markdown]
# # Experiment 16 — Transformer (Peer Adapted)
#
# Fusion of the rigorous data pipeline from exp 15 (chronological sorting, 
# global normalization, correct -999.0 null handling, GroupKFold) with the 
# high-performing architectural choices from the peer's 2nd-place solution:
#   - Shallow but wide network: num_layers=1, n_head=8
#   - Massive Deep Ensembling: n_ensemble=30
#   - Larger receptive field: context_length=16
#   - Stable training: lr=1e-2, epochs=100, no early stopping

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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupKFold
from skore import evaluate

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")

@dataclass
class Config:
    context_length: int = 16       # Peer's longer context
    hidden_dim: int = 64
    n_head: int = 8                # Peer's wide attention
    num_layers: int = 1            # Peer's shallow network (prevents overfitting)
    mul: int = 4
    dropout_att: float = 0.2
    dropout_mlp: float = 0.2
    lr: float = 1e-2               # Peer's higher learning rate
    weight_decay: float = 1e-2
    batch_size: int = 64
    epochs: int = 100              # Fixed epochs, no early stopping
    step_size: int = 20
    gamma: float = 0.5
    n_ensemble: int = 30           # Peer's massive deep ensembling
    n_folds: int = 5
    null_value: float = -999.0     # Retained from exp 15 to correctly score true 0s
    deviation: float = 0.02
    device: torch.device = field(default_factory=lambda: DEVICE)

cfg = Config()

COLS_TO_NORMALIZE = [
    "time_since_intake_on", "time_since_intake_off", "ledd",
    "age_at_diagnosis", "time_since_diagnosis", "age", "on", "off",
]
COLS_WITH_NULLS = [
    "time_since_intake_on", "time_since_intake_off", "ledd",
    "age_at_diagnosis", "time_since_diagnosis", "on", "off",
]
GENE_DUMMIES = ["GBA+", "LRRK2+", "OTHER+", "No Mutation"]

def build_patient_matrix(
    df_X: pl.DataFrame,
    df_y: pl.DataFrame | None,
    context_length: int,
    norm_stats: dict | None,
) -> tuple[np.ndarray, np.ndarray | None, dict, list[str]]:
    df = df_X.clone()

    for g in GENE_DUMMIES:
        df = df.with_columns((pl.col("gene") == g).cast(pl.Int8).alias(f"gene_{g}"))
    df = df.with_columns(pl.col("cohort").replace({"A": "0", "B": "1"}).cast(pl.Int8))
    df = df.with_columns((pl.col("age") - pl.col("age_at_diagnosis")).alias("time_since_diagnosis"))

    if df_y is not None:
        lbl = df_y.rename({"target": "_target"}).select(["Index", "_target"])
        df = df.join(lbl, on="Index", how="left")

    df = df.sort(["patient_id", "age"])

    computed_stats: dict = {}
    for col in COLS_TO_NORMALIZE:
        if norm_stats is not None:
            mu, sigma = norm_stats[col]
        else:
            mu = float(df[col].mean() or 0.0)
            sigma = float(df[col].std() or 1.0) + 1e-8
        computed_stats[col] = (mu, sigma)
        df = df.with_columns(((pl.col(col) - mu) / sigma).alias(col))

    for col in COLS_WITH_NULLS:
        df = df.with_columns(pl.col(col).is_null().cast(pl.Int8).alias(f"flag_{col}"))

    df = df.fill_null(0)

    base_feats = [
        "cohort", "sexM", "gene_GBA+", "gene_LRRK2+", "gene_OTHER+", "gene_No Mutation",
        "age_at_diagnosis", "age", "ledd", "time_since_intake_on", "time_since_intake_off",
        "on", "off", "time_since_diagnosis",
    ]
    flag_feats = [f"flag_{c}" for c in COLS_WITH_NULLS]
    feat_cols = base_feats + flag_feats

    all_cols = feat_cols + (["_target"] if df_y is not None else [])
    agg_cols = [pl.col(c) for c in all_cols]
    df_grouped = df.group_by("patient_id").agg(agg_cols)

    n_patients = df_grouped.height
    n_feats = len(feat_cols)
    X_pat = np.zeros((n_patients, context_length * n_feats), dtype=np.float32)
    y_pat = np.full((n_patients, context_length), cfg.null_value, dtype=np.float32) if df_y is not None else None

    for p_idx in range(n_patients):
        row = df_grouped.row(p_idx, named=True)
        visits_feats = np.array(
            [[row[c][v] if v < len(row[c]) and row[c][v] is not None else 0.0 for c in feat_cols]
             for v in range(context_length)],
            dtype=np.float32
        )
        X_pat[p_idx] = visits_feats.flatten()

        if df_y is not None and y_pat is not None:
            tgt_list = row["_target"]
            for v in range(min(len(tgt_list), context_length)):
                val = tgt_list[v]
                y_pat[p_idx, v] = float(val) if val is not None else cfg.null_value

    stats_out = computed_stats if norm_stats is None else norm_stats
    return X_pat, y_pat, stats_out, feat_cols


class MLP(nn.Module):
    def __init__(self, hidden_dim: int, mul: int, dropout: float):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(hidden_dim, mul * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mul * hidden_dim, hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=cfg.hidden_dim, num_heads=cfg.n_head,
            dropout=cfg.dropout_att, batch_first=True
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
            nn.Linear(cfg.mul * cfg.hidden_dim, cfg.hidden_dim)
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
        return self.head(h)


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
def eval_ensemble(
    models: list[VisitTransformer],
    loader: DataLoader,
    cfg: Config,
) -> float:
    for m in models:
        m.eval()
    total_loss, total_n = 0.0, 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(cfg.device), y_b.to(cfg.device)
        padding_mask = make_padding_mask(y_b, cfg.null_value)
        preds = torch.stack([m(X_b, padding_mask).squeeze(-1) for m in models])
        pred_mean = preds.mean(0)
        valid = ~padding_mask
        if valid.sum() == 0:
            continue
        loss = F.mse_loss(pred_mean[valid], y_b[valid])
        n = int(valid.sum().item())
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
    train_ds = TensorDataset(X_pat[train_idx], y_pat[train_idx])
    val_ds = TensorDataset(X_pat[val_idx], y_pat[val_idx])
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    models = [VisitTransformer(cfg, n_features).to(cfg.device) for _ in range(cfg.n_ensemble)]
    optimizers = [optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay) for m in models]
    schedulers = [optim.lr_scheduler.StepLR(opt, step_size=cfg.step_size, gamma=cfg.gamma) for opt in optimizers]

    for epoch in range(cfg.epochs):
        for m, opt, sched in zip(models, optimizers, schedulers):
            train_one_epoch(m, train_loader, opt, cfg)
            sched.step()
        
        if (epoch + 1) % 20 == 0:
            val_mse = eval_ensemble(models, val_loader, cfg)
            print(f"  Fold {fold_id} epoch {epoch+1:3d}: val_mse={val_mse:.4f}")

    final_val_mse = eval_ensemble(models, val_loader, cfg)
    print(f"  Fold {fold_id} FINAL val MSE={final_val_mse:.4f}  RMSE={final_val_mse**0.5:.4f}")
    return models, final_val_mse


print("Loading data...")
df_X = pl.read_csv("data/X_train.csv")
df_y = pl.read_csv("data/y_train.csv")
df_Xt = pl.read_csv("data/X_test.csv")
X_test_raw = pd.read_csv("data/X_test.csv")

print("Building patient-level matrices...")
X_pat_np, y_pat_np, train_stats, feat_cols = build_patient_matrix(
    df_X, df_y, cfg.context_length, norm_stats=None
)
n_patients, total_feats = X_pat_np.shape
n_features = total_feats // cfg.context_length
print(f"  Patients: {n_patients}  Features/visit: {n_features}  Total: {total_feats}")

X_pat = torch.tensor(X_pat_np, dtype=torch.float32)
y_pat = torch.tensor(y_pat_np, dtype=torch.float32)

print(f"\n[16] Transformer Peer Adapted GroupKFold-{cfg.n_folds} CV")
cv = GroupKFold(n_splits=cfg.n_folds)
dummy_X = np.zeros((n_patients, 1))
dummy_groups = np.arange(n_patients)
cv_splits = list(cv.split(dummy_X, dummy_groups, groups=dummy_groups))

fold_val_mses: list[float] = []
fold_models_list: list[list[VisitTransformer]] = []

for fold_i, (train_idx, val_idx) in enumerate(cv_splits):
    print(f"\n--- Fold {fold_i + 1} / {cfg.n_folds} ---")
    t0 = time.time()
    fold_models, val_mse = train_fold(X_pat, y_pat, train_idx, val_idx, cfg, n_features, fold_i + 1)
    fold_val_mses.append(val_mse)
    fold_models_list.append(fold_models)
    print(f"  Fold {fold_i+1} done in {(time.time()-t0)/60:.1f} min")

mean_val_mse = float(np.mean(fold_val_mses))
mean_val_rmse = mean_val_mse ** 0.5
print(f"\n{'='*60}")
print(f"  CV MSE  = {mean_val_mse:.4f}")
print(f"  CV RMSE = {mean_val_rmse:.4f}")
print(f"{'='*60}")

print("\n[Final] Training on ALL patients for submission...")
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

print("\n[Submission] Building test predictions...")
X_test_np, _, _, _ = build_patient_matrix(df_Xt, None, cfg.context_length, norm_stats=train_stats)
X_test_t = torch.tensor(X_test_np, dtype=torch.float32).to(cfg.device)

test_patient_ids = (
    df_Xt.sort(["patient_id", "age"])
    .group_by("patient_id")
    .agg(pl.col("age").first())
    ["patient_id"].to_list()
)

for m in final_models:
    m.eval()

with torch.inference_mode():
    pad_mask_test = torch.zeros(len(test_patient_ids), cfg.context_length, dtype=torch.bool, device=cfg.device)
    preds_stack = torch.stack([m(X_test_t, pad_mask_test).squeeze(-1) for m in final_models])
    pred_mean = preds_stack.mean(0).cpu().numpy()

pid_to_preds: dict[str, np.ndarray] = {
    str(pid): pred_mean[i] for i, pid in enumerate(test_patient_ids)
}

rows: list[dict] = []
test_sorted = X_test_raw.sort_values(["patient_id", "age"]).reset_index(drop=True)
for pid, grp in test_sorted.groupby("patient_id", sort=False):
    preds = pid_to_preds[str(pid)]
    for v_idx, (_, row) in enumerate(grp.iterrows()):
        if v_idx < cfg.context_length:
            rows.append({"Index": int(row["Index"]), "target": float(preds[v_idx])})

submission = pd.DataFrame(rows).sort_values("Index").reset_index(drop=True)
Path("submissions").mkdir(exist_ok=True)
submission.to_csv("submissions/16_transformer_peer_adapted.csv", index=False)
print(f"  Wrote submissions/16_transformer_peer_adapted.csv  ({len(submission)} rows)")