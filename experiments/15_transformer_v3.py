# %% [markdown]
# # Experiment 15 — Transformer v3: pipeline camarade + régularisation forte
#
# Reprend la data pipeline Polars du camarade (null_value=0, tri par âge dans
# le group_by, flags réordonnés par timestep) mais avec :
#   - GroupKFold(5) sur patient_id (son KFold était leaky)
#   - dropout 0.3 + weight_decay 0.1 pour corriger l'overfitting de exp 14
#   - lr=5e-3 avec StepLR agressif (coupe par 0.5 tous les 15 epochs)
#   - Features enrichies de exp 14 (47 features/visite)
#   - Early stopping (patience=15 checks × 5 epochs)

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
    n_head: int = 4
    num_layers: int = 4
    mul: int = 4
    dropout_att: float = 0.3      # ×3 vs exp 13/14 pour combattre l'overfitting
    dropout_mlp: float = 0.3
    lr: float = 5e-3              # plus agressif, le scheduler descend vite
    weight_decay: float = 0.1    # ×10 vs exp 14
    batch_size: int = 64
    epochs: int = 200
    step_size: int = 15           # LR ÷2 tous les 15 epochs
    gamma: float = 0.5
    n_ensemble: int = 5
    n_folds: int = 5
    null_value: float = -999.0    # sentinelle : -999 n'est jamais une vraie valeur OFF
    deviation: float = 0.02
    es_patience: int = 15
    es_check_every: int = 5
    es_min_delta: float = 0.01
    device: torch.device = field(default_factory=lambda: DEVICE)

cfg = Config()

# ── Data pipeline (inspirée du camarade, adaptée) ─────────────────────────────
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
    """Pipeline Polars : encode → normalise → flag → group_by patient (trié par âge).

    null_value=0 : les positions paddées valent 0, les nulls aussi → pas de sentinelle.
    Retourne X [n_patients, context_length * n_features], y ou None, stats, feat_cols.
    """
    df = df_X.clone()

    # Gene → one-hot
    for g in GENE_DUMMIES:
        df = df.with_columns(
            (pl.col("gene") == g).cast(pl.Int8).alias(f"gene_{g}")
        )
    # Cohort → 0/1
    df = df.with_columns(
        pl.col("cohort").replace({"A": "0", "B": "1"}).cast(pl.Int8)
    )
    # time_since_diagnosis
    df = df.with_columns(
        (pl.col("age") - pl.col("age_at_diagnosis")).alias("time_since_diagnosis")
    )

    # Attacher les labels
    if df_y is not None:
        lbl = df_y.rename({"target": "_target"}).select(["Index", "_target"])
        df = df.join(lbl, on="Index", how="left")

    # Trier par âge pour avoir les visites dans l'ordre chronologique
    df = df.sort(["patient_id", "age"])

    # Normalisation
    computed_stats: dict = {}
    for col in COLS_TO_NORMALIZE:
        if norm_stats is not None:
            mu, sigma = norm_stats[col]
        else:
            mu = float(df[col].mean() or 0.0)
            sigma = float(df[col].std() or 1.0) + 1e-8
        computed_stats[col] = (mu, sigma)
        df = df.with_columns(((pl.col(col) - mu) / sigma).alias(col))

    # Colonnes de flags (1 = était null avant normalisation)
    for col in COLS_WITH_NULLS:
        df = df.with_columns(
            pl.col(col).is_null().cast(pl.Int8).alias(f"flag_{col}")
        )

    # Remplir les nulls par 0 (= moyenne après normalisation)
    df = df.fill_null(0)

    # Colonnes de features dans l'ordre : base + flags (intercalés par timestep comme le camarade)
    base_feats = [
        "cohort", "sexM",
        "gene_GBA+", "gene_LRRK2+", "gene_OTHER+", "gene_No Mutation",
        "age_at_diagnosis", "age", "ledd",
        "time_since_intake_on", "time_since_intake_off",
        "on", "off", "time_since_diagnosis",
    ]
    flag_feats = [f"flag_{c}" for c in COLS_WITH_NULLS]
    feat_cols = base_feats + flag_feats   # 21 features / visite

    # Group by patient → liste de visites (déjà triée par âge), pad à context_length
    all_cols = feat_cols + (["_target"] if df_y is not None else [])
    agg_cols = [pl.col(c) for c in all_cols]
    df_grouped = df.group_by("patient_id").agg(agg_cols)

    n_patients = df_grouped.height
    n_feats = len(feat_cols)
    X_pat = np.zeros((n_patients, context_length * n_feats), dtype=np.float32)
    y_pat = np.full((n_patients, context_length), cfg.null_value, dtype=np.float32) if df_y is not None else None

    # Remplir les matrices
    for p_idx in range(n_patients):
        row = df_grouped.row(p_idx, named=True)
        visits_feats = np.array(
            [[row[c][v] if v < len(row[c]) and row[c][v] is not None else 0.0
              for c in feat_cols]
             for v in range(context_length)],
            dtype=np.float32
        )  # [context_length, n_feats]
        X_pat[p_idx] = visits_feats.flatten()

        if df_y is not None and y_pat is not None:
            tgt_list = row["_target"]
            for v in range(min(len(tgt_list), context_length)):
                val = tgt_list[v]
                y_pat[p_idx, v] = float(val) if val is not None else cfg.null_value

    stats_out = computed_stats if norm_stats is None else norm_stats
    return X_pat, y_pat, stats_out, feat_cols


# ── Patient IDs dans le même ordre que df_grouped ─────────────────────────────
def get_patient_order(df_X: pl.DataFrame) -> list[str]:
    return (
        df_X.sort(["patient_id", "age"])
        .group_by("patient_id")
        .agg(pl.col("age").first())
        ["patient_id"].to_list()
    )


# ── Model (identique exp 13/14) ───────────────────────────────────────────────
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
    """True où la position est paddée (aucune vraie visite)."""
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

    best_val = float("inf")
    best_state = [m.state_dict() for m in models]
    patience_count = 0
    stopped_epoch = cfg.epochs

    for epoch in range(cfg.epochs):
        for m, opt, sched in zip(models, optimizers, schedulers):
            train_one_epoch(m, train_loader, opt, cfg)
            sched.step()

        if (epoch + 1) % cfg.es_check_every == 0:
            val_mse = eval_ensemble(models, val_loader, cfg)
            if val_mse < best_val - cfg.es_min_delta:
                best_val = val_mse
                best_state = [m.state_dict() for m in models]
                patience_count = 0
            else:
                patience_count += 1

            if (epoch + 1) % 10 == 0:
                train_loss = train_one_epoch(models[0], train_loader, optimizers[0], cfg)
                print(f"  Fold {fold_id} epoch {epoch+1:3d}: "
                      f"val_mse={val_mse:.4f}  best={best_val:.4f}  "
                      f"patience={patience_count}/{cfg.es_patience}")

            if patience_count >= cfg.es_patience:
                stopped_epoch = epoch + 1
                print(f"  Fold {fold_id} early stop at epoch {stopped_epoch}")
                break

    for m, state in zip(models, best_state):
        m.load_state_dict(state)

    print(f"  Fold {fold_id} FINAL val MSE={best_val:.4f}  RMSE={best_val**0.5:.4f}  (stopped ep {stopped_epoch})")
    return models, best_val


# ── Main ──────────────────────────────────────────────────────────────────────
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

# ── Cross-validation ──────────────────────────────────────────────────────────
print(f"\n[15] Transformer v3 GroupKFold-{cfg.n_folds} CV")
cv = GroupKFold(n_splits=cfg.n_folds)
dummy_X = np.zeros((n_patients, 1))
dummy_groups = np.arange(n_patients)
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

mean_val_mse = float(np.mean(fold_val_mses))
mean_val_rmse = mean_val_mse ** 0.5
print(f"\n{'='*60}")
print(f"  CV MSE  = {mean_val_mse:.4f}")
print(f"  CV RMSE = {mean_val_rmse:.4f}  (exp 14 était 3.90)")
print(f"{'='*60}")

print("\n[Results] CV summary:")
print(f"  experiment      : 15_transformer_v3")
print(f"  cv_mse          : {mean_val_mse:.4f}")
print(f"  cv_rmse         : {mean_val_rmse:.4f}")
print(f"  fold_mses       : {[round(x, 4) for x in fold_val_mses]}")
print(f"  dropout         : {cfg.dropout_att}")
print(f"  weight_decay    : {cfg.weight_decay}")

# ── Entraînement final ────────────────────────────────────────────────────────
print("\n[Final] Training on ALL patients for submission...")
full_ds = TensorDataset(X_pat, y_pat)
full_loader = DataLoader(full_ds, batch_size=cfg.batch_size, shuffle=True)

# Epochs finales = médiane des epochs auxquelles l'early stopping a triggeré
final_epochs = 100  # valeur par défaut raisonnable
print(f"  Training for {final_epochs} epochs")

final_models = [VisitTransformer(cfg, n_features).to(cfg.device) for _ in range(cfg.n_ensemble)]
final_optimizers = [optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay) for m in final_models]
final_schedulers = [optim.lr_scheduler.StepLR(opt, step_size=cfg.step_size, gamma=cfg.gamma) for opt in final_optimizers]

t0 = time.time()
for epoch in range(final_epochs):
    for m, opt, sched in zip(final_models, final_optimizers, final_schedulers):
        train_one_epoch(m, full_loader, opt, cfg)
        sched.step()
    if (epoch + 1) % 20 == 0:
        print(f"  Final epoch {epoch+1}/{final_epochs} ({(time.time()-t0)/60:.1f} min)")
print(f"  Final training done in {(time.time()-t0)/60:.1f} min")

# ── Prédictions test ──────────────────────────────────────────────────────────
print("\n[Submission] Building test predictions...")
X_test_np, _, _, _ = build_patient_matrix(
    df_Xt, None, cfg.context_length, norm_stats=train_stats
)
X_test_t = torch.tensor(X_test_np, dtype=torch.float32).to(cfg.device)

# Ordre des patients test (même que build_patient_matrix → group_by)
test_patient_ids = (
    df_Xt.sort(["patient_id", "age"])
    .group_by("patient_id")
    .agg(pl.col("age").first())
    ["patient_id"].to_list()
)

for m in final_models:
    m.eval()

with torch.inference_mode():
    # Pas de positions paddées au test — masque entièrement False
    pad_mask_test = torch.zeros(len(test_patient_ids), cfg.context_length, dtype=torch.bool, device=cfg.device)
    preds_stack = torch.stack([
        m(X_test_t, pad_mask_test).squeeze(-1) for m in final_models
    ])
    pred_mean = preds_stack.mean(0).cpu().numpy()  # [n_test_patients, 12]

pid_to_preds: dict[str, np.ndarray] = {
    str(pid): pred_mean[i] for i, pid in enumerate(test_patient_ids)
}

rows: list[dict] = []
test_sorted = X_test_raw.sort_values(["patient_id", "age"]).reset_index(drop=True)
for pid, grp in test_sorted.groupby("patient_id", sort=False):
    preds = pid_to_preds[str(pid)]
    for v_idx, (_, row) in enumerate(grp.iterrows()):
        if v_idx < cfg.context_length:
            rows.append({"Index": int(row["Index"]), "target": float(preds[v_idx])})  # type: ignore[arg-type]

submission = pd.DataFrame(rows).sort_values("Index").reset_index(drop=True)
Path("submissions").mkdir(exist_ok=True)
submission.to_csv("submissions/15_transformer_v3.csv", index=False)
print(f"  Wrote submissions/15_transformer_v3.csv  ({len(submission)} rows)")

print(f"\n{'='*60}")
print(f"FINAL RESULTS")
print(f"  13_transformer CV:      RMSE 3.9161")
print(f"  14_transformer_v2 CV:   RMSE 3.8962")
print(f"  15_transformer_v3 CV:   MSE {mean_val_mse:.4f}  RMSE {mean_val_rmse:.4f}")
print(f"{'='*60}")
