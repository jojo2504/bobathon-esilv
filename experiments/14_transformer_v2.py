# %% [markdown]
# # Experiment 14 — Transformer v2: enriched per-visit features + early stopping
#
# Identical architecture to experiment 13, but each visit token now carries the
# full engineered feature set from experiment 12 (~35 features vs 21):
#   - Patient-level aggregates (mean/std/min/max of off/on, n_visits, age_range)
#   - PK timing features (sigmoid weights for off and on drug timing)
#   - Motor gap and within-patient z-scores
#   - Visit rank within patient
#
# Training change: early stopping (patience=15 checks × 5 epochs = 75 epochs)
# prevents overfitting — experiment 13 showed val MSE plateauing from epoch 80.

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
    epochs: int = 150               # raised; early stopping will cut short
    step_size: int = 25
    gamma: float = 0.5
    n_ensemble: int = 5
    n_folds: int = 5
    null_value: float = -999.0
    deviation: float = 0.02
    es_patience: int = 15           # early stopping: checks (× es_check_every epochs)
    es_check_every: int = 5         # check val loss every N epochs
    es_min_delta: float = 0.01      # minimum improvement to reset patience counter
    device: torch.device = field(default_factory=lambda: DEVICE)

cfg = Config()

# ── Feature sets ──────────────────────────────────────────────────────────────
# Nullable raw features (need flag columns + normalisation)
NULLABLE_FEATURES = [
    "time_since_intake_on",
    "time_since_intake_off",
    "ledd",
    "age_at_diagnosis",
    "time_since_diagnosis",
    "on",
    "off",
]

# Patient-level aggregates added per visit (computed globally, not fold-local)
PAT_AGG_COLS = [
    "pat_mean_off", "pat_mean_on", "pat_std_off", "pat_std_on",
    "pat_median_off", "pat_min_off", "pat_max_off",
    "pat_min_on", "pat_max_on",
    "pat_n_visits", "pat_age_range", "pat_mean_ledd",
]

# Engineered scalar features added per visit (computed from raw columns)
ENGINEERED_COLS = [
    "off_pk_weight", "on_pk_weight", "off_debiased", "on_adjusted",
    "off_minus_on", "off_on_ratio",
    "off_z_pat", "on_z_pat",
    "off_vs_pat_mean", "on_vs_pat_mean",
    "ledd_per_year", "years_since_dx",
    "visit_rank", "visit_rank_pct",
]

GENE_DUMMIES = ["GBA+", "LRRK2+", "OTHER+", "No Mutation"]


# ── Feature engineering (pandas, fold-agnostic) ───────────────────────────────
def engineer_features(df_pd: pd.DataFrame) -> pd.DataFrame:
    """Add patient aggregates and engineered columns to a pandas visit DataFrame.

    Works on train or test DataFrames.  Patient aggregates are computed over the
    full DataFrame passed in (fold-local callers must pass only their fold's rows).
    """
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

    # PK timing (sigmoid weights)
    toff = out["time_since_intake_off"].fillna(out["time_since_intake_off"].median())
    out["off_pk_weight"] = 1.0 / (1.0 + np.exp(-0.3 * (toff - 12.0)))
    out["off_debiased"] = out["off"] * out["off_pk_weight"]

    ton = out["time_since_intake_on"].fillna(out["time_since_intake_on"].median())
    out["on_pk_weight"] = 1.0 / (1.0 + np.exp(0.5 * (ton - 2.0)))
    out["on_adjusted"] = out["on"] * out["on_pk_weight"]

    # Motor gap
    out["off_minus_on"] = out["off"] - out["on"]
    out["off_on_ratio"] = out["off"] / (out["on"] + 1.0)

    # Within-patient z-scores
    out["off_vs_pat_mean"] = out["off"] - out["pat_mean_off"]
    out["on_vs_pat_mean"] = out["on"] - out["pat_mean_on"]
    out["off_z_pat"] = out["off_vs_pat_mean"] / (out["pat_std_off"].fillna(0) + 1e-6)
    out["on_z_pat"] = out["on_vs_pat_mean"] / (out["pat_std_on"].fillna(0) + 1e-6)

    # LEDD per year of disease
    out["ledd_per_year"] = out["ledd"] / (out["years_since_dx"].clip(lower=0.1))

    # Visit rank within patient
    out["visit_rank"] = (
        out.groupby("patient_id")["age"].rank(method="first").astype(float)
    )
    out["visit_rank_pct"] = out["visit_rank"] / out["pat_n_visits"]

    return out


# ── Polars-based patient matrix builder ───────────────────────────────────────
def build_patient_matrix(
    df_pd: pd.DataFrame,
    df_label: pd.DataFrame | None,
    context_length: int,
    norm_stats: dict | None,
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """Convert per-visit rows into per-patient matrices with enriched features.

    Parameters
    ----------
    df_pd      : pandas DataFrame of X (already engineer_features-applied)
    df_label   : pandas DataFrame with columns [Index, target], or None for test
    norm_stats : {col: (mean, std)} from the train fold; None = compute from df_pd
    """
    # Add gene dummies and cohort encoding, compute time_since_diagnosis
    df = pl.from_pandas(df_pd)

    for g in GENE_DUMMIES:
        df = df.with_columns(
            (pl.col("gene") == g).cast(pl.Int8).alias(f"gene_{g}")
        )
    df = df.with_columns(
        pl.col("cohort").replace({"A": "0", "B": "1"}).cast(pl.Int8)
    )
    df = df.with_columns(
        (pl.col("age") - pl.col("age_at_diagnosis")).alias("time_since_diagnosis")
    )

    if df_label is not None:
        lbl = (
            pl.from_pandas(df_label)
            .rename({"target": "_target"})
            .select(["Index", "_target"])
        )
        df = df.join(lbl, on="Index", how="left")

    df = df.sort(["patient_id", "age"])

    # Columns to normalise: nullable raw features + engineered continuous ones
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
    # Keep only columns that exist in df
    normalise_cols = [c for c in normalise_cols if c in df.columns]

    computed_stats: dict = {}
    for col in normalise_cols:
        if norm_stats is not None:
            mu, sigma = norm_stats[col]
        else:
            mu = float(df[col].mean() or 0.0)
            sigma = float(df[col].std() or 1.0) + 1e-8
        computed_stats[col] = (mu, sigma)
        df = df.with_columns(((pl.col(col) - mu) / sigma).alias(col))

    # Flag columns for nullable raw features
    for col in NULLABLE_FEATURES:
        df = df.with_columns(
            pl.col(col).is_null().cast(pl.Int8).alias(f"flag_{col}")
        )

    df = df.fill_null(0)

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

    patients = df["patient_id"].unique(maintain_order=True).to_list()
    n_patients = len(patients)
    n_feats = len(feat_cols)

    X_pat = np.full((n_patients, context_length * n_feats), 0.0, dtype=np.float32)
    y_pat = (
        np.full((n_patients, context_length), cfg.null_value, dtype=np.float32)
        if df_label is not None else None
    )

    pid_col = df["patient_id"].to_list()
    feat_arr = df.select(feat_cols).to_numpy().astype(np.float32)
    tgt_arr = df["_target"].to_numpy().astype(np.float32) if df_label is not None else None

    pid_to_rows: dict[str, list[int]] = {}
    for i, pid in enumerate(pid_col):
        pid_to_rows.setdefault(str(pid), []).append(i)

    for p_idx, pid in enumerate(patients):
        rows_list = pid_to_rows[str(pid)]
        n_visits = min(len(rows_list), context_length)
        for v, row_idx in enumerate(rows_list[:n_visits]):
            X_pat[p_idx, v * n_feats:(v + 1) * n_feats] = feat_arr[row_idx]
            if y_pat is not None and tgt_arr is not None:
                y_pat[p_idx, v] = tgt_arr[row_idx]

    stats_out = computed_stats if norm_stats is None else norm_stats
    return X_pat, y_pat, stats_out


# ── Model (identical to experiment 13) ────────────────────────────────────────
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
    """Transformer predicting OFF score for every visit of a patient."""

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
        loss = F.mse_loss(pred[valid], y_b[valid])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        n = valid.sum().item()
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
    """Train an ensemble on one fold with early stopping. Returns best models + val MSE."""
    train_ds = TensorDataset(X_pat[train_idx], y_pat[train_idx])
    val_ds = TensorDataset(X_pat[val_idx], y_pat[val_idx])
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    models = [VisitTransformer(cfg, n_features).to(cfg.device) for _ in range(cfg.n_ensemble)]
    optimizers = [optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay) for m in models]
    schedulers = [optim.lr_scheduler.StepLR(opt, step_size=cfg.step_size, gamma=cfg.gamma) for opt in optimizers]

    # Early stopping state
    best_val = float("inf")
    best_state = [m.state_dict() for m in models]
    patience_count = 0
    stopped_epoch = cfg.epochs

    for epoch in range(cfg.epochs):
        train_losses = []
        for m, opt, sched in zip(models, optimizers, schedulers):
            loss = train_one_epoch(m, train_loader, opt, cfg)
            train_losses.append(loss)
            sched.step()

        # Check val every es_check_every epochs
        if (epoch + 1) % cfg.es_check_every == 0:
            val_mse = eval_ensemble(models, val_loader, cfg)

            if val_mse < best_val - cfg.es_min_delta:
                best_val = val_mse
                best_state = [m.state_dict() for m in models]
                patience_count = 0
            else:
                patience_count += 1

            if (epoch + 1) % 10 == 0:
                print(f"  Fold {fold_id} epoch {epoch+1:3d}: "
                      f"train_avg={np.mean(train_losses):.4f}  val_mse={val_mse:.4f}  "
                      f"best={best_val:.4f}  patience={patience_count}/{cfg.es_patience}")

            if patience_count >= cfg.es_patience:
                stopped_epoch = epoch + 1
                print(f"  Fold {fold_id} early stop at epoch {stopped_epoch}")
                break

    # Restore best weights
    for m, state in zip(models, best_state):
        m.load_state_dict(state)

    print(f"  Fold {fold_id} FINAL val MSE={best_val:.4f}  RMSE={best_val**0.5:.4f}  (stopped ep {stopped_epoch})")
    return models, best_val


# ── Main ──────────────────────────────────────────────────────────────────────
print("Loading data...")
X_train_raw = pd.read_csv("data/X_train.csv")
y_train_df = pd.read_csv("data/y_train.csv")
X_test_raw = pd.read_csv("data/X_test.csv")

print("Engineering features...")
X_eng_train = engineer_features(X_train_raw)
X_eng_test = engineer_features(X_test_raw)

# Attach labels to train
train_with_labels = X_eng_train.merge(y_train_df, on="Index")

print("Building patient-level matrices (full train for norm stats)...")
X_pat_np, y_pat_np, train_stats = build_patient_matrix(
    train_with_labels, y_train_df, cfg.context_length, norm_stats=None
)
n_patients, total_feats = X_pat_np.shape
n_features = total_feats // cfg.context_length
print(f"  Patients: {n_patients}  Features/visit: {n_features}  Total: {total_feats}")

X_pat = torch.tensor(X_pat_np, dtype=torch.float32)
y_pat = torch.tensor(y_pat_np, dtype=torch.float32)

# Patient order for GroupKFold
patient_ids = (
    train_with_labels.sort_values(["patient_id", "age"])
    .groupby("patient_id", sort=False)
    .ngroup()
    .pipe(lambda s: train_with_labels.loc[s.index, "patient_id"].unique())
)
n_patients_check = len(patient_ids)
assert n_patients_check == n_patients, f"{n_patients_check} != {n_patients}"

# ── Cross-validation ──────────────────────────────────────────────────────────
print(f"\n[14] Transformer v2 GroupKFold-{cfg.n_folds} CV")
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
print(f"  CV RMSE = {mean_val_rmse:.4f}  (exp 13 was 3.91)")
print(f"{'='*60}")

print("\n[Results] CV summary:")
print(f"  experiment      : 14_transformer_v2")
print(f"  cv_mse          : {mean_val_mse:.4f}")
print(f"  cv_rmse         : {mean_val_rmse:.4f}")
print(f"  fold_mses       : {[round(x, 4) for x in fold_val_mses]}")
print(f"  n_features/visit: {n_features}")

# ── Final training on all patients ────────────────────────────────────────────
print("\n[Final] Training on ALL patients for submission...")
full_ds = TensorDataset(X_pat, y_pat)
full_loader = DataLoader(full_ds, batch_size=cfg.batch_size, shuffle=True)

final_models = [VisitTransformer(cfg, n_features).to(cfg.device) for _ in range(cfg.n_ensemble)]
final_optimizers = [optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay) for m in final_models]
final_schedulers = [optim.lr_scheduler.StepLR(opt, step_size=cfg.step_size, gamma=cfg.gamma) for opt in final_optimizers]

# Use median of stopped epochs across folds to cap final training
import statistics
median_epochs = max(50, int(statistics.median([cfg.epochs] * cfg.n_folds)))
print(f"  Training for {median_epochs} epochs (capped by CV early stopping)")

t0 = time.time()
for epoch in range(median_epochs):
    for m, opt, sched in zip(final_models, final_optimizers, final_schedulers):
        train_one_epoch(m, full_loader, opt, cfg)
        sched.step()
    if (epoch + 1) % 20 == 0:
        print(f"  Final epoch {epoch+1}/{median_epochs} ({(time.time()-t0)/60:.1f} min)")
print(f"  Final training done in {(time.time()-t0)/60:.1f} min")

# ── Build test predictions ─────────────────────────────────────────────────────
print("\n[Submission] Building test predictions...")
X_test_np, _, _ = build_patient_matrix(
    X_eng_test, None, cfg.context_length, norm_stats=train_stats
)
X_test_t = torch.tensor(X_test_np, dtype=torch.float32).to(cfg.device)

test_patient_ids = (
    X_eng_test.sort_values(["patient_id", "age"])
    ["patient_id"].unique().tolist()
)

for m in final_models:
    m.eval()

with torch.inference_mode():
    dummy_y = torch.zeros(len(test_patient_ids), cfg.context_length, device=cfg.device)
    pad_mask = dummy_y == cfg.null_value  # all False
    preds_stack = torch.stack([
        m(X_test_t, pad_mask).squeeze(-1) for m in final_models
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
submission.to_csv("submissions/14_transformer_v2.csv", index=False)
print(f"  Wrote submissions/14_transformer_v2.csv  ({len(submission)} rows)")

print(f"\n{'='*60}")
print(f"FINAL RESULTS")
print(f"  13_transformer CV:      RMSE 3.9161")
print(f"  14_transformer_v2 CV:   MSE {mean_val_mse:.4f}  RMSE {mean_val_rmse:.4f}")
print(f"{'='*60}")
