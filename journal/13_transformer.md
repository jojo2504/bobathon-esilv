# Experiment 13 — Patient-level Transformer with Visit Tokens

## Question / hypothesis

Can a Transformer that treats each patient's visits as a sequence of tokens — predicting all visit scores jointly — substantially beat the gradient-boosting RMSE of 4.58 by exploiting the full cross-visit context?

## Motivation

The key insight from the reference approach (2nd place, MSE ~7.35 ≈ RMSE ~2.7): **concatenate all visits of a patient into a single row, then use a Transformer where each visit is a token**. This lets the model attend across time — a patient's early visits inform their later ones and vice versa. Gradient boosting treats each visit independently; the Transformer sees the entire disease trajectory at once.

The reference author used:
- context_length = 12 (max visits, padded with null_value = 0)
- MLP token embedding per visit (raw features → hidden_dim)
- Standard Transformer blocks (MHA + LayerNorm + MLP)
- Deep ensemble of 30 models → MSE 7.35 on validation

Our implementation mirrors their architecture faithfully, adapted to our infrastructure (PyTorch + MPS, skore Hub, GroupKFold-5 on patient_id, sklearn-compatible wrapper for submission).

## Method

**Data pipeline (per patient):**
1. One-hot encode `gene` (4 dummies: GBA+, LRRK2+, OTHER+, No Mutation)
2. Map `cohort` A→0, B→1
3. Add `time_since_diagnosis = age - age_at_diagnosis`
4. Sort visits by `age` within each patient
5. For each of the 7 nullable numeric features, create a binary flag column (1=was_null, 0=observed)
6. Normalize numeric columns (mean/std from **train fold only** at CV time)
7. Fill NaN → 0 (equals the mean after normalization)
8. Pad each patient to `context_length=12` visits (null_value = -999 sentinel for masked loss)
9. Shape: `[n_patients, context_length × n_features_per_visit]`
10. Target: `[n_patients, context_length]` — true OFF score per visit, padded with null_value

**Features per visit (n_features_per_visit ≈ 21):**
`cohort, sexM, gene_GBA+, gene_LRRK2+, gene_OTHER+, gene_no_mutation, age_at_diagnosis, age, ledd, time_since_intake_on, time_since_intake_off, on, off, time_since_diagnosis` (14 base) + 7 flag columns = 21

**Model (Transformer):**
- Token embedding: Linear(n_features, mul×hidden_dim) → GELU → Linear(mul×hidden_dim, hidden_dim)
- Positional embedding: learned `nn.Embedding(context_length, hidden_dim)`
- N Transformer blocks: PreNorm MHA + PreNorm MLP (residual)
- Output: LayerNorm → Linear(hidden_dim, 1) per token
- Loss: MSE on non-padded positions only

**Training:**
- GroupKFold(5) on patient_id — all visits of a patient are a single row, so the fold structure is exact
- AdamW, lr=1e-3, weight_decay=1e-2
- StepLR: cut LR by 0.5 every 20 epochs, 100 epochs total
- Batch size: 64 patients
- Deep ensemble: 5 independently-trained models (budget: ~30 min on MPS)
- Device: MPS (Apple Silicon) with CPU fallback

**Hyperparameters:**
- hidden_dim = 64, n_head = 4, num_layers = 4, mul = 4, dropout = 0.1
- context_length = 12

**Submission:**
- For test patients: same pipeline (sort by age, pad, normalize with train stats)
- Predict with all 5 ensemble members, average → test predictions

## Risks

- **Training instability**: transformers on small tabular datasets can be finicky. Use gradient clipping (max_norm=1.0) and careful initialization.
- **MPS memory**: 5 models × 4 layers should fit easily. Fall back to CPU if OOM.
- **Overfitting**: only 5576 train patients. Dropout + weight_decay + early stopping on val loss.
- **Padding mask**: if null_value sentinel leaks into attention, the model will attend to padding. Need causal or padding mask.

## Status

- **State**: approved
- **Approved by user on**: 2025-07-15
- **Headline result**: —
- **Implication for next iteration**: —

**Sourcing strategy**: `user` (reference architecture from challenge 2nd-place writeup)
