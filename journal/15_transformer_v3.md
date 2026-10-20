# Experiment 15 — Transformer v3: teammate pipeline + stronger regularisation

## Question / hypothesis

Can using the teammate's data pipeline (pure Polars, `null_value=0`, age-sorted visits in `group_by`) combined with stronger regularisation (dropout 0.3, weight_decay 0.1) fix the overfitting seen in exp 14 (train MSE ~9 vs val MSE ~15 at convergence) and push RMSE below 3.80?

## Motivation

Experiments 13 and 14 show a growing train/val gap after epoch 40:
- Exp 14 fold 1: train=8.5 vs val=16.2 at epoch 100 → gap ×1.9
- Early stopping triggers around epoch 110 but the best val MSE is already reached around epoch 40

The current regularisation (dropout=0.1, weight_decay=1e-2) is not strong enough.
Two levers to pull:
1. **dropout_att=0.3, dropout_mlp=0.3** (×3 vs exp 13/14)
2. **weight_decay=0.1** (×10 vs exp 13/14)
3. **lr=5e-3** (more aggressive early, the scheduler drops it quickly)

The teammate's Polars pipeline is cleaner:
- `null_value=0`: padded positions = 0 after normalisation = the mean, no sentinel to handle in the attention mask
- Age sort built into the `group_by` (via `list.get` with positional index)
- Flag columns cleanly reordered by timestep

**IMPORTANT**: we keep GroupKFold on patient_id — the teammate's plain KFold leaks patients.

## Status

- **State**: approved
- **Approved by user on**: 2025-07-15
- **Headline result**: —
- **Implication for next iteration**: —

**Sourcing strategy**: `user` (adapted from teammate's code)
