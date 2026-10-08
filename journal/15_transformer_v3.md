# Experiment 15 — Transformer v3: camarade pipeline + régularisation forte

## Question / hypothesis

La data pipeline du camarade (Polars pur, null_value=0, tri par âge dans le group_by)
combinée à une régularisation plus forte (dropout 0.3, weight_decay 0.1) peut-elle
corriger l'overfitting visible dans exp 14 (train MSE ~9 vs val MSE ~15 à convergence)
et passer sous RMSE 3.80 ?

## Motivation

Exp 13 et 14 montrent un gap train/val croissant après epoch 40 :
- Exp 14 fold 1 : train=8.5 vs val=16.2 à epoch 100 → gap ×1.9
- L'early stopping arrête vers epoch 110 mais le meilleur val MSE est atteint vers epoch 40

La régularisation actuelle (dropout=0.1, weight_decay=1e-2) est insuffisante.
Deux leviers :
1. **dropout_att=0.3, dropout_mlp=0.3** (×3 vs exp 13/14)
2. **weight_decay=0.1** (×10 vs exp 13/14)
3. **lr=5e-3** (plus agressif en début, le scheduler descend vite)

La pipeline Polars du camarade est plus propre :
- `null_value=0` : les positions paddées = 0 après normalisation = moyenne, pas de
  sentinelle à gérer dans le masque
- Tri par âge intégré dans le group_by (via list.get avec index positionnel)
- Flag columns réordonnées proprement par timestep

**IMPORTANT** : on garde GroupKFold sur patient_id (son KFold simple est leaky).

## Status

- **State**: approved
- **Approved by user on**: 2025-07-15
- **Headline result**: —
- **Implication for next iteration**: —

**Sourcing strategy**: `user` (code camarade adapté)
