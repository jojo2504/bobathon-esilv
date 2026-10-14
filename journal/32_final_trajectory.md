# Expérience 32 — Ensemble d'arbres et trajectoires par patient

## Idée

Prédire le vrai score OFF en combinant les données de chaque visite avec
l'évolution des observations du même patient. Les trajectoires linéaires et
quadratiques permettent d'exploiter la progression avec l'âge et de réduire
le bruit des mesures ON/OFF.

Le modèle final utilise LightGBM et XGBoost. Il ne dépend d'aucune prédiction
d'un ancien modèle et ne contient aucun réseau de neurones.

## Stratégie

1. **Corriger les mesures ON/OFF.** Deux petits LightGBM apprennent respectivement
   `target - on` et `target - off` sur les visites où le score est observé.
   Ils utilisent les variables de la visite, dont `rater_id`, le traitement et
   les délais depuis la prise. Les scores absents restent absents après correction.
2. **Construire 121 variables.** Ajouter les agrégats patient, les écarts à la
   moyenne, le rang des visites, les trajectoires régularisées des scores et
   de la LEDD, les observations voisines et leurs distances temporelles.
   Les variables pharmacocinétiques utilisent `exp(-k × délai)` pour
   `k ∈ {0.2, 0.5, 1.0}`, seul et multiplié par la LEDD.
3. **Combiner deux modèles.** Calculer la moyenne à poids égaux des prédictions
   de LightGBM et XGBoost.
4. **Lisser par patient.** Mélanger 85 % de la courbe quadratique ajustée aux
   prédictions avec 15 % des prédictions brutes, puis borner le score à `[0, 132]`.
5. **Valider et publier.** Évaluer avec `skore.evaluate` sur cinq groupes de
   patients, réentraîner sur toutes les données et publier le rapport sur Skore.

## Hyperparamètres

| Paramètre | LightGBM final | XGBoost final |
|---|---|---|
| Nombre d'arbres | 2 200 | 1 760 |
| Learning rate | 0.025 | 0.035 |
| Capacité | 63 feuilles | Profondeur 5 |
| Minimum par feuille | 25 observations | `min_child_weight=15` |
| Sous-échantillonnage des lignes | 0.85, chaque itération | 0.85 |
| Sous-échantillonnage des colonnes | 0.85 | 0.85 |
| Régularisation L1 / L2 | 0.05 / 3 | 0.05 / 10 |
| Graine | 2026 | 2026 |

Chaque modèle de correction utilise 350 arbres, 15 feuilles, un learning rate
de 0.035, au moins 100 observations par feuille et une régularisation L2 de 20.

## Validation

`GroupKFold(5)` sépare les patients : aucune visite d'un patient de validation
n'est utilisée pour entraîner les modèles de son fold. Les corrections
supervisées sont également apprises uniquement sur les patients d'entraînement.
La séparation des patients entre les fichiers train et test est vérifiée.

Les agrégats et les trajectoires d'un patient de validation ou de test sont
calculés depuis ses propres observations X, sans ses labels. L'inférence
suppose donc que toutes ses visites disponibles sont fournies ensemble,
comme dans les fichiers du challenge ; ce protocole n'est pas une prévision
strictement limitée aux visites passées.

Le premier fold a servi à comparer les forces de lissage 0, 0.5, 0.85 et 1.
Le résultat global inclut donc cette sélection sur un fold ; il ne constitue
pas un score sur un jeu de test indépendant de tout choix de modèle.

## Résultats

| Mesure | RMSE |
|---|---|
| Fold 1 | 3.56730 |
| Fold 2 | 3.59380 |
| Fold 3 | 3.46349 |
| Fold 4 | 3.47063 |
| Fold 5 | 3.64308 |
| **Global sur toutes les prédictions OOF** | **3.54836** |
| Anciennes prédictions T17 disponibles | 3.93637 |

Le gain relatif est de **9.86 %** par rapport aux prédictions T17 sauvegardées.
Le RMSE global correspond à toutes les visites OOF réunies, et non à une
moyenne des cinq RMSE. Aucun score Kaggle n'a été mesuré pour cette expérience.

Le modèle final est entraîné sur **44 590 visites** et génère **11 013
prédictions**. Le CSV est vérifié : indices uniques, ordre identique au modèle
de soumission et absence de valeurs manquantes ou infinies.

Rapport publié : [32_final_trajectory sur Skore](https://skore.probabl.ai/degensunite/bobathon-esilv/cross-validations/86150).

## Fichiers générés

| Fichier | Contenu |
|---|---|
| `experiments/32_final_trajectory.py` | Estimateur, validation, entraînement et publication |
| `scratch/32_final_trajectory_report.joblib` | Rapport Skore des cinq folds réels |
| `scratch/32_final_trajectory_model.joblib` | Ensemble entraîné sur toutes les données |
| `scratch/32_final_trajectory_oof.csv` | Prédictions de validation alignées sur `Index` |
| `scratch/32_final_trajectory_metrics.json` | Scores, tailles des données et identifiant Skore |
| `submissions/32_final_trajectory.csv` | Soumission au format `Index,target` |

## Exécution

Depuis la racine du dépôt :

```sh
# Comparer le lissage sur le premier fold
.venv/bin/python -u experiments/32_final_trajectory.py --probe

# Valider, entraîner le modèle final et générer le CSV
.venv/bin/python -u experiments/32_final_trajectory.py

# Publier le rapport sur Skore
.venv/bin/python -u experiments/32_final_trajectory.py --push
```

Les exécutions suivantes réutilisent les fichiers `.joblib` existants. Après
modification des données ou du modèle, déplacer ou supprimer explicitement
ces deux fichiers avant de relancer pour refaire l'entraînement. La commande
`--push` publie sur Skore ; elle ne soumet pas le CSV à Kaggle.
