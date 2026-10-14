# Expérience 33 — Calibration des mesures et progression par patient

## Nouvelle approche

Cette expérience repart des CSV bruts. Elle n'importe pas le modèle 32 et
n'utilise aucune de ses prédictions comme variable ou composante d'ensemble.
Ses prédictions OOF servent uniquement à comparer les performances.

L'analyse des données d'entraînement met en évidence trois signaux :

- L'écart entre le score observé et la cible varie avec l'évaluateur et le
  délai depuis la prise du traitement.
- Les mesures ON et la LEDD sont presque toujours absentes aux faibles
  valeurs de la cible. Le début et la fin des observations apportent donc
  de l'information sur la trajectoire.
- Les visites d'un patient décrivent une évolution assez régulière pour
  apprendre conjointement son niveau, sa pente et sa courbure.

## Architecture

1. **Modèle de mesure direct.** Pour ON et OFF séparément, une régression Ridge
   apprend `score_observé = cible × facteur(contexte) + décalage(contexte)`.
   Le contexte comprend des splines du délai, l'évaluateur, la dose,
   la cohorte et les indicateurs de valeurs manquantes. Une pondération
   en fonction de la cible réduit la domination des scores élevés.
2. **Inversion et trajectoire initiale.** Inverser cette relation donne une
   estimation de sévérité depuis chaque mesure disponible. Les observations
   ON/OFF sont combinées dans une courbe quadratique régularisée, avec des
   poids calculés depuis le facteur de calibration et le bruit estimé.
3. **Représentation des visites.** Construire 228 variables : données de la
   visite, agrégats, courbes, moyennes locales à trois échelles temporelles,
   premières et dernières observations, et toutes les mesures ordonnées
   du patient, jusqu'à 12 visites comme dans ce jeu de données.
4. **Apprentissage par visite.** Deux LightGBM corrigent les erreurs de la
   trajectoire initiale. Leurs prédictions sont lissées par patient.
5. **Apprentissage par patient.** Trois autres LightGBM prédisent les
   coefficients de correction de la courbe : niveau, pente et courbure.
   Ces coefficients sont définis sur une base temporelle orthogonale.
6. **Ensemble final.** Combiner 35 % du premier modèle par visite,
   35 % du second et 30 % du modèle par patient. Borner le score à `[0, 132]`.

La calibration et tous les modèles supervisés sont réentraînés dans chaque
fold. Les transformations des patients de validation utilisent seulement
leurs observations X et les paramètres appris sur les patients train.
Les modèles finaux sont des Ridge et des LightGBM, sans réseau de neurones.

## Choix des modèles

Les essais utilisent exclusivement le premier fold de `GroupKFold(5)` :

| Candidat | RMSE sur le fold d'exploration |
|---|---|
| Modèle 32, référence | 3.56730 |
| LightGBM 31 feuilles, 2 600 arbres | 3.41357 |
| LightGBM 63 feuilles, 2 600 arbres | 3.41796 |
| XGBoost profondeur 5, 2 600 arbres | 3.44401 |
| LightGBM 15 feuilles, entraînement prolongé | 3.40006 |
| LightGBM 31 feuilles, entraînement prolongé | 3.39618 |
| Modèle de coefficients par patient | 3.52569 |
| **Ensemble retenu 35 / 35 / 30** | **3.35755** |

Le modèle par patient est moins performant seul, mais ses erreurs sont
complémentaires de celles des modèles par visite. XGBoost n'est pas retenu
dans l'ensemble final.

Les hyperparamètres et les poids sont figés avant la validation sur les
quatre autres folds. Le premier fold a participé à la sélection ; son score
est donc optimiste. La comparaison sur les folds 2 à 5 constitue la
vérification sur d'autres groupes de validation. Cette procédure n'est pas
une validation croisée imbriquée : les patients de ces groupes ont aussi
participé à l'entraînement des modèles exploratoires. Il n'existe pas ici
de jeu de test étiqueté totalement indépendant de la recherche de modèle.

Les visites sont remises dans un ordre canonique avant l'inférence pour
stabiliser les réductions numériques des variables patient. Un contrôle
vérifie que permuter les lignes d'entrée conserve les mêmes prédictions.

## Hyperparamètres retenus

| Composante | Arbres | Feuilles | Learning rate | L2 |
|---|---|---|---|---|
| LightGBM par visite A | 6 499 | 15 | 0.025 | 8 |
| LightGBM par visite B | 6 497 | 31 | 0.025 | 8 |
| LightGBM patient : niveau | 2 998 | 15 | 0.02 | 10 |
| LightGBM patient : pente | 2 333 | 15 | 0.02 | 10 |
| LightGBM patient : courbure | 2 067 | 15 | 0.02 | 10 |

Les nombres d'arbres proviennent des essais avec arrêt anticipé sur le
fold d'exploration. La validation complète utilise ces nombres fixes,
sans consulter les labels des autres folds pour arrêter l'entraînement.

## Résultats de validation complète

| Fold | Modèle 32 | Modèle 33 |
|---|---|---|
| 1 — utilisé pour l'exploration | 3.56730 | 3.35490 |
| 2 | 3.59380 | 3.33285 |
| 3 | 3.46349 | 3.38568 |
| 4 | 3.47063 | 3.33698 |
| 5 | 3.64308 | 3.34332 |
| **RMSE OOF global** | **3.54836** | **3.35080** |

Le gain est positif sur les cinq folds : **5.57 % de réduction du RMSE**
global. Sur les folds 2 à 5 seuls, le RMSE passe de 3.54361 à environ 3.34977.
Un bootstrap apparié par patient, avec 2 000 rééchantillonnages sur ces
quatre folds, donne un intervalle à 95 % du gain absolu de RMSE de
**[0.137 ; 0.250]**. Cet intervalle est conditionnel au modèle retenu et ne
mesure pas l'incertitude liée à toute la recherche d'hyperparamètres.

Les contrôles d'inférence vérifient que le réordonnancement des lignes
et le traitement séparé de patients complets conservent les prédictions.

Le modèle final a été entraîné sur 44 590 visites. Les 11 013 prédictions
du CSV ont été contrôlées : format, indices uniques, ordre attendu,
valeurs finies et scores dans `[0, 132]`.

**Publication Skore confirmée :** [33_generative_progression](https://skore.probabl.ai/degensunite/bobathon-esilv/cross-validations/87766).

## Reproduction et artefacts

```sh
.venv/bin/python -u experiments/33_generative_progression.py
.venv/bin/python -u experiments/33_generative_progression.py --push
```

Le script conserve le rapport Skore, le modèle final, les prédictions OOF
et les métriques sous le préfixe `scratch/33_generative_progression`.
Le CSV final est `submissions/33_generative_progression.csv`.
Les modèles intermédiaires sont mis en cache selon les données du fold
et la graine. Après modification de l'algorithme, changer la version du cache
et déplacer les anciens artefacts avant de relancer.

Le protocole suppose que les visites disponibles d'un patient sont fournies
ensemble, y compris celles situées plus tard dans sa trajectoire. Il
correspond au challenge et ne mesure pas une prévision limitée au passé.
La publication demandée est sur Skore.
