# Rapport d'Exploration - Dataset Lung Cancer Risk
**Date de génération:** 2026-01-09 10:50:47
**Source:** C:\Users\Martin\Documents\travail\master\doc\data\raw\lung_cancer.csv

---

## 1. Vue d'ensemble du Dataset

- **Nombre total d'échantillons:** 5,000
- **Nombre de variables:** 30
- **Variables prédictives:** 29
- **Variable cible:** `lung_cancer_risk`

## 2. Structure des Données

### 2.1 Liste des Variables

| Variable | Type | Description |
|----------|------|-------------|
| `age` | Numérique | Age du patient en années |
| `gender` | Catégorielle | Genre (0=Femme, 1=Homme) |
| `education_years` | Numérique | Nombre d'années d'éducation |
| `income_level` | Catégorielle | Niveau de revenu (1-4) |
| `smoker` | Catégorielle | Statut fumeur (0=Non, 1=Oui) |
| `smoking_years` | Numérique | Années de tabagisme |
| `cigarettes_per_day` | Numérique | Cigarettes par jour |
| `pack_years` | Numérique | Paquets-années |
| `passive_smoking` | Catégorielle | Tabagisme passif (0=Non, 1=Oui) |
| `air_pollution_index` | Numérique | Indice de pollution de l'air |
| `occupational_exposure` | Catégorielle | Exposition professionnelle (0=Non, 1=Oui) |
| `radon_exposure` | Catégorielle | Exposition au radon (0=Non, 1=Oui) |
| `family_history_cancer` | Catégorielle | Antécédents familiaux (0=Non, 1=Oui) |
| `copd` | Catégorielle | BPCO (0=Non, 1=Oui) |
| `asthma` | Catégorielle | Asthme (0=Non, 1=Oui) |
| `previous_tb` | Catégorielle | Antécédents tuberculose (0=Non, 1=Oui) |
| `chronic_cough` | Catégorielle | Toux chronique (0=Non, 1=Oui) |
| `chest_pain` | Catégorielle | Douleur thoracique (0=Non, 1=Oui) |
| `shortness_of_breath` | Catégorielle | Essoufflement (0=Non, 1=Oui) |
| `fatigue` | Catégorielle | Fatigue (0=Non, 1=Oui) |
| `bmi` | Numérique | Indice de masse corporelle |
| `oxygen_saturation` | Numérique | Saturation en oxygène (%) |
| `fev1_x10` | Numérique | FEV1 multiplié par 10 |
| `crp_level` | Numérique | Niveau de protéine C-réactive |
| `xray_abnormal` | Catégorielle | Radiographie anormale (0=Normal, 1=Anormal) |
| `exercise_hours_per_week` | Numérique | Heures d'exercice par semaine |
| `diet_quality` | Catégorielle | Qualité de l'alimentation (1-5) |
| `alcohol_units_per_week` | Numérique | Unités d'alcool par semaine |
| `healthcare_access` | Catégorielle | Accès aux soins (1-4) |
| `lung_cancer_risk` | Catégorielle | Risque de cancer (0=Faible, 1=Élevé) - VARIABLE CIBLE |

## 3. Statistiques Descriptives

### 3.1 Variables Numériques

| Variable | Moyenne | Médiane | Écart-type | Min | Max |
|----------|---------|---------|------------|-----|-----|
| `age` | 54.57 | 55.00 | 11.93 | 18.00 | 90.00 |
| `gender` | 0.49 | 0.00 | 0.50 | 0.00 | 1.00 |
| `education_years` | 11.51 | 11.00 | 2.95 | 5.00 | 20.00 |
| `income_level` | 2.55 | 3.00 | 0.98 | 1.00 | 5.00 |
| `smoker` | 0.45 | 0.00 | 0.50 | 0.00 | 1.00 |
| `smoking_years` | 8.82 | 0.00 | 11.65 | 0.00 | 52.00 |
| `cigarettes_per_day` | 6.69 | 0.00 | 9.03 | 0.00 | 44.00 |
| `pack_years` | 6.25 | 0.00 | 9.96 | 0.00 | 60.00 |
| `passive_smoking` | 0.35 | 0.00 | 0.48 | 0.00 | 1.00 |
| `air_pollution_index` | 64.28 | 64.00 | 19.48 | 20.00 | 130.00 |
| `occupational_exposure` | 0.25 | 0.00 | 0.43 | 0.00 | 1.00 |
| `radon_exposure` | 0.14 | 0.00 | 0.35 | 0.00 | 1.00 |
| `family_history_cancer` | 0.20 | 0.00 | 0.40 | 0.00 | 1.00 |
| `copd` | 0.16 | 0.00 | 0.37 | 0.00 | 1.00 |
| `asthma` | 0.15 | 0.00 | 0.35 | 0.00 | 1.00 |
| `previous_tb` | 0.09 | 0.00 | 0.29 | 0.00 | 1.00 |
| `chronic_cough` | 0.17 | 0.00 | 0.38 | 0.00 | 1.00 |
| `chest_pain` | 0.25 | 0.00 | 0.43 | 0.00 | 1.00 |
| `shortness_of_breath` | 0.20 | 0.00 | 0.40 | 0.00 | 1.00 |
| `fatigue` | 0.40 | 0.00 | 0.49 | 0.00 | 1.00 |
| `bmi` | 23.60 | 24.00 | 3.94 | 16.00 | 37.00 |
| `oxygen_saturation` | 96.09 | 97.00 | 3.41 | 85.00 | 100.00 |
| `fev1_x10` | 31.48 | 33.00 | 5.21 | 5.00 | 37.00 |
| `crp_level` | 4.59 | 3.00 | 5.33 | 0.00 | 33.00 |
| `xray_abnormal` | 0.20 | 0.00 | 0.40 | 0.00 | 1.00 |
| `exercise_hours_per_week` | 2.58 | 2.00 | 1.83 | 0.00 | 10.00 |
| `diet_quality` | 2.53 | 3.00 | 0.99 | 1.00 | 5.00 |
| `alcohol_units_per_week` | 5.88 | 6.00 | 4.42 | 0.00 | 23.00 |
| `healthcare_access` | 2.54 | 3.00 | 0.98 | 1.00 | 5.00 |
| `lung_cancer_risk` | 0.25 | 0.00 | 0.43 | 0.00 | 1.00 |

## 4. Distribution de la Variable Cible

- **Classe 0 (Faible risque):** 3,756 (75.12%)
- **Classe 1 (Risque élevé):** 1,244 (24.88%)
- **Ratio d'équilibre:** 3.02:1

⚠️ **Attention:** Le dataset présente un déséquilibre de classes.

## 5. Analyse des Valeurs Manquantes

| Variable | Valeurs manquantes | Pourcentage |
|----------|-------------------|-------------|
| *Aucune* | 0 | 0.00% |

## 6. Analyse des Duplicatas

- **Nombre de lignes dupliquées:** 0
- **Pourcentage:** 0.00%

## 7. Corrélations avec la Variable Cible

| Variable | Corrélation |
|----------|-------------|
| `pack_years` | 0.8307 |
| `crp_level` | 0.7804 |
| `cigarettes_per_day` | 0.7510 |
| `xray_abnormal` | 0.7480 |
| `smoking_years` | 0.7216 |
| `shortness_of_breath` | 0.6590 |
| `smoker` | 0.6301 |
| `chronic_cough` | 0.6296 |
| `copd` | 0.6164 |
| `age` | 0.0835 |
| `family_history_cancer` | 0.0632 |
| `air_pollution_index` | 0.0501 |
| `radon_exposure` | 0.0472 |
| `passive_smoking` | 0.0326 |
| `gender` | 0.0247 |
| `alcohol_units_per_week` | 0.0141 |
| `occupational_exposure` | 0.0117 |
| `chest_pain` | 0.0051 |
| `asthma` | 0.0021 |
| `previous_tb` | -0.0027 |
| `income_level` | -0.0036 |
| `fatigue` | -0.0046 |
| `diet_quality` | -0.0072 |
| `education_years` | -0.0125 |
| `healthcare_access` | -0.0196 |
| `bmi` | -0.0239 |
| `exercise_hours_per_week` | -0.0439 |
| `oxygen_saturation` | -0.7663 |
| `fev1_x10` | -0.7870 |

## 8. Analyse par Groupes

### 8.1 Distribution par Genre

| Genre | Total | Risque élevé | Taux de risque |
|-------|------|--------------|----------------|
| Femme (0) | 2,559 | 610 | 23.84% |
| Homme (1) | 2,441 | 634 | 25.97% |

### 8.2 Distribution par Statut Fumeur

| Statut | Total | Risque élevé | Taux de risque |
|--------|------|--------------|----------------|
| Non-fumeur (0) | 2,726 | 0 | 0.00% |
| Fumeur (1) | 2,274 | 1,244 | 54.71% |

## 9. Observations et Recommandations

### 9.1 Points Clés

- Dataset bien structuré avec 30 variables
- Aucune valeur manquante détectée
- Distribution de la variable cible: 3.02:1

### 9.2 Variables les Plus Corrélées avec le Risque

- `pack_years`: corrélation = 0.8307
- `fev1_x10`: corrélation = -0.7870
- `crp_level`: corrélation = 0.7804
- `oxygen_saturation`: corrélation = -0.7663
- `cigarettes_per_day`: corrélation = 0.7510

### 9.3 Recommandations pour la Modélisation

1. **Préprocessing:**
   - Normaliser/standardiser les variables numériques
   - Encoder les variables catégorielles si nécessaire
   - Gérer le déséquilibre de classes si présent

2. **Feature Engineering:**
   - Créer des interactions entre variables importantes
   - Considérer des transformations non-linéaires

3. **Modélisation:**
   - Tester plusieurs algorithmes (Random Forest, XGBoost, Logistic Regression)
   - Utiliser la validation croisée
   - Évaluer avec plusieurs métriques (accuracy, precision, recall, F1, AUC-ROC)

## 10. Prochaines Étapes

1. Analyse statistique approfondie (tests d'hypothèses)
2. Visualisations avancées (distributions, corrélations)
3. Feature selection
4. Développement de modèles prédictifs
5. Validation et interprétation des résultats

---

*Rapport généré automatiquement le 2026-01-09 10:50:47*
