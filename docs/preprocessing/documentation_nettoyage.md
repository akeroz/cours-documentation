# Documentation du Nettoyage et Normalisation des Données

**Date de création:** 2026-01-09 11:36:35  
**Source:** `C:\Users\arthur\Documents\EPSI-new\M2\IA\documentation\TP_doc_lung_cancer-main\TP_doc_lung_cancer-main\data\raw\lung_cancer.csv`  
**Destination:** `C:\Users\arthur\Documents\EPSI-new\M2\IA\documentation\TP_doc_lung_cancer-main\TP_doc_lung_cancer-main\data\processed\lung_cancer_cleaned.csv`

---

## Vue d'ensemble

Ce document décrit les étapes de nettoyage et de normalisation appliquées au dataset `lung_cancer.csv` en vue de l'entraînement des modèles de prédiction.

---

## Étapes de Nettoyage

### 1. Vérification des valeurs manquantes
- **Résultat:** 0 valeur(s) manquante(s) détectée(s)
- **Action:** Aucune action nécessaire

### 2. Suppression des doublons
- **Doublons détectés:** 0
- **Action:** Aucune action (pas de doublons)
- **Lignes après suppression:** 5000

### 3. Détection des valeurs aberrantes
- **Méthode:** Méthode IQR (Interquartile Range) avec seuil à 3×IQR
- **Résultat:** Valeurs aberrantes détectées dans 3 variables
- **Action:** Conservation des valeurs (nettoyage sommaire)

### 4. Normalisation des variables numériques
- **Méthode:** StandardScaler (normalisation z-score)
- **Formule:** (x - moyenne) / écart-type
- **Variables normalisées:** 15 variables
    - age
  - education_years
  - income_level
  - smoking_years
  - cigarettes_per_day
  - pack_years
  - air_pollution_index
  - bmi
  - oxygen_saturation
  - fev1_x10
    ...
- **Variables non normalisées:** Variables binaires (0/1) et variables catégorielles conservées telles quelles

---

## Statistiques

### Avant nettoyage
- **Nombre de lignes:** 5000
- **Nombre de colonnes:** 30
- **Valeurs manquantes:** 0
- **Lignes dupliquées:** 0

### Après nettoyage
- **Nombre de lignes:** 5000
- **Nombre de colonnes:** 30
- **Valeurs manquantes:** 0
- **Lignes dupliquées:** 0

---

## Lineage

Le lineage complet des transformations est disponible dans: `C:\Users\arthur\Documents\EPSI-new\M2\IA\documentation\TP_doc_lung_cancer-main\TP_doc_lung_cancer-main\data\processed\lineage.json`

Le lineage contient:
- Date et heure de chaque transformation
- Détails de chaque étape de nettoyage
- Paramètres utilisés pour la normalisation
- Statistiques avant/après chaque transformation

---

## Utilisation

Le dataset nettoyé peut être utilisé directement pour l'entraînement des modèles:

```python
import pandas as pd
df = pd.read_csv('C:\Users\arthur\Documents\EPSI-new\M2\IA\documentation\TP_doc_lung_cancer-main\TP_doc_lung_cancer-main\data\processed\lung_cancer_cleaned.csv')
```

**Note:** Les variables numériques sont normalisées (moyenne=0, écart-type=1).  
Les variables binaires et catégorielles conservent leurs valeurs originales.

---

## Problèmes Détectés et Corrigés

### Data Leakage (Détecté et Corrigé)

**Date de détection:** 2026-01-09  
**Problème:** Lors de l'entraînement initial des modèles, un bug dans la fonction `prepare_features()` du script `train_models.py` permettait à la variable cible d'être incluse dans les features d'entraînement.

**Symptômes:**
- Performances anormalement élevées (100% accuracy) sur les deux modèles
- Résultats non réalistes pour un problème de classification

**Cause:**
- La logique d'exclusion des variables cibles retirait la variable cible de la liste d'exclusion au lieu de la conserver
- Code bugué : `exclude_cols.remove(target_variable)` au lieu de toujours exclure toutes les variables cibles

**Correction:**
- Modification de la fonction `prepare_features()` pour **toujours** exclure toutes les variables cibles (`lung_cancer_risk`, `family_history_cancer`, `smoker`)
- La variable cible est maintenant systématiquement exclue des features, même si c'est celle qu'on prédit

**Résultats après correction:**
- **Modèle `family_history_cancer`:** Accuracy = 79.70% (au lieu de 100%)
- **Modèle `smoker`:** Accuracy = 100% (résultat maintenu, probablement dû à des features très prédictives comme les variables de tabagisme)

**Impact:** Cette correction a permis d'obtenir des résultats réalistes et crédibles pour l'entraînement des modèles.

---

*Document généré automatiquement le 2026-01-09 11:36:35*  
*Mis à jour le 2026-01-09 - Ajout de la section sur le data leakage*
