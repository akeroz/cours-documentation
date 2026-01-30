# Model Card: Prédiction de `smoker`

**Date de développement:** 2026-01-09  
**Version:** 1.0  
**Variable cible:** `smoker`

---

## 1. Informations générales

### Quand a-t-il été développé ?

- **Date de développement:** 2026-01-09
- **Contexte:** Projet d'analyse du risque de cancer du poumon
- **Phase:** Modélisation prédictive (Phase 2)

---

## 2. Architecture du modèle

### Quelle architecture ?

Random Forest Classifier (scikit-learn)


- **Algorithme:** Random Forest (Ensemble de Decision Trees)
- **Bibliothèque:** scikit-learn
- **Hyperparamètres optimisés:**
  - n_estimators: 50
  - max_depth: 5
  - min_samples_split: 2


---

## 3. Données d'entraînement

### Sur quelles données ?


- **Dataset source:** `data/raw/lung_cancer.csv`
- **Dataset nettoyé:** `data/processed/lung_cancer_cleaned.csv`
- **Nombre d'échantillons d'entraînement:** 5000
- **Split train/test:** 80% / 20%
- **Stratification:** Oui (pour maintenir la distribution des classes)
- **Nombre de features:** 28
- **Distribution de la variable cible:**
  - Classe 0: 2726 échantillons
  - Classe 1: 2274 échantillons

- **Préprocessing appliqué:**
  - Normalisation: StandardScaler (variables numériques)
  - Nettoyage: Suppression des doublons, vérification des valeurs manquantes
  - Date de preprocessing: 2026-01-09


### Features utilisées

Le modèle utilise 28 features prédictives, incluant:
- Variables démographiques (âge, genre, éducation, revenu)
- Variables de tabagisme (années de tabagisme, cigarettes/jour, etc.)
- Expositions environnementales (pollution, radon, exposition professionnelle)
- Antécédents médicaux (BPCO, asthme, tuberculose)
- Symptômes (toux chronique, douleur thoracique, essoufflement, fatigue)
- Paramètres cliniques (BMI, saturation O2, FEV1, CRP, radiographie)
- Mode de vie (exercice, alimentation, alcool, accès aux soins)

**Note:** La variable `smoker` a été exclue des features lors de l'entraînement.

---

## 4. Métriques de performance

### Quelles métriques de performance ?


### Métriques sur le jeu de test

| Métrique | Valeur |
|----------|--------|
| **Accuracy** | 1.0000 |
| **Precision** | 1.0000 |
| **Recall** | 1.0000 |
| **F1-Score** | 1.0000 |
| **ROC-AUC** | 1.0000 |

### Validation croisée (5-fold)

- **F1-Score moyen:** 1.0000
- **Écart-type:** 0.0000

### Matrice de confusion

```
                Prédit 0    Prédit 1
Réel 0               545           0
Réel 1                 0         455
```


---

## 5. Hyperparamètres

### Quels hyperparamètres ?


### Hyperparamètres finaux

Les hyperparamètres suivants ont été sélectionnés via une recherche par grille (GridSearchCV):

- **max_depth:** 5
- **min_samples_split:** 2
- **n_estimators:** 50

### Méthode de recherche

- **Méthode:** GridSearchCV (Recherche exhaustive sur grille)
- **Validation croisée:** 5-fold cross-validation
- **Métrique d'optimisation:** F1-Score
- **Espace de recherche:** 3 hyperparamètres testés


---

## 6. Limitations et considérations

### Limitations

- Le modèle a été entraîné sur un dataset spécifique (5000 patients)
- Les performances peuvent varier sur d'autres populations
- Le modèle ne capture que les relations présentes dans les données d'entraînement
- Pas de garantie de causalité (modèle prédictif, non explicatif)

### Considérations éthiques

- Les prédictions ne doivent pas être utilisées comme seul critère de décision médicale
- Biais potentiels liés aux données d'entraînement
- Confidentialité des données patients

---

## 7. Utilisation

### Chargement du modèle

```python
import pickle

with open('models/model_smoker.pkl', 'rb') as f:
    model = pickle.load(f)

# Prédiction
prediction = model.predict(X_new)
prediction_proba = model.predict_proba(X_new)
```

### Prérequis

- Les données d'entrée doivent être normalisées (StandardScaler)
- Format: DataFrame pandas avec les mêmes features que lors de l'entraînement
- Variables numériques normalisées, variables binaires/catégorielles non modifiées

---

## 8. Références

- **Bibliothèque:** scikit-learn
- **Version Python recommandée:** 3.8+
- **Documentation:** https://scikit-learn.org/

---

*Model Card générée automatiquement le 2026-01-09 11:39:14*
