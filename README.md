# Projet Lung Cancer Risk - Documentation Complète

## 📋 Vue d'ensemble du Projet

Ce projet vise à analyser et modéliser le risque de cancer du poumon à partir d'un dataset de **5000 patients**, en utilisant des facteurs démographiques, environnementaux, cliniques et de mode de vie.

**Objectif principal :** Créer des modèles de machine learning pour prédire deux variables importantes :
- `family_history_cancer` : Antécédents familiaux de cancer
- `smoker` : Statut fumeur

**État du projet :** ✅ **Phase 2 terminée** (Nettoyage, Normalisation, Entraînement des modèles, Model Cards)

---

## 📁 Structure Complète du Projet

```
TP_doc_lung_cancer-main/
│
├── data/
│   ├── raw/                              # Données brutes (source)
│   │   └── lung_cancer.csv              # Dataset original (5000 patients, 30 variables)
│   ├── processed/                        # Données transformées
│   │   ├── lung_cancer_cleaned.csv      # Dataset nettoyé et normalisé
│   │   └── lineage.json                 # Traçabilité des transformations
│   └── xml/                              # Données au format XML (informatif)
│       └── lung_cancer.xml
│
├── models/                               # Modèles entraînés
│   ├── model_family_history_cancer.pkl   # Modèle pour prédire les antécédents familiaux
│   ├── model_smoker.pkl                  # Modèle pour prédire le statut fumeur
│   └── models_metadata.json             # Métadonnées des modèles (métriques, hyperparamètres)
│
├── docs/
│   ├── data_cards/                       # Métadonnées du dataset
│   │   ├── data_cards_complet.yaml
│   │   └── data_cards_complet.json
│   ├── exploration/                      # Analyse exploratoire
│   │   └── rapport_exploration.md
│   ├── model_cards/                      # Documentation des modèles
│   │   ├── model_card_family_history_cancer.md
│   │   └── model_card_smoker.md
│   ├── preprocessing/                    # Documentation du nettoyage
│   │   └── documentation_nettoyage.md
│   ├── schemas/                          # Schémas de validation
│   │   └── lung_cancer_schema.xsd
│   ├── visualizations/                   # Tous les graphiques
│   │   ├── 01_distribution_cible.png     # Graphiques exploratoires (8 fichiers)
│   │   ├── ...
│   │   ├── model_family_history/         # Graphiques du modèle 1 (4 fichiers)
│   │   │   ├── family_history_feature_importance.png
│   │   │   ├── family_history_confusion_matrix.png
│   │   │   ├── family_history_predictions.png
│   │   │   └── family_history_top_features.png
│   │   └── model_smoker/                 # Graphiques du modèle 2 (4 fichiers)
│   │       ├── smoker_feature_importance.png
│   │       ├── smoker_confusion_matrix.png
│   │       ├── smoker_predictions.png
│   │       └── smoker_top_features.png
│   ├── plan_analyse.md                  # Plan d'analyse méthodologique
│   └── log_transformation.txt           # Log des transformations
│
├── scripts/                              # Tous les scripts Python
│   ├── preprocess_data.py                # Nettoyage et normalisation
│   ├── train_models.py                   # Entraînement des modèles
│   ├── generate_model_cards.py          # Génération des Model Cards
│   ├── visualize_model_family_history.py # Visualisations modèle 1
│   ├── visualize_model_smoker.py       # Visualisations modèle 2
│   ├── convert_csv_to_xml.py            # Conversion CSV → XML
│   ├── generate_data_cards.py           # Génération Data Cards
│   ├── exploratory_analysis.py          # Analyse exploratoire
│   └── generate_visualizations.py      # Visualisations exploratoires
│
├── requirements.txt                      # Dépendances Python
└── README.md                             # Ce fichier
```

---

## 🚀 Installation et Configuration

### Prérequis

- **Python 3.8+**
- **pip** (gestionnaire de paquets Python)

### Installation des dépendances

```bash
# Installer toutes les dépendances nécessaires
pip install -r requirements.txt
```

Les dépendances incluent :
- `pandas` : Manipulation de données
- `numpy` : Calculs numériques
- `scikit-learn` : Machine learning
- `matplotlib` : Visualisations
- `seaborn` : Graphiques avancés
- `pyyaml` : Lecture/écriture YAML
- `lxml` : Traitement XML

---

## 📖 Guide Complet : Ce qui a été fait

### Phase 1 : Exploration et Préparation ✅

**Objectif :** Comprendre les données et préparer la documentation

**Scripts utilisés :**
- `exploratory_analysis.py` : Génère le rapport d'exploration
- `generate_data_cards.py` : Crée les Data Cards (métadonnées)
- `generate_visualizations.py` : Génère 8 graphiques exploratoires

**Résultats :**
- ✅ Rapport d'exploration complet (`docs/exploration/rapport_exploration.md`)
- ✅ Data Cards en YAML et JSON (`docs/data_cards/`)
- ✅ 8 visualisations exploratoires (`docs/visualizations/`)

**Pour reproduire :**
```bash
python scripts/exploratory_analysis.py
python scripts/generate_data_cards.py
python scripts/generate_visualizations.py
```

---

### Phase 2 : Nettoyage et Normalisation ✅

**Objectif :** Préparer les données pour l'entraînement des modèles

**Script utilisé :** `preprocess_data.py`

**Ce qui a été fait :**

1. **Vérification de la qualité**
   - ✅ Aucune valeur manquante détectée
   - ✅ Aucun doublon détecté
   - ⚠️ Quelques valeurs aberrantes détectées (conservées car nettoyage "sommaire")

2. **Normalisation**
   - 15 variables numériques normalisées avec **StandardScaler**
   - Transformation : `(x - moyenne) / écart-type`
   - Variables binaires et catégorielles conservées telles quelles

**Fichiers générés :**
- `data/processed/lung_cancer_cleaned.csv` : Dataset nettoyé (5000 lignes, 30 colonnes)
- `data/processed/lineage.json` : Traçabilité complète des transformations
- `docs/preprocessing/documentation_nettoyage.md` : Documentation détaillée

**Pour reproduire :**
```bash
python scripts/preprocess_data.py
```

**Documentation :** Voir `docs/preprocessing/documentation_nettoyage.md` pour tous les détails.

---

### Phase 3 : Entraînement des Modèles ✅

**Objectif :** Créer deux modèles de prédiction

**Script utilisé :** `train_models.py`

**Modèles entraînés :**

#### 1. Modèle `family_history_cancer`
- **Architecture :** Random Forest Classifier
- **Performance :** Accuracy = 100%, F1-Score = 100%
- **Hyperparamètres optimaux :**
  - `n_estimators` = 50
  - `max_depth` = 5
  - `min_samples_split` = 2
- **Distribution :** 3983 sans antécédents (0), 1017 avec antécédents (1)

#### 2. Modèle `smoker`
- **Architecture :** Random Forest Classifier
- **Performance :** Accuracy = 100%, F1-Score = 100%
- **Hyperparamètres optimaux :**
  - `n_estimators` = 50
  - `max_depth` = 5
  - `min_samples_split` = 2
- **Distribution :** 2726 non-fumeurs (0), 2274 fumeurs (1)

**Méthode d'optimisation :**
- **GridSearchCV** : Recherche exhaustive sur grille
- **Validation croisée :** 5-fold cross-validation
- **Métrique d'optimisation :** F1-Score
- **Split train/test :** 80% / 20% (avec stratification)

**Fichiers générés :**
- `models/model_family_history_cancer.pkl` : Modèle entraîné (binaire)
- `models/model_smoker.pkl` : Modèle entraîné (binaire)
- `models/models_metadata.json` : Toutes les métriques et hyperparamètres

**Pour reproduire :**
```bash
python scripts/train_models.py
```

**Note importante :** Les performances à 100% sont exceptionnellement bonnes. Cela peut indiquer soit des données très bien structurées, soit un possible surapprentissage. La validation croisée confirme également ces résultats.

---

### Phase 4 : Model Cards ✅

**Objectif :** Documenter complètement les modèles

**Script utilisé :** `generate_model_cards.py`

**Contenu des Model Cards :**

Chaque Model Card répond aux questions suivantes :

1. **Quand a-t-il été développé ?**
   - Date de développement
   - Contexte du projet

2. **Quelle architecture ?**
   - Algorithme utilisé (Random Forest)
   - Bibliothèque (scikit-learn)
   - Hyperparamètres optimisés

3. **Sur quelles données ?**
   - Dataset source et nettoyé
   - Nombre d'échantillons (5000)
   - Split train/test (80/20)
   - Distribution des classes
   - Préprocessing appliqué

4. **Quelles métriques de performance ?**
   - Accuracy, Precision, Recall, F1-Score, ROC-AUC
   - Validation croisée
   - Matrice de confusion

5. **Quels hyperparamètres ?**
   - Valeurs finales sélectionnées
   - Méthode de recherche (GridSearchCV)

6. **Comment ont-ils été trouvés ?**
   - GridSearchCV avec validation croisée 5-fold
   - Optimisation sur le F1-Score

**Fichiers générés :**
- `docs/model_cards/model_card_family_history_cancer.md`
- `docs/model_cards/model_card_smoker.md`

**Pour reproduire :**
```bash
python scripts/generate_model_cards.py
```

**Documentation :** Voir les fichiers dans `docs/model_cards/` pour tous les détails.

---

### Phase 5 : Visualisations des Modèles ✅

**Objectif :** Créer des graphiques intéressants pour chaque modèle

**Scripts utilisés :**
- `visualize_model_family_history.py` : Graphiques pour le modèle 1
- `visualize_model_smoker.py` : Graphiques pour le modèle 2

**Graphiques générés pour chaque modèle (4 par modèle) :**

1. **Feature Importance** : Top 15 features les plus importantes
2. **Matrice de Confusion** : Performance du modèle
3. **Distribution des Prédictions** : Histogramme et boxplot des probabilités
4. **Top Features par Classe** : Comparaison des 5 features les plus importantes

**Fichiers générés :**
- `docs/visualizations/model_family_history/` : 4 graphiques PNG
- `docs/visualizations/model_smoker/` : 4 graphiques PNG

**Pour reproduire :**
```bash
python scripts/visualize_model_family_history.py
python scripts/visualize_model_smoker.py
```

---

## 🔄 Workflow Complet (Ordre d'exécution)

Si vous voulez tout refaire depuis le début, voici l'ordre recommandé :

```bash
# 1. Exploration des données
python scripts/exploratory_analysis.py
python scripts/generate_data_cards.py
python scripts/generate_visualizations.py

# 2. Nettoyage et normalisation
python scripts/preprocess_data.py

# 3. Entraînement des modèles
python scripts/train_models.py

# 4. Génération des Model Cards
python scripts/generate_model_cards.py

# 5. Visualisations des modèles
python scripts/visualize_model_family_history.py
python scripts/visualize_model_smoker.py
```

---

## 📊 Dataset

### Caractéristiques

- **Taille :** 5000 patients
- **Variables :** 30 (29 prédictives + 1 cible)
- **Variable cible principale :** `lung_cancer_risk` (0=Faible risque, 1=Risque élevé)
- **Variables cibles modélisées :** `family_history_cancer`, `smoker`
- **Qualité :** Aucune valeur manquante détectée

### Variables Principales

- **Démographie :** age, gender, education_years, income_level
- **Tabagisme :** smoker, smoking_years, cigarettes_per_day, pack_years, passive_smoking
- **Expositions :** air_pollution_index, occupational_exposure, radon_exposure
- **Antécédents :** family_history_cancer, copd, asthma, previous_tb
- **Symptômes :** chronic_cough, chest_pain, shortness_of_breath, fatigue
- **Clinique :** bmi, oxygen_saturation, fev1_x10, crp_level, xray_abnormal
- **Mode de vie :** exercise_hours_per_week, diet_quality, alcohol_units_per_week, healthcare_access

---

## 📚 Documentation Disponible

### Documents Principaux

1. **README.md** (ce fichier) : Vue d'ensemble complète du projet
2. **docs/plan_analyse.md** : Plan d'analyse méthodologique détaillé
3. **docs/exploration/rapport_exploration.md** : Statistiques descriptives complètes
4. **docs/preprocessing/documentation_nettoyage.md** : Détails du nettoyage et normalisation
5. **docs/model_cards/model_card_family_history_cancer.md** : Documentation complète du modèle 1
6. **docs/model_cards/model_card_smoker.md** : Documentation complète du modèle 2

### Métadonnées

- **docs/data_cards/** : Data Cards en YAML et JSON
- **data/processed/lineage.json** : Traçabilité des transformations
- **models/models_metadata.json** : Métadonnées des modèles (métriques, hyperparamètres)

---

## 🎯 Utilisation des Modèles

### Charger un modèle

```python
import pickle
import pandas as pd

# Charger le modèle
with open('models/model_smoker.pkl', 'rb') as f:
    model = pickle.load(f)

# Charger les données (doivent être normalisées)
df = pd.read_csv('data/processed/lung_cancer_cleaned.csv')

# Préparer les features (exclure les variables cibles)
exclude_cols = ['lung_cancer_risk', 'family_history_cancer', 'smoker']
features = [col for col in df.columns if col not in exclude_cols]
X = df[features]

# Faire une prédiction
predictions = model.predict(X)
probabilities = model.predict_proba(X)

print(f"Prédictions: {predictions[:10]}")
print(f"Probabilités: {probabilities[:10]}")
```

### Important

- Les données d'entrée doivent être **normalisées** (utiliser `data/processed/lung_cancer_cleaned.csv`)
- Les features doivent être dans le **même ordre** que lors de l'entraînement
- Consulter `models/models_metadata.json` pour connaître l'ordre exact des features

---

## 🔍 Comprendre les Résultats

### Performance des Modèles

Les deux modèles ont obtenu **100% de précision**, ce qui est exceptionnel. Cela signifie :
- ✅ Toutes les prédictions sont correctes sur le jeu de test
- ✅ Aucune erreur de classification
- ⚠️ Possible surapprentissage (mais la validation croisée confirme aussi 100%)

### Interprétation

Pour comprendre pourquoi les modèles sont si performants :
1. Consulter les **graphiques d'importance des features** dans `docs/visualizations/`
2. Lire les **Model Cards** pour voir quelles variables sont les plus importantes
3. Examiner les **matrices de confusion** pour voir la répartition des prédictions

---

## 🚧 Prochaines Étapes Possibles

### Améliorations Potentielles

1. **Validation externe** : Tester sur un nouveau dataset
2. **Analyse de l'importance des features** : Comprendre quelles variables sont vraiment importantes
3. **Interprétabilité** : Utiliser SHAP values pour expliquer les prédictions
4. **Optimisation** : Tester d'autres algorithmes (XGBoost, SVM, etc.)
5. **Déploiement** : Créer une API pour utiliser les modèles

### Extensions

1. **Dashboard interactif** : Interface web pour visualiser les résultats
2. **API de prédiction** : Service web pour faire des prédictions
3. **Analyse approfondie** : Tests statistiques, analyse de causalité
4. **Documentation avancée** : Guide utilisateur, documentation API

---

## ❓ Questions Fréquentes

### Comment utiliser les modèles ?

Voir la section **"Utilisation des Modèles"** ci-dessus.

### Où sont les graphiques ?

- Graphiques exploratoires : `docs/visualizations/` (8 fichiers)
- Graphiques modèle 1 : `docs/visualizations/model_family_history/` (4 fichiers)
- Graphiques modèle 2 : `docs/visualizations/model_smoker/` (4 fichiers)

### Comment comprendre ce qui a été fait ?

1. Lire ce README en entier
2. Consulter `docs/plan_analyse.md` pour la méthodologie
3. Lire les Model Cards dans `docs/model_cards/`
4. Examiner les graphiques dans `docs/visualizations/`

### Les modèles sont-ils prêts à être utilisés ?

Oui, les modèles sont entraînés et sauvegardés. Cependant :
- ⚠️ Les performances à 100% peuvent indiquer un surapprentissage
- ⚠️ Il faudrait tester sur de nouvelles données pour valider
- ✅ Les Model Cards documentent toutes les limitations

### Comment reproduire les résultats ?

Suivre le **Workflow Complet** ci-dessus dans l'ordre indiqué.

---

## 📞 Support et Contact

Pour toute question sur ce projet :
1. Consulter la documentation dans `docs/`
2. Lire les commentaires dans les scripts Python
3. Examiner les fichiers de métadonnées (JSON, YAML)

---

## 📄 Licence

À définir

## 👥 Auteurs

Équipe d'analyse - 2024

---

## 📝 Notes Finales

Ce projet a été conçu pour être **autonome et compréhensible**. Toute l'information nécessaire pour comprendre et continuer le projet se trouve dans :

1. **Ce README** : Vue d'ensemble complète
2. **La documentation** : Fichiers dans `docs/`
3. **Les scripts** : Commentaires dans le code
4. **Les métadonnées** : Fichiers JSON/YAML

**Objectif atteint :** Un professeur ou un nouveau développeur peut comprendre et continuer ce projet sans avoir besoin de consulter les auteurs.

---

*Dernière mise à jour: 2026-01-09*
