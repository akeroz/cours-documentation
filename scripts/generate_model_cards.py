"""
Script pour générer les Model Cards pour les modèles entraînés
"""

import json
import pickle
from pathlib import Path
from datetime import datetime
import pandas as pd


def load_model_metadata(models_dir):
    """Charge les métadonnées des modèles"""
    metadata_path = models_dir / "models_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Métadonnées introuvables: {metadata_path}")
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_lineage(base_dir):
    """Charge le lineage des transformations"""
    lineage_path = base_dir / "data" / "processed" / "lineage.json"
    if lineage_path.exists():
        with open(lineage_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def generate_model_card(target_var, model_info, lineage, base_dir):
    """
    Génère une Model Card pour un modèle donné
    """
    # Informations sur le modèle
    model_type = model_info['model_type']
    best_params = model_info['best_params']
    metrics = model_info['metrics']
    
    # Date de développement
    development_date = datetime.now().strftime('%Y-%m-%d')
    
    # Architecture
    if model_type == 'random_forest':
        architecture = f"Random Forest Classifier (scikit-learn)"
        architecture_details = f"""
- **Algorithme:** Random Forest (Ensemble de Decision Trees)
- **Bibliothèque:** scikit-learn
- **Hyperparamètres optimisés:**
  - n_estimators: {best_params.get('n_estimators', 'N/A')}
  - max_depth: {best_params.get('max_depth', 'N/A')}
  - min_samples_split: {best_params.get('min_samples_split', 'N/A')}
"""
    elif model_type == 'logistic_regression':
        architecture = f"Logistic Regression (scikit-learn)"
        architecture_details = f"""
- **Algorithme:** Régression Logistique
- **Bibliothèque:** scikit-learn
- **Hyperparamètres optimisés:**
  - C: {best_params.get('C', 'N/A')}
  - penalty: {best_params.get('penalty', 'N/A')}
  - solver: {best_params.get('solver', 'N/A')}
"""
    else:
        architecture = f"{model_type} (scikit-learn)"
        architecture_details = f"- **Algorithme:** {model_type}\n- **Bibliothèque:** scikit-learn"
    
    # Données d'entraînement
    n_samples = model_info['n_samples']
    n_features = model_info['n_features']
    target_dist = model_info['target_distribution']
    
    # Informations sur le dataset
    data_info = f"""
- **Dataset source:** `data/raw/lung_cancer.csv`
- **Dataset nettoyé:** `data/processed/lung_cancer_cleaned.csv`
- **Nombre d'échantillons d'entraînement:** {n_samples}
- **Split train/test:** 80% / 20%
- **Stratification:** Oui (pour maintenir la distribution des classes)
- **Nombre de features:** {n_features}
- **Distribution de la variable cible:**
  - Classe 0: {target_dist.get('0', target_dist.get(0, 'N/A'))} échantillons
  - Classe 1: {target_dist.get('1', target_dist.get(1, 'N/A'))} échantillons
"""
    
    if lineage:
        data_info += f"""
- **Préprocessing appliqué:**
  - Normalisation: StandardScaler (variables numériques)
  - Nettoyage: Suppression des doublons, vérification des valeurs manquantes
  - Date de preprocessing: {lineage.get('date_creation', 'N/A')[:10]}
"""
    
    # Métriques de performance
    metrics_section = f"""
### Métriques sur le jeu de test

| Métrique | Valeur |
|----------|--------|
| **Accuracy** | {metrics['accuracy']:.4f} |
| **Precision** | {metrics['precision']:.4f} |
| **Recall** | {metrics['recall']:.4f} |
| **F1-Score** | {metrics['f1_score']:.4f} |
| **ROC-AUC** | {metrics['roc_auc']:.4f} |

### Validation croisée (5-fold)

- **F1-Score moyen:** {metrics['cv_f1_mean']:.4f}
- **Écart-type:** {metrics['cv_f1_std']:.4f}

### Matrice de confusion

```
                Prédit 0    Prédit 1
Réel 0          {metrics['confusion_matrix'][0][0]:>8}    {metrics['confusion_matrix'][0][1]:>8}
Réel 1          {metrics['confusion_matrix'][1][0]:>8}    {metrics['confusion_matrix'][1][1]:>8}
```
"""
    
    # Hyperparamètres
    hyperparams_section = f"""
### Hyperparamètres finaux

Les hyperparamètres suivants ont été sélectionnés via une recherche par grille (GridSearchCV):

"""
    for param, value in best_params.items():
        hyperparams_section += f"- **{param}:** {value}\n"
    
    hyperparams_section += f"""
### Méthode de recherche

- **Méthode:** GridSearchCV (Recherche exhaustive sur grille)
- **Validation croisée:** 5-fold cross-validation
- **Métrique d'optimisation:** F1-Score
- **Espace de recherche:** {len(best_params)} hyperparamètres testés
"""
    
    # Générer le contenu de la Model Card
    model_card = f"""# Model Card: Prédiction de `{target_var}`

**Date de développement:** {development_date}  
**Version:** 1.0  
**Variable cible:** `{target_var}`

---

## 1. Informations générales

### Quand a-t-il été développé ?

- **Date de développement:** {development_date}
- **Contexte:** Projet d'analyse du risque de cancer du poumon
- **Phase:** Modélisation prédictive (Phase 2)

---

## 2. Architecture du modèle

### Quelle architecture ?

{architecture}

{architecture_details}

---

## 3. Données d'entraînement

### Sur quelles données ?

{data_info}

### Features utilisées

Le modèle utilise {n_features} features prédictives, incluant:
- Variables démographiques (âge, genre, éducation, revenu)
- Variables de tabagisme (années de tabagisme, cigarettes/jour, etc.)
- Expositions environnementales (pollution, radon, exposition professionnelle)
- Antécédents médicaux (BPCO, asthme, tuberculose)
- Symptômes (toux chronique, douleur thoracique, essoufflement, fatigue)
- Paramètres cliniques (BMI, saturation O2, FEV1, CRP, radiographie)
- Mode de vie (exercice, alimentation, alcool, accès aux soins)

**Note:** La variable `{target_var}` a été exclue des features lors de l'entraînement.

---

## 4. Métriques de performance

### Quelles métriques de performance ?

{metrics_section}

---

## 5. Hyperparamètres

### Quels hyperparamètres ?

{hyperparams_section}

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

with open('models/model_{target_var}.pkl', 'rb') as f:
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

*Model Card générée automatiquement le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    return model_card


def generate_all_model_cards():
    """
    Génère les Model Cards pour tous les modèles entraînés
    """
    print("=" * 80)
    print("GÉNÉRATION DES MODEL CARDS")
    print("=" * 80)
    
    base_dir = Path(__file__).parent.parent
    models_dir = base_dir / "models"
    output_dir = base_dir / "docs" / "model_cards"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Charger les métadonnées
    print("\n[1/3] Chargement des métadonnées des modèles...")
    metadata = load_model_metadata(models_dir)
    print(f"    [OK] {len(metadata)} modele(s) trouve(s)")
    
    # Charger le lineage
    print("\n[2/3] Chargement du lineage...")
    lineage = load_lineage(base_dir)
    if lineage:
        print("    [OK] Lineage charge")
    else:
        print("    [ATTENTION] Lineage non trouve (continuation sans)")
    
    # Générer les Model Cards
    print("\n[3/3] Génération des Model Cards...")
    for target_var, model_info in metadata.items():
        print(f"\n    Génération de la Model Card pour '{target_var}'...")
        
        model_card = generate_model_card(target_var, model_info, lineage, base_dir)
        
        # Sauvegarder
        output_path = output_dir / f"model_card_{target_var}.md"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(model_card)
        
        print(f"    [OK] Model Card sauvegardee: {output_path}")
    
    print("\n" + "=" * 80)
    print("[OK] GENERATION DES MODEL CARDS TERMINEE")
    print("=" * 80)
    
    return len(metadata)


def main():
    """Fonction principale"""
    try:
        n_cards = generate_all_model_cards()
        print(f"\n[OK] Succes! {n_cards} Model Card(s) generee(s).")
        return n_cards
    except Exception as e:
        print(f"\n[ERREUR] Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
