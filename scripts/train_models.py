"""
Script d'entraînement des modèles pour prédire family_history_cancer et smoker
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')


def prepare_features(df, target_variable):
    """
    Prépare les features en excluant la variable cible et les autres variables cibles
    """
    # Exclure TOUJOURS toutes les variables cibles possibles
    # On ne doit JAMAIS utiliser la variable qu'on essaie de prédire comme feature !
    exclude_cols = ['lung_cancer_risk', 'family_history_cancer', 'smoker']
    
    # Features = toutes les colonnes sauf celles à exclure
    # La variable cible est TOUJOURS exclue
    features = [col for col in df.columns if col not in exclude_cols]
    
    X = df[features]
    y = df[target_variable]
    
    return X, y, features


def train_model(X, y, model_name='random_forest'):
    """
    Entraîne un modèle avec recherche d'hyperparamètres
    """
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Choisir le modèle de base
    if model_name == 'random_forest':
        base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10]
        }
    elif model_name == 'logistic_regression':
        base_model = LogisticRegression(random_state=42, max_iter=1000)
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
    else:
        raise ValueError(f"Modèle non reconnu: {model_name}")
    
    # Recherche d'hyperparamètres avec GridSearchCV
    print(f"    Recherche d'hyperparamètres ({len(param_grid)} paramètres)...")
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    # Meilleur modèle
    best_model = grid_search.best_estimator_
    
    # Prédictions
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Métriques
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0
    }
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Rapport de classification
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    metrics['classification_report'] = report
    
    # Validation croisée
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='f1')
    metrics['cv_f1_mean'] = cv_scores.mean()
    metrics['cv_f1_std'] = cv_scores.std()
    
    return {
        'model': best_model,
        'best_params': grid_search.best_params_,
        'metrics': metrics,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'param_grid': param_grid
    }


def train_all_models():
    """
    Entraîne les deux modèles pour family_history_cancer et smoker
    """
    print("=" * 80)
    print("ENTRAÎNEMENT DES MODÈLES")
    print("=" * 80)
    
    # Charger les données nettoyées
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / "data" / "processed" / "lung_cancer_cleaned.csv"
    
    if not data_path.exists():
        print(f"\n[ERREUR] Fichier {data_path} introuvable.")
        print("   Veuillez d'abord exécuter preprocess_data.py")
        return None
    
    print(f"\n[1/4] Chargement des données nettoyées...")
    df = pd.read_csv(data_path)
    print(f"    [OK] {len(df)} lignes chargees")
    print(f"    [OK] {len(df.columns)} colonnes")
    
    # Variables cibles
    target_variables = ['family_history_cancer', 'smoker']
    
    results = {}
    models_dir = base_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    for i, target_var in enumerate(target_variables, 1):
        print(f"\n[{(i+1)}/4] Entraînement du modèle pour '{target_var}'...")
        
        # Préparer les features
        X, y, feature_names = prepare_features(df, target_var)
        print(f"    [OK] {len(feature_names)} features preparees")
        print(f"    [OK] Distribution de la cible: {y.value_counts().to_dict()}")
        
        # Choisir le modèle (Random Forest pour les deux)
        model_name = 'random_forest'
        print(f"    [OK] Modele: {model_name}")
        
        # Entraîner le modèle
        result = train_model(X, y, model_name=model_name)
        
        # Afficher les résultats
        print(f"\n    Résultats:")
        print(f"      - Accuracy: {result['metrics']['accuracy']:.4f}")
        print(f"      - Precision: {result['metrics']['precision']:.4f}")
        print(f"      - Recall: {result['metrics']['recall']:.4f}")
        print(f"      - F1-Score: {result['metrics']['f1_score']:.4f}")
        print(f"      - ROC-AUC: {result['metrics']['roc_auc']:.4f}")
        print(f"      - CV F1 (mean ± std): {result['metrics']['cv_f1_mean']:.4f} ± {result['metrics']['cv_f1_std']:.4f}")
        print(f"\n    Meilleurs hyperparamètres:")
        for param, value in result['best_params'].items():
            print(f"      - {param}: {value}")
        
        # Sauvegarder le modèle
        model_path = models_dir / f"model_{target_var}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(result['model'], f)
        print(f"\n    [OK] Modele sauvegarde: {model_path}")
        
        # Stocker les résultats
        results[target_var] = {
            'model_path': str(model_path),
            'model_type': model_name,
            'best_params': result['best_params'],
            'metrics': result['metrics'],
            'feature_names': feature_names,
            'n_features': len(feature_names),
            'n_samples': len(X),
            'target_distribution': y.value_counts().to_dict()
        }
    
    # Sauvegarder les métadonnées des modèles
    metadata_path = models_dir / "models_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[OK] Metadonnees des modeles sauvegardees: {metadata_path}")
    
    print("\n" + "=" * 80)
    print("[OK] ENTRAINEMENT TERMINE")
    print("=" * 80)
    
    return results


def main():
    """Fonction principale"""
    try:
        results = train_all_models()
        if results:
            print(f"\n[OK] Succes! {len(results)} modele(s) entraine(s).")
        return results
    except Exception as e:
        print(f"\n[ERREUR] Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
