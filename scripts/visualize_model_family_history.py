"""
Script simple pour visualiser le modèle family_history_cancer
Génère des graphiques intéressants sur les features importantes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration simple
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)

def load_model_and_data():
    """Charge le modèle et les données"""
    import json
    base_dir = Path(__file__).parent.parent
    
    # Charger les métadonnées pour avoir les feature names exacts
    metadata_path = base_dir / "models" / "models_metadata.json"
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    feature_names = metadata['family_history_cancer']['feature_names']
    
    # Charger le modèle
    model_path = base_dir / "models" / "model_family_history_cancer.pkl"
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Charger les données nettoyées
    data_path = base_dir / "data" / "processed" / "lung_cancer_cleaned.csv"
    df = pd.read_csv(data_path)
    
    # Utiliser les features dans le même ordre que lors de l'entraînement
    X = df[feature_names]
    y = df['family_history_cancer']
    
    return model, X, y, feature_names

def plot_feature_importance(model, X, output_dir):
    """Graphique 1: Importance des features (TOP 15)"""
    importances = model.feature_importances_
    feature_names = list(X.columns)
    indices = np.argsort(importances)[::-1][:15]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importances[indices], color='steelblue', alpha=0.7)
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance', fontsize=12, fontweight='bold')
    plt.title('Top 15 Features Importantes\n(Modèle: family_history_cancer)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'family_history_feature_importance.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("[OK] Graphique 1: Importance des features")

def plot_confusion_matrix(model, X, y, output_dir):
    """Graphique 2: Matrice de confusion"""
    from sklearn.metrics import confusion_matrix
    
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Pas d\'antécédents', 'Avec antécédents'],
                yticklabels=['Pas d\'antécédents', 'Avec antécédents'])
    plt.ylabel('Vraie valeur', fontsize=12, fontweight='bold')
    plt.xlabel('Prédiction', fontsize=12, fontweight='bold')
    plt.title('Matrice de Confusion\n(Modèle: family_history_cancer)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'family_history_confusion_matrix.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("[OK] Graphique 2: Matrice de confusion")

def plot_prediction_distribution(model, X, y, output_dir):
    """Graphique 3: Distribution des probabilités de prédiction"""
    y_proba = model.predict_proba(X)[:, 1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogramme des probabilités
    ax1.hist(y_proba[y==0], bins=30, alpha=0.6, label='Pas d\'antécédents', color='green')
    ax1.hist(y_proba[y==1], bins=30, alpha=0.6, label='Avec antécédents', color='red')
    ax1.set_xlabel('Probabilité prédite', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Fréquence', fontsize=11, fontweight='bold')
    ax1.set_title('Distribution des Probabilités', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Boxplot par classe
    data_to_plot = [y_proba[y==0], y_proba[y==1]]
    ax2.boxplot(data_to_plot, labels=['Pas d\'antécédents', 'Avec antécédents'])
    ax2.set_ylabel('Probabilité prédite', fontsize=11, fontweight='bold')
    ax2.set_title('Distribution par Classe', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    plt.suptitle('Analyse des Prédictions - family_history_cancer', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'family_history_predictions.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("[OK] Graphique 3: Distribution des prédictions")

def plot_top_features_comparison(model, X, y, output_dir):
    """Graphique 4: Comparaison des top 5 features par classe"""
    importances = model.feature_importances_
    feature_names = list(X.columns)
    top5_indices = np.argsort(importances)[::-1][:5]
    top5_features = [feature_names[i] for i in top5_indices]
    
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    
    for idx, (feat_idx, feat_name) in enumerate(zip(top5_indices, top5_features)):
        ax = axes[idx]
        
        # Boxplot par classe
        data_0 = X[y==0][feat_name]
        data_1 = X[y==1][feat_name]
        
        ax.boxplot([data_0, data_1], labels=['0', '1'])
        ax.set_title(f'{feat_name}\n(Importance: {importances[feat_idx]:.3f})', 
                    fontsize=10, fontweight='bold')
        ax.set_ylabel('Valeur normalisée', fontsize=9)
        ax.grid(alpha=0.3)
    
    plt.suptitle('Top 5 Features Importantes par Classe\n(Modèle: family_history_cancer)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'family_history_top_features.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("[OK] Graphique 4: Top features par classe")

def main():
    """Fonction principale"""
    print("=" * 60)
    print("VISUALISATION MODELE: family_history_cancer")
    print("=" * 60)
    
    # Créer le répertoire de sortie
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "docs" / "visualizations" / "model_family_history"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Charger modèle et données
    print("\n[1/5] Chargement du modele et des donnees...")
    model, X, y, features = load_model_and_data()
    print(f"    [OK] Modele charge")
    print(f"    [OK] {len(X)} echantillons, {len(features)} features")
    
    # Générer les graphiques
    print("\n[2/5] Generation des graphiques...")
    plot_feature_importance(model, X, output_dir)
    plot_confusion_matrix(model, X, y, output_dir)
    plot_prediction_distribution(model, X, y, output_dir)
    plot_top_features_comparison(model, X, y, output_dir)
    
    print("\n" + "=" * 60)
    print("[OK] VISUALISATION TERMINEE!")
    print("=" * 60)
    print(f"\nGraphiques sauvegardes dans: {output_dir}")
    print("Total: 4 graphiques generes")

if __name__ == "__main__":
    main()
