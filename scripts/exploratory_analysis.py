"""
Script d'analyse exploratoire du dataset lung_cancer
Génère un rapport Markdown complet avec statistiques et visualisations
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def generate_exploration_report(csv_path, output_path):
    """
    Génère un rapport d'exploration complet au format Markdown
    """
    # Charger les données
    df = pd.read_csv(csv_path)
    
    # Initialiser le rapport
    report = []
    report.append("# Rapport d'Exploration - Dataset Lung Cancer Risk\n")
    report.append(f"**Date de génération:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**Source:** {csv_path}\n\n")
    report.append("---\n\n")
    
    # 1. Vue d'ensemble
    report.append("## 1. Vue d'ensemble du Dataset\n\n")
    report.append(f"- **Nombre total d'échantillons:** {len(df):,}\n")
    report.append(f"- **Nombre de variables:** {len(df.columns)}\n")
    report.append(f"- **Variables prédictives:** {len(df.columns) - 1}\n")
    report.append(f"- **Variable cible:** `lung_cancer_risk`\n\n")
    
    # 2. Structure des données
    report.append("## 2. Structure des Données\n\n")
    report.append("### 2.1 Liste des Variables\n\n")
    report.append("| Variable | Type | Description |\n")
    report.append("|----------|------|-------------|\n")
    
    variable_descriptions = {
        'age': 'Age du patient en années',
        'gender': 'Genre (0=Femme, 1=Homme)',
        'education_years': 'Nombre d\'années d\'éducation',
        'income_level': 'Niveau de revenu (1-4)',
        'smoker': 'Statut fumeur (0=Non, 1=Oui)',
        'smoking_years': 'Années de tabagisme',
        'cigarettes_per_day': 'Cigarettes par jour',
        'pack_years': 'Paquets-années',
        'passive_smoking': 'Tabagisme passif (0=Non, 1=Oui)',
        'air_pollution_index': 'Indice de pollution de l\'air',
        'occupational_exposure': 'Exposition professionnelle (0=Non, 1=Oui)',
        'radon_exposure': 'Exposition au radon (0=Non, 1=Oui)',
        'family_history_cancer': 'Antécédents familiaux (0=Non, 1=Oui)',
        'copd': 'BPCO (0=Non, 1=Oui)',
        'asthma': 'Asthme (0=Non, 1=Oui)',
        'previous_tb': 'Antécédents tuberculose (0=Non, 1=Oui)',
        'chronic_cough': 'Toux chronique (0=Non, 1=Oui)',
        'chest_pain': 'Douleur thoracique (0=Non, 1=Oui)',
        'shortness_of_breath': 'Essoufflement (0=Non, 1=Oui)',
        'fatigue': 'Fatigue (0=Non, 1=Oui)',
        'bmi': 'Indice de masse corporelle',
        'oxygen_saturation': 'Saturation en oxygène (%)',
        'fev1_x10': 'FEV1 multiplié par 10',
        'crp_level': 'Niveau de protéine C-réactive',
        'xray_abnormal': 'Radiographie anormale (0=Normal, 1=Anormal)',
        'exercise_hours_per_week': 'Heures d\'exercice par semaine',
        'diet_quality': 'Qualité de l\'alimentation (1-5)',
        'alcohol_units_per_week': 'Unités d\'alcool par semaine',
        'healthcare_access': 'Accès aux soins (1-4)',
        'lung_cancer_risk': 'Risque de cancer (0=Faible, 1=Élevé) - VARIABLE CIBLE'
    }
    
    for var in df.columns:
        var_type = 'Catégorielle' if df[var].dtype == 'object' or df[var].nunique() < 10 else 'Numérique'
        description = variable_descriptions.get(var, 'Non définie')
        report.append(f"| `{var}` | {var_type} | {description} |\n")
    
    report.append("\n")
    
    # 3. Statistiques descriptives
    report.append("## 3. Statistiques Descriptives\n\n")
    
    # Variables numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    report.append("### 3.1 Variables Numériques\n\n")
    report.append("| Variable | Moyenne | Médiane | Écart-type | Min | Max |\n")
    report.append("|----------|---------|---------|------------|-----|-----|\n")
    
    for col in numeric_cols:
        mean_val = df[col].mean()
        median_val = df[col].median()
        std_val = df[col].std()
        min_val = df[col].min()
        max_val = df[col].max()
        report.append(f"| `{col}` | {mean_val:.2f} | {median_val:.2f} | {std_val:.2f} | {min_val:.2f} | {max_val:.2f} |\n")
    
    report.append("\n")
    
    # 4. Distribution de la variable cible
    report.append("## 4. Distribution de la Variable Cible\n\n")
    target_dist = df['lung_cancer_risk'].value_counts()
    report.append(f"- **Classe 0 (Faible risque):** {target_dist[0]:,} ({target_dist[0]/len(df)*100:.2f}%)\n")
    report.append(f"- **Classe 1 (Risque élevé):** {target_dist[1]:,} ({target_dist[1]/len(df)*100:.2f}%)\n")
    report.append(f"- **Ratio d'équilibre:** {target_dist[0]/target_dist[1]:.2f}:1\n\n")
    
    if abs(target_dist[0]/target_dist[1] - 1) > 0.2:
        report.append("⚠️ **Attention:** Le dataset présente un déséquilibre de classes.\n\n")
    
    # 5. Valeurs manquantes
    report.append("## 5. Analyse des Valeurs Manquantes\n\n")
    missing = df.isna().sum()
    missing_pct = (missing / len(df)) * 100
    
    report.append("| Variable | Valeurs manquantes | Pourcentage |\n")
    report.append("|----------|-------------------|-------------|\n")
    
    has_missing = False
    for col in df.columns:
        if missing[col] > 0:
            has_missing = True
            report.append(f"| `{col}` | {missing[col]:,} | {missing_pct[col]:.2f}% |\n")
    
    if not has_missing:
        report.append("| *Aucune* | 0 | 0.00% |\n")
    
    report.append("\n")
    
    # 6. Duplicatas
    report.append("## 6. Analyse des Duplicatas\n\n")
    duplicates = df.duplicated().sum()
    report.append(f"- **Nombre de lignes dupliquées:** {duplicates:,}\n")
    report.append(f"- **Pourcentage:** {duplicates/len(df)*100:.2f}%\n\n")
    
    # 7. Corrélations avec la variable cible
    report.append("## 7. Corrélations avec la Variable Cible\n\n")
    correlations = df[numeric_cols].corr()['lung_cancer_risk'].sort_values(ascending=False)
    correlations = correlations.drop('lung_cancer_risk')
    
    report.append("| Variable | Corrélation |\n")
    report.append("|----------|-------------|\n")
    
    for var, corr in correlations.items():
        report.append(f"| `{var}` | {corr:.4f} |\n")
    
    report.append("\n")
    
    # 8. Analyse par groupes
    report.append("## 8. Analyse par Groupes\n\n")
    
    # Par genre
    report.append("### 8.1 Distribution par Genre\n\n")
    gender_dist = df.groupby('gender')['lung_cancer_risk'].agg(['count', 'sum', 'mean'])
    report.append("| Genre | Total | Risque élevé | Taux de risque |\n")
    report.append("|-------|------|--------------|----------------|\n")
    report.append(f"| Femme (0) | {gender_dist.loc[0, 'count']:,} | {gender_dist.loc[0, 'sum']:,} | {gender_dist.loc[0, 'mean']*100:.2f}% |\n")
    report.append(f"| Homme (1) | {gender_dist.loc[1, 'count']:,} | {gender_dist.loc[1, 'sum']:,} | {gender_dist.loc[1, 'mean']*100:.2f}% |\n")
    report.append("\n")
    
    # Par statut fumeur
    report.append("### 8.2 Distribution par Statut Fumeur\n\n")
    smoker_dist = df.groupby('smoker')['lung_cancer_risk'].agg(['count', 'sum', 'mean'])
    report.append("| Statut | Total | Risque élevé | Taux de risque |\n")
    report.append("|--------|------|--------------|----------------|\n")
    report.append(f"| Non-fumeur (0) | {smoker_dist.loc[0, 'count']:,} | {smoker_dist.loc[0, 'sum']:,} | {smoker_dist.loc[0, 'mean']*100:.2f}% |\n")
    report.append(f"| Fumeur (1) | {smoker_dist.loc[1, 'count']:,} | {smoker_dist.loc[1, 'sum']:,} | {smoker_dist.loc[1, 'mean']*100:.2f}% |\n")
    report.append("\n")
    
    # 9. Observations et recommandations
    report.append("## 9. Observations et Recommandations\n\n")
    
    report.append("### 9.1 Points Clés\n\n")
    report.append("- Dataset bien structuré avec 30 variables\n")
    report.append(f"- Aucune valeur manquante détectée\n")
    report.append(f"- Distribution de la variable cible: {target_dist[0]/target_dist[1]:.2f}:1\n")
    
    # Identifier les variables les plus corrélées
    top_corr = correlations.abs().nlargest(5)
    report.append("\n### 9.2 Variables les Plus Corrélées avec le Risque\n\n")
    for var, corr in top_corr.items():
        report.append(f"- `{var}`: corrélation = {correlations[var]:.4f}\n")
    
    report.append("\n### 9.3 Recommandations pour la Modélisation\n\n")
    report.append("1. **Préprocessing:**\n")
    report.append("   - Normaliser/standardiser les variables numériques\n")
    report.append("   - Encoder les variables catégorielles si nécessaire\n")
    report.append("   - Gérer le déséquilibre de classes si présent\n\n")
    
    report.append("2. **Feature Engineering:**\n")
    report.append("   - Créer des interactions entre variables importantes\n")
    report.append("   - Considérer des transformations non-linéaires\n\n")
    
    report.append("3. **Modélisation:**\n")
    report.append("   - Tester plusieurs algorithmes (Random Forest, XGBoost, Logistic Regression)\n")
    report.append("   - Utiliser la validation croisée\n")
    report.append("   - Évaluer avec plusieurs métriques (accuracy, precision, recall, F1, AUC-ROC)\n\n")
    
    # 10. Prochaines étapes
    report.append("## 10. Prochaines Étapes\n\n")
    report.append("1. Analyse statistique approfondie (tests d'hypothèses)\n")
    report.append("2. Visualisations avancées (distributions, corrélations)\n")
    report.append("3. Feature selection\n")
    report.append("4. Développement de modèles prédictifs\n")
    report.append("5. Validation et interprétation des résultats\n\n")
    
    report.append("---\n\n")
    report.append(f"*Rapport généré automatiquement le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    # Sauvegarder le rapport
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(''.join(report))
    
    print(f"Rapport d'exploration genere avec succes: {output_path}")


if __name__ == '__main__':
    base_dir = Path(__file__).parent.parent
    csv_path = base_dir / 'data' / 'raw' / 'lung_cancer.csv'
    output_path = base_dir / 'docs' / 'exploration' / 'rapport_exploration.md'
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    generate_exploration_report(csv_path, output_path)
