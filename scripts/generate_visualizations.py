"""
Script pour générer des visualisations du dataset lung_cancer
Crée plusieurs graphiques d'analyse exploratoire
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration du style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Configuration pour le français
plt.rcParams['font.family'] = 'DejaVu Sans'


def create_output_dir():
    """Crée le répertoire de sortie pour les visualisations"""
    output_dir = Path(__file__).parent.parent / 'docs' / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def plot_target_distribution(df, output_dir):
    """Distribution de la variable cible"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Graphique en barres
    target_counts = df['lung_cancer_risk'].value_counts().sort_index()
    colors = ['#2ecc71', '#e74c3c']
    bars = ax1.bar(['Faible risque (0)', 'Risque élevé (1)'], 
                    target_counts.values, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Nombre de patients', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution de la Variable Cible\n(Risque de Cancer du Poumon)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.grid(axis='y', alpha=0.3)
    
    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Graphique en camembert
    ax2.pie(target_counts.values, labels=['Faible risque', 'Risque élevé'], 
            autopct='%1.1f%%', colors=colors, startangle=90,
            textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax2.set_title('Répartition des Classes', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / '01_distribution_cible.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Graphique 1: Distribution de la variable cible")


def plot_numeric_distributions(df, output_dir):
    """Distributions des variables numériques importantes"""
    key_vars = ['age', 'bmi', 'pack_years', 'oxygen_saturation', 'fev1_x10', 'crp_level']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, var in enumerate(key_vars):
        if var in df.columns:
            ax = axes[idx]
            
            # Histogramme avec courbe de densité
            df[var].hist(bins=30, ax=ax, alpha=0.7, edgecolor='black', color='steelblue')
            df[var].plot.density(ax=ax, color='red', linewidth=2, linestyle='--')
            
            ax.set_xlabel(var.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            ax.set_ylabel('Fréquence', fontsize=11, fontweight='bold')
            ax.set_title(f'Distribution de {var.replace("_", " ").title()}', 
                        fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3)
            
            # Ajouter statistiques
            mean_val = df[var].mean()
            median_val = df[var].median()
            ax.axvline(mean_val, color='red', linestyle='-', linewidth=2, label=f'Moyenne: {mean_val:.1f}')
            ax.axvline(median_val, color='green', linestyle='-', linewidth=2, label=f'Médiane: {median_val:.1f}')
            ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / '02_distributions_numeriques.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Graphique 2: Distributions des variables numeriques")


def plot_correlation_heatmap(df, output_dir):
    """Heatmap de corrélations"""
    # Sélectionner les variables numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df[numeric_cols].corr()
    
    # Trier par corrélation avec la variable cible
    target_corr = corr_matrix['lung_cancer_risk'].abs().sort_values(ascending=False)
    top_vars = target_corr.head(15).index.tolist()
    corr_subset = df[top_vars].corr()
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    mask = np.triu(np.ones_like(corr_subset, dtype=bool))
    sns.heatmap(corr_subset, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1, ax=ax, annot_kws={'fontsize': 9})
    
    ax.set_title('Matrice de Corrélation\n(Top 15 variables corrélées avec le risque)', 
                 fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / '03_heatmap_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Graphique 3: Heatmap de correlations")


def plot_top_correlations(df, output_dir):
    """Top 10 variables les plus corrélées avec la variable cible"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    correlations = df[numeric_cols].corr()['lung_cancer_risk'].abs().sort_values(ascending=False)
    correlations = correlations.drop('lung_cancer_risk').head(10)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['#e74c3c' if x > 0.5 else '#3498db' for x in correlations.values]
    bars = ax.barh(range(len(correlations)), correlations.values, color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_yticks(range(len(correlations)))
    ax.set_yticklabels([var.replace('_', ' ').title() for var in correlations.index], fontsize=11)
    ax.set_xlabel('Corrélation absolue avec le risque', fontsize=12, fontweight='bold')
    ax.set_title('Top 10 Variables les Plus Corrélées avec le Risque de Cancer', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    # Ajouter les valeurs
    for i, (var, corr) in enumerate(correlations.items()):
        ax.text(corr + 0.01, i, f'{corr:.3f}', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / '04_top_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Graphique 4: Top 10 correlations")


def plot_group_analyses(df, output_dir):
    """Analyses par groupes"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Par genre
    ax1 = axes[0, 0]
    gender_risk = df.groupby('gender')['lung_cancer_risk'].agg(['mean', 'count'])
    labels = ['Femme', 'Homme']
    bars = ax1.bar(labels, gender_risk['mean'] * 100, color=['#e91e63', '#2196f3'], 
                   alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Taux de risque (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Taux de Risque par Genre', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for i, (bar, count) in enumerate(zip(bars, gender_risk['count'])):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%\n(n={count})', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. Par statut fumeur
    ax2 = axes[0, 1]
    smoker_risk = df.groupby('smoker')['lung_cancer_risk'].agg(['mean', 'count'])
    labels = ['Non-fumeur', 'Fumeur']
    bars = ax2.bar(labels, smoker_risk['mean'] * 100, color=['#2ecc71', '#e74c3c'], 
                   alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Taux de risque (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Taux de Risque par Statut Fumeur', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for i, (bar, count) in enumerate(zip(bars, smoker_risk['count'])):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%\n(n={count})', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. Par antécédents familiaux
    ax3 = axes[1, 0]
    family_risk = df.groupby('family_history_cancer')['lung_cancer_risk'].agg(['mean', 'count'])
    labels = ['Sans antécédents', 'Avec antécédents']
    bars = ax3.bar(labels, family_risk['mean'] * 100, color=['#95a5a6', '#f39c12'], 
                   alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Taux de risque (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Taux de Risque par Antécédents Familiaux', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    for i, (bar, count) in enumerate(zip(bars, family_risk['count'])):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%\n(n={count})', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 4. Par radiographie
    ax4 = axes[1, 1]
    xray_risk = df.groupby('xray_abnormal')['lung_cancer_risk'].agg(['mean', 'count'])
    labels = ['Radiographie normale', 'Radiographie anormale']
    bars = ax4.bar(labels, xray_risk['mean'] * 100, color=['#3498db', '#e74c3c'], 
                   alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Taux de risque (%)', fontsize=11, fontweight='bold')
    ax4.set_title('Taux de Risque par État de la Radiographie', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    for i, (bar, count) in enumerate(zip(bars, xray_risk['count'])):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%\n(n={count})', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / '05_analyses_par_groupes.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Graphique 5: Analyses par groupes")


def plot_boxplots_by_risk(df, output_dir):
    """Boxplots des variables numériques par niveau de risque"""
    key_vars = ['age', 'pack_years', 'bmi', 'oxygen_saturation', 'fev1_x10', 'crp_level']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, var in enumerate(key_vars):
        if var in df.columns:
            ax = axes[idx]
            
            data_to_plot = [df[df['lung_cancer_risk'] == 0][var].dropna(),
                           df[df['lung_cancer_risk'] == 1][var].dropna()]
            
            bp = ax.boxplot(data_to_plot, labels=['Faible risque', 'Risque élevé'],
                           patch_artist=True, showmeans=True)
            
            # Colorier les boxplots
            colors_box = ['#2ecc71', '#e74c3c']
            for patch, color in zip(bp['boxes'], colors_box):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_ylabel(var.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            ax.set_title(f'Distribution de {var.replace("_", " ").title()}\npar Niveau de Risque', 
                        fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '06_boxplots_par_risque.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Graphique 6: Boxplots par niveau de risque")


def plot_age_smoking_interaction(df, output_dir):
    """Interaction entre âge et tabagisme"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Créer des groupes d'âge
    df['age_group'] = pd.cut(df['age'], bins=[0, 40, 50, 60, 70, 100], 
                             labels=['<40', '40-50', '50-60', '60-70', '70+'])
    
    # Calculer les taux de risque par groupe
    interaction = df.groupby(['age_group', 'smoker'])['lung_cancer_risk'].mean().unstack()
    
    x = np.arange(len(interaction.index))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, interaction[0] * 100, width, label='Non-fumeur', 
                   color='#2ecc71', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, interaction[1] * 100, width, label='Fumeur', 
                   color='#e74c3c', alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Groupe d\'âge', fontsize=12, fontweight='bold')
    ax.set_ylabel('Taux de risque (%)', fontsize=12, fontweight='bold')
    ax.set_title('Interaction Âge × Tabagisme sur le Risque de Cancer', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(interaction.index)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Ajouter les valeurs
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / '07_interaction_age_tabagisme.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Graphique 7: Interaction age x tabagisme")


def plot_symptoms_analysis(df, output_dir):
    """Analyse des symptômes"""
    symptoms = ['chronic_cough', 'chest_pain', 'shortness_of_breath', 'fatigue']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    symptom_risk = {}
    for symptom in symptoms:
        if symptom in df.columns:
            symptom_risk[symptom] = df.groupby(symptom)['lung_cancer_risk'].mean()
    
    x = np.arange(len(symptoms))
    width = 0.35
    
    no_symptom = [symptom_risk[s][0] * 100 if 0 in symptom_risk[s].index else 0 for s in symptoms]
    with_symptom = [symptom_risk[s][1] * 100 if 1 in symptom_risk[s].index else 0 for s in symptoms]
    
    bars1 = ax.bar(x - width/2, no_symptom, width, label='Sans symptôme', 
                   color='#3498db', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, with_symptom, width, label='Avec symptôme', 
                   color='#e74c3c', alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Symptômes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Taux de risque (%)', fontsize=12, fontweight='bold')
    ax.set_title('Impact des Symptômes sur le Risque de Cancer', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_', ' ').title() for s in symptoms], rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Ajouter les valeurs
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / '08_analyse_symptomes.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Graphique 8: Analyse des symptomes")


def main():
    """Fonction principale"""
    print("=" * 60)
    print("GENERATION DES VISUALISATIONS")
    print("=" * 60)
    
    # Charger les données
    base_dir = Path(__file__).parent.parent
    csv_path = base_dir / 'data' / 'raw' / 'lung_cancer.csv'
    
    print(f"\nChargement des donnees: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Dataset charge: {len(df)} patients, {len(df.columns)} variables\n")
    
    # Créer le répertoire de sortie
    output_dir = create_output_dir()
    print(f"Repertoire de sortie: {output_dir}\n")
    
    # Générer les visualisations
    try:
        plot_target_distribution(df, output_dir)
        plot_numeric_distributions(df, output_dir)
        plot_correlation_heatmap(df, output_dir)
        plot_top_correlations(df, output_dir)
        plot_group_analyses(df, output_dir)
        plot_boxplots_by_risk(df, output_dir)
        plot_age_smoking_interaction(df, output_dir)
        plot_symptoms_analysis(df, output_dir)
        
        print("\n" + "=" * 60)
        print("GENERATION TERMINEE AVEC SUCCES!")
        print("=" * 60)
        print(f"\nToutes les visualisations ont ete sauvegardees dans:")
        print(f"{output_dir}")
        print(f"\nTotal: 8 graphiques generes")
        
    except Exception as e:
        print(f"\nERREUR lors de la generation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
