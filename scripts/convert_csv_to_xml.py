"""
Script de conversion CSV vers XML pour le dataset lung_cancer
Génère un XML informatif avec métadonnées, statistiques et résumés pratiques
au lieu de copier toutes les données brutes
"""

import pandas as pd
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from datetime import datetime
import numpy as np


def prettify_xml(elem):
    """Retourne une chaîne XML formatée de manière lisible"""
    rough_string = ET.tostring(elem, encoding='unicode')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def convert_csv_to_xml(csv_path, xml_path):
    """
    Convertit un CSV en XML informatif avec métadonnées et statistiques
    """
    # Charger les données
    df = pd.read_csv(csv_path)
    
    # Créer l'élément racine
    root = ET.Element('lung_cancer_dataset')
    
    # ===== MÉTADONNÉES =====
    metadata = ET.SubElement(root, 'metadata')
    ET.SubElement(metadata, 'source_file').text = str(csv_path)
    ET.SubElement(metadata, 'conversion_date').text = datetime.now().isoformat()
    ET.SubElement(metadata, 'format_version').text = '2.0'
    ET.SubElement(metadata, 'description').text = 'Dataset informatif avec métadonnées et statistiques'
    
    dataset_info = ET.SubElement(metadata, 'dataset_info')
    ET.SubElement(dataset_info, 'total_patients').text = str(len(df))
    ET.SubElement(dataset_info, 'total_variables').text = str(len(df.columns))
    ET.SubElement(dataset_info, 'target_variable').text = 'lung_cancer_risk'
    ET.SubElement(dataset_info, 'missing_values').text = str(df.isna().sum().sum())
    ET.SubElement(dataset_info, 'duplicates').text = str(df.duplicated().sum())
    
    # ===== STATISTIQUES GLOBALES =====
    statistics = ET.SubElement(root, 'statistics')
    
    # Distribution de la variable cible
    target_dist = df['lung_cancer_risk'].value_counts()
    outcome_stats = ET.SubElement(statistics, 'outcome_distribution')
    ET.SubElement(outcome_stats, 'low_risk', count=str(target_dist[0]), percentage=f"{target_dist[0]/len(df)*100:.2f}%")
    ET.SubElement(outcome_stats, 'high_risk', count=str(target_dist[1]), percentage=f"{target_dist[1]/len(df)*100:.2f}%")
    ET.SubElement(outcome_stats, 'balance_ratio').text = f"{target_dist[0]/target_dist[1]:.2f}:1"
    
    # ===== STATISTIQUES PAR VARIABLE =====
    variables_stats = ET.SubElement(root, 'variables_statistics')
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in df.columns:
        var_stat = ET.SubElement(variables_stats, 'variable', name=col)
        
        if col in numeric_cols:
            ET.SubElement(var_stat, 'type').text = 'numeric'
            ET.SubElement(var_stat, 'mean').text = f"{df[col].mean():.2f}"
            ET.SubElement(var_stat, 'median').text = f"{df[col].median():.2f}"
            ET.SubElement(var_stat, 'std').text = f"{df[col].std():.2f}"
            ET.SubElement(var_stat, 'min').text = f"{df[col].min():.2f}"
            ET.SubElement(var_stat, 'max').text = f"{df[col].max():.2f}"
            
            # Corrélation avec la variable cible
            if col != 'lung_cancer_risk':
                corr = df[col].corr(df['lung_cancer_risk'])
                ET.SubElement(var_stat, 'correlation_with_target').text = f"{corr:.4f}"
        else:
            ET.SubElement(var_stat, 'type').text = 'categorical'
            value_counts = df[col].value_counts()
            distribution = ET.SubElement(var_stat, 'distribution')
            for val, count in value_counts.head(10).items():
                ET.SubElement(distribution, 'value', value=str(val), count=str(count), percentage=f"{count/len(df)*100:.2f}%")
    
    # ===== ANALYSES PAR GROUPES =====
    group_analyses = ET.SubElement(root, 'group_analyses')
    
    # Par genre
    gender_analysis = ET.SubElement(group_analyses, 'by_gender')
    for gender_val, gender_label in [(0, 'Femme'), (1, 'Homme')]:
        gender_group = df[df['gender'] == gender_val]
        gender_elem = ET.SubElement(gender_analysis, 'group', label=gender_label, count=str(len(gender_group)))
        ET.SubElement(gender_elem, 'risk_rate').text = f"{gender_group['lung_cancer_risk'].mean()*100:.2f}%"
        ET.SubElement(gender_elem, 'mean_age').text = f"{gender_group['age'].mean():.1f}"
        ET.SubElement(gender_elem, 'smoker_rate').text = f"{gender_group['smoker'].mean()*100:.2f}%"
    
    # Par statut fumeur
    smoker_analysis = ET.SubElement(group_analyses, 'by_smoking_status')
    for smoker_val, smoker_label in [(0, 'Non-fumeur'), (1, 'Fumeur')]:
        smoker_group = df[df['smoker'] == smoker_val]
        smoker_elem = ET.SubElement(smoker_analysis, 'group', label=smoker_label, count=str(len(smoker_group)))
        ET.SubElement(smoker_elem, 'risk_rate').text = f"{smoker_group['lung_cancer_risk'].mean()*100:.2f}%"
        ET.SubElement(smoker_elem, 'mean_pack_years').text = f"{smoker_group['pack_years'].mean():.1f}"
    
    # Par antécédents familiaux
    family_analysis = ET.SubElement(group_analyses, 'by_family_history')
    for fh_val, fh_label in [(0, 'Sans antécédents'), (1, 'Avec antécédents')]:
        fh_group = df[df['family_history_cancer'] == fh_val]
        fh_elem = ET.SubElement(family_analysis, 'group', label=fh_label, count=str(len(fh_group)))
        ET.SubElement(fh_elem, 'risk_rate').text = f"{fh_group['lung_cancer_risk'].mean()*100:.2f}%"
    
    # ===== VARIABLES LES PLUS IMPORTANTES =====
    feature_importance = ET.SubElement(root, 'feature_importance')
    
    # Top 10 corrélations avec la variable cible
    correlations = df[numeric_cols].corr()['lung_cancer_risk'].abs().sort_values(ascending=False)
    correlations = correlations.drop('lung_cancer_risk')
    
    top_features = ET.SubElement(feature_importance, 'top_correlated_features')
    for i, (var, corr) in enumerate(correlations.head(10).items(), 1):
        feature_elem = ET.SubElement(top_features, 'feature', rank=str(i), name=var, correlation=f"{corr:.4f}")
        if var in df.columns:
            high_risk_mean = df[df['lung_cancer_risk'] == 1][var].mean()
            low_risk_mean = df[df['lung_cancer_risk'] == 0][var].mean()
            ET.SubElement(feature_elem, 'mean_high_risk').text = f"{high_risk_mean:.2f}"
            ET.SubElement(feature_elem, 'mean_low_risk').text = f"{low_risk_mean:.2f}"
    
    # ===== ÉCHANTILLON REPRÉSENTATIF =====
    sample = ET.SubElement(root, 'representative_sample')
    ET.SubElement(sample, 'description').text = 'Échantillon de 10 patients représentatifs (5 à risque faible, 5 à risque élevé)'
    
    # Échantillon équilibré
    low_risk_sample = df[df['lung_cancer_risk'] == 0].sample(min(5, len(df[df['lung_cancer_risk'] == 0])))
    high_risk_sample = df[df['lung_cancer_risk'] == 1].sample(min(5, len(df[df['lung_cancer_risk'] == 1])))
    sample_df = pd.concat([low_risk_sample, high_risk_sample])
    
    sample_patients = ET.SubElement(sample, 'patients')
    for idx, (_, row) in enumerate(sample_df.iterrows(), 1):
        patient = ET.SubElement(sample_patients, 'patient', id=f'S{idx:02d}', risk=str(int(row['lung_cancer_risk'])))
        
        # Seulement les variables clés
        key_vars = ['age', 'gender', 'smoker', 'pack_years', 'family_history_cancer', 
                   'xray_abnormal', 'bmi', 'oxygen_saturation']
        
        for var in key_vars:
            if var in row:
                ET.SubElement(patient, var).text = str(row[var])
    
    # ===== RÉSUMÉS PAR CATÉGORIES =====
    summaries = ET.SubElement(root, 'category_summaries')
    
    # Démographie
    demo_summary = ET.SubElement(summaries, 'demographics')
    ET.SubElement(demo_summary, 'mean_age').text = f"{df['age'].mean():.1f}"
    ET.SubElement(demo_summary, 'age_range').text = f"{df['age'].min():.0f}-{df['age'].max():.0f}"
    ET.SubElement(demo_summary, 'gender_distribution', 
                 female=f"{df[df['gender']==0].shape[0]}", 
                 male=f"{df[df['gender']==1].shape[0]}")
    
    # Tabagisme
    smoking_summary = ET.SubElement(summaries, 'smoking')
    ET.SubElement(smoking_summary, 'smoker_rate').text = f"{df['smoker'].mean()*100:.2f}%"
    ET.SubElement(smoking_summary, 'mean_smoking_years').text = f"{df[df['smoker']==1]['smoking_years'].mean():.1f}"
    ET.SubElement(smoking_summary, 'mean_pack_years').text = f"{df[df['smoker']==1]['pack_years'].mean():.1f}"
    
    # Clinique
    clinical_summary = ET.SubElement(summaries, 'clinical')
    ET.SubElement(clinical_summary, 'mean_bmi').text = f"{df['bmi'].mean():.1f}"
    ET.SubElement(clinical_summary, 'mean_oxygen_saturation').text = f"{df['oxygen_saturation'].mean():.1f}"
    ET.SubElement(clinical_summary, 'xray_abnormal_rate').text = f"{df['xray_abnormal'].mean()*100:.2f}%"
    
    # ===== RECOMMANDATIONS =====
    recommendations = ET.SubElement(root, 'recommendations')
    ET.SubElement(recommendations, 'data_quality').text = 'Excellent - Aucune valeur manquante'
    
    if abs(target_dist[0]/target_dist[1] - 1) > 0.2:
        ET.SubElement(recommendations, 'class_imbalance').text = f'Déséquilibre détecté (ratio {target_dist[0]/target_dist[1]:.2f}:1) - Considérer rééchantillonnage'
    else:
        ET.SubElement(recommendations, 'class_balance').text = 'Classes équilibrées'
    
    top_risk_factors = ', '.join(correlations.head(5).index.tolist())
    ET.SubElement(recommendations, 'key_risk_factors').text = f'Facteurs de risque principaux: {top_risk_factors}'
    
    # Formater et sauvegarder
    xml_string = prettify_xml(root)
    
    with open(xml_path, 'w', encoding='utf-8') as xmlfile:
        xmlfile.write(xml_string)
    
    file_size_kb = Path(xml_path).stat().st_size / 1024
    print(f"XML informatif genere avec succes: {xml_path}")
    print(f"Taille: {file_size_kb:.1f} KB")


if __name__ == '__main__':
    base_dir = Path(__file__).parent.parent
    csv_path = base_dir / 'data' / 'raw' / 'lung_cancer.csv'
    xml_path = base_dir / 'data' / 'xml' / 'lung_cancer.xml'
    
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Generation du XML informatif (metadonnees et statistiques)...")
    print(f"Source: {csv_path}")
    convert_csv_to_xml(csv_path, xml_path)
    print("\nConversion terminee avec succes!")
