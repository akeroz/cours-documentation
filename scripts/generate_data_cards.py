"""
Script pour générer les Data Cards (métadonnées complètes du dataset)
"""

import pandas as pd
import json
import yaml
from pathlib import Path
from datetime import datetime


def generate_data_cards(csv_path, output_yaml, output_json):
    """
    Génère les Data Cards complètes avec métadonnées, dictionnaire et statistiques
    """
    # Charger les données
    df = pd.read_csv(csv_path)
    
    # Dictionnaire des variables
    variable_dict = {
        'age': {
            'type': 'integer',
            'description': 'Age du patient en années',
            'unit': 'années',
            'range': [int(df['age'].min()), int(df['age'].max())],
            'missing_values': int(df['age'].isna().sum())
        },
        'gender': {
            'type': 'categorical',
            'description': 'Genre du patient (0=Femme, 1=Homme)',
            'values': {0: 'Femme', 1: 'Homme'},
            'distribution': df['gender'].value_counts().to_dict(),
            'missing_values': int(df['gender'].isna().sum())
        },
        'education_years': {
            'type': 'integer',
            'description': 'Nombre d\'années d\'éducation',
            'unit': 'années',
            'range': [int(df['education_years'].min()), int(df['education_years'].max())],
            'missing_values': int(df['education_years'].isna().sum())
        },
        'income_level': {
            'type': 'categorical',
            'description': 'Niveau de revenu (1=Faible, 2=Moyen, 3=Élevé, 4=Très élevé)',
            'values': {1: 'Faible', 2: 'Moyen', 3: 'Élevé', 4: 'Très élevé'},
            'distribution': df['income_level'].value_counts().to_dict(),
            'missing_values': int(df['income_level'].isna().sum())
        },
        'smoker': {
            'type': 'binary',
            'description': 'Statut fumeur (0=Non-fumeur, 1=Fumeur)',
            'values': {0: 'Non-fumeur', 1: 'Fumeur'},
            'distribution': df['smoker'].value_counts().to_dict(),
            'missing_values': int(df['smoker'].isna().sum())
        },
        'smoking_years': {
            'type': 'integer',
            'description': 'Nombre d\'années de tabagisme',
            'unit': 'années',
            'range': [int(df['smoking_years'].min()), int(df['smoking_years'].max())],
            'missing_values': int(df['smoking_years'].isna().sum())
        },
        'cigarettes_per_day': {
            'type': 'integer',
            'description': 'Nombre de cigarettes par jour',
            'unit': 'cigarettes/jour',
            'range': [int(df['cigarettes_per_day'].min()), int(df['cigarettes_per_day'].max())],
            'missing_values': int(df['cigarettes_per_day'].isna().sum())
        },
        'pack_years': {
            'type': 'integer',
            'description': 'Paquets-années (mesure cumulative d\'exposition au tabac)',
            'unit': 'paquets-années',
            'range': [int(df['pack_years'].min()), int(df['pack_years'].max())],
            'missing_values': int(df['pack_years'].isna().sum())
        },
        'passive_smoking': {
            'type': 'binary',
            'description': 'Exposition au tabagisme passif (0=Non, 1=Oui)',
            'values': {0: 'Non', 1: 'Oui'},
            'distribution': df['passive_smoking'].value_counts().to_dict(),
            'missing_values': int(df['passive_smoking'].isna().sum())
        },
        'air_pollution_index': {
            'type': 'continuous',
            'description': 'Indice de pollution de l\'air',
            'unit': 'index',
            'range': [float(df['air_pollution_index'].min()), float(df['air_pollution_index'].max())],
            'mean': float(df['air_pollution_index'].mean()),
            'std': float(df['air_pollution_index'].std()),
            'missing_values': int(df['air_pollution_index'].isna().sum())
        },
        'occupational_exposure': {
            'type': 'binary',
            'description': 'Exposition professionnelle à des toxiques (0=Non, 1=Oui)',
            'values': {0: 'Non', 1: 'Oui'},
            'distribution': df['occupational_exposure'].value_counts().to_dict(),
            'missing_values': int(df['occupational_exposure'].isna().sum())
        },
        'radon_exposure': {
            'type': 'binary',
            'description': 'Exposition au radon (0=Non, 1=Oui)',
            'values': {0: 'Non', 1: 'Oui'},
            'distribution': df['radon_exposure'].value_counts().to_dict(),
            'missing_values': int(df['radon_exposure'].isna().sum())
        },
        'family_history_cancer': {
            'type': 'binary',
            'description': 'Antécédents familiaux de cancer (0=Non, 1=Oui)',
            'values': {0: 'Non', 1: 'Oui'},
            'distribution': df['family_history_cancer'].value_counts().to_dict(),
            'missing_values': int(df['family_history_cancer'].isna().sum())
        },
        'copd': {
            'type': 'binary',
            'description': 'BPCO (Bronchopneumopathie chronique obstructive) (0=Non, 1=Oui)',
            'values': {0: 'Non', 1: 'Oui'},
            'distribution': df['copd'].value_counts().to_dict(),
            'missing_values': int(df['copd'].isna().sum())
        },
        'asthma': {
            'type': 'binary',
            'description': 'Asthme (0=Non, 1=Oui)',
            'values': {0: 'Non', 1: 'Oui'},
            'distribution': df['asthma'].value_counts().to_dict(),
            'missing_values': int(df['asthma'].isna().sum())
        },
        'previous_tb': {
            'type': 'binary',
            'description': 'Antécédents de tuberculose (0=Non, 1=Oui)',
            'values': {0: 'Non', 1: 'Oui'},
            'distribution': df['previous_tb'].value_counts().to_dict(),
            'missing_values': int(df['previous_tb'].isna().sum())
        },
        'chronic_cough': {
            'type': 'binary',
            'description': 'Toux chronique (0=Non, 1=Oui)',
            'values': {0: 'Non', 1: 'Oui'},
            'distribution': df['chronic_cough'].value_counts().to_dict(),
            'missing_values': int(df['chronic_cough'].isna().sum())
        },
        'chest_pain': {
            'type': 'binary',
            'description': 'Douleur thoracique (0=Non, 1=Oui)',
            'values': {0: 'Non', 1: 'Oui'},
            'distribution': df['chest_pain'].value_counts().to_dict(),
            'missing_values': int(df['chest_pain'].isna().sum())
        },
        'shortness_of_breath': {
            'type': 'binary',
            'description': 'Essoufflement (0=Non, 1=Oui)',
            'values': {0: 'Non', 1: 'Oui'},
            'distribution': df['shortness_of_breath'].value_counts().to_dict(),
            'missing_values': int(df['shortness_of_breath'].isna().sum())
        },
        'fatigue': {
            'type': 'binary',
            'description': 'Fatigue (0=Non, 1=Oui)',
            'values': {0: 'Non', 1: 'Oui'},
            'distribution': df['fatigue'].value_counts().to_dict(),
            'missing_values': int(df['fatigue'].isna().sum())
        },
        'bmi': {
            'type': 'continuous',
            'description': 'Indice de masse corporelle',
            'unit': 'kg/m²',
            'range': [float(df['bmi'].min()), float(df['bmi'].max())],
            'mean': float(df['bmi'].mean()),
            'std': float(df['bmi'].std()),
            'missing_values': int(df['bmi'].isna().sum())
        },
        'oxygen_saturation': {
            'type': 'continuous',
            'description': 'Saturation en oxygène',
            'unit': '%',
            'range': [float(df['oxygen_saturation'].min()), float(df['oxygen_saturation'].max())],
            'mean': float(df['oxygen_saturation'].mean()),
            'std': float(df['oxygen_saturation'].std()),
            'missing_values': int(df['oxygen_saturation'].isna().sum())
        },
        'fev1_x10': {
            'type': 'continuous',
            'description': 'Volume expiratoire maximal en 1 seconde (multiplié par 10)',
            'unit': 'L x 10',
            'range': [float(df['fev1_x10'].min()), float(df['fev1_x10'].max())],
            'mean': float(df['fev1_x10'].mean()),
            'std': float(df['fev1_x10'].std()),
            'missing_values': int(df['fev1_x10'].isna().sum())
        },
        'crp_level': {
            'type': 'continuous',
            'description': 'Niveau de protéine C-réactive',
            'unit': 'mg/L',
            'range': [float(df['crp_level'].min()), float(df['crp_level'].max())],
            'mean': float(df['crp_level'].mean()),
            'std': float(df['crp_level'].std()),
            'missing_values': int(df['crp_level'].isna().sum())
        },
        'xray_abnormal': {
            'type': 'binary',
            'description': 'Radiographie anormale (0=Normal, 1=Anormal)',
            'values': {0: 'Normal', 1: 'Anormal'},
            'distribution': df['xray_abnormal'].value_counts().to_dict(),
            'missing_values': int(df['xray_abnormal'].isna().sum())
        },
        'exercise_hours_per_week': {
            'type': 'continuous',
            'description': 'Heures d\'exercice par semaine',
            'unit': 'heures/semaine',
            'range': [float(df['exercise_hours_per_week'].min()), float(df['exercise_hours_per_week'].max())],
            'mean': float(df['exercise_hours_per_week'].mean()),
            'std': float(df['exercise_hours_per_week'].std()),
            'missing_values': int(df['exercise_hours_per_week'].isna().sum())
        },
        'diet_quality': {
            'type': 'categorical',
            'description': 'Qualité de l\'alimentation (1=Faible, 2=Moyenne, 3=Bonne, 4=Excellente, 5=Très excellente)',
            'values': {1: 'Faible', 2: 'Moyenne', 3: 'Bonne', 4: 'Excellente', 5: 'Très excellente'},
            'distribution': df['diet_quality'].value_counts().to_dict(),
            'missing_values': int(df['diet_quality'].isna().sum())
        },
        'alcohol_units_per_week': {
            'type': 'continuous',
            'description': 'Unités d\'alcool par semaine',
            'unit': 'unités/semaine',
            'range': [float(df['alcohol_units_per_week'].min()), float(df['alcohol_units_per_week'].max())],
            'mean': float(df['alcohol_units_per_week'].mean()),
            'std': float(df['alcohol_units_per_week'].std()),
            'missing_values': int(df['alcohol_units_per_week'].isna().sum())
        },
        'healthcare_access': {
            'type': 'categorical',
            'description': 'Accès aux soins de santé (1=Très limité, 2=Limité, 3=Moyen, 4=Bon)',
            'values': {1: 'Très limité', 2: 'Limité', 3: 'Moyen', 4: 'Bon'},
            'distribution': df['healthcare_access'].value_counts().to_dict(),
            'missing_values': int(df['healthcare_access'].isna().sum())
        },
        'lung_cancer_risk': {
            'type': 'binary',
            'description': 'Risque de cancer du poumon (variable cible) (0=Faible risque, 1=Risque élevé)',
            'values': {0: 'Faible risque', 1: 'Risque élevé'},
            'distribution': df['lung_cancer_risk'].value_counts().to_dict(),
            'missing_values': int(df['lung_cancer_risk'].isna().sum())
        }
    }
    
    # Statistiques globales
    stats = {
        'total_samples': len(df),
        'total_features': len(df.columns),
        'target_variable': 'lung_cancer_risk',
        'class_distribution': {
            'low_risk': int((df['lung_cancer_risk'] == 0).sum()),
            'high_risk': int((df['lung_cancer_risk'] == 1).sum()),
            'imbalance_ratio': float((df['lung_cancer_risk'] == 0).sum() / (df['lung_cancer_risk'] == 1).sum())
        },
        'missing_data': {
            'total_missing': int(df.isna().sum().sum()),
            'columns_with_missing': [col for col in df.columns if df[col].isna().sum() > 0]
        }
    }
    
    # Biais potentiels
    biases = {
        'gender_balance': {
            'description': 'Distribution du genre dans le dataset',
            'distribution': df['gender'].value_counts().to_dict(),
            'potential_bias': 'À vérifier si la distribution reflète la population réelle'
        },
        'age_distribution': {
            'description': 'Distribution de l\'âge',
            'mean': float(df['age'].mean()),
            'std': float(df['age'].std()),
            'potential_bias': 'Dataset peut être biaisé vers certaines tranches d\'âge'
        },
        'smoking_distribution': {
            'description': 'Distribution des fumeurs vs non-fumeurs',
            'distribution': df['smoker'].value_counts().to_dict(),
            'potential_bias': 'Proportion de fumeurs peut ne pas refléter la population générale'
        }
    }
    
    # Structure complète des Data Cards
    data_cards = {
        'metadata': {
            'dataset_name': 'Lung Cancer Risk Dataset',
            'version': '1.0',
            'creation_date': datetime.now().isoformat(),
            'source': str(csv_path),
            'description': 'Dataset pour l\'analyse du risque de cancer du poumon basé sur des facteurs démographiques, environnementaux et cliniques',
            'license': 'À définir',
            'citation': 'À définir'
        },
        'statistics': stats,
        'variables': variable_dict,
        'biases': biases,
        'data_quality': {
            'completeness': float((1 - df.isna().sum().sum() / (len(df) * len(df.columns))) * 100),
            'duplicates': int(df.duplicated().sum()),
            'outliers': 'À analyser'
        }
    }
    
    # Sauvegarder en YAML
    with open(output_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(data_cards, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    
    # Sauvegarder en JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(data_cards, f, indent=2, ensure_ascii=False)
    
    print(f"Data Cards generees avec succes!")
    print(f"YAML: {output_yaml}")
    print(f"JSON: {output_json}")


if __name__ == '__main__':
    base_dir = Path(__file__).parent.parent
    csv_path = base_dir / 'data' / 'raw' / 'lung_cancer.csv'
    output_yaml = base_dir / 'docs' / 'data_cards' / 'data_cards_complet.yaml'
    output_json = base_dir / 'docs' / 'data_cards' / 'data_cards_complet.json'
    
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    
    generate_data_cards(csv_path, output_yaml, output_json)
