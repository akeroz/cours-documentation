"""
Script de nettoyage et normalisation des données pour l'entraînement des modèles
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def clean_and_normalize_data(input_path, output_path, lineage_path, doc_path):
    """
    Nettoie et normalise les données, puis exporte le dataset nettoyé
    avec documentation et lineage
    """
    print("=" * 80)
    print("NETTOYAGE ET NORMALISATION DES DONNÉES")
    print("=" * 80)
    
    # Charger les données brutes
    print("\n[1/5] Chargement des données brutes...")
    df = pd.read_csv(input_path)
    print(f"    [OK] {len(df)} lignes chargees")
    print(f"    [OK] {len(df.columns)} colonnes")
    
    # Initialiser le lineage
    lineage = {
        "date_creation": datetime.now().isoformat(),
        "source": str(input_path),
        "etapes": [],
        "transformations": {},
        "statistiques_avant": {},
        "statistiques_apres": {}
    }
    
    # Statistiques avant nettoyage
    lineage["statistiques_avant"] = {
        "nombre_lignes": len(df),
        "nombre_colonnes": len(df.columns),
        "valeurs_manquantes": int(df.isnull().sum().sum()),
        "lignes_dupliquees": int(df.duplicated().sum())
    }
    
    # ÉTAPE 1: Vérification des valeurs manquantes
    print("\n[2/5] Vérification des valeurs manquantes...")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"    [ATTENTION] {missing.sum()} valeurs manquantes detectees")
        lineage["etapes"].append({
            "etape": "verification_valeurs_manquantes",
            "date": datetime.now().isoformat(),
            "resultat": f"{missing.sum()} valeurs manquantes trouvées",
            "action": "Aucune action nécessaire (valeurs manquantes acceptables)"
        })
    else:
        print("    [OK] Aucune valeur manquante")
        lineage["etapes"].append({
            "etape": "verification_valeurs_manquantes",
            "date": datetime.now().isoformat(),
            "resultat": "Aucune valeur manquante",
            "action": "Aucune action"
        })
    
    # ÉTAPE 2: Suppression des doublons
    print("\n[3/5] Suppression des doublons...")
    duplicates_before = df.duplicated().sum()
    if duplicates_before > 0:
        df = df.drop_duplicates()
        print(f"    [OK] {duplicates_before} doublons supprimes")
        lineage["etapes"].append({
            "etape": "suppression_doublons",
            "date": datetime.now().isoformat(),
            "resultat": f"{duplicates_before} doublons supprimés",
            "action": "drop_duplicates()"
        })
    else:
        print("    [OK] Aucun doublon detecte")
        lineage["etapes"].append({
            "etape": "suppression_doublons",
            "date": datetime.now().isoformat(),
            "resultat": "Aucun doublon",
            "action": "Aucune action"
        })
    
    # ÉTAPE 3: Détection et gestion des valeurs aberrantes (sommaire)
    print("\n[4/5] Détection des valeurs aberrantes (sommaire)...")
    outliers_info = {}
    
    # Variables numériques à vérifier
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclure les variables binaires et la variable cible
    numeric_cols = [col for col in numeric_cols 
                   if col not in ['gender', 'smoker', 'passive_smoking', 'occupational_exposure', 
                                 'radon_exposure', 'family_history_cancer', 'copd', 'asthma', 
                                 'previous_tb', 'chronic_cough', 'chest_pain', 'shortness_of_breath',
                                 'fatigue', 'xray_abnormal', 'lung_cancer_risk']]
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR  # Utilisation de 3*IQR pour être moins strict
        upper_bound = Q3 + 3 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if len(outliers) > 0:
            outliers_info[col] = {
                "nombre": len(outliers),
                "pourcentage": round(len(outliers) / len(df) * 100, 2),
                "borne_inf": round(lower_bound, 2),
                "borne_sup": round(upper_bound, 2)
            }
    
    if outliers_info:
        print(f"    [ATTENTION] Valeurs aberrantes detectees dans {len(outliers_info)} variables")
        print("    [INFO] Conservation des valeurs aberrantes (nettoyage sommaire)")
        lineage["etapes"].append({
            "etape": "detection_valeurs_aberrantes",
            "date": datetime.now().isoformat(),
            "resultat": f"Valeurs aberrantes détectées dans {len(outliers_info)} variables",
            "action": "Conservation des valeurs (nettoyage sommaire)",
            "details": outliers_info
        })
    else:
        print("    [OK] Aucune valeur aberrante detectee")
        lineage["etapes"].append({
            "etape": "detection_valeurs_aberrantes",
            "date": datetime.now().isoformat(),
            "resultat": "Aucune valeur aberrante",
            "action": "Aucune action"
        })
    
    # ÉTAPE 4: Normalisation des variables numériques
    print("\n[5/5] Normalisation des variables numériques...")
    
    # Séparer les variables numériques et catégorielles
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclure les variables binaires et la variable cible
    binary_cols = ['gender', 'smoker', 'passive_smoking', 'occupational_exposure', 
                   'radon_exposure', 'family_history_cancer', 'copd', 'asthma', 
                   'previous_tb', 'chronic_cough', 'chest_pain', 'shortness_of_breath',
                   'fatigue', 'xray_abnormal']
    
    # Variables numériques à normaliser (exclure binaires et cible)
    cols_to_normalize = [col for col in numeric_features 
                        if col not in binary_cols + ['lung_cancer_risk']]
    
    # Sauvegarder les valeurs avant normalisation pour le lineage
    df_normalized = df.copy()
    
    # Normalisation StandardScaler (moyenne=0, écart-type=1)
    scaler = StandardScaler()
    df_normalized[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
    
    print(f"    [OK] {len(cols_to_normalize)} variables normalisees (StandardScaler)")
    print(f"    [OK] Variables normalisees: {', '.join(cols_to_normalize[:5])}...")
    
    lineage["transformations"]["normalisation"] = {
        "methode": "StandardScaler (sklearn)",
        "variables_normalisees": cols_to_normalize,
        "nombre_variables": len(cols_to_normalize),
        "description": "Normalisation z-score: (x - mean) / std"
    }
    
    lineage["etapes"].append({
        "etape": "normalisation",
        "date": datetime.now().isoformat(),
        "resultat": f"{len(cols_to_normalize)} variables normalisées",
        "action": "StandardScaler.fit_transform()",
        "variables": cols_to_normalize
    })
    
    # Statistiques après nettoyage
    lineage["statistiques_apres"] = {
        "nombre_lignes": len(df_normalized),
        "nombre_colonnes": len(df_normalized.columns),
        "valeurs_manquantes": int(df_normalized.isnull().sum().sum()),
        "lignes_dupliquees": int(df_normalized.duplicated().sum())
    }
    
    # Sauvegarder le dataset nettoyé
    print("\n" + "=" * 80)
    print("EXPORT DU DATASET NETTOYÉ")
    print("=" * 80)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_normalized.to_csv(output_path, index=False)
    print(f"\n[OK] Dataset nettoye sauvegarde: {output_path}")
    print(f"  - {len(df_normalized)} lignes")
    print(f"  - {len(df_normalized.columns)} colonnes")
    
    # Sauvegarder le lineage
    lineage_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lineage_path, 'w', encoding='utf-8') as f:
        json.dump(lineage, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Lineage sauvegarde: {lineage_path}")
    
    # Créer la documentation
    doc_content = f"""# Documentation du Nettoyage et Normalisation des Données

**Date de création:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Source:** `{input_path}`  
**Destination:** `{output_path}`

---

## Vue d'ensemble

Ce document décrit les étapes de nettoyage et de normalisation appliquées au dataset `lung_cancer.csv` en vue de l'entraînement des modèles de prédiction.

---

## Étapes de Nettoyage

### 1. Vérification des valeurs manquantes
- **Résultat:** {lineage['statistiques_avant']['valeurs_manquantes']} valeur(s) manquante(s) détectée(s)
- **Action:** {'Aucune action nécessaire' if lineage['statistiques_avant']['valeurs_manquantes'] == 0 else 'Analyse des valeurs manquantes'}

### 2. Suppression des doublons
- **Doublons détectés:** {lineage['statistiques_avant']['lignes_dupliquees']}
- **Action:** {'Aucune action (pas de doublons)' if lineage['statistiques_avant']['lignes_dupliquees'] == 0 else 'Suppression des doublons avec drop_duplicates()'}
- **Lignes après suppression:** {lineage['statistiques_apres']['nombre_lignes']}

### 3. Détection des valeurs aberrantes
- **Méthode:** Méthode IQR (Interquartile Range) avec seuil à 3×IQR
- **Résultat:** {'Aucune valeur aberrante détectée' if not outliers_info else f"Valeurs aberrantes détectées dans {len(outliers_info)} variables"}
- **Action:** Conservation des valeurs (nettoyage sommaire)

### 4. Normalisation des variables numériques
- **Méthode:** StandardScaler (normalisation z-score)
- **Formule:** (x - moyenne) / écart-type
- **Variables normalisées:** {len(cols_to_normalize)} variables
  {chr(10).join(f'  - {col}' for col in cols_to_normalize[:10])}
  {'  ...' if len(cols_to_normalize) > 10 else ''}
- **Variables non normalisées:** Variables binaires (0/1) et variables catégorielles conservées telles quelles

---

## Statistiques

### Avant nettoyage
- **Nombre de lignes:** {lineage['statistiques_avant']['nombre_lignes']}
- **Nombre de colonnes:** {lineage['statistiques_avant']['nombre_colonnes']}
- **Valeurs manquantes:** {lineage['statistiques_avant']['valeurs_manquantes']}
- **Lignes dupliquées:** {lineage['statistiques_avant']['lignes_dupliquees']}

### Après nettoyage
- **Nombre de lignes:** {lineage['statistiques_apres']['nombre_lignes']}
- **Nombre de colonnes:** {lineage['statistiques_apres']['nombre_colonnes']}
- **Valeurs manquantes:** {lineage['statistiques_apres']['valeurs_manquantes']}
- **Lignes dupliquées:** {lineage['statistiques_apres']['lignes_dupliquees']}

---

## Lineage

Le lineage complet des transformations est disponible dans: `{lineage_path}`

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
df = pd.read_csv('{output_path}')
```

**Note:** Les variables numériques sont normalisées (moyenne=0, écart-type=1).  
Les variables binaires et catégorielles conservent leurs valeurs originales.

---

*Document généré automatiquement le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    with open(doc_path, 'w', encoding='utf-8') as f:
        f.write(doc_content)
    print(f"[OK] Documentation sauvegardee: {doc_path}")
    
    print("\n" + "=" * 80)
    print("[OK] NETTOYAGE ET NORMALISATION TERMINES")
    print("=" * 80)
    
    return df_normalized, lineage


def main():
    """Fonction principale"""
    # Définir les chemins
    base_dir = Path(__file__).parent.parent
    input_path = base_dir / "data" / "raw" / "lung_cancer.csv"
    output_path = base_dir / "data" / "processed" / "lung_cancer_cleaned.csv"
    lineage_path = base_dir / "data" / "processed" / "lineage.json"
    doc_path = base_dir / "docs" / "preprocessing" / "documentation_nettoyage.md"
    
    try:
        df_cleaned, lineage = clean_and_normalize_data(
            input_path, output_path, lineage_path, doc_path
        )
        print(f"\n[OK] Succes! Dataset nettoye pret pour l'entrainement.")
        return df_cleaned, lineage
    except Exception as e:
        print(f"\n[ERREUR] Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    main()
