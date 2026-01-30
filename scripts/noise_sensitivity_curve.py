"""
TP - Courbe de sensibilité au bruit (Noise Sensitivity Curve)

- Prédiction puis bruitage d'une feature numérique
- Bruit = 1%, 3%, 5%, 10%, 20% de l'écart-type de la feature
- Métriques : MSE, F1-score pour chaque niveau de bruit
- Tracé de l'évolution de la MSE en fonction du pourcentage de bruit
- Répétition pour les 3 features numériques les plus corrélées à la cible
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score

# Répertoire du projet
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "lung_cancer_cleaned.csv"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "docs" / "visualizations" / "noise_sensitivity"
RANDOM_STATE = 42

# Niveaux de bruit (% de l'écart-type)
NOISE_LEVELS_PCT = [0, 1, 3, 5, 10, 20]


def prepare_features(df, target_variable):
    """Prépare X, y et la liste des features (même logique que train_models)."""
    exclude_cols = ['lung_cancer_risk', 'family_history_cancer', 'smoker']
    features = [col for col in df.columns if col not in exclude_cols]
    X = df[features]
    y = df[target_variable]
    return X, y, features


def get_numerical_features(X):
    """Retourne les noms des colonnes numériques (continues)."""
    return X.select_dtypes(include=[np.number]).columns.tolist()


def top_correlated_numerical_features(X, y, n=3):
    """
    Retourne les n features numériques les plus corrélées (en valeur absolue)
    avec la variable cible y.
    """
    numerical = get_numerical_features(X)
    if not numerical:
        return []
    corr = X[numerical].corrwith(y).abs().sort_values(ascending=False)
    return corr.head(n).index.tolist()


def add_noise_to_feature(X, feature_name, noise_pct, random_state=None):
    """
    Ajoute un bruit gaussien à la feature : bruit = (noise_pct/100) * std * N(0,1).
    X est modifié sur une copie, pas en place.
    """
    rng = np.random.default_rng(random_state)
    X_noisy = X.copy()
    col = X_noisy[feature_name]
    std_col = col.std()
    if std_col == 0:
        std_col = 1e-10  # éviter division par zéro
    noise = (noise_pct / 100.0) * std_col * rng.standard_normal(size=len(X_noisy))
    X_noisy[feature_name] = col + noise
    return X_noisy


def evaluate_noise_sensitivity(model, X_test, y_test, feature_name, noise_levels_pct, random_state=42):
    """
    Pour une feature donnée, applique chaque niveau de bruit, prédit et calcule
    MSE et F1-score. Retourne deux listes (mse_list, f1_list).
    """
    mse_list = []
    f1_list = []
    for pct in noise_levels_pct:
        X_noisy = add_noise_to_feature(X_test, feature_name, pct, random_state=random_state)
        y_pred = model.predict(X_noisy)
        mse = mean_squared_error(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mse_list.append(mse)
        f1_list.append(f1)
    return mse_list, f1_list


def run_noise_sensitivity_one_feature(model, X_test, y_test, feature_name, target_name):
    """Exécute l'analyse pour une feature et affiche les métriques + courbe MSE."""
    mse_list, f1_list = evaluate_noise_sensitivity(
        model, X_test, y_test, feature_name, NOISE_LEVELS_PCT
    )
    print(f"\n--- Feature: {feature_name} (cible: {target_name}) ---")
    print(f"{'Bruit %':>8} | {'MSE':>8} | {'F1-score':>10}")
    print("-" * 32)
    for i, pct in enumerate(NOISE_LEVELS_PCT):
        print(f"{pct:>7}% | {mse_list[i]:>8.4f} | {f1_list[i]:>10.4f}")
    return mse_list, f1_list


def plot_noise_sensitivity_curves(results_by_feature, output_path):
    """
    Trace la courbe "Noise sensitivity curve" : MSE en fonction du % de bruit.
    results_by_feature : dict { feature_name: (noise_levels, mse_list) }
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for feat_name, (levels, mse_list) in results_by_feature.items():
        ax.plot(levels, mse_list, marker='o', label=feat_name.replace('_', ' ').title(), linewidth=2)
    ax.set_xlabel("Bruit (% de l'écart-type)", fontsize=12)
    ax.set_ylabel("MSE (Mean Squared Error)", fontsize=12)
    ax.set_title("Noise Sensitivity Curve\nÉvolution de la MSE selon le bruit ajouté à la feature", fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(NOISE_LEVELS_PCT)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[OK] Courbe sauvegardée : {output_path}")


def main():
    print("=" * 60)
    print("TP - Courbe de sensibilité au bruit (Noise Sensitivity Curve)")
    print("=" * 60)

    if not DATA_PATH.exists():
        print(f"[ERREUR] Fichier {DATA_PATH} introuvable. Exécutez d'abord preprocess_data.py")
        return

    # Charger les données
    df = pd.read_csv(DATA_PATH)
    target_var = "smoker"  # modèle avec de bonnes métriques
    X, y, feature_names = prepare_features(df, target_var)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Charger le modèle
    model_path = MODELS_DIR / f"model_{target_var}.pkl"
    if not model_path.exists():
        print(f"[ERREUR] Modèle {model_path} introuvable. Exécutez train_models.py")
        return

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    # Éviter les erreurs de multiprocessing (sandbox / Windows)
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1

    numerical_features = get_numerical_features(X)
    top3 = top_correlated_numerical_features(X, y, n=3)
    print(f"\nFeatures numériques les plus corrélées avec '{target_var}' (top 3) : {top3}")

    # --- Étape 1 : une première feature (la plus corrélée) ---
    first_feature = top3[0] if top3 else numerical_features[0]
    print(f"\n[1] Analyse sur une feature : {first_feature}")
    run_noise_sensitivity_one_feature(model, X_test, y_test, first_feature, target_var)

    # --- Étape 2 : les 3 features les plus corrélées ---
    results_for_plot = {}
    for feat in top3:
        mse_list, f1_list = evaluate_noise_sensitivity(
            model, X_test, y_test, feat, NOISE_LEVELS_PCT
        )
        results_for_plot[feat] = (NOISE_LEVELS_PCT, mse_list)
        print(f"\n--- Top feature: {feat} ---")
        print(f"{'Bruit %':>8} | {'MSE':>8} | {'F1-score':>10}")
        for i, pct in enumerate(NOISE_LEVELS_PCT):
            print(f"{pct:>7}% | {mse_list[i]:>8.4f} | {f1_list[i]:>10.4f}")

    # Tracer la courbe de sensibilité au bruit (MSE vs % bruit)
    plot_path = OUTPUT_DIR / "noise_sensitivity_curve.png"
    plot_noise_sensitivity_curves(results_for_plot, plot_path)

    print("\n" + "=" * 60)
    print("TP terminé.")
    print("=" * 60)


if __name__ == "__main__":
    main()
