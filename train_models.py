"""
train_models.py
================
Entrena y compara 5 modelos para predecir `prioridad` (ALTA/MEDIA/BAJA) de
incidentes en El Porvenir:

  Clásicos:  Regresión Logística, Random Forest, LightGBM
  Híbridos:  Híbrido 1 = Stacking(LR+RF+LightGBM) + meta-MLP
             Híbrido 2 = GA + MLP

Flujo: split train/test -> optimización de hiperparámetros (GridSearchCV /
Optuna / GA) sobre el train -> validación cruzada repetida (configurable) ->
ajuste final sobre todo el train -> métricas y figuras sobre el test -> guardar
modelos (.pkl / .h5).
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve, auc,
                              confusion_matrix)
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.features import FEATURE_COLS, CLASS_NAMES  # noqa: E402
from hyperparameter_search import (optimize_logistic_regression,
                                    optimize_random_forest, optimize_lightgbm)
from genetic_algorithm import GeneticAlgorithmMLP
from hybrid_models import build_keras_mlp, KerasMLPWrapper, StackingHybrid

# ------------------------------------------------------------------
# Configuración (todo lo "configurable" que pide el enunciado vive aquí)
# ------------------------------------------------------------------
RANDOM_STATE = 42
N_SPLITS = 5          # folds de la validación cruzada repetida (pedido: 5, configurable)
N_REPEATS = 1         # repeticiones -> súbelo a 2+ si tienes más CPU disponible
TEST_SIZE = 0.20       # hold-out final para figuras (ROC, matriz de confusión)
GRIDSEARCH_CV_SPLITS = 3
OPTUNA_TRIALS = 10     # súbelo a 20-50 con más cómputo disponible
GA_POBLACION = 4       # súbelo a 8-10 con más cómputo disponible
GA_GENERACIONES = 2    # súbelo a 4-6 con más cómputo disponible
# NOTA: estos valores están calibrados para correr en ~15-20 min en un
# entorno de 1 sola CPU. Si vas a correr esto en tu propia máquina o en un
# servidor con más núcleos, sube estos números para una búsqueda más fina.

DATA_PATH = Path("data/dataset_features.csv")
MODELS_DIR = Path("models")
FIGURES_DIR = Path("figuras_entrenamiento")
RESULTS_DIR = Path("resultados")

TARGET_COL = "prioridad_encoded"
# FEATURE_COLS y CLASS_NAMES ahora vienen de common/features.py (fuente única
# de verdad compartida con la API de inferencia)

sns.set_theme(style="whitegrid")


def cargar_datos():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLS].to_numpy(dtype="float64")
    y = df[TARGET_COL].to_numpy(dtype="int64")
    return X, y


# ------------------------------------------------------------------
# Evaluación por validación cruzada (idéntica para los 5 modelos)
# ------------------------------------------------------------------
def evaluar_cv(nombre, builder_fn, X, y, cv, needs_scaling=True):
    filas = []
    for fold_i, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        if needs_scaling:
            scaler = StandardScaler().fit(X_tr)
            X_tr, X_val = scaler.transform(X_tr), scaler.transform(X_val)

        X_tr_res, y_tr_res = SMOTE(random_state=RANDOM_STATE + fold_i).fit_resample(X_tr, y_tr)

        modelo = builder_fn()
        t0 = time.time()
        modelo.fit(X_tr_res, y_tr_res)
        tiempo = time.time() - t0

        y_pred = modelo.predict(X_val)
        y_proba = modelo.predict_proba(X_val)

        filas.append({
            "modelo": nombre, "fold": fold_i,
            "accuracy": accuracy_score(y_val, y_pred),
            "precision_macro": precision_score(y_val, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_val, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_val, y_pred, average="macro", zero_division=0),
            "roc_auc_macro": roc_auc_score(y_val, y_proba, multi_class="ovr", average="macro"),
            "tiempo_entrenamiento_s": tiempo,
        })
        print(f"    fold {fold_i + 1}/{cv.get_n_splits()} -> "
              f"F1={filas[-1]['f1_macro']:.4f}  tiempo={tiempo:.1f}s")
    return pd.DataFrame(filas)


def main():
    for d in [MODELS_DIR, FIGURES_DIR, RESULTS_DIR]:
        d.mkdir(exist_ok=True)

    print("Cargando datos...")
    X, y = cargar_datos()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # ----------------------------------------------------------------
    # 1. OPTIMIZACIÓN DE HIPERPARÁMETROS (sobre el train, con SMOTE interno)
    # ----------------------------------------------------------------
    print("\n=== Optimización de hiperparámetros ===")
    print("Regresión Logística (GridSearchCV)...")
    lr_params, lr_cv_score = optimize_logistic_regression(
        X_train, y_train, cv_splits=GRIDSEARCH_CV_SPLITS, random_state=RANDOM_STATE)

    print("Random Forest (GridSearchCV)...")
    rf_params, rf_cv_score = optimize_random_forest(
        X_train, y_train, cv_splits=GRIDSEARCH_CV_SPLITS, random_state=RANDOM_STATE)

    print("LightGBM (Optuna)...")
    lgbm_params, lgbm_cv_score = optimize_lightgbm(
        X_train, y_train, n_trials=OPTUNA_TRIALS, cv_splits=GRIDSEARCH_CV_SPLITS,
        random_state=RANDOM_STATE)

    print("Híbrido 2 - MLP (Algoritmo Genético)...")
    scaler_ga = StandardScaler().fit(X_train)
    X_train_scaled_for_ga = scaler_ga.transform(X_train)
    ga = GeneticAlgorithmMLP(build_keras_mlp, poblacion=GA_POBLACION,
                              generaciones=GA_GENERACIONES, muestra_fitness=3000,
                              epochs_fitness=8, random_state=RANDOM_STATE)
    ga_params = ga.run(X_train_scaled_for_ga, y_train)
    print(f"  [GA] Mejores hiperparámetros MLP -> {ga_params} (F1-macro={ga.mejor_fitness_:.4f})")

    optimizacion_log = {
        "logistic_regression": {"params": lr_params, "f1_macro_cv": lr_cv_score},
        "random_forest": {"params": rf_params, "f1_macro_cv": rf_cv_score},
        "lightgbm": {"params": lgbm_params, "f1_macro_cv": lgbm_cv_score},
        "ga_mlp": {"params": ga_params, "f1_macro_fitness": ga.mejor_fitness_,
                   "historial_generaciones": ga.historial},
    }
    with open(RESULTS_DIR / "hiperparametros_optimizados.json", "w", encoding="utf-8") as f:
        json.dump(optimizacion_log, f, indent=2, default=str, ensure_ascii=False)

    # ----------------------------------------------------------------
    # 2. CONSTRUCTORES DE MODELOS (instancia fresca por cada fold)
    # ----------------------------------------------------------------
    def build_lr():
        return LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, **lr_params)

    def build_rf():
        return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, **rf_params)

    def build_lgbm():
        return LGBMClassifier(**lgbm_params)

    def build_hybrid1():
        return StackingHybrid(base_estimators={
            "lr": build_lr(), "rf": build_rf(), "lgbm": build_lgbm(),
        }, random_state=RANDOM_STATE)

    def build_hybrid2():
        return KerasMLPWrapper(**ga_params, epochs=40)

    modelos = {
        "Regresion_Logistica": build_lr,
        "Random_Forest": build_rf,
        "LightGBM": build_lgbm,
        "Hibrido1_Stacking_MLP": build_hybrid1,
        "Hibrido2_GA_MLP": build_hybrid2,
    }

    # ----------------------------------------------------------------
    # 3. VALIDACIÓN CRUZADA REPETIDA (Repeated Stratified KFold, configurable)
    # ----------------------------------------------------------------
    print(f"\n=== Validación cruzada: RepeatedStratifiedKFold "
          f"(n_splits={N_SPLITS}, n_repeats={N_REPEATS}) ===")
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)

    resultados_todos = []
    for nombre, builder in modelos.items():
        print(f"\n  Modelo: {nombre}")
        df_res = evaluar_cv(nombre, builder, X_train, y_train, cv)
        resultados_todos.append(df_res)

    cv_results = pd.concat(resultados_todos, ignore_index=True)
    cv_results.to_csv(RESULTS_DIR / "cv_resultados_por_fold.csv", index=False)

    tabla_comparativa = cv_results.groupby("modelo").agg(
        ["mean", "std"]).round(4)
    tabla_comparativa.to_csv(RESULTS_DIR / "tabla_comparativa_modelos.csv")
    print("\n=== Tabla comparativa (media ± std sobre", cv.get_n_splits(), "folds) ===")
    print(tabla_comparativa)

    # ----------------------------------------------------------------
    # 4. AJUSTE FINAL sobre todo el train, evaluación sobre el TEST hold-out
    # ----------------------------------------------------------------
    print("\n=== Ajuste final sobre el conjunto de entrenamiento completo ===")
    scaler_final = StandardScaler().fit(X_train)
    X_train_scaled = scaler_final.transform(X_train)
    X_test_scaled = scaler_final.transform(X_test)
    X_train_res, y_train_res = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train_scaled, y_train)

    modelos_finales = {}
    y_proba_test = {}
    y_pred_test = {}
    for nombre, builder in modelos.items():
        print(f"  Entrenando modelo final: {nombre}")
        modelo = builder()
        modelo.fit(X_train_res, y_train_res)
        modelos_finales[nombre] = modelo
        y_proba_test[nombre] = modelo.predict_proba(X_test_scaled)
        y_pred_test[nombre] = modelo.predict(X_test_scaled)

    # --- guardar modelos ---
    with open(MODELS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler_final, f)

    import joblib
    joblib.dump(modelos_finales["Regresion_Logistica"], MODELS_DIR / "regresion_logistica.pkl")
    joblib.dump(modelos_finales["Random_Forest"], MODELS_DIR / "random_forest.pkl")
    joblib.dump(modelos_finales["LightGBM"], MODELS_DIR / "lightgbm.pkl")
    modelos_finales["Hibrido1_Stacking_MLP"].save(str(MODELS_DIR / "hibrido1_stacking_mlp"))
    modelos_finales["Hibrido2_GA_MLP"].save(str(MODELS_DIR / "hibrido2_ga_mlp.h5"))
    print(f"\nModelos guardados en {MODELS_DIR}/")

    # ----------------------------------------------------------------
    # 5. FIGURAS
    # ----------------------------------------------------------------
    print("\n=== Generando figuras ===")
    generar_figura_roc(y_test, y_proba_test, modelos.keys())
    generar_figura_matrices_confusion(y_test, y_pred_test, modelos.keys())
    generar_figura_importancia_variables(modelos_finales, X_test_scaled, y_test)
    generar_figura_comparacion_modelos(cv_results)

    # --- selección del mejor modelo por F1-macro promedio ---
    mejor_modelo = tabla_comparativa[("f1_macro", "mean")].idxmax()
    mejor_f1 = tabla_comparativa.loc[mejor_modelo, ("f1_macro", "mean")]
    print(f"\n>>> Mejor modelo según F1-macro (CV): {mejor_modelo} (F1={mejor_f1:.4f})")

    resumen_final = {
        "mejor_modelo": mejor_modelo,
        "f1_macro_cv_promedio": float(mejor_f1),
        "feature_cols": FEATURE_COLS,
        "class_names": CLASS_NAMES,
        "config": {"n_splits": N_SPLITS, "n_repeats": N_REPEATS, "test_size": TEST_SIZE,
                   "random_state": RANDOM_STATE},
    }
    with open(RESULTS_DIR / "resumen_entrenamiento.json", "w", encoding="utf-8") as f:
        json.dump(resumen_final, f, indent=2, ensure_ascii=False)

    print("\nListo. Revisa resultados/ , models/ y figuras_entrenamiento/")


# ------------------------------------------------------------------
# Figuras
# ------------------------------------------------------------------
def generar_figura_roc(y_test, y_proba_dict, nombres_modelos):
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    for ax, nombre in zip(axes, nombres_modelos):
        proba = y_proba_dict[nombre]
        for i, clase in enumerate(CLASS_NAMES):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], proba[:, i])
            ax.plot(fpr, tpr, label=f"{clase} (AUC={auc(fpr, tpr):.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_title(nombre.replace("_", " "), fontsize=10)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend(fontsize=7)
    plt.suptitle("Curvas ROC (One-vs-Rest) por modelo - conjunto de prueba")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_curvas_roc.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  guardado: {FIGURES_DIR}/01_curvas_roc.png")


def generar_figura_matrices_confusion(y_test, y_pred_dict, nombres_modelos):
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    for ax, nombre in zip(axes, nombres_modelos):
        cm = confusion_matrix(y_test, y_pred_dict[nombre])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar=False)
        ax.set_title(nombre.replace("_", " "), fontsize=10)
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
    plt.suptitle("Matrices de confusión por modelo - conjunto de prueba")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "02_matrices_confusion.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  guardado: {FIGURES_DIR}/02_matrices_confusion.png")


def permutation_importance_manual(modelo, X, y, n_repeats=5, random_state=42):
    """Permutation importance implementada a mano (no depende de la API
    interna de sklearn `__sklearn_tags__`, que nuestras clases custom
    -KerasMLPWrapper y StackingHybrid- no implementan por no heredar de
    BaseEstimator). Funciona igual para cualquier modelo con .predict(X)."""
    rng = np.random.default_rng(random_state)
    baseline = f1_score(y, modelo.predict(X), average="macro", zero_division=0)

    n_features = X.shape[1]
    importancias = np.zeros((n_repeats, n_features))
    for r in range(n_repeats):
        for j in range(n_features):
            X_perm = X.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            score_perm = f1_score(y, modelo.predict(X_perm), average="macro", zero_division=0)
            importancias[r, j] = baseline - score_perm
    return importancias.mean(axis=0), importancias.std(axis=0)


def generar_figura_importancia_variables(modelos_finales, X_test_scaled, y_test):
    # sub-muestra del test para acotar el costo computacional (1 CPU); con más
    # cómputo disponible se puede usar X_test_scaled/y_test completos.
    rng = np.random.default_rng(RANDOM_STATE)
    n_sample = min(2000, len(y_test))
    idx = rng.choice(len(y_test), n_sample, replace=False)
    X_imp, y_imp = X_test_scaled[idx], y_test[idx]

    fig, axes = plt.subplots(1, 5, figsize=(28, 6))
    for ax, (nombre, modelo) in zip(axes, modelos_finales.items()):
        medias, stds = permutation_importance_manual(
            modelo, X_imp, y_imp, n_repeats=3, random_state=RANDOM_STATE)
        orden = np.argsort(medias)[-10:]
        ax.barh(np.array(FEATURE_COLS)[orden], medias[orden],
                xerr=stds[orden], color="steelblue")
        ax.set_title(nombre.replace("_", " "), fontsize=10)
        ax.set_xlabel("Importancia (caída de F1-macro)")
    plt.suptitle("Importancia de variables (permutation importance) - top 10 por modelo")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "03_importancia_variables.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  guardado: {FIGURES_DIR}/03_importancia_variables.png")


def generar_figura_comparacion_modelos(cv_results):
    resumen = cv_results.groupby("modelo").agg(
        f1_mean=("f1_macro", "mean"), f1_std=("f1_macro", "std"),
        acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
        auc_mean=("roc_auc_macro", "mean"), auc_std=("roc_auc_macro", "std"),
        tiempo_mean=("tiempo_entrenamiento_s", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(len(resumen))
    width = 0.25
    axes[0].bar(x - width, resumen["acc_mean"], width, yerr=resumen["acc_std"], label="Accuracy")
    axes[0].bar(x, resumen["f1_mean"], width, yerr=resumen["f1_std"], label="F1-macro")
    axes[0].bar(x + width, resumen["auc_mean"], width, yerr=resumen["auc_std"], label="ROC-AUC")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(resumen["modelo"], rotation=30, ha="right")
    axes[0].set_title("Comparación de métricas (media ± std, CV)")
    axes[0].legend()

    axes[1].bar(x, resumen["tiempo_mean"], color="darkorange")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(resumen["modelo"], rotation=30, ha="right")
    axes[1].set_title("Tiempo de entrenamiento promedio por fold (s)")
    axes[1].set_ylabel("segundos")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_comparacion_modelos.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  guardado: {FIGURES_DIR}/04_comparacion_modelos.png")


if __name__ == "__main__":
    main()
