"""
hyperparameter_search.py
=========================
Optimización de hiperparámetros:
  - Regresión Logística y Random Forest -> GridSearchCV
  - LightGBM -> Optuna (búsqueda bayesiana)

Todo aplicado dentro de un Pipeline con SMOTE (imblearn) para que el
sobremuestreo de la clase minoritaria (MEDIA) ocurra SOLO dentro de cada
fold de entrenamiento y no contamine la validación.
"""

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier


def optimize_logistic_regression(X, y, cv_splits=3, random_state=42):
    pipe = ImbPipeline([
        ("smote", SMOTE(random_state=random_state)),
        ("clf", LogisticRegression(max_iter=2000, random_state=random_state)),
    ])
    param_grid = {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs"],
    }
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    search = GridSearchCV(pipe, param_grid, scoring="f1_macro", cv=cv, n_jobs=1)
    search.fit(X, y)
    best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    print(f"  [GridSearchCV] Regresión Logística -> {best_params} (F1-macro CV={search.best_score_:.4f})")
    return best_params, search.best_score_


def optimize_random_forest(X, y, cv_splits=3, random_state=42):
    pipe = ImbPipeline([
        ("smote", SMOTE(random_state=random_state)),
        ("clf", RandomForestClassifier(random_state=random_state, n_jobs=1)),
    ])
    param_grid = {
        "clf__n_estimators": [150, 300],
        "clf__max_depth": [10, 20],
        "clf__min_samples_split": [2, 5],
    }
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    search = GridSearchCV(pipe, param_grid, scoring="f1_macro", cv=cv, n_jobs=1)
    search.fit(X, y)
    best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    print(f"  [GridSearchCV] Random Forest -> {best_params} (F1-macro CV={search.best_score_:.4f})")
    return best_params, search.best_score_


def optimize_lightgbm(X, y, n_trials=20, cv_splits=3, random_state=42):
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
            "random_state": random_state,
            "n_jobs": 1,
            "verbose": -1,
        }
        scores = []
        for train_idx, val_idx in cv.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            X_tr_res, y_tr_res = SMOTE(random_state=random_state).fit_resample(X_tr, y_tr)
            model = LGBMClassifier(**params)
            model.fit(X_tr_res, y_tr_res)
            from sklearn.metrics import f1_score
            scores.append(f1_score(y_val, model.predict(X_val), average="macro"))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_params
    best_params.update({"random_state": random_state, "verbose": -1, "n_jobs": 1})
    print(f"  [Optuna] LightGBM -> {best_params} (F1-macro CV={study.best_value:.4f})")
    return best_params, study.best_value
