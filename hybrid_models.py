"""
hybrid_models.py
================
Componentes de los dos modelos híbridos:

  Híbrido 1 (Stacking + MLP): 3 aprendices base (Regresión Logística, Random
  Forest, LightGBM, ya optimizados) generan probabilidades out-of-fold que
  alimentan un meta-modelo MLP (red neuronal) que aprende a combinarlas.

  Híbrido 2 (GA + MLP): una red neuronal MLP entrenada directamente sobre
  todas las features, cuya arquitectura/hiperparámetros fueron optimizados
  con un Algoritmo Genético (ver genetic_algorithm.py).

Ambos exponen una interfaz consistente fit/predict/predict_proba para poder
evaluarse exactamente igual que los modelos clásicos dentro del mismo bucle
de validación cruzada.
"""

from __future__ import annotations
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import tensorflow as tf
from tensorflow import keras


def build_keras_mlp(input_dim: int, n_classes: int, n_capas=2, unidades=64,
                     dropout=0.2, learning_rate=0.001, activacion="relu") -> keras.Model:
    """Fábrica de un MLP Keras simple, parametrizable (usada por ambos híbridos)."""
    model = keras.Sequential(name="mlp")
    model.add(keras.layers.Input(shape=(input_dim,)))
    for _ in range(n_capas):
        model.add(keras.layers.Dense(unidades, activation=activacion))
        if dropout > 0:
            model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(n_classes, activation="softmax"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


class KerasMLPWrapper:
    """Envoltorio sklearn-like (fit/predict/predict_proba/save) sobre un
    modelo Keras, para poder usarlo en los mismos bucles de evaluación que
    los modelos scikit-learn."""

    def __init__(self, n_capas=2, unidades=64, dropout=0.2, learning_rate=0.001,
                 activacion="relu", epochs=40, batch_size=128, verbose=0):
        self.n_capas = n_capas
        self.unidades = unidades
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.activacion = activacion
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model_: keras.Model | None = None
        self.classes_ = None

    def get_params(self):
        return dict(n_capas=self.n_capas, unidades=self.unidades, dropout=self.dropout,
                     learning_rate=self.learning_rate, activacion=self.activacion,
                     epochs=self.epochs, batch_size=self.batch_size)

    def clone(self) -> "KerasMLPWrapper":
        return KerasMLPWrapper(**self.get_params())

    def fit(self, X, y):
        X = np.asarray(X, dtype="float32")
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        early_stop = keras.callbacks.EarlyStopping(
            monitor="loss", patience=5, restore_best_weights=True)

        self.model_ = build_keras_mlp(
            X.shape[1], n_classes, self.n_capas, self.unidades,
            self.dropout, self.learning_rate, self.activacion)
        self.model_.fit(X, y, epochs=self.epochs, batch_size=self.batch_size,
                         verbose=self.verbose, callbacks=[early_stop])
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype="float32")
        return self.model_.predict(X, verbose=0)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def save(self, path: str):
        self.model_.save(path)  # extensión .h5 o .keras

    @classmethod
    def load(cls, path: str, classes: np.ndarray) -> "KerasMLPWrapper":
        """Reconstruye el wrapper a partir de un .h5 guardado (para inferencia,
        sin necesidad de reentrenar)."""
        obj = cls()
        obj.model_ = keras.models.load_model(path)
        obj.classes_ = np.asarray(classes)
        return obj


class StackingHybrid:
    """Híbrido 1: Stacking (LR + RF + LightGBM) con meta-learner MLP (Keras).

    Usa cross_val_predict interno sobre el fold de entrenamiento para generar
    probabilidades out-of-fold de los 3 aprendices base (evita que el
    meta-modelo se sobreajuste viendo predicciones "de memoria").
    """

    def __init__(self, base_estimators: dict, meta_hparams: dict | None = None,
                 internal_cv=3, random_state=42):
        self.base_estimators = base_estimators  # dict nombre -> estimador (ya con mejores hparams)
        self.meta_hparams = meta_hparams or dict(n_capas=1, unidades=16, dropout=0.1,
                                                  learning_rate=0.005, epochs=30)
        self.internal_cv = internal_cv
        self.random_state = random_state
        self.fitted_base_ = {}
        self.meta_model_: KerasMLPWrapper | None = None
        self.classes_ = None

    def clone(self) -> "StackingHybrid":
        return StackingHybrid(
            {k: clone(v) for k, v in self.base_estimators.items()},
            self.meta_hparams.copy(), self.internal_cv, self.random_state)

    def _meta_features(self, base_probas: list[np.ndarray]) -> np.ndarray:
        return np.hstack(base_probas)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        cv = StratifiedKFold(n_splits=self.internal_cv, shuffle=True, random_state=self.random_state)

        oof_probas = []
        for name, est in self.base_estimators.items():
            oof = cross_val_predict(clone(est), X, y, cv=cv, method="predict_proba")
            oof_probas.append(oof)
            # ahora sí, entrenar el aprendiz base en TODO el fold para usarlo en predict()
            fitted = clone(est)
            fitted.fit(X, y)
            self.fitted_base_[name] = fitted

        meta_X = self._meta_features(oof_probas)
        self.meta_model_ = KerasMLPWrapper(**self.meta_hparams)
        self.meta_model_.fit(meta_X, y)
        return self

    def predict_proba(self, X):
        base_probas = [est.predict_proba(X) for est in self.fitted_base_.values()]
        meta_X = self._meta_features(base_probas)
        return self.meta_model_.predict_proba(meta_X)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def save(self, dir_path: str):
        import os, joblib
        os.makedirs(dir_path, exist_ok=True)
        for name, est in self.fitted_base_.items():
            joblib.dump(est, os.path.join(dir_path, f"base_{name}.pkl"))
        self.meta_model_.save(os.path.join(dir_path, "meta_mlp.h5"))

    @classmethod
    def load(cls, dir_path: str, classes: np.ndarray) -> "StackingHybrid":
        """Reconstruye el híbrido de stacking a partir de los .pkl (base
        learners) y el .h5 (meta MLP) guardados por `save()`."""
        import os, joblib
        obj = cls(base_estimators={})
        obj.classes_ = np.asarray(classes)
        obj.fitted_base_ = {}
        for fname in sorted(os.listdir(dir_path)):
            if fname.startswith("base_") and fname.endswith(".pkl"):
                nombre = fname[len("base_"):-len(".pkl")]
                obj.fitted_base_[nombre] = joblib.load(os.path.join(dir_path, fname))
        obj.meta_model_ = KerasMLPWrapper.load(
            os.path.join(dir_path, "meta_mlp.h5"), classes=obj.classes_)
        return obj
