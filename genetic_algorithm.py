"""
genetic_algorithm.py
=====================
Algoritmo Genético (GA) simple, implementado desde cero con numpy, para
optimizar la arquitectura/hiperparámetros de una red neuronal MLP (Keras).
Es el método de optimización del Híbrido 2 (GA + MLP).

Cromosoma (gen) = diccionario de hiperparámetros:
    n_capas         : 1, 2 o 3 capas ocultas
    unidades        : neuronas por capa (mismo valor para todas las capas)
    dropout         : tasa de dropout
    learning_rate   : tasa de aprendizaje del optimizador Adam
    activacion      : función de activación

Fitness = F1-macro en una validación rápida (hold-out) sobre una muestra del
conjunto de entrenamiento, para mantener el costo computacional razonable.
"""

from __future__ import annotations
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

SEARCH_SPACE = {
    "n_capas": [1, 2, 3],
    "unidades": [16, 32, 64, 128],
    "dropout": [0.0, 0.1, 0.2, 0.3],
    "learning_rate": [0.01, 0.005, 0.001, 0.0005],
    "activacion": ["relu", "tanh"],
}


def _random_individuo(rng: np.random.Generator) -> dict:
    return {k: v[rng.integers(len(v))] for k, v in SEARCH_SPACE.items()}


def _crossover(padre: dict, madre: dict, rng: np.random.Generator) -> dict:
    hijo = {}
    for k in SEARCH_SPACE:
        hijo[k] = padre[k] if rng.random() < 0.5 else madre[k]
    return hijo


def _mutar(individuo: dict, rng: np.random.Generator, p_mutacion: float = 0.25) -> dict:
    individuo = individuo.copy()
    for k, v in SEARCH_SPACE.items():
        if rng.random() < p_mutacion:
            individuo[k] = v[rng.integers(len(v))]
    return individuo


class GeneticAlgorithmMLP:
    """GA que busca hiperparámetros de un MLP Keras maximizando F1-macro."""

    def __init__(self, build_mlp_fn, poblacion=8, generaciones=4,
                 muestra_fitness=6000, epochs_fitness=12, random_state=42):
        self.build_mlp_fn = build_mlp_fn  # callable(input_dim, n_classes, **hparams) -> keras.Model
        self.poblacion_size = poblacion
        self.generaciones = generaciones
        self.muestra_fitness = muestra_fitness
        self.epochs_fitness = epochs_fitness
        self.rng = np.random.default_rng(random_state)
        self.historial = []  # (generacion, mejor_fitness, promedio_fitness)
        self.mejor_individuo_ = None
        self.mejor_fitness_ = -np.inf

    def _fitness(self, individuo: dict, X: np.ndarray, y: np.ndarray) -> float:
        n = len(X)
        if n > self.muestra_fitness:
            idx = self.rng.choice(n, self.muestra_fitness, replace=False)
            X, y = X[idx], y[idx]

        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=42)

        n_classes = len(np.unique(y))
        model = self.build_mlp_fn(X.shape[1], n_classes, **individuo)
        model.fit(X_tr, y_tr, epochs=self.epochs_fitness, batch_size=128,
                  verbose=0, validation_split=0.0)
        y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
        return f1_score(y_val, y_pred, average="macro")

    def run(self, X: np.ndarray, y: np.ndarray) -> dict:
        poblacion = [_random_individuo(self.rng) for _ in range(self.poblacion_size)]

        for gen in range(self.generaciones):
            t0 = time.time()
            fitnesses = [self._fitness(ind, X, y) for ind in poblacion]

            idx_mejor = int(np.argmax(fitnesses))
            if fitnesses[idx_mejor] > self.mejor_fitness_:
                self.mejor_fitness_ = fitnesses[idx_mejor]
                self.mejor_individuo_ = poblacion[idx_mejor]

            self.historial.append({
                "generacion": gen + 1,
                "mejor_fitness": float(np.max(fitnesses)),
                "promedio_fitness": float(np.mean(fitnesses)),
                "tiempo_s": round(time.time() - t0, 1),
            })
            print(f"  [GA] Gen {gen + 1}/{self.generaciones} - "
                  f"mejor F1={np.max(fitnesses):.4f} - promedio={np.mean(fitnesses):.4f} "
                  f"({time.time() - t0:.1f}s)")

            # selección por torneo + elitismo (mejor pasa directo)
            tam_torneo = min(3, len(poblacion))
            nueva_poblacion = [poblacion[idx_mejor]]
            while len(nueva_poblacion) < self.poblacion_size:
                candidatos = self.rng.choice(len(poblacion), size=tam_torneo, replace=False)
                padre = poblacion[candidatos[np.argmax([fitnesses[c] for c in candidatos])]]
                candidatos = self.rng.choice(len(poblacion), size=tam_torneo, replace=False)
                madre = poblacion[candidatos[np.argmax([fitnesses[c] for c in candidatos])]]
                hijo = _mutar(_crossover(padre, madre, self.rng), self.rng)
                nueva_poblacion.append(hijo)
            poblacion = nueva_poblacion

        return self.mejor_individuo_
