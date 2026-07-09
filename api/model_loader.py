"""
api/model_loader.py
=====================
Repository (patrón Repository) para los modelos entrenados: los carga UNA
vez (al iniciar la API) y expone una interfaz simple de predicción, sin que
el resto de la app necesite saber si un modelo es un .pkl de sklearn o un
.h5 de Keras/stacking.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.features import CLASS_NAMES, FEATURE_COLS, build_feature_vector, feature_dict_to_array  # noqa: E402
from config import settings  # noqa: E402
from exceptions import ModeloNoDisponibleError, PrediccionError  # noqa: E402
from hybrid_models import KerasMLPWrapper, StackingHybrid  # noqa: E402
from logging_config import configurar_logging  # noqa: E402

logger = configurar_logging(__name__)

CLASSES_ARRAY = np.array([0, 1, 2])  # ALTA=0, MEDIA=1, BAJA=2 (ver common.features.CLASS_NAMES)


class ModelRepository:
    """Carga y expone los 5 modelos entrenados + el scaler + las métricas de
    CV. Pensado para instanciarse UNA vez (singleton) al arrancar la API."""

    def __init__(self, models_dir: Path | None = None, resultados_dir: Path | None = None):
        self.models_dir = models_dir or settings.models_dir
        self.resultados_dir = resultados_dir or settings.resultados_dir
        self._modelos: dict[str, object] = {}
        self._scaler = None
        self._metricas: dict[str, dict] = {}
        self._cargado = False

    @property
    def cargado(self) -> bool:
        return self._cargado

    @property
    def modelos_disponibles(self) -> list[str]:
        return list(self._modelos.keys())

    def cargar(self) -> None:
        if self._cargado:
            return
        logger.info("Cargando modelos desde %s", self.models_dir)

        scaler_path = self.models_dir / "scaler.pkl"
        if not scaler_path.exists():
            raise ModeloNoDisponibleError(
                "No se encontró el scaler entrenado.", detalle={"ruta": str(scaler_path)})
        with open(scaler_path, "rb") as f:
            self._scaler = pickle.load(f)

        try:
            self._modelos["Regresion_Logistica"] = joblib.load(self.models_dir / "regresion_logistica.pkl")
            self._modelos["Random_Forest"] = joblib.load(self.models_dir / "random_forest.pkl")
            self._modelos["LightGBM"] = joblib.load(self.models_dir / "lightgbm.pkl")
            self._modelos["Hibrido1_Stacking_MLP"] = StackingHybrid.load(
                str(self.models_dir / "hibrido1_stacking_mlp"), classes=CLASSES_ARRAY)
            self._modelos["Hibrido2_GA_MLP"] = KerasMLPWrapper.load(
                str(self.models_dir / "hibrido2_ga_mlp.h5"), classes=CLASSES_ARRAY)
        except FileNotFoundError as exc:
            raise ModeloNoDisponibleError(
                f"Falta un archivo de modelo: {exc}. ¿Corriste train_models.py?"
            ) from exc

        self._cargar_metricas()
        self._cargado = True
        logger.info("Modelos cargados: %s", self.modelos_disponibles)

    def _cargar_metricas(self) -> None:
        path = self.resultados_dir / "tabla_comparativa_modelos.csv"
        if not path.exists():
            logger.warning("No se encontró %s; /models no tendrá métricas de CV", path)
            return
        import pandas as pd
        tabla = pd.read_csv(path, header=[0, 1], index_col=0)
        for modelo in tabla.index:
            self._metricas[modelo] = {
                "f1_macro_cv": float(tabla.loc[modelo, ("f1_macro", "mean")]),
                "accuracy_cv": float(tabla.loc[modelo, ("accuracy", "mean")]),
                "roc_auc_cv": float(tabla.loc[modelo, ("roc_auc_macro", "mean")]),
            }

    def metricas(self, nombre_modelo: str) -> dict:
        return self._metricas.get(nombre_modelo, {})

    def todas_las_metricas(self) -> dict[str, dict]:
        return self._metricas

    def _obtener_modelo(self, nombre_modelo: str):
        if nombre_modelo not in self._modelos:
            raise ModeloNoDisponibleError(
                f"Modelo '{nombre_modelo}' no disponible.",
                detalle={"modelos_disponibles": self.modelos_disponibles})
        return self._modelos[nombre_modelo]

    def predecir(self, latitud: float, longitud: float, hora: int, dia_semana: int,
                 mes: int, tipo: str, nombre_modelo: str | None = None) -> dict:
        """Ejecuta la inferencia completa: features -> escalado -> predicción."""
        nombre_modelo = nombre_modelo or settings.api_default_model
        modelo = self._obtener_modelo(nombre_modelo)

        t0 = time.perf_counter()
        try:
            features_dict = build_feature_vector(latitud, longitud, hora, dia_semana, mes, tipo)
            X = feature_dict_to_array(features_dict)
            X_scaled = self._scaler.transform(X)
            proba = modelo.predict_proba(X_scaled)[0]
            pred_idx = int(np.argmax(proba))
        except (ValueError, KeyError) as exc:
            raise PrediccionError(f"Error al predecir: {exc}") from exc

        tiempo_ms = (time.perf_counter() - t0) * 1000
        return {
            "prioridad": CLASS_NAMES[pred_idx],
            "confianza": float(proba[pred_idx]),
            "probabilidades": {CLASS_NAMES[i]: float(p) for i, p in enumerate(proba)},
            "modelo_usado": nombre_modelo,
            "tiempo_inferencia_ms": round(tiempo_ms, 3),
        }


# Instancia única (singleton) usada por la API vía dependencia de FastAPI
repository = ModelRepository()
