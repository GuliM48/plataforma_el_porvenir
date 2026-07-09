"""
common/features.py
====================
Fuente única de verdad para todo lo relacionado a ingeniería de
características: hotspots, distancia Haversine, codificación cíclica, y la
lista canónica de columnas de features (FEATURE_COLS).

Tanto el pipeline batch (build_seed.py, feature_engineering.py,
train_models.py) como la API de inferencia (api/) importan de este módulo,
para garantizar que el vector de features usado en entrenamiento sea
EXACTAMENTE el mismo que se usa en producción (evita el clásico bug de
"training/serving skew").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Hotspot:
    name: str
    lat: float
    lng: float
    weight: float


BOUNDING_BOX = {
    "lat_min": -8.085, "lat_max": -7.995,
    "lng_min": -79.025, "lng_max": -78.965,
}

HOTSPOTS: tuple[Hotspot, ...] = (
    Hotspot("Hermelinda", -8.044, -79.003, 0.35),
    Hotspot("Sanchez Carrion", -8.031, -78.993, 0.25),
    Hotspot("Parque Industrial", -8.058, -79.012, 0.20),
    Hotspot("Cesar Vallejo", -8.021, -78.982, 0.20),
)

TIPOS_VALIDOS: tuple[str, ...] = (
    "Estafa", "Extorsión", "Hurto", "Otros", "Robo", "Secuestro",
    "Violencia contra la mujer e integrantes",
)

CLASS_NAMES: tuple[str, ...] = ("ALTA", "MEDIA", "BAJA")  # orden fijo: 0,1,2

# Orden EXACTO de columnas que espera cada modelo entrenado. Si esto cambia,
# hay que reentrenar los modelos (o versionar el esquema).
FEATURE_COLS: tuple[str, ...] = (
    "hora_sin", "hora_cos", "dia_semana_sin", "dia_semana_cos", "mes_sin", "mes_cos",
    "es_fin_de_semana", "latitud", "longitud",
    "dist_hermelinda_km", "dist_sanchez_carrion_km", "dist_parque_industrial_km",
    "dist_cesar_vallejo_km", "dist_hotspot_mas_cercano_km",
    "tipo_Estafa", "tipo_Extorsión", "tipo_Hurto", "tipo_Otros", "tipo_Robo",
    "tipo_Secuestro", "tipo_Violencia contra la mujer e integrantes",
)


def haversine_km(lat1, lng1, lat2, lng2):
    """Distancia Haversine en km. Acepta escalares o arrays numpy."""
    R = 6371.0
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def cyclic_encode_scalar(value: float, period: int) -> tuple[float, float]:
    """Codificación cíclica seno/coseno para un único valor escalar."""
    radians = 2 * np.pi * value / period
    return float(np.sin(radians)), float(np.cos(radians))


def cyclic_encode_series(series, period: int, prefix: str):
    """Versión vectorizada (pandas Series) de la codificación cíclica, usada
    por feature_engineering.py al procesar el dataset completo en batch."""
    import pandas as pd
    radians = 2 * np.pi * series / period
    return pd.DataFrame({f"{prefix}_sin": np.sin(radians), f"{prefix}_cos": np.cos(radians)})


def build_feature_vector(lat: float, lng: float, hora: int, dia_semana: int,
                          mes: int, tipo: str) -> dict[str, float]:
    """Construye el vector de features (dict ordenado según FEATURE_COLS) para
    UN incidente, a partir de los datos crudos que reporta el ciudadano/app.

    Usado por la API de inferencia (`/predict`) - debe reflejar EXACTAMENTE
    la misma lógica que `feature_engineering.py` aplica en batch.
    """
    if tipo not in TIPOS_VALIDOS:
        raise ValueError(f"tipo inválido: {tipo!r}. Debe ser uno de {TIPOS_VALIDOS}")
    if not (0 <= hora <= 23):
        raise ValueError(f"hora fuera de rango [0,23]: {hora}")
    if not (0 <= dia_semana <= 6):
        raise ValueError(f"dia_semana fuera de rango [0,6]: {dia_semana}")
    if not (1 <= mes <= 12):
        raise ValueError(f"mes fuera de rango [1,12]: {mes}")

    hora_sin, hora_cos = cyclic_encode_scalar(hora, 24)
    dsem_sin, dsem_cos = cyclic_encode_scalar(dia_semana, 7)
    mes_sin, mes_cos = cyclic_encode_scalar(mes, 12)
    es_finde = 1 if dia_semana in (5, 6) else 0

    distancias = {
        f"dist_{h.name.lower().replace(' ', '_')}_km": float(haversine_km(lat, lng, h.lat, h.lng))
        for h in HOTSPOTS
    }
    dist_min = min(distancias.values())

    features = {
        "hora_sin": hora_sin, "hora_cos": hora_cos,
        "dia_semana_sin": dsem_sin, "dia_semana_cos": dsem_cos,
        "mes_sin": mes_sin, "mes_cos": mes_cos,
        "es_fin_de_semana": es_finde,
        "latitud": lat, "longitud": lng,
        **distancias,
        "dist_hotspot_mas_cercano_km": dist_min,
    }
    for t in TIPOS_VALIDOS:
        features[f"tipo_{t}"] = 1 if t == tipo else 0

    return features


def feature_dict_to_array(features: dict[str, float]) -> np.ndarray:
    """Convierte el dict de features a un array numpy en el orden canónico
    de FEATURE_COLS (crítico: el orden debe coincidir con el usado al
    entrenar los modelos)."""
    return np.array([[features[col] for col in FEATURE_COLS]], dtype="float64")
