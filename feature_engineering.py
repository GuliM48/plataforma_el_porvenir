"""
feature_engineering.py
=======================
Toma data/dataset_definitivo.csv (salida de build_seed.py) y construye las
variables finales para el pipeline de Machine Learning:

- hora_bin        : bloques de 3 horas (0-7)
- hora_sin/cos     : codificación cíclica de la hora (útil para redes neuronales)
- dia_semana_sin/cos: codificación cíclica del día de la semana
- mes_sin/cos      : codificación cíclica del mes
- lat_bin/lng_bin  : celdas espaciales de ~1.1 km (redondeo a 2 decimales)
- dist_*_km        : distancias Haversine a cada hotspot (ya vienen del seed)
- hotspot_cercano  : nombre del hotspot más próximo (categórica)
- tipo_encoded     : codificación numérica (LabelEncoder) del tipo de delito
- tipo_<categoria> : one-hot encoding del tipo de delito
- prioridad_encoded: variable objetivo codificada (ALTA=0, MEDIA=1, BAJA=2)

Salida: data/dataset_features.csv + label_encoders.pkl (para reutilizar la
misma codificación en entrenamiento/inferencia).
"""

import numpy as np
import pandas as pd
import pickle
import sys
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.features import cyclic_encode_series  # noqa: E402

IN_PATH = Path("data/dataset_definitivo.csv")
OUT_PATH = Path("data/dataset_features.csv")
ENCODERS_PATH = Path("data/label_encoders.pkl")

DIST_COLS = ["dist_hermelinda_km", "dist_sanchez_carrion_km",
             "dist_parque_industrial_km", "dist_cesar_vallejo_km"]

HOTSPOT_DISPLAY_NAMES = {
    "dist_hermelinda_km": "Hermelinda",
    "dist_sanchez_carrion_km": "Sanchez Carrion",
    "dist_parque_industrial_km": "Parque Industrial",
    "dist_cesar_vallejo_km": "Cesar Vallejo",
}


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()

    # --- temporales ---
    df["hora_bin"] = df["hora"] // 3
    df = pd.concat([df, cyclic_encode_series(df["hora"], 24, "hora")], axis=1)
    df = pd.concat([df, cyclic_encode_series(df["dia_semana"], 7, "dia_semana")], axis=1)
    df = pd.concat([df, cyclic_encode_series(df["mes"], 12, "mes")], axis=1)
    df["es_fin_de_semana"] = df["dia_semana"].isin([5, 6]).astype(int)

    # --- espaciales ---
    df["lat_bin"] = df["latitud"].round(2)
    df["lng_bin"] = df["longitud"].round(2)
    df["hotspot_cercano"] = df[DIST_COLS].idxmin(axis=1).map(HOTSPOT_DISPLAY_NAMES)
    df["dist_hotspot_mas_cercano_km"] = df[DIST_COLS].min(axis=1)

    encoders = {}

    # --- categóricas: tipo de delito ---
    le_tipo = LabelEncoder()
    df["tipo_encoded"] = le_tipo.fit_transform(df["tipo"])
    encoders["tipo"] = le_tipo
    tipo_dummies = pd.get_dummies(df["tipo"], prefix="tipo", dtype=int)
    df = pd.concat([df, tipo_dummies], axis=1)

    # --- categórica: hotspot cercano ---
    le_hotspot = LabelEncoder()
    df["hotspot_encoded"] = le_hotspot.fit_transform(df["hotspot_cercano"])
    encoders["hotspot"] = le_hotspot

    # --- variable objetivo ---
    le_prioridad = LabelEncoder()
    # orden fijo y explícito para que ALTA=0, MEDIA=1, BAJA=2 sea reproducible
    le_prioridad.classes_ = np.array(["ALTA", "MEDIA", "BAJA"])
    mapping = {c: i for i, c in enumerate(le_prioridad.classes_)}
    df["prioridad_encoded"] = df["prioridad"].map(mapping)
    encoders["prioridad"] = le_prioridad

    return df, encoders


def main():
    df = pd.read_csv(IN_PATH, parse_dates=["fecha"])
    features, encoders = build_features(df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUT_PATH, index=False)
    with open(ENCODERS_PATH, "wb") as f:
        pickle.dump(encoders, f)

    print(f"Dataset con features: {features.shape[0]} filas, {features.shape[1]} columnas -> {OUT_PATH}")
    print(f"Encoders guardados en -> {ENCODERS_PATH}")
    print("\nColumnas finales:")
    print(list(features.columns))


if __name__ == "__main__":
    main()
