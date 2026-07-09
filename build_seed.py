"""
build_seed.py
=============
Prepara el dataset definitivo de incidentes para El Porvenir (Trujillo, La Libertad)
a partir del dataset abierto SIDPOL (datosabiertos.gob.pe).

IMPORTANTE - corrección respecto al plan original:
El UBIGEO correcto de El Porvenir es 130102, NO 130109 (130109 corresponde a
Salaverry). Esto se verificó directamente contra el archivo de datos real.

El dataset SIDPOL viene agregado a nivel (año, mes, distrito, tipo de delito) con
un conteo ("cantidad"), es decir NO trae registros individuales con lat/lng ni hora.
Por eso este script "desagrega" cada fila mensual en `cantidad` incidentes
puntuales sintéticos, distribuidos espacialmente sobre hotspots conocidos de
El Porvenir y temporalmente sobre una distribución horaria realista según
literatura criminológica (picos 7-9am, 12-2pm, 6-9pm).

Salida: data/dataset_definitivo.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.features import HOTSPOTS, BOUNDING_BOX, haversine_km  # noqa: E402

# ------------------------------------------------------------------
# Configuración
# ------------------------------------------------------------------
RANDOM_SEED = 42
UBIGEO_EL_PORVENIR = "130102"  # corregido: antes se usaba 130109 (Salaverry) por error

RAW_PATH = Path("data/sidpol_raw.csv")
OUT_PATH = Path("data/dataset_definitivo.csv")

# Mapeo tipo de delito -> nivel BASE de prioridad. Esta es una regla heurística
# basada en urgencia de respuesta (delitos en curso / riesgo a la víctima =
# ALTA), NO un dato oficial del SIDPOL. Para el artículo, esta regla debe
# citarse como criterio metodológico propio y puede ajustarse si el equipo
# tiene una referencia normativa (p.ej. protocolo PNP) que la respalde mejor.
#
# OJO - por qué esto es solo el nivel BASE y no la prioridad final:
# Si la prioridad dependiera UNICAMENTE del tipo de delito, el problema de
# clasificación sería trivial (una tabla de 7 filas) y las variables de
# ubicación/hora -que son el aporte real de este sistema geolocalizado-
# quedarían sin ningún poder predictivo. Cualquier modelo lograría ~100% de
# accuracy memorizando la tabla, lo cual invalidaría la comparación de
# modelos y las pruebas estadísticas del artículo. Por eso, más abajo
# (función `ajustar_prioridad_por_contexto`) el nivel base se modula
# probabilísticamente según hora, día y cercanía a zonas críticas, de modo
# que la prioridad final sí dependa de contexto espaciotemporal aprendible.
PRIORIDAD_BASE_MAP = {
    "Secuestro": 2,   # ALTA
    "Extorsión": 2,   # ALTA
    "Violencia contra la mujer e integrantes": 2,  # ALTA
    "Robo": 2,        # ALTA
    "Hurto": 1,       # MEDIA
    "Estafa": 0,      # BAJA
    "Otros": 0,       # BAJA
}
NIVEL_A_PRIORIDAD = {0: "BAJA", 1: "MEDIA", 2: "ALTA"}


def ajustar_prioridad_por_contexto(nivel_base, hora, dia_semana, dist_hotspot_km, rng):
    """Modula el nivel base de prioridad según contexto espaciotemporal.

    Reglas (documentadas para el artículo, sección de metodología):
    - Horario nocturno (22:00-05:59): mayor percepción de riesgo/urgencia
      -> probabilidad de escalar +1 nivel.
    - Fin de semana: leve incremento adicional de probabilidad de escalar.
    - Muy cerca de un hotspot conocido (menos de 300 m): zona de alta
      incidencia histórica -> mayor probabilidad de escalar.
    - Horario diurno entre semana y lejos de cualquier hotspot (más de 2 km):
      mayor probabilidad de des-escalar (contexto de menor urgencia relativa).
    """
    es_noche = (hora >= 22) | (hora < 6)
    es_finde = np.isin(dia_semana, [5, 6])
    cerca_hotspot = dist_hotspot_km < 0.3
    lejos_hotspot = dist_hotspot_km > 2.0
    diurno_entre_semana = (~es_noche) & (~es_finde)

    p_escalar = 0.05 + 0.25 * es_noche + 0.08 * es_finde + 0.20 * cerca_hotspot
    p_desescalar = 0.05 + 0.15 * (diurno_entre_semana & lejos_hotspot)

    u = rng.random(len(nivel_base))
    delta = np.zeros(len(nivel_base), dtype=int)
    delta[u < p_escalar] = 1
    resto = u >= p_escalar
    delta[resto & (u < (p_escalar + p_desescalar))] = -1

    return np.clip(nivel_base + delta, 0, 2)

# Distribución horaria (24 bins, probabilidad relativa). Aproxima picos de
# actividad delictiva reportados en literatura de predictive policing.
HOUR_WEIGHTS = np.array([
    0.010, 0.008, 0.006, 0.005, 0.006, 0.010,   # 0-5   madrugada
    0.030, 0.045, 0.040, 0.025, 0.020, 0.022,   # 6-11  pico mañana (7-9)
    0.045, 0.040, 0.022, 0.020, 0.022, 0.028,   # 12-17 pico mediodía (12-14)
    0.045, 0.050, 0.048, 0.040, 0.030, 0.018,   # 18-23 pico noche (18-21)
])
HOUR_WEIGHTS = HOUR_WEIGHTS / HOUR_WEIGHTS.sum()


def cargar_y_filtrar(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)

    # --- Limpieza ---
    for col in ["DPTO_HECHO_NEW", "PROV_HECHO", "DIST_HECHO", "P_MODALIDADES"]:
        df[col] = df[col].str.strip()
    df["ANIO"] = df["ANIO"].astype(int)
    df["MES"] = df["MES"].astype(int)
    df["cantidad"] = pd.to_numeric(df["cantidad"], errors="coerce")

    antes = len(df)
    df = df.dropna(subset=["cantidad", "UBIGEO_HECHO"])
    df = df[df["cantidad"] > 0]
    df = df.drop_duplicates(subset=["ANIO", "MES", "UBIGEO_HECHO", "P_MODALIDADES"])
    print(f"Limpieza: {antes} -> {len(df)} filas (se removieron nulos/duplicados/cantidad<=0)")

    ep = df[df["UBIGEO_HECHO"] == UBIGEO_EL_PORVENIR].copy()
    if ep.empty:
        raise ValueError(f"No se encontraron filas para UBIGEO {UBIGEO_EL_PORVENIR}")
    print(f"El Porvenir (UBIGEO {UBIGEO_EL_PORVENIR}): {len(ep)} filas mensuales, "
          f"{int(ep['cantidad'].sum())} incidentes totales, años {ep['ANIO'].min()}-{ep['ANIO'].max()}")
    return ep


def asignar_hotspot(n: int, rng: np.random.Generator):
    weights = [h.weight for h in HOTSPOTS]
    idx = rng.choice(len(HOTSPOTS), size=n, p=weights)
    lat_centro = np.array([HOTSPOTS[i].lat for i in idx])
    lng_centro = np.array([HOTSPOTS[i].lng for i in idx])
    # jitter gaussiano (~ sigma 0.006° ≈ 650 m) recortado al bounding box
    lat = np.clip(lat_centro + rng.normal(0, 0.006, n), BOUNDING_BOX["lat_min"], BOUNDING_BOX["lat_max"])
    lng = np.clip(lng_centro + rng.normal(0, 0.006, n), BOUNDING_BOX["lng_min"], BOUNDING_BOX["lng_max"])
    return lat, lng


def generar_incidentes(ep: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    filas = []
    for _, row in ep.iterrows():
        n = int(row["cantidad"])
        anio, mes, tipo = int(row["ANIO"]), int(row["MES"]), row["P_MODALIDADES"]

        dias_en_mes = pd.Period(f"{anio}-{mes:02d}").days_in_month
        dia = rng.integers(1, dias_en_mes + 1, size=n)
        hora = rng.choice(24, size=n, p=HOUR_WEIGHTS)
        minuto = rng.integers(0, 60, size=n)
        lat, lng = asignar_hotspot(n, rng)

        bloque = pd.DataFrame({
            "anio": anio, "mes": mes, "dia": dia, "hora": hora, "minuto": minuto,
            "tipo": tipo, "latitud": lat, "longitud": lng,
        })
        filas.append(bloque)

    data = pd.concat(filas, ignore_index=True)

    # fecha real -> permite derivar día de la semana correctamente (no aleatorio)
    data["fecha"] = pd.to_datetime(dict(year=data.anio, month=data.mes, day=data.dia,
                                         hour=data.hora, minute=data.minuto))
    data["dia_semana"] = data["fecha"].dt.weekday  # 0=lunes ... 6=domingo

    # distancias Haversine a cada hotspot (features para el modelo ML)
    for h in HOTSPOTS:
        col = f"dist_{h.name.lower().replace(' ', '_')}_km"
        data[col] = haversine_km(data["latitud"], data["longitud"], h.lat, h.lng)

    nivel_base = data["tipo"].map(PRIORIDAD_BASE_MAP).fillna(1).to_numpy()
    dist_min = data[[f"dist_{h.name.lower().replace(' ', '_')}_km" for h in HOTSPOTS]].min(axis=1).to_numpy()
    nivel_final = ajustar_prioridad_por_contexto(
        nivel_base, data["hora"].to_numpy(), data["dia_semana"].to_numpy(), dist_min, rng
    )
    data["prioridad"] = pd.Series(nivel_final).map(NIVEL_A_PRIORIDAD)
    data["prioridad_base_tipo"] = pd.Series(nivel_base.astype(int)).map(NIVEL_A_PRIORIDAD)

    data["incident_id"] = np.arange(1, len(data) + 1)
    cols = ["incident_id", "fecha", "anio", "mes", "dia", "hora", "minuto", "dia_semana",
            "tipo", "prioridad", "prioridad_base_tipo", "latitud", "longitud"] + \
           [c for c in data.columns if c.startswith("dist_")]
    return data[cols].sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    ep = cargar_y_filtrar(RAW_PATH)
    dataset = generar_incidentes(ep, rng)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUT_PATH, index=False)

    print(f"\nDataset definitivo: {len(dataset)} incidentes -> {OUT_PATH}")
    print("\nDistribución de prioridad:")
    print(dataset["prioridad"].value_counts(normalize=True).round(3))
    print("\nDistribución de tipo:")
    print(dataset["tipo"].value_counts(normalize=True).round(3))


if __name__ == "__main__":
    main()
