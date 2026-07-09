"""
eda.py
======
Análisis Exploratorio de Datos (EDA) sobre data/dataset_features.csv.
Genera y guarda TODAS las figuras en figures/ (300 dpi, listas para el
artículo) y un resumen de estadísticos en data/eda_resumen.txt.

Cubre:
  - Valores faltantes
  - Outliers (regla IQR)
  - Balance de clases (prioridad y tipo)
  - Estadísticos descriptivos
  - Correlaciones (heatmap)
  - Heatmap geográfico de densidad de incidentes
  - Histogramas
  - Boxplots
  - Distribuciones (series temporales)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_PATH = Path("data/dataset_features.csv")
FIG_DIR = Path("figures")
RESUMEN_PATH = Path("data/eda_resumen.txt")

sns.set_theme(style="whitegrid")
PRIORIDAD_ORDER = ["ALTA", "MEDIA", "BAJA"]
PRIORIDAD_PALETTE = {"ALTA": "#d62728", "MEDIA": "#ff7f0e", "BAJA": "#7f7f7f"}

NUMERIC_COLS = [
    "hora", "dia_semana", "mes", "dist_hermelinda_km", "dist_sanchez_carrion_km",
    "dist_parque_industrial_km", "dist_cesar_vallejo_km", "dist_hotspot_mas_cercano_km",
]


def savefig(name: str):
    path = FIG_DIR / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  guardado: {path}")


def seccion_valores_faltantes(df: pd.DataFrame, log: list):
    nulos = df.isnull().sum()
    nulos = nulos[nulos > 0]
    log.append("=== VALORES FALTANTES ===")
    if nulos.empty:
        log.append("No se encontraron valores nulos en ninguna columna.\n")
    else:
        log.append(nulos.to_string())
        log.append("")

    plt.figure(figsize=(10, 5))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="viridis")
    plt.title("Mapa de valores faltantes")
    savefig("01_valores_faltantes")


def seccion_outliers(df: pd.DataFrame, log: list):
    log.append("=== DETECCIÓN DE OUTLIERS (regla IQR) ===")
    for col in ["dist_hermelinda_km", "dist_sanchez_carrion_km",
                "dist_parque_industrial_km", "dist_cesar_vallejo_km"]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        pct = n_outliers / len(df) * 100
        log.append(f"{col}: {n_outliers} outliers ({pct:.2f}%) fuera de [{lower:.2f}, {upper:.2f}] km")
    log.append("")


def seccion_balance_clases(df: pd.DataFrame, log: list):
    log.append("=== BALANCE DE CLASES ===")
    log.append("Prioridad (variable objetivo):")
    log.append(df["prioridad"].value_counts(normalize=True).round(3).to_string())
    log.append("\nTipo de delito:")
    log.append(df["tipo"].value_counts(normalize=True).round(3).to_string())
    log.append("")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    order_p = df["prioridad"].value_counts().loc[PRIORIDAD_ORDER].index
    sns.countplot(data=df, x="prioridad", order=order_p, hue="prioridad", palette=PRIORIDAD_PALETTE, legend=False, ax=axes[0])
    axes[0].set_title("Balance de clases: prioridad")
    for p in axes[0].patches:
        axes[0].annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()),
                          ha="center", va="bottom")

    order_t = df["tipo"].value_counts().index
    sns.countplot(data=df, y="tipo", order=order_t, ax=axes[1], color="steelblue")
    axes[1].set_title("Distribución por tipo de delito")
    savefig("02_balance_de_clases")


def seccion_estadisticos_descriptivos(df: pd.DataFrame, log: list):
    log.append("=== ESTADÍSTICOS DESCRIPTIVOS (variables numéricas) ===")
    desc = df[NUMERIC_COLS].describe().round(3)
    log.append(desc.to_string())
    log.append("")
    desc.to_csv("data/estadisticos_descriptivos.csv")


def seccion_correlaciones(df: pd.DataFrame, log: list):
    corr = df[NUMERIC_COLS + ["prioridad_encoded"]].corr()
    log.append("=== MATRIZ DE CORRELACIÓN ===")
    log.append(corr.round(3).to_string())
    log.append("")

    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
    plt.title("Matriz de correlación - variables numéricas")
    savefig("03_matriz_correlacion")


def seccion_heatmap_geografico(df: pd.DataFrame, log: list):
    plt.figure(figsize=(8, 7))
    sns.kdeplot(data=df, x="longitud", y="latitud", fill=True, cmap="Reds",
                thresh=0.02, levels=30)
    plt.scatter(df["longitud"], df["latitud"], s=1, alpha=0.05, color="black")
    plt.title("Mapa de calor geográfico de incidentes - El Porvenir")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    savefig("04_heatmap_geografico")


def seccion_histogramas(df: pd.DataFrame, log: list):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    sns.histplot(df["hora"], bins=24, ax=axes[0, 0], color="teal")
    axes[0, 0].set_title("Distribución horaria de incidentes")
    sns.histplot(df["dia_semana"], bins=7, ax=axes[0, 1], color="darkorange")
    axes[0, 1].set_title("Distribución por día de la semana (0=lunes)")
    sns.histplot(df["dist_hotspot_mas_cercano_km"], bins=30, ax=axes[1, 0], color="purple")
    axes[1, 0].set_title("Distancia al hotspot más cercano (km)")
    sns.histplot(df["mes"], bins=12, ax=axes[1, 1], color="steelblue")
    axes[1, 1].set_title("Distribución mensual")
    savefig("05_histogramas")


def seccion_boxplots(df: pd.DataFrame, log: list):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.boxplot(data=df, x="prioridad", y="dist_hotspot_mas_cercano_km", hue="prioridad", legend=False,
                order=PRIORIDAD_ORDER, palette=PRIORIDAD_PALETTE, ax=axes[0])
    axes[0].set_title("Distancia al hotspot por prioridad")

    sns.boxplot(data=df, x="prioridad", y="hora", hue="prioridad", legend=False,
                order=PRIORIDAD_ORDER, palette=PRIORIDAD_PALETTE, ax=axes[1])
    axes[1].set_title("Hora del incidente por prioridad")
    savefig("06_boxplots_por_prioridad")


def seccion_distribuciones_temporales(df: pd.DataFrame, log: list):
    serie = df.groupby(["anio", "mes"]).size().reset_index(name="incidentes")
    serie["periodo"] = pd.to_datetime(serie["anio"].astype(str) + "-" + serie["mes"].astype(str))

    plt.figure(figsize=(13, 5))
    plt.plot(serie["periodo"], serie["incidentes"], marker="o", markersize=3)
    plt.title("Serie temporal de incidentes por mes - El Porvenir (2018-2026)")
    plt.xlabel("Periodo")
    plt.ylabel("N° de incidentes")
    savefig("07_serie_temporal_mensual")

    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x="hotspot_cercano",
                  order=df["hotspot_cercano"].value_counts().index, color="firebrick")
    plt.title("Incidentes por zona (hotspot más cercano)")
    savefig("08_incidentes_por_zona")


def main():
    FIG_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_PATH, parse_dates=["fecha"])
    log = [f"EDA - dataset_features.csv ({len(df)} filas, {df.shape[1]} columnas)\n"]

    print("Generando EDA...")
    seccion_valores_faltantes(df, log)
    seccion_outliers(df, log)
    seccion_balance_clases(df, log)
    seccion_estadisticos_descriptivos(df, log)
    seccion_correlaciones(df, log)
    seccion_heatmap_geografico(df, log)
    seccion_histogramas(df, log)
    seccion_boxplots(df, log)
    seccion_distribuciones_temporales(df, log)

    RESUMEN_PATH.write_text("\n".join(log), encoding="utf-8")
    print(f"\nResumen de texto guardado en -> {RESUMEN_PATH}")
    print(f"Figuras guardadas en -> {FIG_DIR}/ (8 archivos PNG, 300 dpi)")


if __name__ == "__main__":
    main()
