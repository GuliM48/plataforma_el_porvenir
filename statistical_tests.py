"""
statistical_tests.py
======================
Validación estadística de la comparación de los 5 modelos, usando los
resultados por fold ya generados por train_models.py
(resultados/cv_resultados_por_fold.csv).

- Friedman: ¿hay diferencias significativas entre los 5 modelos?
- Nemenyi (post-hoc): si Friedman es significativo, ¿qué pares difieren?
- Wilcoxon pareado: comparación directa del mejor modelo contra cada uno de
  los demás, con corrección de Holm-Bonferroni por comparaciones múltiples.

Uso: python statistical_tests.py
Salida: resultados/pruebas_estadisticas.json,
        resultados/matriz_nemenyi.csv,
        figuras_entrenamiento/05_boxplot_comparacion_estadistica.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests
import scikit_posthocs as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exceptions import DatosNoEncontradosError  # noqa: E402
from logging_config import configurar_logging  # noqa: E402

logger = configurar_logging(__name__)

CV_RESULTS_PATH = Path("resultados/cv_resultados_por_fold.csv")
OUT_JSON = Path("resultados/pruebas_estadisticas.json")
OUT_NEMENYI = Path("resultados/matriz_nemenyi.csv")
FIGURES_DIR = Path("figuras_entrenamiento")
METRICA = "f1_macro"
ALPHA = 0.05

sns.set_theme(style="whitegrid")


def cargar_resultados_wide() -> pd.DataFrame:
    if not CV_RESULTS_PATH.exists():
        raise DatosNoEncontradosError(
            f"No se encontró {CV_RESULTS_PATH}. Corre primero train_models.py.",
            detalle={"ruta_esperada": str(CV_RESULTS_PATH)})

    df = pd.read_csv(CV_RESULTS_PATH)
    wide = df.pivot(index="fold", columns="modelo", values=METRICA)
    return wide


def prueba_friedman(wide: pd.DataFrame) -> dict:
    stat, p = friedmanchisquare(*[wide[col].values for col in wide.columns])
    resultado = {
        "metrica_evaluada": METRICA,
        "n_folds": len(wide),
        "n_modelos": len(wide.columns),
        "estadistico_chi2": float(stat),
        "p_value": float(p),
        "significativo_alpha_0.05": bool(p < ALPHA),
    }
    logger.info("Friedman: chi2=%.4f, p=%.4f, n_folds=%d", stat, p, len(wide))
    if len(wide) < 10:
        resultado["advertencia"] = (
            f"Solo hay {len(wide)} folds (bloques). Para Friedman/Nemenyi se "
            "recomienda >=10 bloques para tener potencia estadística "
            "razonable; con menos, un resultado 'no significativo' puede "
            "deberse a falta de potencia y no a que los modelos sean "
            "realmente equivalentes. Sube N_REPEATS en train_models.py si "
            "tienes más cómputo disponible."
        )
    return resultado


def prueba_nemenyi(wide: pd.DataFrame) -> pd.DataFrame:
    matriz = sp.posthoc_nemenyi_friedman(wide.to_numpy())
    matriz.index = wide.columns
    matriz.columns = wide.columns
    return matriz


def pruebas_wilcoxon_pareadas(wide: pd.DataFrame) -> list[dict]:
    """Wilcoxon pareado para TODOS los pares de modelos, con corrección de
    Holm-Bonferroni por comparaciones múltiples."""
    modelos = list(wide.columns)
    pares, p_values, estadisticos = [], [], []

    for i in range(len(modelos)):
        for j in range(i + 1, len(modelos)):
            a, b = wide[modelos[i]].values, wide[modelos[j]].values
            diff = a - b
            if np.allclose(diff, 0):
                # Wilcoxon no está definido si todas las diferencias son cero
                stat, p = np.nan, 1.0
            else:
                stat, p = wilcoxon(a, b, zero_method="wilcox", method="auto")
            pares.append((modelos[i], modelos[j]))
            p_values.append(p)
            estadisticos.append(stat)

    _, p_corregidos, _, _ = multipletests(p_values, alpha=ALPHA, method="holm")

    resultados = []
    for (m1, m2), stat, p_raw, p_holm in zip(pares, estadisticos, p_values, p_corregidos):
        resultados.append({
            "modelo_a": m1, "modelo_b": m2,
            "estadistico_wilcoxon": None if np.isnan(stat) else float(stat),
            "p_value": float(p_raw),
            "p_value_holm": float(p_holm),
            "significativo_holm_0.05": bool(p_holm < ALPHA),
        })
    return resultados


def generar_boxplot(wide: pd.DataFrame, mejor_modelo: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    orden = wide.mean().sort_values(ascending=False).index
    data_long = wide.melt(var_name="modelo", value_name=METRICA)
    colores = ["#2ca02c" if m == mejor_modelo else "#4c72b0" for m in orden]
    sns.boxplot(data=data_long, x="modelo", y=METRICA, order=orden,
                hue="modelo", palette=colores, legend=False, ax=ax)
    sns.stripplot(data=data_long, x="modelo", y=METRICA, order=orden,
                  color="black", alpha=0.5, size=4, ax=ax)
    ax.set_title(f"Distribución de {METRICA} por modelo (por fold) - "
                 f"mejor modelo en verde")
    ax.set_xlabel("")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    out = FIGURES_DIR / "05_boxplot_comparacion_estadistica.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Figura guardada: %s", out)


def main():
    FIGURES_DIR.mkdir(exist_ok=True)
    wide = cargar_resultados_wide()
    logger.info("Datos cargados: %d folds x %d modelos", *wide.shape)

    friedman = prueba_friedman(wide)

    nemenyi_dict = None
    if friedman["significativo_alpha_0.05"]:
        nemenyi = prueba_nemenyi(wide)
        nemenyi.to_csv(OUT_NEMENYI)
        nemenyi_dict = nemenyi.round(4).to_dict()
        logger.info("Nemenyi guardado en %s", OUT_NEMENYI)
    else:
        logger.info("Friedman no significativo (p>=0.05): se omite Nemenyi "
                     "(no correspondería interpretarlo si el ómnibus no rechaza H0), "
                     "pero igual se reportan los Wilcoxon pareados a modo descriptivo.")

    wilcoxon_resultados = pruebas_wilcoxon_pareadas(wide)

    mejor_modelo = wide.mean().idxmax()
    generar_boxplot(wide, mejor_modelo)

    reporte = {
        "metrica_evaluada": METRICA,
        "mejor_modelo_por_media": mejor_modelo,
        "medias_por_modelo": wide.mean().round(4).to_dict(),
        "friedman": friedman,
        "nemenyi_p_values": nemenyi_dict,
        "wilcoxon_pareado_holm": wilcoxon_resultados,
    }
    OUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)

    print(f"\n=== Prueba de Friedman ({METRICA}) ===")
    print(json.dumps(friedman, indent=2, ensure_ascii=False))
    if nemenyi_dict:
        print("\n=== Matriz de p-values Nemenyi ===")
        print(pd.DataFrame(nemenyi_dict).round(4))
    print("\n=== Wilcoxon pareado (Holm-Bonferroni) ===")
    for r in wilcoxon_resultados:
        marca = "*" if r["significativo_holm_0.05"] else ""
        print(f"  {r['modelo_a']} vs {r['modelo_b']}: "
              f"p_holm={r['p_value_holm']:.4f} {marca}")
    print(f"\nGuardado en {OUT_JSON}")


if __name__ == "__main__":
    main()
