"""
reports/context.py
====================
Carga, UNA sola vez, todos los datos que necesitan los tres generadores de
reportes (PDF/Excel/Word): tabla comparativa de modelos, resultados por
fold, pruebas estadísticas, hiperparámetros óptimos, y rutas de las figuras
ya generadas por train_models.py / statistical_tests.py.

Evita que cada generador tenga su propia lógica de carga (DRY) y hace
explícito qué datos deben existir antes de generar reportes.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402
from exceptions import DatosNoEncontradosError  # noqa: E402


@dataclass
class ReportContext:
    fecha_generacion: str
    tabla_comparativa: pd.DataFrame
    cv_resultados_por_fold: pd.DataFrame
    resumen_entrenamiento: dict
    hiperparametros: dict
    pruebas_estadisticas: dict | None
    figuras: dict[str, Path] = field(default_factory=dict)


def _requerir_archivo(path: Path) -> Path:
    if not path.exists():
        raise DatosNoEncontradosError(
            f"No se encontró {path}. Corre train_models.py (y statistical_tests.py) primero.",
            detalle={"ruta_esperada": str(path)})
    return path


def cargar_contexto_reporte() -> ReportContext:
    resultados_dir = settings.resultados_dir
    figuras_dir = settings.figuras_dir

    tabla_comparativa = pd.read_csv(
        _requerir_archivo(resultados_dir / "tabla_comparativa_modelos.csv"),
        header=[0, 1], index_col=0)
    cv_por_fold = pd.read_csv(_requerir_archivo(resultados_dir / "cv_resultados_por_fold.csv"))

    with open(_requerir_archivo(resultados_dir / "resumen_entrenamiento.json"), encoding="utf-8") as f:
        resumen = json.load(f)
    with open(_requerir_archivo(resultados_dir / "hiperparametros_optimizados.json"), encoding="utf-8") as f:
        hiperparametros = json.load(f)

    pruebas_path = resultados_dir / "pruebas_estadisticas.json"
    pruebas = None
    if pruebas_path.exists():
        with open(pruebas_path, encoding="utf-8") as f:
            pruebas = json.load(f)

    figuras = {
        "roc": figuras_dir / "01_curvas_roc.png",
        "matrices_confusion": figuras_dir / "02_matrices_confusion.png",
        "importancia_variables": figuras_dir / "03_importancia_variables.png",
        "comparacion_modelos": figuras_dir / "04_comparacion_modelos.png",
        "boxplot_estadistico": figuras_dir / "05_boxplot_comparacion_estadistica.png",
    }
    figuras = {k: v for k, v in figuras.items() if v.exists()}

    return ReportContext(
        fecha_generacion=datetime.now().strftime("%Y-%m-%d %H:%M"),
        tabla_comparativa=tabla_comparativa,
        cv_resultados_por_fold=cv_por_fold,
        resumen_entrenamiento=resumen,
        hiperparametros=hiperparametros,
        pruebas_estadisticas=pruebas,
        figuras=figuras,
    )
