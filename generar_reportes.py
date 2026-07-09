"""
generar_reportes.py
=====================
Orquesta la generación automática de los 3 reportes (PDF, Excel, Word) a
partir de los resultados ya generados por train_models.py y
statistical_tests.py.

Uso: python generar_reportes.py
Salida: reportes_generados/reporte_entrenamiento.{pdf,xlsx,docx}
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import settings  # noqa: E402
from exceptions import PlataformaError, ReporteGenerationError  # noqa: E402
from logging_config import configurar_logging  # noqa: E402
from reports.context import cargar_contexto_reporte  # noqa: E402
from reports.excel_report import ExcelReportGenerator  # noqa: E402
from reports.pdf_report import PDFReportGenerator  # noqa: E402
from reports.word_report import WordReportGenerator  # noqa: E402

logger = configurar_logging(__name__)

GENERADORES = [PDFReportGenerator(), ExcelReportGenerator(), WordReportGenerator()]


def main():
    try:
        contexto = cargar_contexto_reporte()
    except PlataformaError as exc:
        logger.error("No se pudo cargar el contexto de reporte: %s", exc.mensaje)
        raise

    settings.reportes_dir.mkdir(parents=True, exist_ok=True)
    for generador in GENERADORES:
        out = settings.reportes_dir / f"reporte_entrenamiento.{generador.extension}"
        try:
            ruta = generador.generar(contexto, out)
            logger.info("Reporte generado: %s", ruta)
            print(f"  ✓ {ruta}")
        except Exception as exc:  # noqa: BLE001 - se envuelve para dar contexto uniforme
            raise ReporteGenerationError(
                f"Falló la generación del reporte {generador.extension}: {exc}"
            ) from exc

    print(f"\nListo. Reportes en {settings.reportes_dir}/")


if __name__ == "__main__":
    main()
