"""
reports/pdf_report.py
=======================
Genera un reporte PDF usando reportlab (platypus): portada, tabla
comparativa, figuras embebidas, hiperparámetros y pruebas estadísticas.
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                 Spacer, Table, TableStyle)

from reports.base import ReportGenerator
from reports.context import ReportContext

ESTILOS = getSampleStyleSheet()
ESTILOS.add(ParagraphStyle(name="TituloPortada", fontSize=22, leading=28,
                            alignment=1, spaceAfter=12, textColor=colors.HexColor("#1F4E78")))
ESTILOS.add(ParagraphStyle(name="Subtitulo", fontSize=13, alignment=1,
                            textColor=colors.HexColor("#444444"), spaceAfter=6))
ESTILOS.add(ParagraphStyle(name="CuerpoJustificado", parent=ESTILOS["BodyText"],
                            alignment=4, spaceAfter=10))


class PDFReportGenerator(ReportGenerator):
    extension = "pdf"

    def generar(self, contexto: ReportContext, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc = SimpleDocTemplate(str(output_path), pagesize=A4,
                                 topMargin=2 * cm, bottomMargin=2 * cm)
        story: list = []

        story += self._portada(contexto)
        story.append(PageBreak())
        story += self._resumen_ejecutivo(contexto)
        story += self._tabla_comparativa(contexto)
        story.append(PageBreak())
        story += self._figuras(contexto)
        story += self._hiperparametros(contexto)
        if contexto.pruebas_estadisticas:
            story += self._pruebas_estadisticas(contexto)

        doc.build(story)
        return output_path

    def _portada(self, contexto: ReportContext) -> list:
        return [
            Spacer(1, 4 * cm),
            Paragraph("Priorización de Incidentes de Seguridad Ciudadana", ESTILOS["TituloPortada"]),
            Paragraph("El Porvenir, Trujillo", ESTILOS["TituloPortada"]),
            Spacer(1, 1 * cm),
            Paragraph("Reporte técnico de entrenamiento y evaluación de modelos", ESTILOS["Subtitulo"]),
            Spacer(1, 0.5 * cm),
            Paragraph(f"Generado automáticamente el {contexto.fecha_generacion}", ESTILOS["Subtitulo"]),
        ]

    def _resumen_ejecutivo(self, contexto: ReportContext) -> list:
        mejor = contexto.resumen_entrenamiento.get("mejor_modelo")
        f1 = contexto.resumen_entrenamiento.get("f1_macro_cv_promedio")
        cfg = contexto.resumen_entrenamiento.get("config", {})
        texto = (
            f"Se entrenaron y compararon 5 modelos de clasificación (3 clásicos: Regresión "
            f"Logística, Random Forest y LightGBM; 2 híbridos: Stacking con meta-modelo MLP, "
            f"y una red neuronal MLP optimizada con Algoritmo Genético) para predecir la "
            f"prioridad (ALTA/MEDIA/BAJA) de incidentes en El Porvenir. La validación se "
            f"realizó con RepeatedStratifiedKFold (n_splits={cfg.get('n_splits')}, "
            f"n_repeats={cfg.get('n_repeats')}). "
            f"<b>El modelo con mejor desempeño promedio fue {mejor} "
            f"(F1-macro = {f1:.4f}).</b>"
        )
        return [
            Paragraph("1. Resumen ejecutivo", ESTILOS["Heading1"]),
            Paragraph(texto, ESTILOS["CuerpoJustificado"]),
            Spacer(1, 0.3 * cm),
        ]

    def _tabla_comparativa(self, contexto: ReportContext) -> list:
        tabla_df = contexto.tabla_comparativa
        metricas = [c for c in tabla_df.columns if c[1] == "mean"]
        mejor_modelo = contexto.resumen_entrenamiento.get("mejor_modelo")

        cabecera = ["Modelo"] + [m[0] for m in metricas]
        filas = [cabecera]
        for modelo in tabla_df.index:
            fila = [modelo] + [f"{tabla_df.loc[modelo, m]:.4f}" for m in metricas]
            filas.append(fila)

        t = Table(filas, hAlign="CENTER")
        estilo = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E78")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ]
        for i, modelo in enumerate(tabla_df.index, start=1):
            if modelo == mejor_modelo:
                estilo.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#C6EFCE")))
        t.setStyle(TableStyle(estilo))

        return [Paragraph("2. Tabla comparativa de modelos", ESTILOS["Heading1"]), t, Spacer(1, 0.5 * cm)]

    def _figuras(self, contexto: ReportContext) -> list:
        elementos = [Paragraph("3. Figuras", ESTILOS["Heading1"])]
        titulos = {
            "roc": "3.1 Curvas ROC (One-vs-Rest)",
            "matrices_confusion": "3.2 Matrices de confusión",
            "importancia_variables": "3.3 Importancia de variables",
            "comparacion_modelos": "3.4 Comparación de métricas y tiempos",
            "boxplot_estadistico": "3.5 Distribución de F1-macro por modelo",
        }
        for clave, titulo in titulos.items():
            ruta = contexto.figuras.get(clave)
            if ruta is None:
                continue
            elementos.append(Paragraph(titulo, ESTILOS["Heading2"]))
            elementos.append(Image(str(ruta), width=16 * cm, height=16 * cm * 0.35))
            elementos.append(Spacer(1, 0.4 * cm))
        return elementos

    def _hiperparametros(self, contexto: ReportContext) -> list:
        elementos = [Paragraph("4. Hiperparámetros óptimos", ESTILOS["Heading1"])]
        for modelo, info in contexto.hiperparametros.items():
            if modelo == "ga_mlp":
                continue
            elementos.append(Paragraph(f"<b>{modelo}:</b> {info.get('params')}",
                                        ESTILOS["CuerpoJustificado"]))
        return elementos

    def _pruebas_estadisticas(self, contexto: ReportContext) -> list:
        elementos = [Paragraph("5. Validación estadística", ESTILOS["Heading1"])]
        friedman = contexto.pruebas_estadisticas["friedman"]
        texto = (
            f"Prueba de Friedman sobre F1-macro: χ²={friedman['estadistico_chi2']:.4f}, "
            f"p={friedman['p_value']:.4f} "
            f"({'significativo' if friedman['significativo_alpha_0.05'] else 'no significativo'} "
            f"a α=0.05, con {friedman['n_folds']} folds)."
        )
        elementos.append(Paragraph(texto, ESTILOS["CuerpoJustificado"]))
        if "advertencia" in friedman:
            elementos.append(Paragraph(f"<i>{friedman['advertencia']}</i>", ESTILOS["CuerpoJustificado"]))

        filas = [["Modelo A", "Modelo B", "p-value", "p-value (Holm)", "Sig."]]
        for r in contexto.pruebas_estadisticas["wilcoxon_pareado_holm"]:
            filas.append([r["modelo_a"], r["modelo_b"], f"{r['p_value']:.4f}",
                          f"{r['p_value_holm']:.4f}", "Sí" if r["significativo_holm_0.05"] else "No"])
        t = Table(filas, hAlign="CENTER")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E78")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        elementos.append(Spacer(1, 0.3 * cm))
        elementos.append(t)
        return elementos
