"""
reports/word_report.py
========================
Genera un reporte Word (.docx) con interpretación en texto de los
resultados, tablas y las figuras generadas por train_models.py /
statistical_tests.py. Pensado como el reporte "narrativo" (a diferencia del
Excel, que es más para inspeccionar números).
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

from reports.base import ReportGenerator
from reports.context import ReportContext


class WordReportGenerator(ReportGenerator):
    extension = "docx"

    def generar(self, contexto: ReportContext, output_path: Path) -> Path:
        doc = Document()
        self._estilos_base(doc)

        self._portada(doc, contexto)
        doc.add_page_break()
        self._resumen_ejecutivo(doc, contexto)
        self._tabla_comparativa(doc, contexto)
        self._figuras(doc, contexto)
        self._hiperparametros(doc, contexto)
        if contexto.pruebas_estadisticas:
            self._pruebas_estadisticas(doc, contexto)
        self._conclusiones(doc, contexto)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(output_path)
        return output_path

    def _estilos_base(self, doc: Document) -> None:
        normal = doc.styles["Normal"]
        normal.font.name = "Calibri"
        normal.font.size = Pt(11)

    def _portada(self, doc: Document, contexto: ReportContext) -> None:
        titulo = doc.add_heading(
            "Priorización de Incidentes de Seguridad Ciudadana\nEl Porvenir, Trujillo", level=0)
        titulo.alignment = WD_ALIGN_PARAGRAPH.CENTER
        sub = doc.add_paragraph("Reporte técnico de entrenamiento y evaluación de modelos")
        sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
        sub.runs[0].font.size = Pt(14)
        sub.runs[0].font.color.rgb = RGBColor(0x44, 0x44, 0x44)
        fecha = doc.add_paragraph(f"Generado automáticamente el {contexto.fecha_generacion}")
        fecha.alignment = WD_ALIGN_PARAGRAPH.CENTER

    def _resumen_ejecutivo(self, doc: Document, contexto: ReportContext) -> None:
        doc.add_heading("1. Resumen ejecutivo", level=1)
        mejor = contexto.resumen_entrenamiento.get("mejor_modelo")
        f1 = contexto.resumen_entrenamiento.get("f1_macro_cv_promedio")
        cfg = contexto.resumen_entrenamiento.get("config", {})
        doc.add_paragraph(
            f"Se entrenaron y compararon 5 modelos de clasificación (3 clásicos: Regresión "
            f"Logística, Random Forest y LightGBM; 2 híbridos: Stacking con meta-modelo MLP, "
            f"y una red neuronal MLP optimizada con Algoritmo Genético) para predecir la "
            f"prioridad (ALTA/MEDIA/BAJA) de incidentes reportados en El Porvenir, Trujillo. "
            f"La validación se realizó con RepeatedStratifiedKFold "
            f"(n_splits={cfg.get('n_splits')}, n_repeats={cfg.get('n_repeats')})."
        )
        doc.add_paragraph(
            f"El modelo con mejor desempeño promedio fue "
        ).add_run(f"{mejor} (F1-macro = {f1:.4f}).").bold = True

    def _tabla_comparativa(self, doc: Document, contexto: ReportContext) -> None:
        doc.add_heading("2. Tabla comparativa de modelos", level=1)
        tabla_df = contexto.tabla_comparativa
        metricas = [c for c in tabla_df.columns if c[1] == "mean"]

        tabla = doc.add_table(rows=1, cols=len(metricas) + 1)
        tabla.style = "Light Grid Accent 1"
        hdr = tabla.rows[0].cells
        hdr[0].text = "Modelo"
        for i, m in enumerate(metricas, start=1):
            hdr[i].text = m[0]

        for modelo in tabla_df.index:
            fila = tabla.add_row().cells
            fila[0].text = str(modelo)
            for i, m in enumerate(metricas, start=1):
                fila[i].text = f"{tabla_df.loc[modelo, m]:.4f}"
        doc.add_paragraph()

    def _figuras(self, doc: Document, contexto: ReportContext) -> None:
        doc.add_heading("3. Figuras", level=1)
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
            doc.add_heading(titulo, level=2)
            doc.add_picture(str(ruta), width=Inches(6.3))

    def _hiperparametros(self, doc: Document, contexto: ReportContext) -> None:
        doc.add_heading("4. Hiperparámetros óptimos", level=1)
        for modelo, info in contexto.hiperparametros.items():
            if modelo == "ga_mlp":
                continue
            doc.add_paragraph(f"{modelo}: {info.get('params')}")

    def _pruebas_estadisticas(self, doc: Document, contexto: ReportContext) -> None:
        doc.add_heading("5. Validación estadística", level=1)
        friedman = contexto.pruebas_estadisticas["friedman"]
        doc.add_paragraph(
            f"Prueba de Friedman sobre F1-macro: χ²={friedman['estadistico_chi2']:.4f}, "
            f"p={friedman['p_value']:.4f} "
            f"({'significativo' if friedman['significativo_alpha_0.05'] else 'no significativo'} "
            f"a α=0.05, con {friedman['n_folds']} folds)."
        )
        if "advertencia" in friedman:
            p = doc.add_paragraph(friedman["advertencia"])
            p.runs[0].italic = True

        doc.add_paragraph("Comparaciones pareadas de Wilcoxon (corrección de Holm-Bonferroni):")
        for r in contexto.pruebas_estadisticas["wilcoxon_pareado_holm"]:
            marca = " (significativo)" if r["significativo_holm_0.05"] else ""
            doc.add_paragraph(
                f"  • {r['modelo_a']} vs {r['modelo_b']}: p_holm={r['p_value_holm']:.4f}{marca}",
                style="List Bullet")

    def _conclusiones(self, doc: Document, contexto: ReportContext) -> None:
        doc.add_heading("6. Conclusiones", level=1)
        mejor = contexto.resumen_entrenamiento.get("mejor_modelo")
        doc.add_paragraph(
            f"El modelo {mejor} obtuvo el mejor F1-macro promedio en validación cruzada. "
            "Sin embargo, la prueba de Friedman/Wilcoxon debe consultarse antes de "
            "afirmar superioridad estadística, especialmente dado el número de folds "
            "usado en esta corrida (ver sección 5)."
        )
