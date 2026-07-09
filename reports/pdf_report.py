from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from dashboard.i18n import get_text
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
            Paragraph(get_text("reports.cover_title", contexto.lang), ESTILOS["TituloPortada"]),
            Paragraph(get_text("reports.cover_subtitle", contexto.lang), ESTILOS["TituloPortada"]),
            Spacer(1, 1 * cm),
            Paragraph(get_text("reports.cover_desc", contexto.lang), ESTILOS["Subtitulo"]),
            Spacer(1, 0.5 * cm),
            Paragraph(f"{get_text('reports.cover_date', contexto.lang)}{contexto.fecha_generacion}", ESTILOS["Subtitulo"]),
        ]

    def _resumen_ejecutivo(self, contexto: ReportContext) -> list:
        mejor = contexto.resumen_entrenamiento.get("mejor_modelo")
        f1 = contexto.resumen_entrenamiento.get("f1_macro_cv_promedio")
        cfg = contexto.resumen_entrenamiento.get("config", {})
        lang = contexto.lang
        texto = (
            f"{get_text('reports.executive_summary', lang)} "
            f"(n_splits={cfg.get('n_splits')}, "
            f"n_repeats={cfg.get('n_repeats')}). "
            f"<b>{get_text('reports.executive_best', lang)} {mejor} "
            f"(F1-macro = {f1:.4f}).</b>"
        )
        return [
            Paragraph(get_text("reports.section1", lang), ESTILOS["Heading1"]),
            Paragraph(texto, ESTILOS["CuerpoJustificado"]),
            Spacer(1, 0.3 * cm),
        ]

    def _tabla_comparativa(self, contexto: ReportContext) -> list:
        tabla_df = contexto.tabla_comparativa
        metricas = [c for c in tabla_df.columns if c[1] == "mean"]
        mejor_modelo = contexto.resumen_entrenamiento.get("mejor_modelo")
        lang = contexto.lang

        cabecera = [get_text("reports.models_header", lang)] + [m[0] for m in metricas]
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

        return [Paragraph(get_text("reports.section2", lang), ESTILOS["Heading1"]), t, Spacer(1, 0.5 * cm)]

    def _figuras(self, contexto: ReportContext) -> list:
        lang = contexto.lang
        elementos = [Paragraph(get_text("reports.section3", lang), ESTILOS["Heading1"])]
        titulos = {
            "roc": get_text("reports.fig_roc", lang),
            "matrices_confusion": get_text("reports.fig_confusion", lang),
            "importancia_variables": get_text("reports.fig_importance", lang),
            "comparacion_modelos": get_text("reports.fig_comparison", lang),
            "boxplot_estadistico": get_text("reports.fig_boxplot", lang),
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
        elementos = [Paragraph(get_text("reports.section4", contexto.lang), ESTILOS["Heading1"])]
        for modelo, info in contexto.hiperparametros.items():
            if modelo == "ga_mlp":
                continue
            elementos.append(Paragraph(f"<b>{modelo}:</b> {info.get('params')}",
                                        ESTILOS["CuerpoJustificado"]))
        return elementos

    def _pruebas_estadisticas(self, contexto: ReportContext) -> list:
        lang = contexto.lang
        elementos = [Paragraph(get_text("reports.section5", lang), ESTILOS["Heading1"])]
        friedman = contexto.pruebas_estadisticas["friedman"]
        sig = (get_text("reports.significant_yes", lang)
               if friedman["significativo_alpha_0.05"]
               else get_text("reports.significant_no", lang))
        texto = (
            f"{get_text('reports.friedman_text', lang)}: "
            f"χ²={friedman['estadistico_chi2']:.4f}, "
            f"p={friedman['p_value']:.4f} "
            f"({sig} "
            f"a α=0.05, con {friedman['n_folds']} folds)."
        )
        elementos.append(Paragraph(texto, ESTILOS["CuerpoJustificado"]))
        if "advertencia" in friedman:
            elementos.append(Paragraph(f"<i>{friedman['advertencia']}</i>", ESTILOS["CuerpoJustificado"]))

        filas = [
            [get_text("reports.wilcoxon_header_a", lang),
             get_text("reports.wilcoxon_header_b", lang),
             get_text("reports.wilcoxon_p", lang),
             get_text("reports.wilcoxon_holm", lang),
             get_text("reports.wilcoxon_sig", lang)]
        ]
        for r in contexto.pruebas_estadisticas["wilcoxon_pareado_holm"]:
            filas.append([r["modelo_a"], r["modelo_b"], f"{r['p_value']:.4f}",
                          f"{r['p_value_holm']:.4f}",
                          get_text("reports.significant_yes", lang) if r["significativo_holm_0.05"]
                          else get_text("reports.significant_no", lang)])
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
