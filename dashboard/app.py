"""
dashboard/app.py
=================
Dashboard interactivo (Streamlit) de la plataforma. Arrancar con:

    streamlit run dashboard/app.py

(ejecutar desde la raíz del proyecto)

Páginas: Predicción en vivo, Comparación de modelos, Curvas ROC, Matrices de
confusión, Heatmaps, Pruebas estadísticas.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api.model_loader import ModelRepository  # noqa: E402
from common.features import TIPOS_VALIDOS  # noqa: E402
from config import settings  # noqa: E402
from dashboard.auth import cerrar_sesion_boton, requiere_login  # noqa: E402
from exceptions import PlataformaError  # noqa: E402
from logging_config import configurar_logging  # noqa: E402

logger = configurar_logging(__name__)

st.set_page_config(page_title="El Porvenir - Priorización de Incidentes",
                    page_icon="🚨", layout="wide")


@st.cache_resource(show_spinner="Cargando modelos entrenados...")
def obtener_repositorio() -> ModelRepository:
    repo = ModelRepository()
    repo.cargar()
    return repo


def cargar_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cargar_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, header=[0, 1], index_col=0)


# ------------------------------------------------------------------
# Login (bloquea el resto de la app hasta autenticarse)
# ------------------------------------------------------------------
requiere_login()

st.sidebar.title("🚨 El Porvenir")
st.sidebar.caption(f"Sesión: {st.session_state.get('usuario', '')}")
pagina = st.sidebar.radio("Navegación", [
    "Predicción en vivo",
    "Comparación de modelos",
    "Curvas ROC",
    "Matrices de confusión",
    "Heatmaps",
    "Pruebas estadísticas",
])
cerrar_sesion_boton()

try:
    repo = obtener_repositorio()
except PlataformaError as exc:
    st.error(f"No se pudieron cargar los modelos: {exc.mensaje}")
    st.info("Corre `python train_models.py` antes de usar el dashboard.")
    st.stop()


# ------------------------------------------------------------------
# Página: Predicción en vivo
# ------------------------------------------------------------------
if pagina == "Predicción en vivo":
    st.title("Predicción en vivo")
    st.caption("Simula un incidente y observa qué prioridad le asignaría cada modelo.")

    col1, col2 = st.columns(2)
    with col1:
        lat = st.slider("Latitud", -8.085, -7.995, -8.044, 0.001, format="%.3f")
        hora = st.slider("Hora del día", 0, 23, 22)
        mes = st.slider("Mes", 1, 12, 7)
    with col2:
        lng = st.slider("Longitud", -79.025, -78.965, -79.003, 0.001, format="%.3f")
        dia_semana = st.selectbox("Día de la semana", list(range(7)),
                                   format_func=lambda d: ["Lunes", "Martes", "Miércoles",
                                                           "Jueves", "Viernes", "Sábado",
                                                           "Domingo"][d])
        tipo = st.selectbox("Tipo de delito", TIPOS_VALIDOS)

    modelo_elegido = st.selectbox("Modelo a usar", repo.modelos_disponibles)

    if st.button("Predecir prioridad", type="primary"):
        try:
            resultado = repo.predecir(lat, lng, hora, dia_semana, mes, tipo, modelo_elegido)
            color = {"ALTA": "🔴", "MEDIA": "🟠", "BAJA": "⚪"}[resultado["prioridad"]]
            st.metric("Prioridad predicha", f"{color} {resultado['prioridad']}",
                      f"confianza {resultado['confianza']:.1%}")
            st.bar_chart(pd.Series(resultado["probabilidades"]))
            st.caption(f"Modelo: {resultado['modelo_usado']} · "
                       f"Tiempo de inferencia: {resultado['tiempo_inferencia_ms']:.1f} ms")
        except PlataformaError as exc:
            st.error(exc.mensaje)


# ------------------------------------------------------------------
# Página: Comparación de modelos
# ------------------------------------------------------------------
elif pagina == "Comparación de modelos":
    st.title("Comparación de modelos")
    tabla = cargar_csv(settings.resultados_dir / "tabla_comparativa_modelos.csv")
    if tabla is None:
        st.warning("No se encontró la tabla comparativa. Corre train_models.py primero.")
    else:
        metricas_key = [c for c in tabla.columns if c[1] == "mean"]
        resumen = tabla[metricas_key].copy()
        resumen.columns = [c[0] for c in metricas_key]
        st.dataframe(resumen.style.highlight_max(axis=0, color="#c6efce"), width='stretch')
        st.bar_chart(resumen[["accuracy", "f1_macro", "roc_auc_macro"]])
        st.caption("Tiempo de entrenamiento promedio por fold (segundos)")
        st.bar_chart(resumen["tiempo_entrenamiento_s"])

    img = settings.figuras_dir / "04_comparacion_modelos.png"
    if img.exists():
        st.image(str(img), width='stretch')


# ------------------------------------------------------------------
# Página: Curvas ROC
# ------------------------------------------------------------------
elif pagina == "Curvas ROC":
    st.title("Curvas ROC (One-vs-Rest)")
    img = settings.figuras_dir / "01_curvas_roc.png"
    if img.exists():
        st.image(str(img), width='stretch')
    else:
        st.warning("No se encontró la figura de curvas ROC. Corre train_models.py primero.")


# ------------------------------------------------------------------
# Página: Matrices de confusión
# ------------------------------------------------------------------
elif pagina == "Matrices de confusión":
    st.title("Matrices de confusión")
    img = settings.figuras_dir / "02_matrices_confusion.png"
    if img.exists():
        st.image(str(img), width='stretch')
    else:
        st.warning("No se encontró la figura de matrices de confusión. Corre train_models.py primero.")
    st.divider()
    st.subheader("Importancia de variables")
    img_imp = settings.figuras_dir / "03_importancia_variables.png"
    if img_imp.exists():
        st.image(str(img_imp), width='stretch')


# ------------------------------------------------------------------
# Página: Heatmaps
# ------------------------------------------------------------------
elif pagina == "Heatmaps":
    st.title("Heatmaps")
    tab1, tab2 = st.tabs(["Geográfico (incidentes)", "Correlación (features)"])
    with tab1:
        img = Path("figures/04_heatmap_geografico.png")
        if img.exists():
            st.image(str(img), width='stretch')
        else:
            st.warning("No se encontró el heatmap geográfico. Corre eda.py primero.")
    with tab2:
        img = Path("figures/03_matriz_correlacion.png")
        if img.exists():
            st.image(str(img), width='stretch')
        else:
            st.warning("No se encontró la matriz de correlación. Corre eda.py primero.")


# ------------------------------------------------------------------
# Página: Pruebas estadísticas
# ------------------------------------------------------------------
elif pagina == "Pruebas estadísticas":
    st.title("Validación estadística")
    resultado = cargar_json(settings.resultados_dir / "pruebas_estadisticas.json")
    if resultado is None:
        st.warning("No se encontró pruebas_estadisticas.json. Corre statistical_tests.py primero.")
    else:
        friedman = resultado["friedman"]
        st.subheader("Prueba de Friedman")
        col1, col2, col3 = st.columns(3)
        col1.metric("Estadístico χ²", f"{friedman['estadistico_chi2']:.3f}")
        col2.metric("p-value", f"{friedman['p_value']:.4f}")
        col3.metric("¿Significativo? (α=0.05)",
                    "Sí" if friedman["significativo_alpha_0.05"] else "No")
        if "advertencia" in friedman:
            st.info(friedman["advertencia"])

        if resultado.get("nemenyi_p_values"):
            st.subheader("Post-hoc de Nemenyi (p-values)")
            st.dataframe(pd.DataFrame(resultado["nemenyi_p_values"]).round(4),
                         width='stretch')

        st.subheader("Wilcoxon pareado (corrección de Holm-Bonferroni)")
        wilcoxon_df = pd.DataFrame(resultado["wilcoxon_pareado_holm"])
        st.dataframe(wilcoxon_df, width='stretch')

        img = settings.figuras_dir / "05_boxplot_comparacion_estadistica.png"
        if img.exists():
            st.image(str(img), width='stretch')
