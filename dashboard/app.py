"""
dashboard/app.py
=================
Dashboard interactivo (Streamlit + Plotly) de la plataforma. Todas las
visualizaciones se generan por código (Plotly) a partir de los datos
guardados en resultados/ - no son imágenes PNG pegadas, así que se pueden
hacer zoom, hover con valores exactos, y se ven nítidas en cualquier
resolución.

Arrancar con: streamlit run dashboard/app.py  (desde la raíz del proyecto)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api.model_loader import ModelRepository  # noqa: E402
from common.features import CLASS_NAMES, FEATURE_COLS, TIPOS_VALIDOS  # noqa: E402
from config import settings  # noqa: E402
from dashboard.auth import cerrar_sesion_boton, requiere_login  # noqa: E402
from exceptions import PlataformaError  # noqa: E402
from logging_config import configurar_logging  # noqa: E402

logger = configurar_logging(__name__)

st.set_page_config(page_title="El Porvenir - Priorización de Incidentes",
                    page_icon="🚨", layout="wide")

COLOR_PRIORIDAD = {"ALTA": "#d62728", "MEDIA": "#ff7f0e", "BAJA": "#7f7f7f"}
PLANTILLA = "plotly_white"


# ------------------------------------------------------------------
# Carga de datos (cacheada - no se recalcula en cada interacción)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando modelos entrenados...")
def obtener_repositorio() -> ModelRepository:
    repo = ModelRepository()
    repo.cargar()
    return repo


@st.cache_data(show_spinner=False)
def cargar_cv_resultados() -> pd.DataFrame | None:
    path = settings.resultados_dir / "cv_resultados_por_fold.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def cargar_predicciones_test() -> dict | None:
    path = settings.resultados_dir / "predicciones_test.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return {k: data[k] for k in data.files}


@st.cache_data(show_spinner=False)
def cargar_importancia() -> pd.DataFrame | None:
    path = settings.resultados_dir / "importancia_variables.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def cargar_pruebas_estadisticas() -> dict | None:
    path = settings.resultados_dir / "pruebas_estadisticas.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def cargar_dataset_features() -> pd.DataFrame | None:
    path = Path("data/dataset_features.csv")
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def cargar_dataset_definitivo() -> pd.DataFrame | None:
    path = Path("data/dataset_definitivo.csv")
    return pd.read_csv(path) if path.exists() else None


# ------------------------------------------------------------------
# Login
# ------------------------------------------------------------------
requiere_login()

st.sidebar.title("🚨 El Porvenir")
st.sidebar.caption(f"Sesión: {st.session_state.get('usuario', '')}")
pagina = st.sidebar.radio("Navegación", [
    "Predicción en vivo",
    "Comparación de modelos",
    "Curvas ROC",
    "Matrices de confusión",
    "Importancia de variables",
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

            probs = resultado["probabilidades"]
            fig = go.Figure(go.Bar(
                x=list(probs.keys()), y=list(probs.values()),
                marker_color=[COLOR_PRIORIDAD[k] for k in probs.keys()],
                text=[f"{v:.1%}" for v in probs.values()], textposition="outside"))
            fig.update_layout(template=PLANTILLA, yaxis_title="Probabilidad",
                               yaxis_range=[0, 1], height=350,
                               margin=dict(t=20, b=20))
            st.plotly_chart(fig, width="stretch")
            st.caption(f"Modelo: {resultado['modelo_usado']} · "
                       f"Tiempo de inferencia: {resultado['tiempo_inferencia_ms']:.1f} ms")
        except PlataformaError as exc:
            st.error(exc.mensaje)


# ------------------------------------------------------------------
# Página: Comparación de modelos
# ------------------------------------------------------------------
elif pagina == "Comparación de modelos":
    st.title("Comparación de modelos")
    cv = cargar_cv_resultados()
    if cv is None:
        st.warning("No se encontró cv_resultados_por_fold.csv. Corre train_models.py primero.")
    else:
        resumen = cv.groupby("modelo").agg(
            accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
            f1_mean=("f1_macro", "mean"), f1_std=("f1_macro", "std"),
            auc_mean=("roc_auc_macro", "mean"), auc_std=("roc_auc_macro", "std"),
            tiempo_mean=("tiempo_entrenamiento_s", "mean"),
        ).reset_index().sort_values("f1_mean", ascending=False)

        st.dataframe(resumen.style.highlight_max(subset=["f1_mean"], color="#c6efce")
                     .format({c: "{:.4f}" for c in resumen.columns if c != "modelo"}),
                     width="stretch")

        fig = go.Figure()
        for metrica, nombre in [("accuracy_mean", "Accuracy"), ("f1_mean", "F1-macro"),
                                 ("auc_mean", "ROC-AUC")]:
            std_col = metrica.replace("_mean", "_std")
            fig.add_trace(go.Bar(name=nombre, x=resumen["modelo"], y=resumen[metrica],
                                  error_y=dict(type="data", array=resumen[std_col])))
        fig.update_layout(barmode="group", template=PLANTILLA,
                           title="Métricas por modelo (media ± std, validación cruzada)",
                           yaxis_title="Valor", height=450)
        st.plotly_chart(fig, width="stretch")

        fig2 = px.bar(resumen, x="modelo", y="tiempo_mean", template=PLANTILLA,
                      title="Tiempo de entrenamiento promedio por fold (segundos)",
                      labels={"tiempo_mean": "segundos", "modelo": ""},
                      color="tiempo_mean", color_continuous_scale="Oranges")
        fig2.update_layout(height=400, coloraxis_showscale=False)
        st.plotly_chart(fig2, width="stretch")


# ------------------------------------------------------------------
# Página: Curvas ROC
# ------------------------------------------------------------------
elif pagina == "Curvas ROC":
    st.title("Curvas ROC (One-vs-Rest)")
    pred = cargar_predicciones_test()
    if pred is None:
        st.warning("No se encontró predicciones_test.npz. Corre export_resultados_dashboard.py primero.")
    else:
        modelo_sel = st.selectbox("Modelo", repo.modelos_disponibles)
        y_test = pred["y_test"]
        proba = pred[f"proba_{modelo_sel}"]
        y_bin = label_binarize(y_test, classes=[0, 1, 2])

        fig = go.Figure()
        for i, clase in enumerate(CLASS_NAMES):
            fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{clase} (AUC={auc(fpr, tpr):.3f})",
                                      line=dict(color=COLOR_PRIORIDAD[clase], width=2.5)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                  line=dict(dash="dash", color="gray"), showlegend=False))
        fig.update_layout(template=PLANTILLA, xaxis_title="FPR", yaxis_title="TPR",
                           title=f"Curvas ROC - {modelo_sel}", height=520,
                           legend=dict(x=0.6, y=0.1))
        st.plotly_chart(fig, width="stretch")

        st.caption("Comparación de AUC (clase ALTA) entre todos los modelos")
        auc_por_modelo = []
        for m in repo.modelos_disponibles:
            p = pred[f"proba_{m}"]
            fpr, tpr, _ = roc_curve(y_bin[:, 0], p[:, 0])
            auc_por_modelo.append({"modelo": m, "auc_alta": auc(fpr, tpr)})
        df_auc = pd.DataFrame(auc_por_modelo).sort_values("auc_alta", ascending=True)
        fig2 = px.bar(df_auc, x="auc_alta", y="modelo", orientation="h", template=PLANTILLA,
                      range_x=[0.5, 1], labels={"auc_alta": "AUC (clase ALTA)", "modelo": ""})
        st.plotly_chart(fig2, width="stretch")


# ------------------------------------------------------------------
# Página: Matrices de confusión
# ------------------------------------------------------------------
elif pagina == "Matrices de confusión":
    st.title("Matrices de confusión")
    pred = cargar_predicciones_test()
    if pred is None:
        st.warning("No se encontró predicciones_test.npz. Corre export_resultados_dashboard.py primero.")
    else:
        cols = st.columns(len(repo.modelos_disponibles))
        y_test = pred["y_test"]
        for col, modelo in zip(cols, repo.modelos_disponibles):
            with col:
                cm = confusion_matrix(y_test, pred[f"pred_{modelo}"])
                fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                 x=CLASS_NAMES, y=CLASS_NAMES, template=PLANTILLA,
                                 labels=dict(x="Predicho", y="Real", color="N°"))
                fig.update_layout(title=dict(text=modelo.replace("_", " "), font=dict(size=11)),
                                  height=320, margin=dict(t=40, b=10, l=10, r=10),
                                  coloraxis_showscale=False)
                st.plotly_chart(fig, width="stretch")


# ------------------------------------------------------------------
# Página: Importancia de variables
# ------------------------------------------------------------------
elif pagina == "Importancia de variables":
    st.title("Importancia de variables")
    st.caption("Permutation importance: cuánto cae el F1-macro al aleatorizar cada feature.")
    imp = cargar_importancia()
    if imp is None:
        st.warning("No se encontró importancia_variables.csv. Corre export_resultados_dashboard.py primero.")
    else:
        modelo_sel = st.selectbox("Modelo", repo.modelos_disponibles, key="imp_modelo")
        df_m = imp[imp["modelo"] == modelo_sel].sort_values("importancia_media", ascending=True).tail(12)
        fig = px.bar(df_m, x="importancia_media", y="feature", orientation="h",
                     error_x="importancia_std", template=PLANTILLA,
                     labels={"importancia_media": "Caída de F1-macro", "feature": ""},
                     color="importancia_media", color_continuous_scale="Blues")
        fig.update_layout(height=500, coloraxis_showscale=False,
                           title=f"Top 12 variables - {modelo_sel}")
        st.plotly_chart(fig, width="stretch")


# ------------------------------------------------------------------
# Página: Heatmaps
# ------------------------------------------------------------------
elif pagina == "Heatmaps":
    st.title("Heatmaps")
    tab1, tab2 = st.tabs(["Geográfico (densidad de incidentes)", "Correlación (features)"])

    with tab1:
        df_def = cargar_dataset_definitivo()
        if df_def is None:
            st.warning("No se encontró dataset_definitivo.csv. Corre build_seed.py primero.")
        else:
            fig = px.density_heatmap(df_def, x="longitud", y="latitud", nbinsx=60, nbinsy=60,
                                      color_continuous_scale="Reds", template=PLANTILLA,
                                      title="Densidad de incidentes - El Porvenir")
            fig.update_layout(height=600)
            st.plotly_chart(fig, width="stretch")

    with tab2:
        df_feat = cargar_dataset_features()
        if df_feat is None:
            st.warning("No se encontró dataset_features.csv. Corre feature_engineering.py primero.")
        else:
            numeric_cols = ["hora", "dia_semana", "mes", "dist_hermelinda_km",
                             "dist_sanchez_carrion_km", "dist_parque_industrial_km",
                             "dist_cesar_vallejo_km", "dist_hotspot_mas_cercano_km",
                             "prioridad_encoded"]
            corr = df_feat[numeric_cols].corr().round(2)
            fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                             zmin=-1, zmax=1, template=PLANTILLA, aspect="auto",
                             title="Matriz de correlación")
            fig.update_layout(height=650)
            st.plotly_chart(fig, width="stretch")


# ------------------------------------------------------------------
# Página: Pruebas estadísticas
# ------------------------------------------------------------------
elif pagina == "Pruebas estadísticas":
    st.title("Validación estadística")
    resultado = cargar_pruebas_estadisticas()
    cv = cargar_cv_resultados()
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
            st.dataframe(pd.DataFrame(resultado["nemenyi_p_values"]).round(4), width="stretch")

        st.subheader("Wilcoxon pareado (corrección de Holm-Bonferroni)")
        st.dataframe(pd.DataFrame(resultado["wilcoxon_pareado_holm"]), width="stretch")

        if cv is not None:
            st.subheader("Distribución de F1-macro por modelo (por fold)")
            orden = cv.groupby("modelo")["f1_macro"].mean().sort_values(ascending=False).index
            fig = px.box(cv, x="modelo", y="f1_macro", points="all", template=PLANTILLA,
                         category_orders={"modelo": list(orden)})
            fig.update_layout(height=480, xaxis_title="", yaxis_title="F1-macro")
            st.plotly_chart(fig, width="stretch")
