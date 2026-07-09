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
from dashboard.chatbot import mostrar_chatbot  # noqa: E402
from dashboard.i18n import (cambiar_idioma, get_text, traducir_clase,  # noqa: E402
                             traducir_clases)
from exceptions import PlataformaError  # noqa: E402
from logging_config import configurar_logging  # noqa: E402

logger = configurar_logging(__name__)

st.set_page_config(page_title=get_text("app_title"),
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
cambiar_idioma()
st.sidebar.caption(f"{get_text('session')}: {st.session_state.get('usuario', '')}")
pagina = st.sidebar.radio(get_text("navigation"), [
    get_text("nav.prediction"),
    get_text("nav.comparison"),
    get_text("nav.roc"),
    get_text("nav.confusion"),
    get_text("nav.importance"),
    get_text("nav.heatmaps"),
    get_text("nav.stats"),
])
cerrar_sesion_boton()

try:
    repo = obtener_repositorio()
except PlataformaError as exc:
    st.error(f"{get_text('errors.no_models')}: {exc.mensaje}")
    st.info(get_text("errors.run_train"))
    st.stop()


# ------------------------------------------------------------------
# Página: Predicción en vivo
# ------------------------------------------------------------------
if pagina == get_text("nav.prediction"):
    st.title(get_text("prediction.title"))
    st.caption(get_text("prediction.caption"))

    col1, col2 = st.columns(2)
    weekdays = get_text("weekdays")
    with col1:
        lat = st.slider(get_text("prediction.lat"), -8.085, -7.995, -8.044, 0.001, format="%.3f")
        hora = st.slider(get_text("prediction.hour"), 0, 23, 22)
        mes = st.slider(get_text("prediction.month"), 1, 12, 7)
    with col2:
        lng = st.slider(get_text("prediction.lng"), -79.025, -78.965, -79.003, 0.001, format="%.3f")
        dia_semana = st.selectbox(get_text("prediction.weekday"), list(range(7)),
                                   format_func=lambda d: weekdays[d])
        tipo = st.selectbox(get_text("prediction.type"), TIPOS_VALIDOS)

    modelo_elegido = st.selectbox(get_text("prediction.model"), repo.modelos_disponibles)

    if st.button(get_text("prediction.predict_btn"), type="primary"):
        try:
            resultado = repo.predecir(lat, lng, hora, dia_semana, mes, tipo, modelo_elegido)
            color = {"ALTA": "🔴", "MEDIA": "🟠", "BAJA": "⚪"}[resultado["prioridad"]]
            prioridad_traducida = traducir_clase(resultado["prioridad"])
            st.metric(get_text("prediction.priority"), f"{color} {prioridad_traducida}",
                      f"{get_text('prediction.confidence')} {resultado['confianza']:.1%}")

            probs = resultado["probabilidades"]
            fig = go.Figure(go.Bar(
                x=list(probs.keys()), y=list(probs.values()),
                marker_color=[COLOR_PRIORIDAD[k] for k in probs.keys()],
                text=[f"{v:.1%}" for v in probs.values()], textposition="outside"))
            fig.update_layout(template=PLANTILLA,
                               yaxis_title=get_text("prediction.probabilities"),
                               yaxis_range=[0, 1], height=350,
                               margin=dict(t=20, b=20))
            st.plotly_chart(fig, width="stretch")
            st.caption(f"{get_text('prediction.model')}: {resultado['modelo_usado']} \u00b7 "
                       f"{get_text('prediction.inference_time')}: {resultado['tiempo_inferencia_ms']:.1f} ms")
        except PlataformaError as exc:
            st.error(exc.mensaje)

    mostrar_chatbot()


# ------------------------------------------------------------------
# Página: Comparación de modelos
# ------------------------------------------------------------------
elif pagina == get_text("nav.comparison"):
    st.title(get_text("comparison.title"))
    cv = cargar_cv_resultados()
    if cv is None:
        st.warning(get_text("errors.no_cv"))
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
                           title=get_text("comparison.metrics"),
                           yaxis_title=get_text("comparison.title"), height=450)
        st.plotly_chart(fig, width="stretch")

        fig2 = px.bar(resumen, x="modelo", y="tiempo_mean", template=PLANTILLA,
                      title=get_text("comparison.time"),
                      labels={"tiempo_mean": "segundos", "modelo": ""},
                      color="tiempo_mean", color_continuous_scale="Oranges")
        fig2.update_layout(height=400, coloraxis_showscale=False)
        st.plotly_chart(fig2, width="stretch")

    mostrar_chatbot()


# ------------------------------------------------------------------
# Página: Curvas ROC
# ------------------------------------------------------------------
elif pagina == get_text("nav.roc"):
    st.title(get_text("roc.title"))
    pred = cargar_predicciones_test()
    if pred is None:
        st.warning(get_text("errors.no_pred"))
    else:
        modelo_sel = st.selectbox(get_text("prediction.model"), repo.modelos_disponibles)
        y_test = pred["y_test"]
        proba = pred[f"proba_{modelo_sel}"]
        y_bin = label_binarize(y_test, classes=[0, 1, 2])

        clases_traducidas = traducir_clases(CLASS_NAMES)

        fig = go.Figure()
        for i, clase in enumerate(CLASS_NAMES):
            fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                      name=f"{clases_traducidas[i]} (AUC={auc(fpr, tpr):.3f})",
                                      line=dict(color=COLOR_PRIORIDAD[clase], width=2.5)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                  line=dict(dash="dash", color="gray"), showlegend=False))
        fig.update_layout(template=PLANTILLA,
                           xaxis_title=get_text("roc.fpr"),
                           yaxis_title=get_text("roc.tpr"),
                           title=f"{get_text('roc.title')} - {modelo_sel}", height=520,
                           legend=dict(x=0.6, y=0.1))
        st.plotly_chart(fig, width="stretch")

        st.caption(get_text("roc.auc_comparison"))
        auc_por_modelo = []
        for m in repo.modelos_disponibles:
            p = pred[f"proba_{m}"]
            fpr, tpr, _ = roc_curve(y_bin[:, 0], p[:, 0])
            auc_por_modelo.append({"modelo": m, "auc_alta": auc(fpr, tpr)})
        df_auc = pd.DataFrame(auc_por_modelo).sort_values("auc_alta", ascending=True)
        fig2 = px.bar(df_auc, x="auc_alta", y="modelo", orientation="h", template=PLANTILLA,
                      range_x=[0.5, 1],
                      labels={"auc_alta": get_text("roc.auc_comparison"), "modelo": ""})
        st.plotly_chart(fig2, width="stretch")

    mostrar_chatbot()


# ------------------------------------------------------------------
# Página: Matrices de confusión
# ------------------------------------------------------------------
elif pagina == get_text("nav.confusion"):
    st.title(get_text("confusion.title"))
    pred = cargar_predicciones_test()
    if pred is None:
        st.warning(get_text("errors.no_pred"))
    else:
        cols = st.columns(len(repo.modelos_disponibles))
        y_test = pred["y_test"]
        clases_traducidas = traducir_clases(CLASS_NAMES)
        for col, modelo in zip(cols, repo.modelos_disponibles):
            with col:
                cm = confusion_matrix(y_test, pred[f"pred_{modelo}"])
                fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                 x=clases_traducidas, y=clases_traducidas, template=PLANTILLA,
                                 labels=dict(x=get_text("confusion.predicted"),
                                              y=get_text("confusion.actual"),
                                              color="N°"))
                fig.update_layout(title=dict(text=modelo.replace("_", " "), font=dict(size=11)),
                                  height=320, margin=dict(t=40, b=10, l=10, r=10),
                                  coloraxis_showscale=False)
                st.plotly_chart(fig, width="stretch")

    mostrar_chatbot()


# ------------------------------------------------------------------
# Página: Importancia de variables
# ------------------------------------------------------------------
elif pagina == get_text("nav.importance"):
    st.title(get_text("importance.title"))
    st.caption(get_text("importance.caption"))
    imp = cargar_importancia()
    if imp is None:
        st.warning(get_text("errors.no_importance"))
    else:
        modelo_sel = st.selectbox(get_text("prediction.model"), repo.modelos_disponibles, key="imp_modelo")
        df_m = imp[imp["modelo"] == modelo_sel].sort_values("importancia_media", ascending=True).tail(12)
        fig = px.bar(df_m, x="importancia_media", y="feature", orientation="h",
                     error_x="importancia_std", template=PLANTILLA,
                     labels={"importancia_media": get_text("importance.drop"), "feature": ""},
                     color="importancia_media", color_continuous_scale="Blues")
        fig.update_layout(height=500, coloraxis_showscale=False,
                           title=f"Top 12 variables - {modelo_sel}")
        st.plotly_chart(fig, width="stretch")

    mostrar_chatbot()


# ------------------------------------------------------------------
# Página: Heatmaps
# ------------------------------------------------------------------
elif pagina == get_text("nav.heatmaps"):
    st.title(get_text("heatmaps.title"))
    tab1, tab2 = st.tabs([get_text("heatmaps.geo"), get_text("heatmaps.corr")])

    with tab1:
        df_def = cargar_dataset_definitivo()
        if df_def is None:
            st.warning(get_text("errors.no_dataset"))
        else:
            fig = px.density_heatmap(df_def, x="longitud", y="latitud", nbinsx=60, nbinsy=60,
                                      color_continuous_scale="Reds", template=PLANTILLA,
                                      title=get_text("heatmaps.geo_title"))
            fig.update_layout(height=600)
            st.plotly_chart(fig, width="stretch")

    with tab2:
        df_feat = cargar_dataset_features()
        if df_feat is None:
            st.warning(get_text("errors.no_features"))
        else:
            numeric_cols = ["hora", "dia_semana", "mes", "dist_hermelinda_km",
                             "dist_sanchez_carrion_km", "dist_parque_industrial_km",
                             "dist_cesar_vallejo_km", "dist_hotspot_mas_cercano_km",
                             "prioridad_encoded"]
            corr = df_feat[numeric_cols].corr().round(2)
            fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                             zmin=-1, zmax=1, template=PLANTILLA, aspect="auto",
                             title=get_text("heatmaps.corr_title"))
            fig.update_layout(height=650)
            st.plotly_chart(fig, width="stretch")

    mostrar_chatbot()


# ------------------------------------------------------------------
# Página: Pruebas estadísticas
# ------------------------------------------------------------------
elif pagina == get_text("nav.stats"):
    st.title(get_text("stats.title"))
    resultado = cargar_pruebas_estadisticas()
    cv = cargar_cv_resultados()
    if resultado is None:
        st.warning(get_text("errors.no_stats"))
    else:
        friedman = resultado["friedman"]
        st.subheader(get_text("stats.friedman"))
        col1, col2, col3 = st.columns(3)
        col1.metric(get_text("stats.statistic"), f"{friedman['estadistico_chi2']:.3f}")
        col2.metric(get_text("stats.pvalue"), f"{friedman['p_value']:.4f}")
        col3.metric(get_text("stats.significant"),
                    get_text("stats.yes") if friedman["significativo_alpha_0.05"] else get_text("stats.no"))
        if "advertencia" in friedman:
            st.info(friedman["advertencia"])

        if resultado.get("nemenyi_p_values"):
            st.subheader(get_text("stats.nemenyi"))
            st.dataframe(pd.DataFrame(resultado["nemenyi_p_values"]).round(4), width="stretch")

        st.subheader(get_text("stats.wilcoxon"))
        st.dataframe(pd.DataFrame(resultado["wilcoxon_pareado_holm"]), width="stretch")

        if cv is not None:
            st.subheader(get_text("stats.f1_distribution"))
            orden = cv.groupby("modelo")["f1_macro"].mean().sort_values(ascending=False).index
            fig = px.box(cv, x="modelo", y="f1_macro", points="all", template=PLANTILLA,
                         category_orders={"modelo": list(orden)})
            fig.update_layout(height=480, xaxis_title="", yaxis_title="F1-macro")
            st.plotly_chart(fig, width="stretch")

    mostrar_chatbot()
