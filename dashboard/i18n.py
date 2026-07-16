import streamlit as st

I18N = {
    "es": {
        "login": {"title": "Iniciar sesión", "user": "Usuario", "pass": "Contraseña", "btn": "Ingresar", "error": "Usuario o contraseña incorrectos.", "caption": "Plataforma de Priorización de Incidentes - El Porvenir"},
        "nav": {"prediction": "Predicción en vivo", "comparison": "Comparación de modelos", "roc": "Curvas ROC", "confusion": "Matrices de confusión", "importance": "Importancia de variables", "heatmaps": "Heatmaps", "stats": "Pruebas estadísticas"},
        "prediction": {"title": "Predicción en vivo", "caption": "Simula un incidente y observa qué prioridad le asignaría cada modelo.", "lat": "Latitud", "lng": "Longitud", "hour": "Hora del día", "month": "Mes", "weekday": "Día de la semana", "type": "Tipo de delito", "model": "Modelo a usar", "predict_btn": "Predecir prioridad", "priority": "Prioridad predicha", "confidence": "confianza", "probabilities": "Probabilidades", "inference_time": "Tiempo de inferencia"},
        "priority": {"ALTA": "ALTA", "MEDIA": "MEDIA", "BAJA": "BAJA"},
        "weekdays": ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"],
        "comparison": {"title": "Comparación de modelos", "metrics": "Métricas por modelo (media ± std, validación cruzada)", "time": "Tiempo de entrenamiento promedio por fold (segundos)"},
        "roc": {"title": "Curvas ROC (One-vs-Rest)", "fpr": "FPR", "tpr": "TPR", "auc_comparison": "Comparación de AUC (clase ALTA) entre todos los modelos"},
        "confusion": {"title": "Matrices de confusión", "predicted": "Predicho", "actual": "Real"},
        "importance": {"title": "Importancia de variables", "caption": "Permutation importance: cuánto cae el F1-macro al aleatorizar cada feature.", "drop": "Caída de F1-macro"},
        "heatmaps": {"geo": "Geográfico (densidad de incidentes)", "corr": "Correlación (features)", "geo_title": "Densidad de incidentes - El Porvenir", "corr_title": "Matriz de correlación"},
        "stats": {"title": "Validación estadística", "friedman": "Prueba de Friedman", "statistic": "Estadístico χ²", "pvalue": "p-value", "significant": "¿Significativo? (α=0.05)", "yes": "Sí", "no": "No", "nemenyi": "Post-hoc de Nemenyi (p-values)", "wilcoxon": "Wilcoxon pareado (corrección de Holm-Bonferroni)", "f1_distribution": "Distribución de F1-macro por modelo (por fold)"},
        "errors": {"no_models": "No se pudieron cargar los modelos", "run_train": "Corre `python train_models.py` antes de usar el dashboard.", "no_cv": "No se encontró cv_resultados_por_fold.csv. Corre train_models.py primero.", "no_pred": "No se encontró predicciones_test.npz. Corre export_resultados_dashboard.py primero.", "no_importance": "No se encontró importancia_variables.csv. Corre export_resultados_dashboard.py primero.", "no_stats": "No se encontró pruebas_estadisticas.json. Corre statistical_tests.py primero.", "no_dataset": "No se encontró dataset_definitivo.csv. Corre build_seed.py primero.", "no_features": "No se encontró dataset_features.csv. Corre feature_engineering.py primero."},
        "reports": {"pdf": "Reporte PDF", "excel": "Reporte Excel", "word": "Reporte Word", "generate": "Generar reporte", "downloading": "Descargando...",
            "cover_title": "Priorización de Incidentes de Seguridad Ciudadana",
            "cover_subtitle": "El Porvenir, Trujillo",
            "cover_desc": "Reporte técnico de entrenamiento y evaluación de modelos",
            "cover_date": "Generado automáticamente el ",
            "section1": "1. Resumen ejecutivo",
            "section2": "2. Tabla comparativa de modelos",
            "section3": "3. Figuras",
            "section4": "4. Hiperparámetros óptimos",
            "section5": "5. Validación estadística",
            "section6": "6. Conclusiones",
            "fig_roc": "3.1 Curvas ROC (One-vs-Rest)",
            "fig_confusion": "3.2 Matrices de confusión",
            "fig_importance": "3.3 Importancia de variables",
            "fig_comparison": "3.4 Comparación de métricas y tiempos",
            "fig_boxplot": "3.5 Distribución de F1-macro por modelo",
            "models_header": "Modelo",
            "metric_mean": "(media)",
            "metric_std": "(std)",
            "hyperparam_header": "Hiperparámetro",
            "hyperparam_value": "Valor",
            "cv_sheet": "CV por fold",
            "hyperparams_sheet": "Hiperparámetros",
            "stats_sheet": "Pruebas estadísticas",
            "stats_chi2": "Estadístico χ²",
            "stats_sig": "¿Significativo? (α=0.05)",
            "wilcoxon_header_a": "Modelo A",
            "wilcoxon_header_b": "Modelo B",
            "wilcoxon_p": "p-value",
            "wilcoxon_holm": "p-value (Holm)",
            "wilcoxon_sig": "Significativo",
            "executive_summary": "Se entrenaron y compararon 5 modelos de clasificación (3 clásicos: Regresión Logística, Random Forest y LightGBM; 2 híbridos: Stacking con meta-modelo MLP, y una red neuronal MLP optimizada con Algoritmo Genético) para predecir la prioridad (ALTA/MEDIA/BAJA) de incidentes en El Porvenir. La validación se realizó con RepeatedStratifiedKFold",
            "executive_best": "El modelo con mejor desempeño promedio fue",
            "friedman_text": "Prueba de Friedman sobre F1-macro",
            "wilcoxon_title": "Comparaciones pareadas de Wilcoxon (corrección de Holm-Bonferroni):",
            "significant_yes": "significativo",
            "significant_no": "no significativo",
            "conclusions": "Sin embargo, la prueba de Friedman/Wilcoxon debe consultarse antes de afirmar superioridad estadística, especialmente dado el número de folds usado en esta corrida (ver sección 5).",
        },
        "chatbot": {"title": "Asistente virtual", "placeholder": "Escribe tu pregunta...", "voice": "🎤 Hablar", "send": "Enviar", "clear": "Limpiar chat", "fab_open": "💬", "fab_close": "✕", "greeting": "¡Hola! Soy tu asistente del dashboard. ¿En qué puedo ayudarte?", "suggestions": ["¿Cómo predigo un incidente?", "¿Qué modelos están disponibles?", "¿Cómo interpreto las curvas ROC?", "¿Qué son los heatmaps?"], "help_prediction": "En 'Predicción en vivo' puedes simular un incidente ajustando latitud, longitud, hora, día, mes y tipo de delito.", "help_comparison": "La comparación de modelos muestra métricas de validación cruzada: accuracy, F1-macro y ROC-AUC.", "help_roc": "Las curvas ROC muestran el rendimiento One-vs-Rest para cada clase de prioridad.", "help_confusion": "Las matrices de confusión comparan predicciones vs valores reales por modelo.", "help_importance": "La importancia de variables usa permutation importance para identificar features más relevantes.", "help_heatmaps": "Los heatmaps incluyen densidad geográfica de incidentes y correlación entre variables.", "help_stats": "Las pruebas estadísticas validan si las diferencias entre modelos son significativas (Friedman, Nemenyi, Wilcoxon).", "default": "Puedo ayudarte con: predicciones en vivo, comparación de modelos, curvas ROC, matrices de confusión, importancia de variables, heatmaps y pruebas estadísticas. ¿Sobre qué tema necesitas ayuda?"},
        "session": "Sesión",
        "logout": "Cerrar sesión",
        "navigation": "Navegación",
        "app_title": "El Porvenir - Priorización de Incidentes",
    },
    "en": {
        "login": {"title": "Sign in", "user": "Username", "pass": "Password", "btn": "Sign in", "error": "Incorrect username or password.", "caption": "Incident Prioritization Platform - El Porvenir"},
        "nav": {"prediction": "Live Prediction", "comparison": "Model Comparison", "roc": "ROC Curves", "confusion": "Confusion Matrices", "importance": "Feature Importance", "heatmaps": "Heatmaps", "stats": "Statistical Tests"},
        "prediction": {"title": "Live Prediction", "caption": "Simulate an incident and see which priority each model would assign.", "lat": "Latitude", "lng": "Longitude", "hour": "Hour of day", "month": "Month", "weekday": "Day of week", "type": "Crime type", "model": "Model to use", "predict_btn": "Predict priority", "priority": "Predicted priority", "confidence": "confidence", "probabilities": "Probabilities", "inference_time": "Inference time"},
        "priority": {"ALTA": "HIGH", "MEDIA": "MEDIUM", "BAJA": "LOW"},
        "weekdays": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        "comparison": {"title": "Model Comparison", "metrics": "Metrics by model (mean ± std, cross-validation)", "time": "Average training time per fold (seconds)"},
        "roc": {"title": "ROC Curves (One-vs-Rest)", "fpr": "FPR", "tpr": "TPR", "auc_comparison": "AUC comparison (HIGH class) across all models"},
        "confusion": {"title": "Confusion Matrices", "predicted": "Predicted", "actual": "Actual"},
        "importance": {"title": "Feature Importance", "caption": "Permutation importance: how much F1-macro drops when randomizing each feature.", "drop": "F1-macro drop"},
        "heatmaps": {"geo": "Geographic (incident density)", "corr": "Correlation (features)", "geo_title": "Incident Density - El Porvenir", "corr_title": "Correlation Matrix"},
        "stats": {"title": "Statistical Validation", "friedman": "Friedman Test", "statistic": "χ² statistic", "pvalue": "p-value", "significant": "Significant? (α=0.05)", "yes": "Yes", "no": "No", "nemenyi": "Nemenyi Post-hoc (p-values)", "wilcoxon": "Paired Wilcoxon (Holm-Bonferroni correction)", "f1_distribution": "F1-macro distribution by model (per fold)"},
        "errors": {"no_models": "Could not load models", "run_train": "Run `python train_models.py` before using the dashboard.", "no_cv": "cv_resultados_por_fold.csv not found. Run train_models.py first.", "no_pred": "predicciones_test.npz not found. Run export_resultados_dashboard.py first.", "no_importance": "importancia_variables.csv not found. Run export_resultados_dashboard.py first.", "no_stats": "pruebas_estadisticas.json not found. Run statistical_tests.py first.", "no_dataset": "dataset_definitivo.csv not found. Run build_seed.py first.", "no_features": "dataset_features.csv not found. Run feature_engineering.py first."},
        "reports": {"pdf": "PDF Report", "excel": "Excel Report", "word": "Word Report", "generate": "Generate report", "downloading": "Downloading...",
            "cover_title": "Citizen Security Incident Prioritization",
            "cover_subtitle": "El Porvenir, Trujillo",
            "cover_desc": "Technical report of model training and evaluation",
            "cover_date": "Auto-generated on ",
            "section1": "1. Executive Summary",
            "section2": "2. Model Comparison Table",
            "section3": "3. Figures",
            "section4": "4. Optimal Hyperparameters",
            "section5": "5. Statistical Validation",
            "section6": "6. Conclusions",
            "fig_roc": "3.1 ROC Curves (One-vs-Rest)",
            "fig_confusion": "3.2 Confusion Matrices",
            "fig_importance": "3.3 Feature Importance",
            "fig_comparison": "3.4 Metrics and Time Comparison",
            "fig_boxplot": "3.5 F1-macro Distribution by Model",
            "models_header": "Model",
            "metric_mean": "(mean)",
            "metric_std": "(std)",
            "hyperparam_header": "Hyperparameter",
            "hyperparam_value": "Value",
            "cv_sheet": "CV by fold",
            "hyperparams_sheet": "Hyperparameters",
            "stats_sheet": "Statistical tests",
            "stats_chi2": "χ² statistic",
            "stats_sig": "Significant? (α=0.05)",
            "wilcoxon_header_a": "Model A",
            "wilcoxon_header_b": "Model B",
            "wilcoxon_p": "p-value",
            "wilcoxon_holm": "p-value (Holm)",
            "wilcoxon_sig": "Significant",
            "executive_summary": "5 classification models were trained and compared (3 classic: Logistic Regression, Random Forest and LightGBM; 2 hybrid: Stacking with MLP meta-model, and a Genetic Algorithm-optimized MLP neural network) to predict incident priority (HIGH/MEDIUM/LOW) in El Porvenir. Validation used RepeatedStratifiedKFold",
            "executive_best": "The model with the best average performance was",
            "friedman_text": "Friedman test on F1-macro",
            "wilcoxon_title": "Paired Wilcoxon comparisons (Holm-Bonferroni correction):",
            "significant_yes": "significant",
            "significant_no": "not significant",
            "conclusions": "However, the Friedman/Wilcoxon test should be consulted before asserting statistical superiority, especially given the number of folds used in this run (see section 5).",
        },
        "chatbot": {"title": "Virtual Assistant", "placeholder": "Type your question...", "voice": "🎤 Speak", "send": "Send", "clear": "Clear chat", "fab_open": "💬", "fab_close": "✕", "greeting": "Hi! I'm your dashboard assistant. How can I help you?", "suggestions": ["How do I predict an incident?", "What models are available?", "How do I interpret ROC curves?", "What are heatmaps?"], "help_prediction": "In 'Live Prediction' you can simulate an incident by adjusting latitude, longitude, hour, day, month and crime type.", "help_comparison": "Model Comparison shows cross-validation metrics: accuracy, F1-macro and ROC-AUC.", "help_roc": "ROC curves show One-vs-Rest performance for each priority class.", "help_confusion": "Confusion matrices compare predictions vs actual values per model.", "help_importance": "Feature Importance uses permutation importance to identify most relevant features.", "help_heatmaps": "Heatmaps include geographic incident density and variable correlation.", "help_stats": "Statistical tests validate whether differences between models are significant (Friedman, Nemenyi, Wilcoxon).", "default": "I can help you with: live predictions, model comparison, ROC curves, confusion matrices, feature importance, heatmaps and statistical tests. What topic do you need help with?"},
        "session": "Session",
        "logout": "Log out",
        "navigation": "Navigation",
        "app_title": "El Porvenir - Incident Prioritization",
    }
}


def get_text(key_path: str, lang: str | None = None) -> str | list | dict:
    if lang is None:
        lang = st.session_state.get("idioma", "es")
    keys = key_path.split(".")
    value = I18N.get(lang, I18N["es"])
    for k in keys:
        if isinstance(value, dict):
            value = value.get(k, key_path)
        else:
            return key_path
    return value


def cambiar_idioma():
    idioma = st.sidebar.selectbox(
        "🌐 Language / Idioma",
        ["Español", "English"],
        index=0 if st.session_state.get("idioma", "es") == "es" else 1,
    )
    st.session_state["idioma"] = "es" if idioma == "Español" else "en"
    return st.session_state["idioma"]


def traducir_clase(clase: str) -> str:
    lang = st.session_state.get("idioma", "es")
    m = {"ALTA": "priority.ALTA", "MEDIA": "priority.MEDIA", "BAJA": "priority.BAJA"}
    return get_text(m.get(clase, clase), lang)


def traducir_clases(clases: tuple[str, ...]) -> list[str]:
    return [traducir_clase(c) for c in clases]
