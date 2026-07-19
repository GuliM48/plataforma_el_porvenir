# Plataforma de Priorización de Incidentes - El Porvenir, Trujillo

Plataforma integral para el artículo científico sobre priorización de
incidentes de seguridad ciudadana en El Porvenir (UBIGEO 130102), usando
datos abiertos del SIDPOL y un pipeline de Machine Learning con 5 modelos
(3 clásicos + 2 híbridos), API de inferencia, dashboard interactivo y
generación automática de reportes.

## Arquitectura

```
ml-pipeline/
├── common/features.py        # Fuente única de verdad: hotspots, Haversine,
│                              # codificación cíclica, FEATURE_COLS
├── config.py                 # Configuración centralizada (.env)
├── logging_config.py         # Logging estructurado (JSON)
├── exceptions.py              # Jerarquía de excepciones propias
│
├── build_seed.py              # 1. Filtra SIDPOL, limpia, genera incidentes sintéticos
├── feature_engineering.py     # 2. Features finales (encodings, cíclicas, distancias)
├── eda.py                     # 3. Análisis exploratorio + 8 figuras
├── hyperparameter_search.py   # 4a. GridSearchCV (LR, RF) + Optuna (LightGBM)
├── genetic_algorithm.py       # 4b. Algoritmo Genético (arquitectura del Híbrido 2)
├── hybrid_models.py           # 4c. Híbrido 1 (Stacking+MLP) e Híbrido 2 (GA+MLP)
├── train_models.py            # 4. Entrena y compara los 5 modelos + figuras
├── statistical_tests.py       # 5. Friedman, Nemenyi, Wilcoxon (Holm-Bonferroni)
├── generar_reportes.py        # 6. Orquesta los 3 reportes (PDF/Excel/Word)
│
├── api/                        # API REST (FastAPI)
│   ├── main.py                 # Endpoints: /predict, /models, /health
│   ├── model_loader.py         # Repository: carga los 5 modelos + scaler
│   └── schemas.py              # Esquemas Pydantic (tipado + validación)
│
├── dashboard/                  # Dashboard interactivo (Streamlit)
│   ├── app.py                  # 6 páginas: predicción, comparación, ROC,
│   │                            # matrices, heatmaps, pruebas estadísticas
│   └── auth.py                 # Login (credenciales vía .env)
│
├── reports/                     # Generación automática de reportes
│   ├── base.py                  # Interfaz común (patrón Strategy)
│   ├── context.py                # Carga de datos compartida (DRY)
│   ├── pdf_report.py             # reportlab
│   ├── excel_report.py           # openpyxl (4 hojas)
│   └── word_report.py            # python-docx
│
├── tests/                       # pytest (15 tests)
│   ├── test_features.py         # Unit tests de common/features.py
│   ├── test_api.py               # Tests de integración de la API
│   └── test_dashboard.py         # Smoke test de las 6 páginas (AppTest)
│
├── data/                       # Datasets (crudo y procesados)
├── models/                     # Modelos entrenados (.pkl / .h5)
├── resultados/                 # Métricas, hiperparámetros, pruebas estadísticas
├── figuras_entrenamiento/      # ROC, matrices, importancia, comparación, boxplot
├── figures/                    # Figuras del EDA
├── reportes_generados/         # PDF / Excel / Word generados
├── .env.example                # Plantilla de configuración
└── requirements.txt
```

### Por qué esta estructura (principios de diseño)

- **Single source of truth**: `common/features.py` define hotspots, distancia
  Haversine, codificación cíclica y el orden de columnas (`FEATURE_COLS`) UNA
  sola vez. Tanto el pipeline batch (`build_seed.py`, `feature_engineering.py`,
  `train_models.py`) como la API de inferencia lo importan — así el vector de
  features en entrenamiento es exactamente igual al de producción (evita el
  bug clásico de "training/serving skew").
- **Repository pattern**: `api/model_loader.py::ModelRepository` encapsula
  la carga de los 5 modelos (mezcla de `.pkl` y `.h5`) detrás de una interfaz
  simple (`predecir(...)`); ni la API ni el dashboard necesitan saber cómo
  está serializado cada modelo.
- **Strategy pattern**: `reports/base.py::ReportGenerator` es la interfaz que
  implementan los 3 generadores de reportes. `generar_reportes.py` no sabe
  (ni le importa) cómo se arma un PDF vs un Excel.
- **Excepciones propias** (`exceptions.py`) en vez de dejar que errores de
  librerías de terceros exploten hasta el usuario — manejo centralizado vía
  `@app.exception_handler` en FastAPI.
- **Tipado estático**: type hints en todas las firmas de función (Python
  3.11+, `from __future__ import annotations` donde aplica).
- **Logging estructurado**: `logging_config.py` emite JSON por línea (fácil
  de indexar en Render/CloudWatch/ELK si se despliega en producción).
- **Configuración por `.env`**: ninguna ruta ni credencial hardcodeada;
  `config.py` centraliza todo vía `pydantic-settings`.

**Demo Dashboard analítico y de monitoreo territorial**: https://porvenirprediccionincidentes.streamlit.app
**User** : admin
**Password**: admin
**API de inferencia y servicios backend**: https://plataforma-el-porvenir.onrender.com

## Instalación

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # y ajustar credenciales/valores
```

## Uso — pipeline completo (en orden)

```bash
python build_seed.py             # -> data/dataset_definitivo.csv
python feature_engineering.py    # -> data/dataset_features.csv
python eda.py                    # -> figures/*.png (8 figuras)
python train_models.py           # -> models/*.pkl,*.h5 ; resultados/ ; figuras_entrenamiento/*.png (4 figuras)
python statistical_tests.py      # -> resultados/pruebas_estadisticas.json ; figuras_entrenamiento/05_*.png
python generar_reportes.py       # -> reportes_generados/reporte_entrenamiento.{pdf,xlsx,docx}
```

## Uso — servir la plataforma

```bash
# API (puerto 8000, docs interactivas en /docs)
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Dashboard (puerto 8501)
streamlit run dashboard/app.py
```

Login del dashboard: usuario/contraseña definidos en `.env`
(`DASHBOARD_USERNAME` / `DASHBOARD_PASSWORD`; por defecto `admin` /
`cambiar_esta_password` — **cámbialo antes de desplegar**).

Ejemplo de request a la API:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"latitud": -8.044, "longitud": -79.003, "hora": 22, "dia_semana": 5, "mes": 7, "tipo": "Robo"}'
```

## Tests

```bash
pytest tests/ -v
```
15 tests: validación de features (Haversine, one-hot, rangos), integración
de la API (`/health`, `/models`, `/predict`, casos de error 422), y smoke
test del dashboard completo (6 páginas, sin excepciones) vía
`streamlit.testing.v1.AppTest`.

## Resultados actuales (referencia)

| Modelo | Accuracy | F1-macro | ROC-AUC |
|---|---|---|---|
| LightGBM | 0.900 | **0.860** | 0.939 |
| Random Forest | 0.899 | 0.860 | 0.938 |
| Regresión Logística | 0.899 | 0.860 | 0.934 |
| Híbrido 1 (Stacking+MLP) | 0.897 | 0.858 | 0.939 |
| Híbrido 2 (GA+MLP) | 0.892 | 0.855 | 0.937 |

Prueba de Friedman sobre F1-macro: χ²=11.52, p=0.021 (significativo a
α=0.05, con 5 folds). Wilcoxon pareado con corrección de Holm-Bonferroni: sin
pares individualmente significativos (ver `resultados/pruebas_estadisticas.json`
para el detalle y la advertencia sobre potencia estadística con pocos folds).

## Notas importantes / decisiones metodológicas

1. **UBIGEO corregido**: El Porvenir es **130102**, no 130109 (Salaverry) —
   verificado contra el dataset real, no contra documentación de terceros.
2. **`prioridad` NO es una función pura de `tipo`**: si lo fuera, el
   problema de clasificación sería trivial (~100% accuracy memorizando una
   tabla de 7 filas) y las variables de ubicación/hora no tendrían ningún
   poder predictivo real. `build_seed.py` modula el nivel base de prioridad
   (definido por tipo) según hora, día y cercanía a hotspots — ver el
   comentario extenso en ese archivo para la justificación completa, útil
   para la sección de Metodología del artículo.
3. **Entorno de cómputo**: desarrollado y probado en un sandbox de **1 sola
   CPU**. Por eso `n_jobs=1` en todos los modelos y los presupuestos de
   búsqueda de hiperparámetros (`OPTUNA_TRIALS`, `GA_POBLACION`,
   `GA_GENERACIONES`, `N_REPEATS` en `train_models.py`) están en valores
   conservadores, documentados como constantes editables al inicio del
   archivo. Con más núcleos disponibles, subir esos valores para una
   búsqueda más fina y más folds (mayor potencia estadística en Friedman/
   Nemenyi).
4. **Login del dashboard**: es autenticación simple (usuario/password desde
   `.env`), apropiada para un proyecto académico. Para producción real,
   reemplazar por autenticación contra una tabla de usuarios con contraseñas
   hasheadas (bcrypt) + JWT — no hay roles ni multi-usuario todavía.

## Despliegue (Render + Vercel + GitHub + Jira)

Según lo indicado por el profesor:
- **Backend/API (FastAPI) -> Render**: usar `uvicorn api.main:app --host
  0.0.0.0 --port $PORT` como start command; variables de entorno del `.env`
  configuradas en el panel de Render (nunca subir `.env` real al repo).
- **Frontend -> Vercel**: si el frontend de la app móvil/web (`frontend-clean`
  en el otro repo) es una app Next.js/web, Vercel es el destino correcto; si
  es una app Expo/React Native pura, el destino real es Expo/EAS, no Vercel.
- **Dashboard (Streamlit)**: Render también soporta Streamlit (start command
  `streamlit run dashboard/app.py --server.port $PORT --server.address
  0.0.0.0`), o Streamlit Community Cloud como alternativa gratuita.
- **Control de versiones**: GitHub (ya en uso).
- **Documentación**: Jira, según lo indicado — este README cubre la parte
  técnica; las historias de usuario/sprints van en Jira aparte.
