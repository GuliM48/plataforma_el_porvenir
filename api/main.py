"""
api/main.py
============
Aplicación FastAPI. Arrancar con:

    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

(ejecutar desde la raíz del proyecto, para que los imports de `common`,
`config`, etc. se resuelvan correctamente)
"""
from __future__ import annotations

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api.model_loader import repository  # noqa: E402
from api.schemas import (HealthResponse, IncidentePayload,  # noqa: E402
                          ModeloInfo, PrediccionResponse)
from config import settings  # noqa: E402
from exceptions import PlataformaError  # noqa: E402
from logging_config import configurar_logging, log_con_contexto  # noqa: E402

logger = configurar_logging(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Iniciando API - cargando modelos...")
    repository.cargar()
    logger.info("API lista. Modelos disponibles: %s", repository.modelos_disponibles)
    yield
    logger.info("Apagando API.")


app = FastAPI(
    title=settings.api_title,
    description="API de inferencia para priorización de incidentes de "
                "seguridad ciudadana en El Porvenir, Trujillo.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# Manejo centralizado de excepciones
# ------------------------------------------------------------------
@app.exception_handler(PlataformaError)
async def plataforma_error_handler(request: Request, exc: PlataformaError):
    logger.error("PlataformaError en %s: %s", request.url.path, exc.mensaje,
                 extra={"extra_data": {"detalle": exc.detalle, "path": request.url.path}})
    return JSONResponse(
        status_code=400,
        content={"error": type(exc).__name__, "mensaje": exc.mensaje, "detalle": exc.detalle},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Error no manejado en %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "InternalServerError", "mensaje": "Error interno del servidor."},
    )


@app.middleware("http")
async def middleware_logging(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    tiempo_ms = (time.perf_counter() - t0) * 1000
    log_con_contexto(logger, "info", "Solicitud procesada",
                      metodo=request.method, ruta=request.url.path,
                      status=response.status_code, tiempo_ms=round(tiempo_ms, 2))
    return response


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------
@app.get("/", tags=["meta"])
def root():
    return {"nombre": settings.api_title, "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    return HealthResponse(
        status="ok" if repository.cargado else "modelos_no_cargados",
        modelos_cargados=repository.modelos_disponibles,
    )


@app.get("/models", response_model=list[ModeloInfo], tags=["modelos"])
def listar_modelos():
    metricas = repository.todas_las_metricas()
    resultado = []
    for nombre in repository.modelos_disponibles:
        m = metricas.get(nombre, {})
        resultado.append(ModeloInfo(
            nombre=nombre,
            tipo="hibrido" if nombre.startswith("Hibrido") else "clasico",
            f1_macro_cv=m.get("f1_macro_cv"),
            accuracy_cv=m.get("accuracy_cv"),
            roc_auc_cv=m.get("roc_auc_cv"),
        ))
    return resultado


@app.post("/predict", response_model=PrediccionResponse, tags=["predicciones"])
def predecir(payload: IncidentePayload, modelo: str | None = None):
    """Predice la prioridad (ALTA/MEDIA/BAJA) de un incidente.

    Parámetro opcional `modelo` (query string) para elegir cuál de los 5
    modelos usar; por defecto usa `settings.api_default_model` (LightGBM).
    """
    resultado = repository.predecir(
        latitud=payload.latitud, longitud=payload.longitud, hora=payload.hora,
        dia_semana=payload.dia_semana, mes=payload.mes, tipo=payload.tipo,
        nombre_modelo=modelo,
    )
    log_con_contexto(logger, "info", "Predicción realizada",
                      prioridad=resultado["prioridad"], modelo=resultado["modelo_usado"])
    return PrediccionResponse(**resultado)
