"""
exceptions.py
=============
Jerarquía de excepciones propias de la plataforma. Permite manejo
centralizado de errores (en la API, vía exception handlers de FastAPI; en el
dashboard, vía try/except uniforme) en lugar de dejar que exploten
excepciones genéricas de librerías de terceros hasta el usuario final.
"""
from __future__ import annotations


class PlataformaError(Exception):
    """Excepción base de la plataforma. Todas las excepciones propias
    heredan de esta, lo que permite un único `except PlataformaError` en los
    puntos de entrada (API, dashboard, scripts) para capturar cualquier
    error "esperado" del dominio."""

    def __init__(self, mensaje: str, detalle: dict | None = None):
        super().__init__(mensaje)
        self.mensaje = mensaje
        self.detalle = detalle or {}


class ModeloNoDisponibleError(PlataformaError):
    """Se solicitó un modelo que no existe o no pudo cargarse desde disco."""


class EntradaInvalidaError(PlataformaError):
    """Los datos de entrada (features del incidente) no son válidos:
    tipo de delito desconocido, hora fuera de rango, coordenadas fuera del
    bounding box de El Porvenir, etc."""


class PrediccionError(PlataformaError):
    """Falló la inferencia del modelo (p.ej. el modelo cargado no coincide
    con la dimensión esperada de features)."""


class DatosNoEncontradosError(PlataformaError):
    """No se encontró un archivo de datos/resultados esperado (dataset,
    métricas de CV, figuras) — normalmente significa que un paso previo del
    pipeline (build_seed / train_models) no se ha corrido todavía."""


class ReporteGenerationError(PlataformaError):
    """Falló la generación de un reporte (PDF/Excel/Word)."""


class AutenticacionError(PlataformaError):
    """Credenciales inválidas al iniciar sesión en el dashboard."""
