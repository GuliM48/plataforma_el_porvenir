"""
logging_config.py
==================
Logging estructurado (JSON) para toda la plataforma. Un log en JSON es
fácil de indexar/consultar si esto se despliega con un colector de logs
(ELK, CloudWatch, etc.) en Render/Vercel.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

from config import settings


class JSONFormatter(logging.Formatter):
    """Formatea cada registro de log como una línea JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        extra = getattr(record, "extra_data", None)
        if extra:
            payload["extra"] = extra
        return json.dumps(payload, ensure_ascii=False)


def configurar_logging(nombre: str = "plataforma_el_porvenir") -> logging.Logger:
    """Configura y devuelve el logger raíz de la aplicación. Llamar una sola
    vez al inicio de cada entrypoint (api/main.py, dashboard/app.py, etc.)."""
    logger = logging.getLogger(nombre)
    if logger.handlers:  # evita handlers duplicados si se llama más de una vez
        return logger

    logger.setLevel(settings.log_level)
    handler = logging.StreamHandler(sys.stdout)
    if settings.log_json:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_con_contexto(logger: logging.Logger, nivel: str, mensaje: str, **contexto):
    """Ayuda a loguear con datos estructurados extra sin repetir boilerplate.

    Ejemplo: log_con_contexto(logger, "info", "Predicción realizada",
                               modelo="LightGBM", prioridad="ALTA", tiempo_ms=12.4)
    """
    log_fn = getattr(logger, nivel.lower())
    log_fn(mensaje, extra={"extra_data": contexto})
