"""
api/schemas.py
===============
Esquemas Pydantic (request/response) de la API. Al ser Pydantic, esto da
tipado estático + validación automática de entrada (rangos de hora,
dia_semana, mes, y el tipo de delito vienen validados por Pydantic antes de
que la lógica de negocio los toque).
"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from common.features import TIPOS_VALIDOS


class IncidentePayload(BaseModel):
    """Datos que reporta el ciudadano/app al crear un incidente."""

    latitud: float = Field(..., ge=-8.085, le=-7.995, description="Latitud dentro de El Porvenir")
    longitud: float = Field(..., ge=-79.025, le=-78.965, description="Longitud dentro de El Porvenir")
    hora: int = Field(..., ge=0, le=23, description="Hora del incidente (0-23)")
    dia_semana: int = Field(..., ge=0, le=6, description="0=lunes ... 6=domingo")
    mes: int = Field(..., ge=1, le=12, description="Mes (1-12)")
    tipo: str = Field(..., description=f"Uno de: {', '.join(TIPOS_VALIDOS)}")

    @field_validator("tipo")
    @classmethod
    def validar_tipo(cls, v: str) -> str:
        if v not in TIPOS_VALIDOS:
            raise ValueError(f"tipo debe ser uno de {TIPOS_VALIDOS}, recibido: {v!r}")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "latitud": -8.044, "longitud": -79.003, "hora": 22,
                "dia_semana": 5, "mes": 7, "tipo": "Robo",
            }
        }
    }


class PrediccionResponse(BaseModel):
    prioridad: str
    confianza: float = Field(..., ge=0, le=1)
    probabilidades: dict[str, float]
    modelo_usado: str
    tiempo_inferencia_ms: float


class ModeloInfo(BaseModel):
    nombre: str
    tipo: str
    f1_macro_cv: float | None = None
    accuracy_cv: float | None = None
    roc_auc_cv: float | None = None


class HealthResponse(BaseModel):
    status: str
    modelos_cargados: list[str]
