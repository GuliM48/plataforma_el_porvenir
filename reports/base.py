"""
reports/base.py
=================
Interfaz común para los generadores de reportes (PDF, Excel, Word). Cada
formato concreto implementa `generar()` — esto es el patrón Strategy: el
módulo que orquesta (generar_reportes.py) no necesita saber CÓMO se arma
cada formato, solo que todos exponen la misma interfaz.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from reports.context import ReportContext


class ReportGenerator(ABC):
    """Contrato que deben cumplir todos los generadores de reportes."""

    extension: str  # ej. "pdf", "xlsx", "docx"

    @abstractmethod
    def generar(self, contexto: ReportContext, output_path: Path) -> Path:
        """Genera el reporte y devuelve la ruta del archivo creado."""
        raise NotImplementedError
