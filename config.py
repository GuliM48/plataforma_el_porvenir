"""
config.py
=========
Configuración centralizada del proyecto, leída desde variables de entorno
(.env). Un único objeto `settings` importable desde cualquier módulo
(API, dashboard, reportes) - evita rutas y credenciales hardcodeadas
dispersas por el código.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- rutas de datos y artefactos ---
    data_dir: Path = BASE_DIR / "data"
    models_dir: Path = BASE_DIR / "models"
    resultados_dir: Path = BASE_DIR / "resultados"
    figuras_dir: Path = BASE_DIR / "figuras_entrenamiento"
    reportes_dir: Path = BASE_DIR / "reportes_generados"

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_title: str = "API de Priorización de Incidentes - El Porvenir"
    api_default_model: str = "LightGBM"  # nombre del modelo usado por defecto en /predict
    cors_allowed_origins: list[str] = Field(default_factory=lambda: ["*"])

    # --- Dashboard (Streamlit) ---
    dashboard_username: str = "admin"
    dashboard_password: str = "cambiar_esta_password"  # noqa: S105 - override via .env en producción

    # --- Gemini (Chatbot) ---
    gemini_api_key: str = ""

    # --- Logging ---
    log_level: str = "INFO"
    log_json: bool = True

    # --- Entrenamiento (referencia; los valores reales viven en train_models.py) ---
    random_state: int = 42


settings = Settings()
