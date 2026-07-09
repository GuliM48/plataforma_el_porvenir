"""
dashboard/auth.py
==================
Autenticación simple basada en sesión de Streamlit, con credenciales
configurables vía .env (config.settings). No es un sistema de usuarios
completo (no hay roles ni base de datos de usuarios) — es un login básico
que cumple el requisito de "Login" antes del dashboard, apropiado para un
proyecto académico. Para producción real, reemplazar por autenticación
contra una tabla de usuarios con contraseñas hasheadas (bcrypt) + JWT.
"""
from __future__ import annotations

import hmac

import streamlit as st

from config import settings
from exceptions import AutenticacionError
from logging_config import configurar_logging

logger = configurar_logging(__name__)


def _credenciales_validas(usuario: str, password: str) -> bool:
    # hmac.compare_digest evita timing attacks en la comparación de strings
    usuario_ok = hmac.compare_digest(usuario, settings.dashboard_username)
    password_ok = hmac.compare_digest(password, settings.dashboard_password)
    return usuario_ok and password_ok


def requiere_login() -> None:
    """Bloquea el resto de la página hasta que el usuario inicie sesión.
    Llamar al inicio de dashboard/app.py."""
    if st.session_state.get("autenticado"):
        return

    st.title("Iniciar sesión")
    st.caption("Plataforma de Priorización de Incidentes - El Porvenir")
    with st.form("login_form"):
        usuario = st.text_input("Usuario")
        password = st.text_input("Contraseña", type="password")
        enviado = st.form_submit_button("Ingresar")

    if enviado:
        if _credenciales_validas(usuario, password):
            st.session_state["autenticado"] = True
            st.session_state["usuario"] = usuario
            logger.info("Login exitoso para usuario=%s", usuario)
            st.rerun()
        else:
            logger.warning("Intento de login fallido para usuario=%s", usuario)
            st.error("Usuario o contraseña incorrectos.")

    st.stop()  # detiene la ejecución del resto del script hasta login exitoso


def cerrar_sesion_boton() -> None:
    if st.sidebar.button("Cerrar sesión"):
        st.session_state.clear()
        st.rerun()
