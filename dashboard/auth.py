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
from dashboard.i18n import get_text
from exceptions import AutenticacionError
from logging_config import configurar_logging

logger = configurar_logging(__name__)


def _credenciales_validas(usuario: str, password: str) -> bool:
    usuario_ok = hmac.compare_digest(usuario, settings.dashboard_username)
    password_ok = hmac.compare_digest(password, settings.dashboard_password)
    return usuario_ok and password_ok


def requiere_login() -> None:
    if st.session_state.get("autenticado"):
        return

    st.title(get_text("login.title"))
    st.caption(get_text("login.caption"))
    with st.form("login_form"):
        usuario = st.text_input(get_text("login.user"))
        password = st.text_input(get_text("login.pass"), type="password")
        enviado = st.form_submit_button(get_text("login.btn"))

    if enviado:
        if _credenciales_validas(usuario, password):
            st.session_state["autenticado"] = True
            st.session_state["usuario"] = usuario
            logger.info("Login exitoso para usuario=%s", usuario)
            st.rerun()
        else:
            logger.warning("Intento de login fallido para usuario=%s", usuario)
            st.error(get_text("login.error"))

    st.stop()


def cerrar_sesion_boton() -> None:
    if st.sidebar.button(get_text("logout")):
        st.session_state.clear()
        st.rerun()
