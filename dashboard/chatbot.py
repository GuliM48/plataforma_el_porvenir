import sys
import os
from pathlib import Path
import html as html_module
import streamlit as st
from streamlit_float import *

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from dashboard.i18n import get_text


if "float_init_done" not in st.session_state:
    float_init()
    st.session_state.float_init_done = True


def _generar_respuesta_gemini(pregunta: str, lang: str) -> str | None:
    """Intenta generar respuesta con Gemini manejando errores y fallbacks dinámicos."""
    try:
        import google.generativeai as genai
    except ImportError:
        st.toast("⚠️ Falta instalar: pip install google-generativeai", icon="📦")
        return None

    # 1. Búsqueda múltiple de la API Key (Settings, Entorno o Secrets de Streamlit)
    api_key = getattr(settings, "gemini_api_key", None)
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")
    if not api_key and hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]

    # Si realmente no hay llave configurada, vamos pacíficamente a las palabras clave
    if not api_key or str(api_key).strip() == "":
        return None

    try:
        genai.configure(api_key=str(api_key).strip())
        
        system_prompt = (
            "Eres un asistente especializado en un dashboard de priorización de incidentes de seguridad ciudadana "
            "en El Porvenir, Trujillo. Responde SOLO preguntas relacionadas con las funcionalidades del dashboard: "
            "predicción en vivo, comparación de modelos, curvas ROC, matrices de confusión, importancia de variables, "
            "heatmaps, pruebas estadísticas, y generación de reportes. Sé conciso y útil. Si la pregunta no está "
            "relacionada con el dashboard, responde amablemente que solo puedes ayudar con el dashboard."
            if lang == "es" else
            "You are a specialized assistant for a citizen security incident prioritization dashboard in "
            "El Porvenir, Trujillo. Answer ONLY questions related to the dashboard features: "
            "live prediction, model comparison, ROC curves, confusion matrices, feature importance, "
            "heatmaps, statistical tests, and report generation. Be concise and helpful. If the question is "
            "not related to the dashboard, politely respond that you can only help with the dashboard."
        )

        # 2. Intentamos en cascada con los identificadores más estables del SDK
        modelos_a_probar = ["gemini-1.5-flash", "gemini-1.5-flash-latest", "gemini-2.0-flash", "gemini-pro"]
        
        ultimo_error = ""
        for model_name in modelos_a_probar:
            try:
                model = genai.GenerativeModel(
                    model_name,
                    system_instruction=system_prompt,
                )
                response = model.generate_content(
                    pregunta,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=500,
                        temperature=0.3,
                    ),
                )
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                ultimo_error = str(e)
                continue
        
        # Si ningún modelo funcionó, notificamos el error exacto
        if ultimo_error:
            print(f"❌ Error de API Gemini (Modelos): {ultimo_error}")
            st.toast(f"⚠️ Error en Gemini: {ultimo_error[:65]}...", icon="🚨")
        return None

    except Exception as e:
        print(f"❌ Error general en Gemini: {e}")
        st.toast(f"⚠️ Error de API Key o Conexión: {str(e)[:65]}...", icon="🚨")
        return None


def _generar_respuesta_keyword(pregunta: str, lang: str) -> str:
    pregunta_lower = pregunta.lower()
    keywords = {
        "prediction": [
            "predicción", "predict", "predecir", "simular", "incidente",
            "live", "latitud", "longitud",
        ],
        "comparison": [
            "comparación", "compare", "modelos", "métricas", "mejor",
            "models", "accuracy", "f1",
        ],
        "roc": ["roc", "curva", "curve", "auc", "curvas"],
        "confusion": ["confusión", "confusion", "matriz", "matrix"],
        "importance": [
            "importancia", "importance", "variables", "features",
            "relevant", "permutation",
        ],
        "heatmaps": [
            "heatmap", "mapa", "geográfico", "correlación",
            "correlation", "density", "calor",
        ],
        "stats": [
            "estadística", "statistical", "friedman", "wilcoxon",
            "nemenyi", "significativo", "significant", "prueba",
        ],
    }
    for tema, palabras in keywords.items():
        if any(p in pregunta_lower for p in palabras):
            return get_text(f"chatbot.help_{tema}", lang)
    return get_text("chatbot.default", lang)


def generar_respuesta(pregunta: str) -> str:
    lang = st.session_state.get("idioma", "es")
    respuesta = _generar_respuesta_gemini(pregunta, lang)
    if respuesta is None:
        respuesta = _generar_respuesta_keyword(pregunta, lang)
    return respuesta


def _escape_html(text: str) -> str:
    """Sanitiza texto para prevenir XSS."""
    return html_module.escape(text)


# ══════════════════════════════════════════════════════════════════════════════
# CSS - Estilos blindados con alineación perfecta
# ══════════════════════════════════════════════════════════════════════════════
_CHAT_CSS = """
<style>
/* ── FAB Botón flotante principal (Estrictamente el hermano de #fab-marker) ── */
div[data-testid="stElementContainer"]:has(#fab-marker) + div[data-testid="stElementContainer"] button {
    width: 60px !important;
    height: 60px !important;
    min-width: 60px !important;
    min-height: 60px !important;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%) !important;
    color: white !important;
    border-radius: 50% !important;
    font-size: 24px !important;
    border: none !important;
    box-shadow: 0 10px 30px rgba(99, 102, 241, 0.4) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    padding: 0 !important;
    z-index: 99999 !important;
}

div[data-testid="stElementContainer"]:has(#fab-marker) + div[data-testid="stElementContainer"] button:hover {
    transform: scale(1.08) !important;
    box-shadow: 0 15px 40px rgba(99, 102, 241, 0.6) !important;
}

/* ── CABECERA Y BOTONES (Limpiar 🗑️ y Cerrar ✕) ── */
div[data-testid="stHorizontalBlock"]:has(#header-marker) {
    align-items: center !important;
    padding-bottom: 12px !important;
    border-bottom: 1px solid #f1f5f9 !important;
    margin-bottom: 12px !important;
}

div[data-testid="stHorizontalBlock"]:has(#header-marker) button {
    background: #f8fafc !important;
    color: #475569 !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    width: 32px !important;
    height: 32px !important;
    min-width: 32px !important;
    min-height: 32px !important;
    font-size: 13px !important;
    padding: 0 !important;
    transition: all 0.2s !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

div[data-testid="stHorizontalBlock"]:has(#header-marker) button:hover {
    background: #f1f5f9 !important;
    color: #ef4444 !important;
    border-color: #cbd5e1 !important;
    transform: translateY(-1px);
}

/* ── SUGERENCIAS (Chips) ── */
button[kind="secondary"] {
    background: #f1f5f9 !important;
    color: #4f46e5 !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 16px !important;
    padding: 6px 14px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
    min-height: auto !important;
    width: 100% !important;
    margin-bottom: 4px !important;
}

button[kind="secondary"]:hover {
    background: #e0e7ff !important;
    border-color: #6366f1 !important;
    transform: translateY(-1px);
}

/* ── FORMULARIO Y BOTÓN DE ENVIAR ↑ (Alineación perfecta a 42px) ── */
div[data-testid="stForm"] {
    border: none !important;
    padding: 0 !important;
    margin: 10px 0 0 0 !important;
}

div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] {
    align-items: center !important;
    gap: 8px !important;
}

div[data-testid="stTextInput"] {
    margin-bottom: 0 !important;
}

div[data-testid="stTextInput"] input {
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 21px !important;
    padding: 0 16px !important;
    font-size: 13px !important;
    background: #f8fafc !important;
    height: 42px !important;
    line-height: 42px !important;
}

div[data-testid="stTextInput"] input:focus {
    border-color: #6366f1 !important;
    background: white !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
}

/* Botón circular enviar dentro de form */
div[data-testid="stFormSubmitButton"] {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    margin: 0 !important;
    padding: 0 !important;
}

div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 50% !important;
    width: 42px !important;
    min-width: 42px !important;
    max-width: 42px !important;
    height: 42px !important;
    min-height: 42px !important;
    max-height: 42px !important;
    font-size: 16px !important;
    padding: 0 !important;
    box-shadow: 0 4px 10px rgba(99, 102, 241, 0.3) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

div[data-testid="stFormSubmitButton"] button:hover {
    transform: scale(1.05);
    box-shadow: 0 6px 14px rgba(99, 102, 241, 0.4) !important;
}

/* ── BURBUJAS DE CHAT ── */
.bot-message, .user-message {
    display: flex;
    gap: 8px;
    margin-bottom: 10px;
    animation: msg-in 0.25s ease-out;
}

.user-message {
    justify-content: flex-end;
}

.bot-message .msg-avatar {
    width: 28px;
    height: 28px;
    min-width: 28px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    color: white;
}

.bot-message .msg-bubble {
    background: #f1f5f9;
    color: #1e293b;
    padding: 8px 12px;
    border-radius: 4px 14px 14px 14px;
    font-size: 13px;
    line-height: 1.45;
    max-width: 85%;
}

.user-message .msg-bubble {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    padding: 8px 12px;
    border-radius: 14px 4px 14px 14px;
    font-size: 13px;
    line-height: 1.45;
    max-width: 85%;
}

@keyframes msg-in {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ── TYPING INDICATOR ── */
.typing-dots {
    display: flex;
    gap: 4px;
    padding: 10px 14px;
    background: #f1f5f9;
    border-radius: 4px 14px 14px 14px;
    width: fit-content;
    align-items: center;
}

.typing-dots span {
    width: 6px;
    height: 6px;
    background: #94a3b8;
    border-radius: 50%;
    animation: bounce 1.4s infinite ease-in-out both;
}

.typing-dots span:nth-child(1) { animation-delay: -0.32s; }
.typing-dots span:nth-child(2) { animation-delay: -0.16s; }

@keyframes bounce {
    0%, 80%, 100% { transform: scale(0); }
    40% { transform: scale(1); }
}

/* ── WELCOME SECTION ── */
.welcome-section {
    text-align: center;
    padding: 10px 0;
}

.welcome-avatar {
    width: 48px;
    height: 48px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    margin: 0 auto 8px;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
}

.welcome-text {
    color: #1e293b;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 2px;
}

.welcome-subtext {
    color: #64748b;
    font-size: 12px;
    margin-bottom: 12px;
}
</style>
"""


def mostrar_chatbot():
    """Renderiza el chatbot flotante optimizado."""
    if "chat_abierto" not in st.session_state:
        st.session_state.chat_abierto = False
    if "chat_historial" not in st.session_state:
        st.session_state.chat_historial = []
    if "chat_typing" not in st.session_state:
        st.session_state.chat_typing = False

    lang = st.session_state.get("idioma", "es")
    is_open = st.session_state.chat_abierto

    st.markdown(_CHAT_CSS, unsafe_allow_html=True)

    # ── FAB (Botón Flotante con Marcador #fab-marker) ──
    fab_container = st.container()
    with fab_container:
        st.markdown('<div id="fab-marker" style="display:none;"></div>', unsafe_allow_html=True)
        fab_label = "✕" if is_open else "💬"
        if st.button(fab_label, key="chatbot_fab_toggle", help="Abrir/cerrar chat"):
            st.session_state.chat_abierto = not st.session_state.chat_abierto
            st.rerun()
    fab_container.float("bottom: 24px; right: 24px; width: 60px;")

    # ── CHATBOX ──
    if is_open:
        chat_container = st.container()
        with chat_container:
            # 1. CABECERA ALINEADA EN UNA SOLA FILA
            header_cols = st.columns([5, 1, 1])
            with header_cols[0]:
                st.markdown(f"""
                <div id="header-marker" style="display:none;"></div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="width: 36px; height: 36px; background: linear-gradient(135deg, #6366f1, #8b5cf6); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 18px; color: white; box-shadow: 0 4px 10px rgba(99, 102, 241, 0.3);">🤖</div>
                    <div>
                        <div style="font-weight: 700; font-size: 15px; color: #1e293b; line-height: 1.2;">{_escape_html(str(get_text("chatbot.title", lang)))}</div>
                        <div style="font-size: 11px; color: #10b981; font-weight: 600; display: flex; align-items: center; gap: 4px; margin-top: 2px;">
                            <span style="width: 6px; height: 6px; background: #10b981; border-radius: 50%; display: inline-block; box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.2);"></span> En línea
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with header_cols[1]:
                if st.button("🗑️", key="chat_clear_btn", help=str(get_text("chatbot.clear", lang))):
                    st.session_state.chat_historial = []
                    st.rerun()
            with header_cols[2]:
                if st.button("✕", key="chat_close_btn", help="Cerrar"):
                    st.session_state.chat_abierto = False
                    st.rerun()

            # 2. ÁREA DE MENSAJES (Altamente equilibrada en 330px para un scroll cómodo)
            messages_area = st.container(height=330)
            with messages_area:
                if not st.session_state.chat_historial:
                    st.markdown(f"""
                    <div class="welcome-section">
                        <div class="welcome-avatar">🤖</div>
                        <div class="welcome-text">{_escape_html(str(get_text("chatbot.greeting", lang)))}</div>
                        <div class="welcome-subtext">¿En qué puedo ayudarte hoy?</div>
                    </div>
                    """, unsafe_allow_html=True)

                    sugerencias = get_text("chatbot.suggestions", lang)
                    if isinstance(sugerencias, list):
                        for i, s in enumerate(sugerencias):
                            if st.button(s, key=f"suggestion_{i}", help=s):
                                st.session_state.chat_historial.append({"role": "user", "content": s})
                                st.session_state.chat_typing = True
                                st.rerun()
                else:
                    for m in st.session_state.chat_historial:
                        if m["role"] == "user":
                            st.markdown(f"""
                            <div class="user-message">
                                <div class="msg-bubble">{_escape_html(m["content"])}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="bot-message">
                                <div class="msg-avatar">🤖</div>
                                <div class="msg-bubble">{_escape_html(m["content"])}</div>
                            </div>
                            """, unsafe_allow_html=True)

                    if st.session_state.chat_typing:
                        st.markdown("""
                        <div class="bot-message">
                            <div class="msg-avatar">🤖</div>
                            <div class="typing-dots">
                                <span></span><span></span><span></span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            # 3. FORMULARIO DE INPUT (Alineado y encapsulado al fondo)
            with st.form(key="chat_input_form", clear_on_submit=True):
                input_cols = st.columns([5, 1])
                with input_cols[0]:
                    user_input = st.text_input(
                        get_text("chatbot.placeholder", lang),
                        key="chat_text_input",
                        label_visibility="collapsed",
                        placeholder=str(get_text("chatbot.placeholder", lang)),
                    )
                with input_cols[1]:
                    send_clicked = st.form_submit_button("↑", help="Enviar")

                if send_clicked and user_input.strip():
                    st.session_state.chat_historial.append({"role": "user", "content": user_input.strip()})
                    st.session_state.chat_typing = True
                    st.rerun()

            # Procesar respuesta de LLM en segundo plano
            if st.session_state.chat_typing and st.session_state.chat_historial:
                last_msg = st.session_state.chat_historial[-1]
                if last_msg["role"] == "user":
                    resp = generar_respuesta(last_msg["content"])
                    st.session_state.chat_historial.append({"role": "assistant", "content": resp})
                    st.session_state.chat_typing = False
                    st.rerun()

        # Configuración del flotante optimizada (540px de alto y overflow hidden para contener todo)
        chat_container.float(
            "bottom: 95px; right: 24px; width: 380px; height: 540px; "
            "background: white; border-radius: 20px; "
            "box-shadow: 0 20px 40px -10px rgba(0,0,0,0.25); "
            "padding: 16px; z-index: 99998; border: 1px solid #f1f5f9; "
            "overflow: hidden !important;"
        )