import sys
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
    try:
        import google.generativeai as genai

        api_key = settings.gemini_api_key
        if not api_key:
            return None

        genai.configure(api_key=api_key)
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
        model = genai.GenerativeModel(
            "gemini-2.0-flash",
            system_instruction=system_prompt,
        )
        response = model.generate_content(
            pregunta,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=500,
                temperature=0.3,
            ),
        )
        return response.text.strip() if response.text else None
    except Exception:
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
# CSS - Selectores específicos y delimitados
# ══════════════════════════════════════════════════════════════════════════════
_CHAT_CSS = """
<style>
/* ═══════════════════════════════════════════════════════════════════════════ */
/* FAB - Botón flotante (usando key específico de Streamlit) */
/* ═══════════════════════════════════════════════════════════════════════════ */
div[data-testid="stVerticalBlock"] > div:has(> div > button[kind="secondary"][aria-label="chatbot_fab_toggle"]) > div > button[kind="secondary"] {
    width: 60px !important;
    height: 60px !important;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%) !important;
    color: white !important;
    border-radius: 50% !important;
    font-size: 24px !important;
    border: none !important;
    box-shadow: 0 10px 40px rgba(99, 102, 241, 0.4) !important;
    transition: all 0.3s ease !important;
    padding: 0 !important;
    min-height: auto !important;
}

div[data-testid="stVerticalBlock"] > div:has(> div > button[kind="secondary"][aria-label="chatbot_fab_toggle"]) > div > button[kind="secondary"]:hover {
    transform: scale(1.1) !important;
    box-shadow: 0 15px 50px rgba(99, 102, 241, 0.5) !important;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* HEADER - Botones nativos de Streamlit estilizados */
/* ═══════════════════════════════════════════════════════════════════════════ */
/* Botón de limpiar (️) */
div[data-testid="stHorizontalBlock"] > div:has(> div > button[kind="secondary"][aria-label="chat_clear_btn"]) > div > button[kind="secondary"] {
    background: rgba(255,255,255,0.15) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 10px !important;
    width: 34px !important;
    height: 34px !important;
    font-size: 16px !important;
    padding: 0 !important;
    min-height: auto !important;
    transition: all 0.2s ease !important;
}

div[data-testid="stHorizontalBlock"] > div:has(> div > button[kind="secondary"][aria-label="chat_clear_btn"]) > div > button[kind="secondary"]:hover {
    background: rgba(255,255,255,0.3) !important;
    transform: translateY(-1px);
}

/* Botón de cerrar (✕) */
div[data-testid="stHorizontalBlock"] > div:has(> div > button[kind="secondary"][aria-label="chat_close_btn"]) > div > button[kind="secondary"] {
    background: rgba(255,255,255,0.15) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 10px !important;
    width: 34px !important;
    height: 34px !important;
    font-size: 16px !important;
    padding: 0 !important;
    min-height: auto !important;
    transition: all 0.2s ease !important;
}

div[data-testid="stHorizontalBlock"] > div:has(> div > button[kind="secondary"][aria-label="chat_close_btn"]) > div > button[kind="secondary"]:hover {
    background: rgba(255,255,255,0.3) !important;
    transform: translateY(-1px);
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* SUGERENCIAS - Chips */
/* ═══════════════════════════════════════════════════════════════════════════ */
/* Los botones de sugerencia tienen keys como "suggestion_0", "suggestion_1", etc. */
button[kind="secondary"][aria-label^="suggestion_"] {
    background: rgba(99, 102, 241, 0.08) !important;
    color: #6366f1 !important;
    border: 1.5px solid rgba(99, 102, 241, 0.2) !important;
    border-radius: 20px !important;
    padding: 6px 14px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    transition: all 0.25s ease !important;
    min-height: auto !important;
    white-space: nowrap !important;
}

button[kind="secondary"][aria-label^="suggestion_"]:hover {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border-color: transparent !important;
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(99, 102, 241, 0.3);
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* INPUT DE TEXTO - st.text_input estilizado */
/* ═══════════════════════════════════════════════════════════════════════════ */
div[data-testid="stTextInput"] input {
    border: 2px solid #e2e8f0 !important;
    border-radius: 24px !important;
    padding: 10px 16px !important;
    font-size: 13.5px !important;
    background: #f8fafc !important;
    transition: all 0.2s ease !important;
}

div[data-testid="stTextInput"] input:focus {
    border-color: #6366f1 !important;
    background: white !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
}

/* Botón de enviar */
div[data-testid="stHorizontalBlock"] > div:has(> div > button[kind="secondary"][aria-label="chat_send_btn"]) > div > button[kind="secondary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 50% !important;
    width: 40px !important;
    height: 40px !important;
    font-size: 18px !important;
    padding: 0 !important;
    min-height: auto !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3) !important;
    transition: all 0.2s ease !important;
}

div[data-testid="stHorizontalBlock"] > div:has(> div > button[kind="secondary"][aria-label="chat_send_btn"]) > div > button[kind="secondary"]:hover {
    transform: scale(1.1);
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* MENSAJES */
/* ═══════════════════════════════════════════════════════════════════════════ */
.bot-message {
    display: flex;
    gap: 10px;
    margin-bottom: 12px;
    animation: msg-in 0.3s ease;
}

.bot-message .msg-avatar {
    width: 30px;
    height: 30px;
    min-width: 30px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 15px;
}

.bot-message .msg-bubble {
    background: #f1f5f9;
    color: #1e293b;
    padding: 10px 14px;
    border-radius: 4px 16px 16px 16px;
    font-size: 13.5px;
    line-height: 1.5;
    max-width: 82%;
}

.user-message {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 12px;
    animation: msg-in 0.3s ease;
}

.user-message .msg-bubble {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    padding: 10px 14px;
    border-radius: 16px 4px 16px 16px;
    font-size: 13.5px;
    line-height: 1.5;
    max-width: 82%;
}

@keyframes msg-in {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* TYPING INDICATOR */
/* ═══════════════════════════════════════════════════════════════════════════ */
.typing-dots {
    display: flex;
    gap: 4px;
    padding: 12px 16px;
    background: #f1f5f9;
    border-radius: 4px 16px 16px 16px;
    width: fit-content;
}

.typing-dots span {
    width: 7px;
    height: 7px;
    background: #94a3b8;
    border-radius: 50%;
    animation: bounce 1.4s ease-in-out infinite;
}

.typing-dots span:nth-child(2) { animation-delay: 0.2s; }
.typing-dots span:nth-child(3) { animation-delay: 0.4s; }

@keyframes bounce {
    0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
    30% { transform: translateY(-7px); opacity: 1; }
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* WELCOME */
/* ═══════════════════════════════════════════════════════════════════════════ */
.welcome-section {
    text-align: center;
    padding: 16px 0;
}

.welcome-avatar {
    width: 56px;
    height: 56px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 28px;
    margin: 0 auto 12px;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.3);
}

.welcome-text {
    color: #1e293b;
    font-size: 15px;
    font-weight: 600;
    margin-bottom: 4px;
}

.welcome-subtext {
    color: #64748b;
    font-size: 12.5px;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* HEADER INFO */
/* ═══════════════════════════════════════════════════════════════════════════ */
.chatbot-title {
    color: white;
    font-size: 16px;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 10px;
}

.chatbot-title .bot-avatar {
    width: 34px;
    height: 34px;
    background: rgba(255,255,255,0.2);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    border: 2px solid rgba(255,255,255,0.3);
}

.online-badge {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    color: rgba(255,255,255,0.9);
    margin-top: 2px;
}

.online-dot {
    width: 7px;
    height: 7px;
    background: #22c55e;
    border-radius: 50%;
    box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.4);
    animation: online-pulse 2s ease-in-out infinite;
}

@keyframes online-pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.4); }
    50% { box-shadow: 0 0 0 5px rgba(34, 197, 94, 0); }
}
</style>
"""


def mostrar_chatbot():
    """Renderiza el chatbot flotante - versión corregida."""
    if "chat_abierto" not in st.session_state:
        st.session_state.chat_abierto = False
    if "chat_historial" not in st.session_state:
        st.session_state.chat_historial = []
    if "chat_typing" not in st.session_state:
        st.session_state.chat_typing = False

    lang = st.session_state.get("idioma", "es")
    is_open = st.session_state.chat_abierto

    st.markdown(_CHAT_CSS, unsafe_allow_html=True)

    # ── FAB ──
    fab_container = st.container()
    with fab_container:
        fab_label = "✕" if is_open else "💬"
        fab_clicked = st.button(
            fab_label, 
            key="chatbot_fab_toggle", 
            help="Abrir/cerrar chat"
        )
    fab_container.float("bottom: 24px; right: 24px; width: 60px;")

    if fab_clicked:
        st.session_state.chat_abierto = not st.session_state.chat_abierto
        st.rerun()

    # ── CHATBOX ──
    if is_open:
        chat_container = st.container()
        with chat_container:
            # HEADER con botones nativos de Streamlit
            header_cols = st.columns([4, 1, 1])
            with header_cols[0]:
                st.markdown(f"""
                <div>
                    <div class="chatbot-title">
                        <span class="bot-avatar">🤖</span>
                        {get_text("chatbot.title", lang)}
                    </div>
                    <div class="online-badge">
                        <span class="online-dot"></span>
                        En línea
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with header_cols[1]:
                if st.button("🗑️", key="chat_clear_btn", help=get_text("chatbot.clear", lang)):
                    st.session_state.chat_historial = []
                    st.rerun()

            with header_cols[2]:
                if st.button("✕", key="chat_close_btn", help="Cerrar"):
                    st.session_state.chat_abierto = False
                    st.rerun()

            # ÁREA DE MENSAJES
            messages_area = st.container()
            with messages_area:
                if not st.session_state.chat_historial:
                    st.markdown(f"""
                    <div class="welcome-section">
                        <div class="welcome-avatar"></div>
                        <div class="welcome-text">{get_text("chatbot.greeting", lang)}</div>
                        <div class="welcome-subtext">¿En qué puedo ayudarte hoy?</div>
                    </div>
                    """, unsafe_allow_html=True)

                    sugerencias = get_text("chatbot.suggestions", lang)
                    if isinstance(sugerencias, list):
                        sug_cols = st.columns(len(sugerencias))
                        for i, s in enumerate(sugerencias):
                            with sug_cols[i]:
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

            # INPUT - Usando st.text_input + botón (respeta el contenedor)
            input_cols = st.columns([5, 1])
            with input_cols[0]:
                user_input = st.text_input(
                    get_text("chatbot.placeholder", lang),
                    key="chat_text_input",
                    label_visibility="collapsed",
                    placeholder=get_text("chatbot.placeholder", lang),
                )
            with input_cols[1]:
                send_clicked = st.button("↑", key="chat_send_btn", help="Enviar")

            # Procesar mensaje
            if user_input and send_clicked:
                st.session_state.chat_historial.append({"role": "user", "content": user_input})
                st.session_state.chat_typing = True
                st.rerun()

            # Procesar respuesta
            if st.session_state.chat_typing and st.session_state.chat_historial:
                last_msg = st.session_state.chat_historial[-1]
                if last_msg["role"] == "user":
                    resp = generar_respuesta(last_msg["content"])
                    st.session_state.chat_historial.append({"role": "assistant", "content": resp})
                    st.session_state.chat_typing = False
                    st.rerun()

        chat_container.float(
            "bottom: 100px; right: 24px; width: 400px; height: 580px; "
            "background: white; border-radius: 20px; "
            "box-shadow: 0 25px 50px -12px rgba(0,0,0,0.25); "
            "overflow: hidden; padding: 16px; z-index: 9999;"
        )