import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from dashboard.i18n import get_text


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
                max_output_tokens=300,
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
            return get_text(f"chatbot.help_{tema}")
    return get_text("chatbot.default")


def generar_respuesta(pregunta: str) -> str:
    lang = st.session_state.get("idioma", "es")
    respuesta = _generar_respuesta_gemini(pregunta, lang)
    if respuesta is None:
        respuesta = _generar_respuesta_keyword(pregunta, lang)
    return respuesta


def mostrar_chatbot():
    if "chat_abierto" not in st.session_state:
        st.session_state.chat_abierto = False
    if "chat_historial" not in st.session_state:
        st.session_state.chat_historial = []

    lang = st.session_state.get("idioma", "es")

    chat_param = st.query_params.get("__chat", "")
    if chat_param == "toggle":
        st.query_params.pop("__chat", None)
        st.session_state.chat_abierto = not st.session_state.chat_abierto
        st.rerun()
    elif chat_param.startswith("q:"):
        msg = chat_param[2:]
        st.query_params.pop("__chat", None)
        st.session_state.chat_historial.append({"role": "user", "content": msg})
        resp = generar_respuesta(msg)
        st.session_state.chat_historial.append({"role": "bot", "content": resp})
        st.rerun()

    is_open = st.session_state.chat_abierto
    display = "flex" if is_open else "none"

    msgs_html = ""
    if not st.session_state.chat_historial:
        msgs_html = f"""<div class="cm cm-bot">{get_text("chatbot.greeting")}</div>"""
    else:
        for m in st.session_state.chat_historial:
            cls = "cm-user" if m["role"] == "user" else "cm-bot"
            safe = m["content"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            msgs_html += f'<div class="cm {cls}">{safe}</div>'

    fab_label = "✕" if is_open else "💬"
    voice_lang = "es-ES" if lang == "es" else "en-US"

    html = f"""
<style>
.cfab {{
    position: fixed; bottom: 20px; right: 20px; z-index: 99999;
    width: 54px; height: 54px; border-radius: 50%;
    background: linear-gradient(135deg, #2563EB, #0EA5E9);
    color: white; text-decoration: none; font-size: 24px;
    box-shadow: 0 4px 15px rgba(37,99,235,0.3);
    display: flex; align-items: center; justify-content: center;
    transition: transform 0.2s;
}}
.cfab:hover {{ transform: scale(1.1); color: white; }}
.cbox {{
    position: fixed; bottom: 84px; right: 20px; z-index: 99998;
    width: 380px; max-height: 520px; background: white;
    border-radius: 16px; box-shadow: 0 8px 32px rgba(0,0,0,0.15);
    display: {display}; flex-direction: column; overflow: hidden;
}}
.chdr {{
    background: linear-gradient(135deg, #2563EB, #0EA5E9); color: white;
    padding: 12px 16px; font-weight: 600; font-size: 15px;
    display: flex; justify-content: space-between; align-items: center;
}}
.cmsgs {{
    padding: 12px; overflow-y: auto; flex: 1; min-height: 200px; max-height: 360px;
}}
.cm {{ padding: 8px 12px; border-radius: 12px; margin: 4px 0;
        max-width: 88%; font-size: 14px; line-height: 1.4; word-wrap: break-word; }}
.cm-user {{ background: #DBEAFE; color: #1E40AF; margin-left: auto; }}
.cm-bot {{ background: #F3F4F6; color: #1F2937; margin-right: auto; }}
.cinp {{
    display: flex; gap: 6px; padding: 10px 14px;
    border-top: 1px solid #E5E7EB; background: #FAFAFA;
}}
.cinp input {{
    flex: 1; border: 1px solid #D1D5DB; border-radius: 20px;
    padding: 8px 14px; font-size: 13px; outline: none;
}}
.cinp input:focus {{ border-color: #2563EB; }}
.csbtn {{
    background: #2563EB; color: white; border: none; border-radius: 50%;
    width: 36px; height: 36px; cursor: pointer; font-size: 13px;
    display: flex; align-items: center; justify-content: center;
}}
.cvbtn {{
    background: #F3F4F6; border: none; border-radius: 50%;
    width: 36px; height: 36px; cursor: pointer; font-size: 15px;
    display: flex; align-items: center; justify-content: center;
}}
.cvbtn:hover {{ background: #E5E7EB; }}
.cclose {{
    color: white; text-decoration: none; font-size: 18px;
    width: 32px; height: 32px; display: flex; align-items: center;
    justify-content: center; border-radius: 8px; transition: background 0.2s;
}}
.cclose:hover {{ background: rgba(255,255,255,0.2); }}
@media (max-width: 480px) {{
    .cbox {{ width: 90vw; right: 5vw; }}
}}
</style>

<a href="?__chat=toggle" class="cfab">{fab_label}</a>

<div class="cbox">
    <div class="chdr">
        <span>🤖 {get_text("chatbot.title")}</span>
        <a href="?__chat=toggle" class="cclose">✕</a>
    </div>
    <div class="cmsgs" id="chatMsgs">
        {msgs_html}
    </div>
    <form class="cinp" method="GET" action="" onsubmit="return enviarMsg()">
        <input type="hidden" name="__chat" id="chatHid">
        <input type="text" id="chatTxt" placeholder="{get_text("chatbot.placeholder")}" autocomplete="off">
        <button type="button" class="cvbtn" onclick="iniciarVoz()" title="{get_text("chatbot.voice")}">🎤</button>
        <button type="submit" class="csbtn">{get_text("chatbot.send")}</button>
    </form>
</div>

<script>
function enviarMsg() {{
    var txt = document.getElementById('chatTxt');
    var text = txt.value.trim();
    if (!text) return false;
    document.getElementById('chatHid').value = 'q:' + encodeURIComponent(text);
    return true;
}}

function iniciarVoz() {{
    if (!('webkitSpeechRecognition' in window)) {{
        alert('Speech recognition requires Chrome.');
        return;
    }}
    var btn = event.currentTarget;
    var recognition = new webkitSpeechRecognition();
    recognition.lang = '{voice_lang}';
    recognition.continuous = false;
    recognition.interimResults = false;
    btn.style.background = '#FEE2E2';
    recognition.start();
    recognition.onresult = function(ev) {{
        var texto = ev.results[0][0].transcript;
        document.getElementById('chatTxt').value = texto;
        btn.style.background = '#F3F4F6';
        document.getElementById('chatHid').value = 'q:' + encodeURIComponent(texto);
        document.querySelector('.cinp').submit();
    }};
    recognition.onerror = function() {{ btn.style.background = '#F3F4F6'; }};
    recognition.onend = function() {{ btn.style.background = '#F3F4F6'; }};
}}
</script>
"""
    st.markdown(html, unsafe_allow_html=True)
