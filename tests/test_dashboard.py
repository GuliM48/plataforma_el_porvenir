import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streamlit.testing.v1 import AppTest


def test_dashboard_todas_las_paginas_sin_excepciones():
    at = AppTest.from_file("dashboard/app.py", default_timeout=60)
    at.run()
    assert not at.exception, f"Error antes del login: {at.exception}"
    assert at.title[0].value == "Iniciar sesión"

    at.session_state["autenticado"] = True
    at.session_state["usuario"] = "test"

    paginas = ["Predicción en vivo", "Comparación de modelos", "Curvas ROC",
               "Matrices de confusión", "Heatmaps", "Pruebas estadísticas"]

    for pagina in paginas:
        at.run()
        radios = at.sidebar.radio
        if radios:
            radios[0].set_value(pagina).run()
        assert not at.exception, f"Error en página '{pagina}': {at.exception}"


if __name__ == "__main__":
    test_dashboard_todas_las_paginas_sin_excepciones()
    print("OK")
