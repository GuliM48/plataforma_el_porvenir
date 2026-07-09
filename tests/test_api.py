import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from api.main import app


def test_health_ok():
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert len(body["modelos_cargados"]) == 5


def test_models_incluye_metricas():
    with TestClient(app) as client:
        resp = client.get("/models")
        assert resp.status_code == 200
        modelos = resp.json()
        assert len(modelos) == 5
        assert all(m["f1_macro_cv"] is not None for m in modelos)


def test_predict_devuelve_prioridad_valida():
    with TestClient(app) as client:
        resp = client.post("/predict", json={
            "latitud": -8.044, "longitud": -79.003, "hora": 22,
            "dia_semana": 5, "mes": 7, "tipo": "Robo",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["prioridad"] in ("ALTA", "MEDIA", "BAJA")
        assert 0 <= body["confianza"] <= 1
        assert abs(sum(body["probabilidades"].values()) - 1.0) < 1e-3


def test_predict_tipo_invalido_devuelve_422():
    with TestClient(app) as client:
        resp = client.post("/predict", json={
            "latitud": -8.044, "longitud": -79.003, "hora": 22,
            "dia_semana": 5, "mes": 7, "tipo": "NoExiste",
        })
        assert resp.status_code == 422


def test_predict_coordenadas_fuera_de_el_porvenir_devuelve_422():
    with TestClient(app) as client:
        resp = client.post("/predict", json={
            "latitud": -12.0, "longitud": -77.0, "hora": 22,
            "dia_semana": 5, "mes": 7, "tipo": "Robo",
        })
        assert resp.status_code == 422


def test_predict_modelo_especifico_por_query_param():
    with TestClient(app) as client:
        resp = client.post("/predict?modelo=Random_Forest", json={
            "latitud": -8.044, "longitud": -79.003, "hora": 22,
            "dia_semana": 5, "mes": 7, "tipo": "Robo",
        })
        assert resp.status_code == 200
        assert resp.json()["modelo_usado"] == "Random_Forest"
