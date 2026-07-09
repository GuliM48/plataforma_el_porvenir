import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from common.features import (FEATURE_COLS, TIPOS_VALIDOS, build_feature_vector,
                              feature_dict_to_array, haversine_km)


def test_haversine_distancia_cero_para_mismo_punto():
    assert haversine_km(-8.04, -79.0, -8.04, -79.0) == pytest.approx(0.0, abs=1e-9)


def test_haversine_distancia_conocida():
    # Hermelinda -> Cesar Vallejo, ~3-4 km según los hotspots definidos
    d = haversine_km(-8.044, -79.003, -8.021, -78.982)
    assert 2.5 < d < 5.0


def test_build_feature_vector_columnas_completas():
    features = build_feature_vector(-8.044, -79.003, hora=22, dia_semana=5, mes=7, tipo="Robo")
    assert set(features.keys()) == set(FEATURE_COLS)


def test_build_feature_vector_one_hot_tipo():
    features = build_feature_vector(-8.044, -79.003, hora=10, dia_semana=1, mes=3, tipo="Hurto")
    assert features["tipo_Hurto"] == 1
    otros_tipos = [t for t in TIPOS_VALIDOS if t != "Hurto"]
    assert all(features[f"tipo_{t}"] == 0 for t in otros_tipos)


def test_build_feature_vector_tipo_invalido():
    with pytest.raises(ValueError):
        build_feature_vector(-8.044, -79.003, hora=10, dia_semana=1, mes=3, tipo="NoExiste")


def test_build_feature_vector_hora_fuera_de_rango():
    with pytest.raises(ValueError):
        build_feature_vector(-8.044, -79.003, hora=25, dia_semana=1, mes=3, tipo="Robo")


def test_feature_dict_to_array_orden_correcto():
    features = build_feature_vector(-8.044, -79.003, hora=22, dia_semana=5, mes=7, tipo="Robo")
    arr = feature_dict_to_array(features)
    assert arr.shape == (1, len(FEATURE_COLS))
    for i, col in enumerate(FEATURE_COLS):
        assert arr[0, i] == pytest.approx(features[col])


def test_es_fin_de_semana():
    f_sabado = build_feature_vector(-8.044, -79.003, hora=10, dia_semana=5, mes=1, tipo="Otros")
    f_martes = build_feature_vector(-8.044, -79.003, hora=10, dia_semana=1, mes=1, tipo="Otros")
    assert f_sabado["es_fin_de_semana"] == 1
    assert f_martes["es_fin_de_semana"] == 0
