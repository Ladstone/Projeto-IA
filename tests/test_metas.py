"""Metas pelas medidas: Mifflin-St Jeor, fator de atividade, ajuste por objetivo e travas de segurança."""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.metas import calcular_metas, OBJETIVOS, normaliza_objetivo

client = TestClient(app)
HOMEM = dict(sexo='masculino', idade=30, peso_kg=80, altura_cm=180, atividade='moderado')


def test_mifflin_and_hypertrophy_by_hand():
    r = calcular_metas(**HOMEM, objetivo='hipertrofia')
    # TMB = 10*80 + 6,25*180 - 5*30 + 5 = 1780; GET = 1780*1,55 = 2759; +10% = 3034,9 -> 3030
    assert r['detalhes']['tmb'] == 1780 and r['detalhes']['gasto_total'] == 2759
    assert r['kcal'] == 3030
    assert r['proteina_g'] == 144                      # 1,8 g/kg x 80
    assert r['lipideos_g'] == 84                       # 25% de 3030 / 9
    assert abs(4 * r['proteina_g'] + 9 * r['lipideos_g'] + 4 * r['carboidrato_g'] - r['kcal']) <= 6


def test_female_formula_and_objectives_order():
    base = dict(sexo='feminino', idade=25, peso_kg=60, altura_cm=165, atividade='leve')
    kcal = {o: calcular_metas(**base, objetivo=o)['kcal'] for o in OBJETIVOS}
    # TMB = 600 + 1031,25 - 125 - 161 = 1345,25
    assert calcular_metas(**base, objetivo='manutencao')['detalhes']['tmb'] == 1345
    assert kcal['emagrecimento'] < kcal['definicao'] < kcal['manutencao'] == kcal['resistencia'] < kcal['hipertrofia']


def test_deficit_never_below_safety_floor():
    r = calcular_metas(sexo='feminino', idade=70, peso_kg=45, altura_cm=150, atividade='sedentario', objetivo='emagrecimento')
    assert r['kcal'] == 1200 and r['avisos']


def test_high_bmi_uses_reference_weight_for_protein():
    r = calcular_metas(sexo='masculino', idade=40, peso_kg=130, altura_cm=175, atividade='leve', objetivo='emagrecimento')
    assert r['proteina_g'] == round(1.8 * 25 * 1.75 ** 2)   # peso de referência de IMC 25, não 130 kg
    assert any('IMC' in a for a in r['avisos'])


@pytest.mark.parametrize('bad', [dict(idade=16), dict(sexo='outro'), dict(atividade='x'), dict(peso_kg=10)])
def test_invalid_inputs_are_rejected(bad):
    with pytest.raises(ValueError):
        calcular_metas(**{**HOMEM, **bad}, objetivo='manutencao')


def test_calculate_endpoint_and_minor_message():
    ok = client.post('/api/profile/calcular', json={**HOMEM, 'objetivo': 'definicao'})
    assert ok.status_code == 200 and ok.json()['kcal'] == round(2759 * 0.9 / 10) * 10
    minor = client.post('/api/profile/calcular', json={**HOMEM, 'idade': 15, 'objetivo': 'emagrecimento'})
    assert minor.status_code == 422 and 'adultos' in minor.json()['detail']


def test_old_objective_ids_still_accepted():
    assert normaliza_objetivo('ganho') == 'hipertrofia' and normaliza_objetivo('perda') == 'emagrecimento'
    opts = client.get('/api/profile/opcoes').json()
    assert {o['id'] for o in opts['objetivos']} == set(OBJETIVOS)
