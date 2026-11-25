"""Histórico de pratos: o app grava cada prato analisado e o calendário lê por mês."""
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as main

LUNCH = Path(__file__).parent / 'fixtures' / 'lunch.jpg'


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Banco temporário: os testes não mexem no histórico de verdade."""
    monkeypatch.setattr(main, 'DB', tmp_path / 'teste.sqlite3')
    main.db_init()
    return TestClient(main.app)


def analisar(client, **extra):
    data = {k: str(v) for k, v in extra.items()}
    r = client.post('/api/analyze', files={'file': ('lunch.jpg', LUNCH.read_bytes(), 'image/jpeg')}, data=data)
    assert r.status_code == 200, r.text
    return r.json()


def test_analise_sem_salvar_nao_grava(client):
    if main.vision is None:
        pytest.skip('modelo indisponível')
    assert analisar(client)['meal_id'] is None
    assert client.get('/api/meals').json()['meals'] == []


def test_prato_analisado_entra_no_historico_com_foto(client):
    if main.vision is None:
        pytest.skip('modelo indisponível')
    res = analisar(client, salvar=1)
    assert res['meal_id'] and res['detected']
    hist = client.get('/api/meals').json()
    assert hist['mes'] == date.today().strftime('%Y-%m') and hist['resumo'] == {'dias': 1, 'pratos': 1}
    meal = hist['meals'][0]
    assert meal['dia'] == date.today().isoformat() and meal['totais']['kcal'] > 0
    assert len(meal['itens']) == len(res['detected'])
    foto = client.get(meal['foto'])
    assert foto.status_code == 200 and foto.headers['content-type'] == 'image/jpeg' and foto.content[:2] == b'\xff\xd8'
    # Refazer o recorte do mesmo prato substitui o registro em vez de duplicar.
    assert analisar(client, salvar=1, meal_id=res['meal_id'])['meal_id'] == res['meal_id']
    assert len(client.get('/api/meals').json()['meals']) == 1


def test_correcao_atualiza_o_registro_e_sem_itens_some(client):
    if main.vision is None:
        pytest.skip('modelo indisponível')
    meal_id = analisar(client, salvar=1)['meal_id']
    item = {'taco_id': 3, 'grams': 200, 'nome': 'Arroz, tipo 1, cozido', 'nome_exibido': 'Arroz'}
    assert client.put(f'/api/meals/{meal_id}', json={'items': [item]}).status_code == 200
    meal = client.get('/api/meals').json()['meals'][0]
    assert meal['itens'] == [{'nome': 'Arroz', 'grams': 200.0, 'taco_id': 3, 'class_id': None}]
    assert meal['totais']['kcal'] == pytest.approx(2 * main.FOODS[3]['kcal'], abs=0.2)
    # Sem itens o prato some do calendário, mas volta se a pessoa adicionar alimentos de novo.
    client.put(f'/api/meals/{meal_id}', json={'items': []})
    assert client.get('/api/meals').json()['meals'] == []
    client.put(f'/api/meals/{meal_id}', json={'items': [item]})
    assert len(client.get('/api/meals').json()['meals']) == 1
    assert client.delete(f'/api/meals/{meal_id}').status_code == 200
    assert client.delete(f'/api/meals/{meal_id}').status_code == 404
    assert client.put(f'/api/meals/{meal_id}', json={'items': [item]}).status_code == 404


def test_mes_invalido(client):
    assert client.get('/api/meals?mes=2026-13').status_code == 422

