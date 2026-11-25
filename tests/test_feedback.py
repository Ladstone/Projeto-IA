"""Feedback escrito pelo LLM: o app calcula os números, o LLM só redige; sem LLM, cai nas dicas por regra.

Nenhum teste chama a Gemini API: a função do LLM é trocada por uma falsa.
"""
import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app import llm

PRATO = [{'taco_id': 3, 'grams': 180, 'nome': 'Arroz, tipo 1, cozido', 'nome_exibido': 'Arroz'},
         {'taco_id': 561, 'grams': 120, 'nome': 'Feijão, carioca, cozido', 'nome_exibido': 'Feijão cozido'}]
TEXTO_PRATO = {'resumo': 'Prato equilibrado.', 'positivo': 'Arroz com feijão.', 'sugestoes': ['Inclua uma salada.']}
TEXTO_MES = {'resumo': 'Bom mês.', 'destaques': ['Feijão frequente.'], 'atencao': ['Poucas verduras.'], 'metas': ['Salada 3x/semana.']}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, 'DB', tmp_path / 'teste.sqlite3')
    main.db_init()
    return TestClient(main.app)


def sem_llm(*_a, **_k):
    raise llm.LLMIndisponivel('teste')


def test_prato_usa_o_texto_do_llm_com_os_numeros_do_app(client, monkeypatch):
    recebido = {}
    def falso(dados):
        recebido.update(dados)
        return TEXTO_PRATO, 'modelo-teste'
    monkeypatch.setattr(llm, 'feedback_prato', falso)
    r = client.post('/api/feedback/prato', json={'items': PRATO}).json()
    assert r == {'texto': TEXTO_PRATO, 'modelo': 'modelo-teste'}
    assert [a['nome'] for a in recebido['alimentos']] == ['Arroz', 'Feijão cozido']
    esperado = main.recalculate(PRATO, main.profile_get())['total']['kcal']
    assert recebido['totais_do_prato']['kcal'] == esperado   # número calculado pelo app, não pelo LLM


def test_prato_sem_llm_devolve_texto_vazio(client, monkeypatch):
    monkeypatch.setattr(llm, 'feedback_prato', sem_llm)
    r = client.post('/api/feedback/prato', json={'items': PRATO})
    assert r.status_code == 200 and r.json()['texto'] is None
    assert client.post('/api/feedback/prato', json={'items': []}).status_code == 422


def inserir_prato(dia, itens, kcal):
    with sqlite3.connect(main.DB) as con:
        con.execute('insert into meals (criado_em,dia,refeicao,foto,itens,totais) values (?,?,?,?,?,?)',
                    (dia + 'T12:00:00', dia, '', b'x', json.dumps(itens), json.dumps({'kcal': kcal, 'proteina_g': 30})))


def test_mes_gera_salva_e_relê(client, monkeypatch):
    inserir_prato('2026-11-03', [{'nome': 'Arroz', 'grams': 150, 'taco_id': 3, 'class_id': 1},
                                 {'nome': 'Salada', 'grams': 50, 'taco_id': 1, 'class_id': 5}], 500)
    inserir_prato('2026-11-04', [{'nome': 'Arroz', 'grams': 150, 'taco_id': 3, 'class_id': 1}], 700)
    recebido = {}
    def falso(dados):
        recebido.update(dados)
        return TEXTO_MES, 'modelo-teste'
    monkeypatch.setattr(llm, 'feedback_mes', falso)
    assert client.get('/api/feedback/mes?mes=2026-11').json()['texto'] is None
    r = client.post('/api/feedback/mes?mes=2026-11').json()
    assert r['texto'] == TEXTO_MES and r['n_pratos'] == 2
    assert recebido['dias_com_registro'] == 2 and recebido['pratos_registrados'] == 2
    assert recebido['media_por_dia_registrado']['kcal'] == 600
    assert recebido['pratos_com_verdura_ou_legume'] == 1
    assert recebido['alimentos_mais_frequentes'][0] == {'nome': 'Arroz', 'pratos': 2}
    salvo = client.get('/api/feedback/mes?mes=2026-11').json()
    assert salvo['texto'] == TEXTO_MES and salvo['modelo'] == 'modelo-teste'


def test_mes_sem_pratos_ou_sem_llm(client, monkeypatch):
    assert client.post('/api/feedback/mes?mes=2026-10').status_code == 422
    inserir_prato('2026-10-10', [{'nome': 'Arroz', 'grams': 150, 'taco_id': 3, 'class_id': 1}], 500)
    monkeypatch.setattr(llm, 'feedback_mes', sem_llm)
    assert client.post('/api/feedback/mes?mes=2026-10').status_code == 503
    assert client.get('/api/feedback/mes?mes=2026-10').json()['texto'] is None


def test_sem_chave_nao_chama_a_api(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', '')
    with pytest.raises(llm.LLMIndisponivel):
        llm.gerar_json('s', 'p', {'type': 'OBJECT'})
