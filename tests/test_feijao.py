"""Feijão: a rede só sabe que é feijão; a cor da região escolhe preto ou carioca na TACO."""
from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import app.main as main

client = TestClient(main.app)
FEIJAO, PRETO, CARIOCA = 2, 567, 561


class FakeVision:
    """Metade de cima da foto marcada como feijão."""
    thresholds = {i: 0.5 for i in range(1, 29)}

    def predict(self, image):
        mask = np.zeros((100, 100), np.uint8)
        mask[:50] = FEIJAO
        probs = np.zeros((28, 100, 100), np.float32)
        probs[FEIJAO - 1][mask == FEIJAO] = 0.9
        return mask, probs


def _feijao(monkeypatch, color):
    monkeypatch.setattr(main, 'vision', FakeVision())
    raw = BytesIO(); Image.new('RGB', (100, 100), color).save(raw, format='PNG')
    data = client.post('/api/analyze', files={'file': ('p.png', raw.getvalue(), 'image/png')}).json()
    return [d for d in data['detected'] if d['class_id'] == FEIJAO][0]


def test_dark_beans_use_black_bean_values(monkeypatch):
    item = _feijao(monkeypatch, (30, 25, 25))
    assert item['taco_id'] == PRETO and item['nome_exibido'] == 'Feijão cozido'


def test_light_beans_use_carioca_values(monkeypatch):
    item = _feijao(monkeypatch, (150, 110, 80))
    assert item['taco_id'] == CARIOCA and item['nome_exibido'] == 'Feijão cozido'


def test_bean_note_does_not_mention_type(monkeypatch):
    note = _feijao(monkeypatch, (150, 110, 80))['note'].lower()
    assert 'preto' not in note and 'carioca' not in note and 'tipo' not in note
    assert not note.startswith('.')


def test_arroz_frango_e_carne_aparecem_com_nome_curto_e_sem_obs():
    from app.unified_classes import NOMES_EXIBIDOS, FOOD_CLASS_MAP
    for class_id, nome, taco_id in [(1, 'Arroz', 3), (3, 'Frango', 410), (4, 'Carne bovina', 377)]:
        assert NOMES_EXIBIDOS[class_id] == nome
        assert FOOD_CLASS_MAP[class_id]['taco_id'] == taco_id  # mesma linha da TACO de antes
        assert FOOD_CLASS_MAP[class_id]['note'] == ''
