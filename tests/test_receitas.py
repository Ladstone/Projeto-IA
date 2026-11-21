"""Classes sem item único na TACO viram vários ingredientes (ex.: Purê = batata + leite + manteiga)."""
from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import app.main as main
from app.nutrition import FOODS
from app.unified_classes import RECIPES, FOOD_CLASS_MAP

client = TestClient(main.app)
PURE, ARROZ = 16, 1


class FakeVision:
    """Máscara fixa: 20% Purê, 20% Arroz, resto fundo."""
    thresholds = {i: 0.5 for i in range(1, 22)}

    def predict(self, image):
        mask = np.zeros((100, 100), np.uint8)
        mask[:20] = PURE
        mask[20:40] = ARROZ
        probs = np.zeros((21, 100, 100), np.float32)
        probs[PURE - 1][mask == PURE] = 0.9
        probs[ARROZ - 1][mask == ARROZ] = 0.9
        return mask, probs


def _analyze(monkeypatch):
    monkeypatch.setattr(main, 'vision', FakeVision())
    raw = BytesIO(); Image.new('RGB', (100, 100), (230, 200, 120)).save(raw, format='PNG')
    return client.post('/api/analyze', files={'file': ('p.png', raw.getvalue(), 'image/png')}).json()


def test_recipes_are_valid():
    for class_id, parts in RECIPES.items():
        assert class_id in FOOD_CLASS_MAP
        assert abs(sum(f for _, _, f in parts) - 1) < 1e-9
        assert all(taco_id in FOODS for _, taco_id, _ in parts)


def test_pure_becomes_potato_milk_and_butter(monkeypatch):
    data = _analyze(monkeypatch)
    pure = [d for d in data['detected'] if d['class_id'] == PURE]
    assert [d['taco_id'] for d in pure] == [91, 458, 261]
    total = main.suggest_grams(FOOD_CLASS_MAP[PURE]['grams'], 0.20)
    assert [d['grams'] for d in pure] == [round(total * 0.80), round(total * 0.16), round(total * 0.04)]
    assert all('receita aproximada' in d['note'] for d in pure)
    # Classes sem receita continuam com um item só.
    arroz = [d for d in data['detected'] if d['class_id'] == ARROZ]
    assert len(arroz) == 1 and arroz[0]['taco_id'] == FOOD_CLASS_MAP[ARROZ]['taco_id']
    # Os nutrientes somam os três ingredientes.
    assert {91, 458, 261} <= {it['taco_id'] for it in data['nutrition']['alimentos']}

