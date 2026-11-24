"""Classes sem item único na TACO: Purê vira um item só calculado pela receita; Hambúrguer e Lasanha, vários ingredientes."""
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
        assert all(taco_id in FOODS for _, taco_id, _ in parts if taco_id is not None)  # None = água


def test_pure_is_one_item_computed_from_the_recipe(monkeypatch):
    data = _analyze(monkeypatch)
    pure = [d for d in data['detected'] if d['class_id'] == PURE]
    assert len(pure) == 1 and pure[0]['taco_id'] == 9016 and pure[0]['nome_exibido'] == 'Purê'
    assert pure[0]['note'] == 'Porção inicial estimada pela área no prato; informe o peso real.'
    total = main.suggest_grams(FOOD_CLASS_MAP[PURE]['grams'], pure[0]['cobertura_prato_percent'] / 100)
    assert pure[0]['grams'] == total
    # 100 g de purê = 80 g de batata + 2 g de leite em pó + 14 g de água + 4 g de manteiga.
    esperado = sum(FOODS[t]['kcal'] * f for _, t, f in RECIPES[PURE] if t is not None)
    assert abs(FOODS[9016]['kcal'] - esperado) < 0.01 and 60 < esperado < 90
    assert all(FOODS[9016][k] is not None for k in ('kcal', 'proteina_g', 'carboidrato_g', 'lipideos_g'))
    # Gordura: a batata não tem o dado na TACO; soma leite em pó + manteiga e avisa.
    assert abs(FOODS[9016]['lipideos_g'] - (0.02 * 26.9 + 0.04 * 82.4)) < 0.01
    assert 'Batata' in FOODS[9016]['marcadores']['lipideos_g']
    item = [it for it in data['nutrition']['alimentos'] if it['taco_id'] == 9016][0]
    assert abs(item['nutrientes']['kcal'] - esperado * total / 100) < 0.01
    # Classes sem receita continuam com um item só.
    arroz = [d for d in data['detected'] if d['class_id'] == ARROZ]
    assert len(arroz) == 1 and arroz[0]['taco_id'] == FOOD_CLASS_MAP[ARROZ]['taco_id']


def test_pure_recipe_details_and_recalculate():
    det = client.get('/api/taco/9016').json()
    assert det['nome'].startswith('Purê (receita: 80%') and 'TACO' in det['fonte']
    res = client.post('/api/recalculate', json={'items': [{'taco_id': 9016, 'grams': 200}]})
    assert res.status_code == 200
    # Não aparece na lista de busca da TACO (continua com os 597 itens oficiais).
    assert all(f['id'] < 9000 for f in client.get('/api/foods').json()['foods'])


def test_crop_is_the_plate_for_grams(monkeypatch):
    """Com recorte, área mínima e gramas usam o recorte (o prato); "% da foto" usa a foto original."""
    monkeypatch.setattr(main, 'vision', FakeVision())
    raw = BytesIO(); Image.new('RGB', (100, 100), (230, 200, 120)).save(raw, format='PNG')
    data = client.post('/api/analyze', files={'file': ('p.png', raw.getvalue(), 'image/png')},
                       data={'area_original': '0.5'}).json()
    arroz = [d for d in data['detected'] if d['class_id'] == ARROZ][0]
    assert arroz['cobertura_imagem_percent'] == 10.0          # 20% do recorte = 10% da foto
    assert arroz['cobertura_prato_percent'] == 20.0
    assert arroz['grams'] == main.suggest_grams(FOOD_CLASS_MAP[ARROZ]['grams'], 0.20)


def test_without_crop_plate_is_box_around_food(monkeypatch):
    """Sem recorte, o prato é o retângulo em volta da comida: a distância da câmera não muda a porção."""
    data = _analyze(monkeypatch)
    arroz = [d for d in data['detected'] if d['class_id'] == ARROZ][0]
    # Comida nas linhas 0-40 da foto; com a margem de 5%, o retângulo cobre 45% da foto.
    assert arroz['cobertura_imagem_percent'] == 20.0
    assert arroz['cobertura_prato_percent'] == round(20 / 45 * 100, 1)


def test_crop_area_out_of_range_is_rejected(monkeypatch):
    monkeypatch.setattr(main, 'vision', FakeVision())
    raw = BytesIO(); Image.new('RGB', (10, 10)).save(raw, format='PNG')
    res = client.post('/api/analyze', files={'file': ('p.png', raw.getvalue(), 'image/png')}, data={'area_original': '1.5'})
    assert res.status_code == 422


def test_crop_suggestion_surrounds_detected_food(monkeypatch):
    monkeypatch.setattr(main, 'vision', FakeVision())   # comida nas linhas 0-39 de 100
    raw = BytesIO(); Image.new('RGB', (100, 100)).save(raw, format='PNG')
    box = client.post('/api/crop-suggestion', files={'file': ('p.png', raw.getvalue(), 'image/png')}).json()['box']
    assert box['x0'] == 0 and box['x1'] == 1 and box['y0'] == 0
    assert 0.40 <= box['y1'] <= 0.50


def test_crop_suggestion_without_food_returns_none(monkeypatch):
    class Empty(FakeVision):
        def predict(self, image):
            return np.zeros((100, 100), np.uint8), np.zeros((21, 100, 100), np.float32)
    monkeypatch.setattr(main, 'vision', Empty())
    raw = BytesIO(); Image.new('RGB', (100, 100)).save(raw, format='PNG')
    assert client.post('/api/crop-suggestion', files={'file': ('p.png', raw.getvalue(), 'image/png')}).json()['box'] is None
