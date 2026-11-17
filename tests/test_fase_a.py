"""Correções da Fase A: EXIF, porção por área, área mínima, mapeamento TACO e limitações conhecidas."""
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app, load_image, suggest_grams, MIN_REGION_FRACTION
from app.unified_classes import FOOD_CLASS_MAP

client = TestClient(app)
EXAMPLES = Path(__file__).parents[1] / 'app' / 'static' / 'examples'


def _jpeg_with_orientation(size, orientation):
    exif = Image.Exif()
    exif[0x0112] = orientation  # tag Orientation
    buffer = BytesIO()
    Image.new('RGB', size, (200, 30, 30)).save(buffer, format='JPEG', exif=exif)
    return buffer.getvalue()


def test_exif_orientation_is_applied():
    # Orientação 6 = foto de celular girada 90°: 400x300 armazenado vira 300x400 exibido.
    image = load_image(_jpeg_with_orientation((400, 300), 6))
    assert image.size == (300, 400)


def test_overlay_matches_displayed_orientation():
    raw = BytesIO()
    lunch = Image.open(EXAMPLES / 'lunch.jpg')
    exif = Image.Exif(); exif[0x0112] = 6
    lunch.save(raw, format='JPEG', exif=exif)
    data = client.post('/api/analyze', files={'file': ('rot.jpg', raw.getvalue(), 'image/jpeg')}).json()
    if data['preview_overlay'] is None:
        pytest.skip('Modelo não produziu regiões para esta foto')
    import base64
    overlay = Image.open(BytesIO(base64.b64decode(data['preview_overlay'].split(',', 1)[1])))
    assert overlay.size == (lunch.height, lunch.width)


def test_invalid_image_still_400():
    with pytest.raises(HTTPException) as err:
        load_image(b'not an image')
    assert err.value.status_code == 400


def test_portion_scales_with_area_and_is_bounded():
    assert suggest_grams(140, 0.20) == 140           # área de referência = porção padrão
    assert suggest_grams(140, 0.10) == 70            # metade da área, metade da porção
    assert suggest_grams(140, 0.01) == 35            # piso de 25%
    assert suggest_grams(140, 0.90) == 210           # teto de 150%
    assert suggest_grams(140, 0.10) % 5 == 0


def test_detections_respect_minimum_area_and_report_uncalibrated_score():
    for name in ['lunch', 'bean', 'salad', 'steak']:
        raw = (EXAMPLES / f'{name}.jpg').read_bytes()
        data = client.post('/api/analyze', files={'file': (f'{name}.jpg', raw, 'image/jpeg')}).json()
        for item in data['detected']:
            assert item['cobertura_imagem_percent'] >= MIN_REGION_FRACTION * 100 - 0.05
            assert 'confianca' not in item and 'score_nao_calibrado' in item
            assert item['grams'] <= item['porcao_padrao_g'] * 1.5


def test_pasta_classes_use_distinct_taco_rows():
    # "Massa com molho vermelho" não deve herdar a bolonhesa (com carne) do "Macarrão".
    assert FOOD_CLASS_MAP[8]['taco_id'] != FOOD_CLASS_MAP[17]['taco_id']


@pytest.mark.xfail(strict=True, reason='Limitação conhecida do modelo atual (Fase B): '
                   'cor marrom lisa é classificada como feijão. Remova este xfail quando o novo modelo passar.')
def test_plain_brown_image_is_not_detected_as_food():
    buffer = BytesIO()
    Image.new('RGB', (400, 300), (120, 80, 40)).save(buffer, format='PNG')
    data = client.post('/api/analyze', files={'file': ('brown.png', buffer.getvalue(), 'image/png')}).json()
    assert data['detected'] == []


@pytest.mark.xfail(strict=True, reason='Limitação conhecida do modelo atual (Fase B): ruído vira arroz.')
def test_random_noise_is_not_detected_as_food():
    buffer = BytesIO()
    noise = np.random.RandomState(0).randint(0, 255, (300, 400, 3), dtype=np.uint8)
    Image.fromarray(noise).save(buffer, format='PNG')
    data = client.post('/api/analyze', files={'file': ('noise.png', buffer.getvalue(), 'image/png')}).json()
    assert data['detected'] == []
