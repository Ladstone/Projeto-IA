"""Modelo ativo do app (v2, 21 classes) e alternativas (v3 de 28 classes, v2 piloto e v1): todos precisam carregar e prever.

O v3 voltou a ser alternativo em 23/11/2026: em fotos reais de prato feito confundia batata frita
com macarrão e bife com frango."""
from pathlib import Path
from PIL import Image
from app.vision import VisionService

ROOT = Path(__file__).parents[1]
LUNCH = Path(__file__).parent / 'fixtures' / 'lunch.jpg'


def test_active_model_is_v2():
    service = VisionService(ROOT / 'models' / 'unified_21class.pt')
    assert service.architecture == 'lraspp_mobilenet_v3'
    assert service.resolution == 224
    assert service.n_class == 21 and service.metadata.get('epochs_trained') == 20
    mask, probs = service.predict(Image.open(LUNCH))
    assert mask.shape == (224, 224) and probs.shape == (21, 224, 224)


def test_v3_alternative_still_loads_and_predicts():
    service = VisionService(ROOT / 'models' / 'unified_28class_v3.pt')
    assert service.architecture == 'lraspp_mobilenet_v3'
    assert service.n_class == 28 and 'MyFoodRepo v2.0' in service.metadata['datasets']
    assert service.metadata.get('epochs_trained') == 15
    mask, probs = service.predict(Image.open(LUNCH))
    assert mask.shape == (224, 224) and probs.shape == (28, 224, 224)


def test_v2_pilot_backup_still_loads_and_predicts():
    service = VisionService(ROOT / 'models' / 'unified_21class_v2_piloto.pt')
    assert 'preliminar' in service.metadata.get('label', '')
    mask, probs = service.predict(Image.open(LUNCH))
    assert mask.shape == (224, 224) and probs.shape == (21, 224, 224)


def test_v1_backup_still_loads_and_predicts():
    service = VisionService(ROOT / 'models' / 'unified_21class_v1.pt')
    assert service.architecture == 'foodunet'
    assert service.resolution == 160
    mask, probs = service.predict(Image.open(LUNCH))
    assert mask.shape == (160, 160) and probs.shape == (21, 160, 160)


def test_thresholds_come_from_validation_only():
    service = VisionService(ROOT / 'models' / 'unified_21class.pt')
    assert 'validação' in service.metadata['selection']
    assert set(service.thresholds) == set(range(1, 22))
