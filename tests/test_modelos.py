"""Modelos do app. Padrão: combinação (base v2 + classes novas do v3 + purê do v3d). Backup: v2 puro.

O v3 sozinho não é usado: em fotos reais de prato feito confundia batata frita com macarrão e bife com frango."""
from pathlib import Path
from PIL import Image
from app.vision import VisionService, CombinedVisionService, ClassOverrideService
from app.main import carregar_visao

ROOT = Path(__file__).parents[1]
LUNCH = Path(__file__).parent / 'fixtures' / 'lunch.jpg'


def test_base_model_is_v2():
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


def test_padrao_e_a_combinacao_e_v2_fica_de_backup():
    padrao = carregar_visao({})
    assert isinstance(padrao, ClassOverrideService) and isinstance(padrao.main, CombinedVisionService)
    assert padrao.n_class == 28 and padrao.main.n_base == 21
    mask, probs = padrao.predict(Image.open(LUNCH))
    assert mask.shape == (224, 224) and probs.shape == (28, 224, 224)
    backup = carregar_visao({'PRATOMETRIA_MODEL': str(ROOT / 'models' / 'unified_21class_v2.pt')})
    assert isinstance(backup, VisionService) and backup.n_class == 21


def test_v3d_do_pure_carrega():
    service = VisionService(ROOT / 'models' / 'unified_28class_v3d.pt')
    assert service.n_class == 28 and 16 in service.thresholds


def test_thresholds_come_from_validation_only():
    service = VisionService(ROOT / 'models' / 'unified_21class.pt')
    assert 'validação' in service.metadata['selection']
    assert set(service.thresholds) == set(range(1, 22))
