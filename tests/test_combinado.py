"""Modelo combinado v2+v3 (porta 8001): o v3 só entra quando a foto tem comida nova (classes 22+)."""
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from app.vision import CombinedVisionService, load_vision

ROOT = Path(__file__).parents[1]
LUNCH = Path(__file__).parent / 'fixtures' / 'lunch.jpg'


class Fake:
    """Imita VisionService com máscara e probabilidades fixas."""

    def __init__(self, n_class, mask, label):
        self.n_class, self.mask = n_class, np.array(mask, dtype='uint8')
        self.resolution, self.architecture = self.mask.shape[0], 'fake'
        self.thresholds = {c: c / 100 for c in range(1, n_class + 1)}
        self.metadata = {'label': label, 'datasets': [label]}

    def predict(self, img, dimension=None):
        return self.mask, np.full((self.n_class, *self.mask.shape), 0.9, dtype=np.float32)


BASE = [[1, 1, 3, 3]] * 4                  # arroz e frango
SEM_NOVA = [[1, 1, 8, 8]] * 4              # v3 vê macarrão: classe antiga, não vale
COM_NOVA = [[1, 1, 28, 28]] * 4            # v3 vê lasanha


def test_sem_comida_nova_vale_o_base():
    c = CombinedVisionService(Fake(21, BASE, 'v2'), Fake(28, SEM_NOVA, 'v3'))
    mask, probs = c.predict(None)
    assert (mask == np.array(BASE)).all() and probs.shape == (28, 4, 4)


def test_com_comida_nova_vale_o_modelo_novo_na_foto_inteira():
    c = CombinedVisionService(Fake(21, BASE, 'v2'), Fake(28, COM_NOVA, 'v3'))
    mask, _ = c.predict(None)
    assert (mask == np.array(COM_NOVA)).all()


def test_comida_nova_pequena_nao_troca_e_so_entra_onde_o_base_nao_viu_comida():
    base = [[1, 1, 3, 3], [1, 1, 3, 3], [1, 1, 3, 3], [0, 0, 0, 0]]
    nova = [[1, 1, 26, 3], [1, 1, 3, 3], [1, 1, 3, 3], [0, 0, 0, 27]]  # 6% de cebola, 6% de sanduíche
    c = CombinedVisionService(Fake(21, base, 'v2'), Fake(28, nova, 'v3'))
    mask, _ = c.predict(None)
    esperado = np.array(base); esperado[3, 3] = 27  # a cebola em cima do frango não entra; o sanduíche no vazio entra
    assert (mask == esperado).all()


def test_modo_pixel_so_troca_os_pixels_da_classe_nova():
    c = CombinedVisionService(Fake(21, BASE, 'v2'), Fake(28, COM_NOVA, 'v3'), modo='pixel')
    mask, _ = c.predict(None)
    assert (mask == np.array([[1, 1, 28, 28]] * 4)).all()
    c = CombinedVisionService(Fake(21, BASE, 'v2'), Fake(28, SEM_NOVA, 'v3'), modo='pixel')
    assert (c.predict(None)[0] == np.array(BASE)).all()


def test_limiares_e_metadados():
    c = CombinedVisionService(Fake(21, BASE, 'v2'), Fake(28, BASE, 'v3'))
    assert c.n_class == 28 and set(c.thresholds) == set(range(1, 29))
    assert c.metadata['classes'] == 28 and c.metadata['datasets'] == ['v3'] and 'v2' in c.metadata['label']
    with pytest.raises(ValueError):
        CombinedVisionService(Fake(21, BASE, 'v2'), Fake(28, BASE, 'v3'), modo='outro')
    with pytest.raises(ValueError):
        CombinedVisionService(Fake(28, BASE, 'v3'), Fake(21, BASE, 'v2'))


def test_v2_mais_v3_reais_carregam_e_preveem():
    c = load_vision('models/unified_21class_v2.pt+models/unified_28class_v3.pt', ROOT)
    assert isinstance(c, CombinedVisionService) and c.n_base == 21 and c.n_class == 28
    mask, probs = c.predict(Image.open(LUNCH))
    assert mask.shape == (224, 224) and probs.shape == (28, 224, 224)


def test_pure_do_especialista_vale_por_cima_do_resto():
    """Porta 8001: onde o v3d marca purê (16), vale purê; o resto continua como o modelo principal disse."""
    from app.vision import ClassOverrideService
    especialista = [[16, 16, 3, 4]] * 4    # purê onde o base viu arroz; frango/carne do especialista não valem
    c = ClassOverrideService(Fake(21, BASE, 'v2'), Fake(28, especialista, 'v3d'))
    mask, probs = c.predict(None)
    assert (mask == np.array([[16, 16, 3, 3]] * 4)).all()
    assert probs.shape == (21, 4, 4) and c.thresholds[16] == 0.16
