"""Leitura do MyFoodRepo: polígonos COCO viram máscara, divisão por hash e teto de imagens por classe."""
import json
from pathlib import Path
import os

import numpy as np
import pytest

from app.unified_classes import N_CLASS
from scripts.data_v2 import (IGNORE, MFR_CAPS, myfoodrepo_split, polygons_to_label, presence_status,
                             select_myfoodrepo, targets_for)

SQUARE_LEFT = json.dumps([[0, 0, 50, 0, 50, 100, 0, 100]])      # metade esquerda de 100x100, como texto (formato do dataset)
SQUARE_RIGHT = [[50, 0, 100, 0, 100, 100, 50, 100]]               # metade direita, como lista


def test_polygons_rasterized_with_class_and_unknown_food():
    label = polygons_to_label([(22, SQUARE_LEFT), (None, SQUARE_RIGHT)], 100, 100, dim=20)
    assert label.shape == (20, 20)
    assert (label[:, :9] == 22).all()          # brócolis
    assert (label[:, 11:] == IGNORE).all()     # categoria fora do mapa = outro alimento


def test_polygons_scaled_from_original_size():
    label = polygons_to_label([(25, [[0, 0, 200, 0, 200, 400, 0, 400]])], 400, 400, dim=40)
    assert (label[:, :19] == 25).all() and (label[:, 21:] == 0).all()


def test_myfoodrepo_presence_and_targets():
    label = np.zeros((40, 40), np.uint8)
    label[:20] = 22          # brócolis anotado
    label[20:25] = IGNORE    # outro alimento
    status = presence_status('myfoodrepo', label, None, True)
    assert status[22 - 1] == 1
    assert status[25 - 1] == 0                 # pão é rotulado pelo MyFoodRepo: ausente com certeza
    assert status[2 - 1] == -1                 # feijão não é rotulado lá: pode estar no "outro alimento"
    target, known = targets_for('myfoodrepo', label, None)
    assert target.shape == (N_CLASS, 40, 40)
    assert target[22 - 1, :20].all() and not target[22 - 1, 20:].any()
    assert not known[:, 20:25].any() and known[:, 25:].all()


def test_split_is_deterministic_and_roughly_80_10_10():
    parts = [myfoodrepo_split(f'{i:06d}.jpg') for i in range(5000)]
    assert parts == [myfoodrepo_split(f'{i:06d}.jpg') for i in range(5000)]
    share = {p: parts.count(p) / len(parts) for p in ('train', 'val', 'test')}
    assert 0.75 < share['train'] < 0.85 and 0.07 < share['val'] < 0.13 and 0.07 < share['test'] < 0.13


def test_selection_respects_caps_and_split():
    images = [dict(id=i, file_name=f'{i:06d}.jpg') for i in range(3000)]
    anns = {i: [(22, None), (1, None)] if i % 2 else [(1, None), (None, None)] for i in range(3000)}
    chosen, count = select_myfoodrepo(images, anns, 'train', (10, 5))
    assert all(myfoodrepo_split(img['file_name']) == 'train' for img, _ in chosen)
    assert count[22] == 10                     # classe nova chega ao teto e para
    assert sum(1 for img, _ in chosen if img['id'] % 2 == 0) <= 5   # fotos só de arroz param no teto de 5
    assert count[1] <= 5 + 10                  # mais as que entram de carona com o brócolis
    assert all(primary == 22 for img, primary in chosen if img['id'] % 2)


@pytest.fixture(scope='module')
def archive():
    path = Path(os.getenv('PRATOMETRIA_DATASET_MYFOODREPO', str(Path(__file__).parents[2] / 'archive.zip')))
    if not path.exists():
        pytest.skip('archive.zip do MyFoodRepo ausente')
    return path


def test_real_myfoodrepo_val_if_zip_available(archive):
    from scripts.data_v2 import load_myfoodrepo
    recs = load_myfoodrepo(archive, 'val', dim=64, caps=(3, 0))
    assert recs and all(r['src'] == 'myfoodrepo' for r in recs)
    assert all(r['label'].shape == (64, 64) and r['img'].shape == (64, 64, 3) for r in recs)
    assert any((r['label'] == 22).any() for r in recs)     # algum brócolis de verdade
    assert MFR_CAPS['train'] == (150, 50)


def test_v3c_old_classes_become_unknown_except_kept(monkeypatch):
    """--mfr-old-classes: no MyFoodRepo, frango vira "outro alimento"; purê (mantido) continua rotulado."""
    import io, zipfile
    from PIL import Image
    import scripts.data_v2 as d
    img = io.BytesIO(); Image.new('RGB', (100, 100), (200, 180, 150)).save(img, format='JPEG')
    ann = dict(images=[dict(id=i, file_name=f'{i}.jpg', width=100, height=100) for i in range(40)],
               categories=[dict(id=1, name='chicken'), dict(id=2, name='mashed-potatoes-prepared-with-full-fat-milk-with-butter'),
                           dict(id=3, name='broccoli')],
               annotations=[a for i in range(40) for a in (dict(image_id=i, category_id=1, segmentation=SQUARE_LEFT),
                                                         dict(image_id=i, category_id=2 if i % 2 else 3, segmentation=SQUARE_RIGHT))])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        z.writestr(d.MFR_PREFIX + 'annotations.json', json.dumps(ann))
        for i in range(40):
            z.writestr(d.MFR_PREFIX + f'images/{i}.jpg', img.getvalue())
    buf.seek(0)
    monkeypatch.setattr(d, 'myfoodrepo_split', lambda name: 'train')
    recs = d.load_myfoodrepo(buf, 'train', dim=20, old_classes={16: 5})
    assert recs and all(r['src'] == 'mfr_novas' for r in recs)
    labels = np.stack([r['label'] for r in recs])
    assert not (labels == 3).any()                       # frango nunca aparece como rótulo
    assert (labels[:, :, :9] == IGNORE).all()            # o lado do frango vira "outro alimento"
    assert sum((r['label'] == 16).any() for r in recs) == 5   # purê limitado ao teto pedido
    # Validação: frango escondido não pode contar como ausente com certeza; brócolis e purê sim.
    st = presence_status('mfr_novas', recs[0]['label'], None, True)
    assert st[3 - 1] == -1 and st[22 - 1] in (0, 1) and st[16 - 1] in (0, 1)


def test_pure_pseudo_only_marked_pixels_teach():
    """Purê com pseudo-rótulo: dentro da máscara é purê (e não as outras classes); fora, desconhecido."""
    import numpy as np
    from scripts.data_v2 import targets_for
    label = np.zeros((4, 4), np.uint8)
    label[:2] = 1
    target, known = targets_for('pure_pseudo', label, 16)
    assert target[15, :2].all() and not target[15, 2:].any()
    assert known[:, :2].all() and not known[:, 2:].any()
    assert not np.delete(target, 15, axis=0).any()
