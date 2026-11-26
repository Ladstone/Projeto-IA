"""Checks actual dataset decoding and class mapping; ZIP-dependent checks are optional."""
from pathlib import Path
import os, pytest
from app.unified_classes import N_CLASS, BRAZIL_LABELS, MYFOOD_LABELS, MYFOODREPO_LABELS, NEW_CLASSES, FOOD_CLASS_MAP, LABEL_COLORS
from app.nutrition import FOODS

def test_28_visual_classes_mapped_to_existing_taco_rows():
    assert N_CLASS==28
    assert len(BRAZIL_LABELS)==20
    assert set(FOOD_CLASS_MAP)==set(range(1,29))
    assert set(LABEL_COLORS)==set(FOOD_CLASS_MAP) and len(set(LABEL_COLORS.values()))==N_CLASS
    assert all(x['taco_id'] in FOODS for x in FOOD_CLASS_MAP.values())
    assert FOOD_CLASS_MAP[3]['taco_id']==410  # chicken, tentative TACO preparation
    assert FOOD_CLASS_MAP[6]['taco_id']==93  # french fries

def test_myfoodrepo_map_covers_every_new_class():
    assert set(MYFOODREPO_LABELS.values())<=set(FOOD_CLASS_MAP)
    assert set(NEW_CLASSES)==set(range(22,29))<=set(MYFOODREPO_LABELS.values())
    assert MYFOODREPO_LABELS['chicken-leg']==3  # coxa de frango junta com Frango (poucas fotos)
    assert MYFOODREPO_LABELS['hamburger']==4  # só a carne; o sanduíche é outra categoria
    assert MYFOODREPO_LABELS['hamburger-bread-meat-ketchup']==27

@pytest.fixture(scope='module')
def dataset_if_available():
    b=Path(os.getenv('PRATOMETRIA_DATASET_BRAZIL',str(Path(__file__).parents[2]/'Brazilian food images.zip')))
    a=Path(os.getenv('PRATOMETRIA_DATASET_MYFOOD',str(Path(__file__).parents[2]/'MyFood Dataset.zip')))
    if not a.exists() or not b.exists():pytest.skip('Uploaded datasets not present, rerun with environment paths')
    from scripts.data_v2 import load_records
    return load_records(a,b,'train',dim=64)

def test_real_brazil_mask_if_zip_available(dataset_if_available):
    from scripts.data_v2 import targets_for
    r=next(r for r in dataset_if_available if r['src']=='brazil' and r['category']=='chicken')
    target,known=targets_for(r['src'],r['label'],r['cls'])
    assert r['cls']==3
    assert known[2].all()  # a classe anotada (frango) é conhecida na foto inteira
    assert 0<target[2].sum()<target[2].size
    assert (known[0]==r['label'].astype(bool)).all() and not known[0].all()  # outras classes: só dentro da máscara

def test_mixed_myfood_lunch_if_zip_available(dataset_if_available):
    from scripts.data_v2 import targets_for, IGNORE
    r=next(r for r in dataset_if_available if r['src']=='myfood' and r['category']=='lunch')
    target,known=targets_for(r['src'],r['label'],r['cls'])
    assert target[0].sum()>0 and target[1].sum()>0  # arroz e feijão anotados no prato misto
    assert (known[5]==(r['label']!=IGNORE)).all()  # todas as classes conhecidas, menos em "outro alimento"
