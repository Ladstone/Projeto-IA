"""Checks actual dataset decoding and class mapping; ZIP-dependent checks are optional."""
from pathlib import Path
import os, pytest
from app.unified_classes import N_CLASS, BRAZIL_LABELS, MYFOOD_LABELS, FOOD_CLASS_MAP
from app.nutrition import FOODS

def test_21_visual_classes_mapped_to_existing_taco_rows():
    assert N_CLASS==21
    assert len(BRAZIL_LABELS)==20
    assert set(FOOD_CLASS_MAP)==set(range(1,22))
    assert all(x['taco_id'] in FOODS for x in FOOD_CLASS_MAP.values())
    assert FOOD_CLASS_MAP[3]['taco_id']==410  # chicken, tentative TACO preparation
    assert FOOD_CLASS_MAP[6]['taco_id']==93  # french fries

@pytest.fixture(scope='module')
def dataset_if_available():
    b=Path(os.getenv('PRATOMETRIA_DATASET_BRAZIL',str(Path(__file__).parents[2]/'Brazilian food images.zip')))
    a=Path(os.getenv('PRATOMETRIA_DATASET_MYFOOD',str(Path(__file__).parents[2]/'MyFood Dataset.zip')))
    if not a.exists() or not b.exists():pytest.skip('Uploaded datasets not present, rerun with environment paths')
    from scripts.train_unified import Combined
    return Combined(a,b,part='train',dim=64)

def test_real_brazil_mask_if_zip_available(dataset_if_available):
    dataset=dataset_if_available
    brazil=next(i for i,r in enumerate(dataset.records) if r[3]=='brazil' and r[4]=='chicken')
    _, mask, sup, valid=dataset[brazil]
    assert sup[2]==1
    assert sup.sum()==1  # only the annotated chicken head, NOT other foods
    assert 0<mask[2].sum()<mask[2].numel()
    assert int(valid.sum())==64*64

def test_mixed_myfood_lunch_if_zip_available(dataset_if_available):
    dataset=dataset_if_available
    idx=next(i for i,r in enumerate(dataset.records) if r[3]=='myfood' and r[4]=='lunch')
    _, mask, sup, valid=dataset[idx]
    assert sup[0]==1 and sup[1]==1 # rice and beans included in supervised heads
    assert sup[5]==0  # fries Brazil-only: not presumed absent from MyFood
    assert mask[0].sum()>0 and mask[1].sum()>0
