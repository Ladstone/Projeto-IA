from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app
from app.nutrition import FOODS,recalculate

client=TestClient(app)


def test_taco_597_items():
    assert len([i for i in FOODS if i < 9000])==597  # 9000+ = receitas calculadas (ex.: purê)
    assert FOODS[3]['nome']=='Arroz, tipo 1, cozido'
    assert FOODS[3]['kcal']==128.0


def test_nutrition_portion_calculation():
    result=recalculate([{'taco_id':3,'grams':125,'nome':'Arroz'}],{'objetivo':'ganho','kcal':2400,'proteina_g':150,'carboidrato_g':290,'lipideos_g':75,'refeicoes':3})
    assert result['total']['kcal']==160.0
    assert result['total']['proteina_g']==3.1
    assert result['total']['carboidrato_g']==35.1
    assert result['comparacao']['kcal']['alvo_refeicao']==800.0


def test_api_health_and_foods():
    assert client.get('/').status_code==200
    assert client.get('/api/health').json()['taco_records']==597
    assert client.get('/api/health').json()['model_ready'] is True
    assert len(client.get('/api/foods').json()['class_map'])==28


def test_profile_persistence():
    before=client.get('/api/profile').json()
    updated={**before,'kcal':2600,'refeicoes':4}
    try:
        res=client.put('/api/profile',json=updated)
        assert res.status_code==200
        assert client.get('/api/profile').json()['kcal']==2600
    finally:
        client.put('/api/profile',json=before)


def test_invalid_portion():
    res=client.post('/api/recalculate',json={'items':[{'taco_id':3,'grams':-5}]})
    assert res.status_code==422


def test_invalid_photo():
    res=client.post('/api/analyze',files={'file':('fake.png',b'not an image','image/png')})
    assert res.status_code==400


def test_local_image_inference_returns_real_data():
    image=(Path(__file__).parent/'fixtures'/'lunch.jpg').read_bytes()
    res=client.post('/api/analyze',files={'file':('test.jpg',image,'image/jpeg')})
    assert res.status_code==200
    data=res.json()
    assert data['mode']=='local'
    assert data['analysis_quality']=='experimental'
    assert isinstance(data['detected'],list)
    assert data['nutrition'] is not None
    assert data['preview_overlay'] is None or data['preview_overlay'].startswith('data:image/png;base64,')
