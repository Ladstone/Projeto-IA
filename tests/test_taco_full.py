from fastapi.testclient import TestClient
from app.main import app
from app.nutrition import FOODS, DATA, META, recalculate

client=TestClient(app)

def test_full_taco_catalog():
    assert len(DATA)==597
    foods=client.get('/api/foods').json()['foods']
    assert len(foods)==597
    assert {i['id'] for i in foods}==set(range(1,598))
    assert len(META)==26
    assert client.get('/api/taco/metadata').json()['alimentos']==597
    assert client.get('/api/taco/metadata').json()['colunas_nutricionais'][0]['campo']=='kcal'

def test_chicken_breast_is_present_and_correct():
    chicken=client.get('/api/taco/410').json()
    assert chicken['nome']=='Frango, peito, sem pele, grelhado'
    x={n['campo']:n['valor'] for n in chicken['nutrientes']}
    assert x['kcal']==159
    assert x['proteina_g']==32
    assert x['lipideos_g']==2.5
    assert x['carboidrato_g']==0
    assert x['sodio_mg']==50
    assert x['potassio_mg']==387

def test_editing_misidentified_food_recalculates_from_taco():
    profile={'objetivo':'ganho','kcal':2400,'proteina_g':150,'carboidrato_g':290,'lipideos_g':75,'refeicoes':3}
    bean=recalculate([{'taco_id':561,'grams':100}],profile)
    chicken=recalculate([{'taco_id':410,'grams':100}],profile)
    assert bean['total']['proteina_g'] == 4.8
    assert chicken['total']['proteina_g'] == 32
    assert len(chicken['detalhamento'])==26

def test_missing_values_not_replaced_by_zero():
    chicken=recalculate([{'taco_id':410,'grams':100}],{'refeicoes':3,'kcal':2400,'proteina_g':150,'carboidrato_g':290,'lipideos_g':75})
    data={x['campo']:x for x in chicken['detalhamento']}
    assert data['vitamina_c_mg']['valor'] is None
    assert data['carboidrato_g']['valor']==0
    assert 'Frango, peito, sem pele, grelhado' in data['vitamina_c_mg']['dados_ausentes_em']

def test_pdf_integral_accessible():
    res=client.get('/api/taco/pdf')
    assert res.status_code==200
    assert res.content.startswith(b'%PDF')

def test_catalog_contains_independent_food_categories():
    # The visual model's 21 labels do not limit which TACO items can be used.
    names={x['id']:x['nome'] for x in client.get('/api/foods').json()['foods']}
    assert names[410]=='Frango, peito, sem pele, grelhado'
    assert names[3]=='Arroz, tipo 1, cozido'
    assert names[567].lower().startswith('feijão')

def test_invalid_food_id_is_404():
    assert client.get('/api/taco/99999').status_code==404
