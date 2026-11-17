"""Alimentos TACO (4ª edição), cálculo por porção e recomendações de demonstração.
Dado ausente não é zero. A identificação visual é um estágio independente.
"""
from pathlib import Path
import json
from math import isfinite

DATA = sorted(json.loads((Path(__file__).resolve().parent/'data'/'taco.json').read_text(encoding='utf-8')),key=lambda d:d['id'])
FOODS = {food['id']:food for food in DATA}

from .unified_classes import FOOD_CLASS_MAP

# A Tabela 1 inclui a composição centesimal + minerais + vitaminas + colesterol.
# Tabelas 2 (ácidos graxos) e 3 (aminoácidos) permanecem integralmente disponíveis no PDF local.
FIELDS=[
 ('kcal','Energia','kcal'),('kj','Energia','kJ'),('proteina_g','Proteínas','g'),
 ('carboidrato_g','Carboidratos','g'),('lipideos_g','Lipídios totais','g'),
 ('fibra_g','Fibra alimentar','g'),('umidade_g','Umidade','g'),('cinzas_g','Cinzas','g'),
 ('colesterol_mg','Colesterol','mg'),('calcio_mg','Cálcio','mg'),
 ('magnesio_mg','Magnésio','mg'),('manganes_mg','Manganês','mg'),
 ('fosforo_mg','Fósforo','mg'),('ferro_mg','Ferro','mg'),('sodio_mg','Sódio','mg'),
 ('potassio_mg','Potássio','mg'),('cobre_mg','Cobre','mg'),('zinco_mg','Zinco','mg'),
 ('retinol_ug','Retinol','µg'),('vitamina_a_re_ug','Vitamina A (RE)','µg'),
 ('vitamina_a_rae_ug','Vitamina A (RAE)','µg'),('tiamina_mg','Tiamina (B1)','mg'),
 ('riboflavina_mg','Riboflavina (B2)','mg'),('piridoxina_mg','Piridoxina (B6)','mg'),
 ('niacina_mg','Niacina (B3)','mg'),('vitamina_c_mg','Vitamina C','mg')]
META=[{'campo':field,'nome':label,'unidade':unit} for field,label,unit in FIELDS]
NUTRIENTS=['kcal','proteina_g','carboidrato_g','lipideos_g','fibra_g']
TARGET_FIELDS=['kcal','proteina_g','carboidrato_g','lipideos_g']

def nval(food,field):
    v=food.get(field)
    return float(v) if isinstance(v,(int,float)) and isfinite(v) else None

def details(food_id):
    try: food=FOODS[int(food_id)]
    except (KeyError,TypeError,ValueError): raise ValueError('ID TACO inválido')
    return {'id':food['id'],'nome':food['nome'],'fonte':food['fonte'],
            'pagina_pdf':food.get('pagina_pdf'),'base':'100 g da parte comestível',
            'nutrientes':[{'campo':m['campo'],'nome':m['nome'],'unidade':m['unidade'],
                           'valor':nval(food,m['campo']),
                           'observacao':food.get('marcadores',{}).get(m['campo'])} for m in META]}

def recalculate(items,profile):
    foods=[]; totals={key:0.0 for key in NUTRIENTS}
    full_totals={field:0.0 for field,_,_ in FIELDS}; incomplete={field:[] for field,_,_ in FIELDS}
    for it in items:
        taco_id=int(it['taco_id']);grams=float(it['grams'])
        if taco_id not in FOODS:raise ValueError(f'Alimento TACO inexistente: {taco_id}')
        if not isfinite(grams) or grams<1 or grams>2500:raise ValueError('Porção deve estar entre 1 e 2500 g')
        src=FOODS[taco_id];contrib={}
        for field,_,_ in FIELDS:
            v=nval(src,field)
            if v is None:
                contrib[field]=None;incomplete[field].append(src['nome'])
            else:
                contrib[field]=round(v*grams/100,3)
                full_totals[field]+=v*grams/100
                if field in totals:totals[field]+=v*grams/100
        foods.append({'nome':src['nome'], 'taco_id':taco_id, 'descricao_taco':src['nome'],
                      'grams':round(grams,1),'class_id':it.get('class_id'),
                      'nutrientes':contrib,'aviso':it.get('note') or '',
                      'ausentes':[key for key,v in contrib.items() if v is None]})
    totals={k:round(v,1) for k,v in totals.items()}
    target={key:round(float(profile.get(key,0))/max(1,int(profile.get('refeicoes',3))),1) for key in TARGET_FIELDS}
    comparison={key:{'obtido':totals[key], 'alvo_refeicao':target[key],
                     'percentual':round(100*totals[key]/target[key]) if target[key]>0 else None} for key in TARGET_FIELDS}
    detailed=[{'campo':field,'nome':label,'unidade':unit,
               'valor':round(full_totals[field],2) if not incomplete[field] and items else None,
               'parcial':round(full_totals[field],2) if incomplete[field] and items else None,
               'dados_ausentes_em':incomplete[field]} for field,label,unit in FIELDS]
    warning=('Alguns valores TACO estão ausentes: não os tratamos como zero. ' if any(incomplete[k] for k in TARGET_FIELDS) else '')
    return {'alimentos':foods,'total':totals,'detalhamento':detailed,'referencia_refeicao':target,
            'comparacao':comparison,'recomendacoes':recommend(totals,target,profile),
            'aviso':warning+'Valores da TACO por 100 g. Gramas, ingredientes e formas de preparo devem ser conferidos.'}

def recommend(total,ref,profile):
    tips=[];objetivo=profile.get('objetivo','equilibrio')
    if ref['proteina_g']>0 and total['proteina_g']<ref['proteina_g']*.8:
        deficit=ref['proteina_g']-total['proteina_g'];g=round(max(40,min(180,deficit/0.32))/10)*10
        tips.append({'titulo':'Aumentar a proteína','descricao':f'Faltam cerca de {deficit:.0f} g de proteína em relação à referência desta refeição. Uma opção é acrescentar aproximadamente {g} g de peito de frango grelhado.','tipo':'proteina'})
    if ref['kcal']>0 and total['kcal']<ref['kcal']*.72 and objetivo in ('ganho','equilibrio'):
        tips.append({'titulo':'Energia abaixo da referência','descricao':'Considere aumentar o arroz, tubérculos ou outra fonte de energia, conforme as demais refeições do dia.','tipo':'energia'})
    elif ref['kcal']>0 and total['kcal']>ref['kcal']*1.25 and objetivo=='perda':
        tips.append({'titulo':'Revisar porções','descricao':'A energia ficou acima da referência simplificada por refeição. Confirme os pesos estimados.','tipo':'energia'})
    if ref['carboidrato_g']>0 and total['carboidrato_g']<ref['carboidrato_g']*.70:
        tips.append({'titulo':'Carboidratos abaixo da referência','descricao':'Arroz ou batata são opções para ajustar o teor de carboidratos.','tipo':'carboidratos'})
    if ref['lipideos_g']>0 and total['lipideos_g']>ref['lipideos_g']*1.35:
        tips.append({'titulo':'Gorduras acima da referência','descricao':'Considere reduzir molhos e óleos ou substituir frituras por preparações grelhadas.','tipo':'gorduras'})
    if total['fibra_g']<4:
        tips.append({'titulo':'Adicionar variedade vegetal','descricao':'Verduras, legumes e leguminosas podem melhorar o aporte de fibras.','tipo':'fibras'})
    if not tips:
        tips.append({'titulo':'Refeição próxima da referência','descricao':'Os macronutrientes parecem próximos das metas por refeição. Confirme os pesos antes do registro.','tipo':'equilibrio'})
    return tips[:4]
