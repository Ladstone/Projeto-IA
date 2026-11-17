"""PratoMetria - API local de segmentação visual e cálculo nutricional."""
from __future__ import annotations
from pathlib import Path
from io import BytesIO
import base64, json, sqlite3, threading
from typing import Optional

import numpy as np
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI,File,UploadFile,HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .nutrition import FOODS, FOOD_CLASS_MAP, DATA, META, details, recalculate
from .vision import VisionService, LABEL_COLORS, CLASS_LABELS

ROOT=Path(__file__).resolve().parents[1];STATIC=ROOT/'app'/'static';MODEL=ROOT/'models'/'unified_21class.pt';DB=ROOT/'app'/'data'/'profile.sqlite3'
app=FastAPI(title='PratoMetria API',version='1.0.0',description='Protótipo acadêmico local: MyFood + Brazilian Food Images + TACO + recomendações')
app.mount('/static',StaticFiles(directory=STATIC),name='static')
lock=threading.Lock();model_lock=threading.Lock()
DEFAULT={'objetivo':'ganho','kcal':2400,'proteina_g':150,'carboidrato_g':290,'lipideos_g':75,'refeicoes':3}

def db_init():
    DB.parent.mkdir(exist_ok=True,parents=True)
    with sqlite3.connect(DB) as con:
        con.execute('create table if not exists profile (id integer primary key check(id=1), data text not null)')
        con.execute('insert or ignore into profile (id,data) values (1,?)',(json.dumps(DEFAULT),))

def profile_get():
    with lock,sqlite3.connect(DB) as con:
        return json.loads(con.execute('select data from profile where id=1').fetchone()[0])

def profile_save(data):
    with lock,sqlite3.connect(DB) as con:con.execute('update profile set data=? where id=1',(json.dumps(data,ensure_ascii=False),))

class Profile(BaseModel):
    objetivo:str='ganho'
    kcal:float=Field(ge=800,le=7000)
    proteina_g:float=Field(ge=0,le=500)
    carboidrato_g:float=Field(ge=0,le=900)
    lipideos_g:float=Field(ge=0,le=300)
    refeicoes:int=Field(ge=1,le=8)
    @field_validator('objetivo')
    @classmethod
    def valid_objetivo(cls,v):
        if v not in {'ganho','perda','equilibrio'}:raise ValueError('Objetivo inválido')
        return v
class FoodInput(BaseModel):
    taco_id:int;grams:float;nome:str='';class_id:Optional[int]=None;note:str=''
class RecalculateRequest(BaseModel):
    items:list[FoodInput]

vision=None
model_error=None
try:
    import torch
    torch.set_num_threads(3)
    if MODEL.exists():vision=VisionService(MODEL)
    else:model_error='Arquivo do modelo ainda não encontrado. Rode scripts/train_model.py.'
except Exception as exc:
    model_error=f'Não foi possível carregar pesos: {exc}'
db_init()

@app.get('/')
def index():return FileResponse(STATIC/'index.html')

@app.get('/api/health')
def health():
    return {'ok':True,'model_ready':vision is not None,'model_error':model_error,'taco_records':len(DATA),'image_classes':len(FOOD_CLASS_MAP),'training_sources':['MyFood Dataset','Brazilian Food Images'],'taco_fields':len(META),'mode':'local-only','model':'FoodUNet 21 classes (MyFood + Brazilian Food Images)','nutrition':'TACO 4ª edição — 2011'}

@app.get('/api/model/info')
def model_info():
    from .unified_classes import CLASS_LABELS
    return {'model':'U-Net compacta, 21 saídas sigmoid',
            'datasets':['MyFood Dataset.zip','Brazilian food images.zip'],
            'mascaras':'Supervisão parcial: apenas classes anotadas são usadas na perda',
            'classes_visuais':CLASS_LABELS,
            'metadata':vision.metadata if vision else {},
            'observacao':'Estas são categorias visuais, não os 597 registros da TACO. Pesos e preparo devem ser conferidos.',
            'limitations':['Máscaras do conjunto brasileiro anotam um alimento por fotografia.',
                'A inferência pode gerar falsos positivos e omitir alimentos.']}

@app.get('/api/profile')
def get_profile():return profile_get()

@app.put('/api/profile')
def put_profile(body:Profile):
    profile_save(body.model_dump());return {'ok':True,'profile':profile_get()}

@app.get('/api/foods')
def all_foods():
    # Disponibilizamos os 597 registros, inclusive aqueles com valor energético ausente.
    return {'foods':[{'id':f['id'],'nome':f['nome']} for f in DATA], 'class_map':FOOD_CLASS_MAP}

@app.get('/api/taco/metadata')
def taco_metadata():
    return {'fonte':'TACO 4ª edição (NEPA/UNICAMP, 2011)', 'alimentos':len(DATA),
            'colunas_nutricionais':META, 'unidade_base':'100 g de parte comestível',
            'observacao':'Tabela 1 estruturada; Tabelas 2 e 3 consultáveis no PDF integral.'}

@app.get('/api/taco/csv')
def taco_csv():
    return FileResponse(ROOT/'app'/'data'/'taco_tabela1_completa.csv',media_type='text/csv',filename='TACO_597_alimentos_26_campos.csv')

@app.get('/api/taco/pdf')
def taco_pdf():
    return FileResponse(ROOT/'app'/'data'/'taco_original.pdf', media_type='application/pdf', filename='TACO_4edicao_NEPA_UNICAMP.pdf')

@app.get('/api/taco/{food_id}')
def get_taco_food(food_id:int):
    try:return details(food_id)
    except ValueError as e:raise HTTPException(status_code=404,detail=str(e))

@app.post('/api/recalculate')
def post_recalculate(body:RecalculateRequest):
    try:return recalculate([b.model_dump() for b in body.items],profile_get())
    except ValueError as e:raise HTTPException(422,str(e))

@app.post('/api/analyze')
async def analyze(file:UploadFile=File(...)):
    """Segmentação local, sem serviços externos, atalhos ou lista fixa de resultados.

    Um pixel pode ter muitas probabilidades independentes; o overlay escolhe só
    a melhor classe elegível. Os candidatos são calculados para CADA imagem.
    Os escores sigmoid NÃO são probabilidades calibradas de acerto.
    """
    if vision is None:
        raise HTTPException(503, model_error or 'Modelo local indisponível')
    raw=await file.read()
    if len(raw)>8*1024*1024:
        raise HTTPException(413,'Imagem deve ter no máximo 8 MB')
    try:
        image=Image.open(BytesIO(raw)); image.verify()
        image=Image.open(BytesIO(raw)).convert('RGB')
        if image.width*image.height>22_000_000:
            raise HTTPException(413,'Resolução excessiva')
    except (UnidentifiedImageError,OSError) as exc:
        raise HTTPException(400,'Arquivo inválido; use JPEG, PNG ou WebP') from exc
    with model_lock:
        mask,probs=vision.predict(image,dimension=vision.metadata.get('resolution',160))

    detected=[]
    diagnostics=[]
    for class_id,record in FOOD_CLASS_MAP.items():
        region=mask==class_id
        frac=float(region.mean())
        confidence=float(probs[class_id-1][region].mean()) if region.any() else 0.0
        threshold=vision.thresholds.get(class_id,0.5)
        candidates=float((probs[class_id-1]>=threshold).mean())
        diagnostics.append({
            'class_id':class_id,'classe_visual':record['nome'],
            'regiao_percent':round(frac*100,2),
            'pixels_acima_limiar_percent':round(candidates*100,2),
            'score_medio_regiao':round(confidence,3),
            'limiar':threshold,
        })
        # Critérios mínimos de região e de escore. NÃO garantem correção semântica.
        if frac<0.012 or confidence<0.35:
            continue
        detected.append({
            'class_id':class_id,'nome':record['nome'],'taco_id':record['taco_id'],
            'grams':record['grams'],'note':record['note'],
            'origem':'modelo_local',
            'confianca':round(confidence,2),
            'cobertura_imagem_percent':round(frac*100,1),
        })
    detected.sort(key=lambda x:x['cobertura_imagem_percent'],reverse=True)
    diagnostics.sort(key=lambda x:x['regiao_percent'],reverse=True)
    overlay=None
    if detected:
        paint=np.zeros((mask.shape[0],mask.shape[1],4),dtype=np.uint8)
        for det in detected:
            color=LABEL_COLORS[det['class_id']].lstrip('#')
            rgb=[int(color[i:i+2],16) for i in (0,2,4)]
            paint[mask==det['class_id']]=(*rgb,115)
        painted=Image.fromarray(paint,'RGBA').resize(image.size,Image.Resampling.NEAREST)
        buffer=BytesIO();painted.save(buffer,format='PNG')
        overlay='data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()
    warning=('A rede local encontrou hipóteses visuais — não há garantia de acerto. '
             'Se estiverem erradas, substitua pela TACO. Informe o peso real e confirme o preparo.'
             if detected else
             'Nenhuma classe ultrapassou os critérios de detecção. Use a pesquisa TACO para adicionar os alimentos presentes.')
    return {
        'detected':detected,'unmatched':[],'preview_overlay':overlay,
        'nutrition':recalculate(detected,profile_get()),
        'warning':warning,'mode':'local','analysis_quality':'experimental',
        'vision_debug':diagnostics,
    }
