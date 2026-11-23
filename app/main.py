"""PratoMetria - API local de segmentação visual e cálculo nutricional."""
from __future__ import annotations
from pathlib import Path
from io import BytesIO
import base64, json, sqlite3, threading
from typing import Optional

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import FastAPI,File,Form,UploadFile,HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .nutrition import FOODS, FOOD_CLASS_MAP, DATA, META, details, recalculate
from .vision import VisionService, LABEL_COLORS, CLASS_LABELS, MIN_REGION_FRACTION, food_crop_box
from .unified_classes import RECIPES
from .metas import calcular_metas, normaliza_objetivo, OBJETIVOS, ATIVIDADES

ROOT=Path(__file__).resolve().parents[1];STATIC=ROOT/'app'/'static';MODEL=ROOT/'models'/'unified_21class.pt';DB=ROOT/'app'/'data'/'profile.sqlite3'
app=FastAPI(title='PratoMetria API',version='1.0.0',description='Protótipo acadêmico local: MyFood + Brazilian Food Images + TACO + recomendações')
app.mount('/static',StaticFiles(directory=STATIC),name='static')
lock=threading.Lock();model_lock=threading.Lock()
DEFAULT={'objetivo':'hipertrofia','kcal':2400,'proteina_g':150,'carboidrato_g':290,'lipideos_g':75,'refeicoes':3}

def db_init():
    DB.parent.mkdir(exist_ok=True,parents=True)
    with sqlite3.connect(DB) as con:
        con.execute('create table if not exists profile (id integer primary key check(id=1), data text not null)')
        con.execute('insert or ignore into profile (id,data) values (1,?)',(json.dumps(DEFAULT),))

def profile_get():
    with lock,sqlite3.connect(DB) as con:
        data=json.loads(con.execute('select data from profile where id=1').fetchone()[0])
    data['objetivo']=normaliza_objetivo(data.get('objetivo','manutencao'))  # perfis salvos com ids antigos
    return data

def profile_save(data):
    with lock,sqlite3.connect(DB) as con:con.execute('update profile set data=? where id=1',(json.dumps(data,ensure_ascii=False),))

class Medidas(BaseModel):
    """Dados para calcular as metas. Opcionais no perfil: a pessoa pode digitar as metas à mão."""
    sexo:Optional[str]=None
    idade:Optional[int]=Field(default=None,ge=18,le=100)
    peso_kg:Optional[float]=Field(default=None,ge=30,le=300)
    altura_cm:Optional[float]=Field(default=None,ge=120,le=230)
    atividade:Optional[str]=None
    @field_validator('sexo')
    @classmethod
    def valid_sexo(cls,v):
        if v not in (None,'feminino','masculino'):raise ValueError('Use feminino ou masculino')
        return v
    @field_validator('atividade')
    @classmethod
    def valid_atividade(cls,v):
        if v is not None and v not in ATIVIDADES:raise ValueError('Nível de atividade inválido')
        return v
class Profile(Medidas):
    objetivo:str='hipertrofia'
    kcal:float=Field(ge=800,le=7000)
    proteina_g:float=Field(ge=0,le=500)
    carboidrato_g:float=Field(ge=0,le=900)
    lipideos_g:float=Field(ge=0,le=300)
    refeicoes:int=Field(ge=1,le=8)
    @field_validator('objetivo')
    @classmethod
    def valid_objetivo(cls,v):
        v=normaliza_objetivo(v)
        if v not in OBJETIVOS:raise ValueError('Objetivo inválido')
        return v
class CalculoMetas(BaseModel):
    sexo:str;idade:int;peso_kg:float;altura_cm:float;atividade:str;objetivo:str
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
    else:model_error='Arquivo do modelo ainda não encontrado. Rode scripts/train_unified.py.'
except Exception as exc:
    model_error=f'Não foi possível carregar pesos: {exc}'
db_init()

@app.get('/')
def index():return FileResponse(STATIC/'index.html')

@app.get('/api/health')
def health():
    return {'ok':True,'model_ready':vision is not None,'model_error':model_error,'taco_records':len(DATA),'image_classes':len(FOOD_CLASS_MAP),'training_sources':['MyFood Dataset','Brazilian Food Images'],'taco_fields':len(META),'mode':'local-only','model':model_description(),'model_label':(vision.metadata.get('label') if vision else None),'nutrition':'TACO 4ª edição — 2011'}

def model_description():
    """Descrição lida do .json do modelo ativo, para não ficar desatualizada ao trocar de modelo."""
    if vision is None:
        return None
    return vision.metadata.get('model') or f'{vision.architecture}, 21 classes'

@app.get('/api/model/info')
def model_info():
    from .unified_classes import CLASS_LABELS
    return {'model':model_description(),
            'label':vision.metadata.get('label') if vision else None,
            'datasets':['MyFood Dataset.zip','Brazilian food images.zip'],
            'mascaras':'Pixels anotados supervisionam todas as classes (positivo na própria, negativo nas demais); '
                       'no conjunto brasileiro, pixels não anotados só contam para a classe anotada',
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

@app.get('/api/profile/opcoes')
def profile_options():
    """Objetivos e níveis de atividade aceitos, com os rótulos para a interface."""
    return {'objetivos':[{'id':k,'nome':v[0]} for k,v in OBJETIVOS.items()],
            'atividades':[{'id':k,'nome':v[1]} for k,v in ATIVIDADES.items()]}

@app.post('/api/profile/calcular')
def profile_calculate(body:CalculoMetas):
    """Calcula metas diárias pelas medidas (Mifflin-St Jeor + objetivo). Não salva: a pessoa revisa antes."""
    try:return calcular_metas(**body.model_dump())
    except ValueError as e:raise HTTPException(422,str(e))

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

# MIN_REGION_FRACTION (área mínima de detecção) fica em app/vision.py para que o app
# e scripts/evaluate_models.py usem exatamente a mesma regra.
# Fração do PRATO que corresponde à porção padrão da categoria. Medir pelo prato (o recorte,
# ou o retângulo em volta da comida) e não pela foto inteira faz o valor não depender da
# distância da câmera. Heurística calibrada a olho num prato feito real, não medição de massa.
REFERENCE_PLATE_FRACTION=0.12
MAX_PORTION_SCALE=2.0  # teto: recorte muito justo ou região errada não vira porção absurda

def load_image(raw:bytes)->Image.Image:
    """Abre a imagem enviada e aplica a orientação EXIF (fotos de celular).

    Sem isso a rede analisa a foto deitada e a máscara não coincide com a
    imagem que o navegador exibe (o navegador aplica a orientação EXIF).
    """
    try:
        probe=Image.open(BytesIO(raw)); probe.verify()
        image=ImageOps.exif_transpose(Image.open(BytesIO(raw))).convert('RGB')
    except Image.DecompressionBombError as exc:
        raise HTTPException(413,'Resolução excessiva') from exc
    except (UnidentifiedImageError,OSError,SyntaxError) as exc:
        raise HTTPException(400,'Arquivo inválido; use JPEG, PNG ou WebP') from exc
    if image.width*image.height>22_000_000:
        raise HTTPException(413,'Resolução excessiva')
    return image

def suggest_grams(default_grams:float,plate_fraction:float)->int:
    """Porção inicial proporcional à área do alimento no prato, limitada a 25%-200% da porção padrão.

    A foto NÃO mede massa: é só um ponto de partida que o usuário deve corrigir.
    """
    scale=min(MAX_PORTION_SCALE,max(0.25,plate_fraction/REFERENCE_PLATE_FRACTION))
    grams=int(round(default_grams*scale/5)*5)
    cap=int(default_grams*MAX_PORTION_SCALE//5*5)  # arredonda o teto para baixo, para não ultrapassá-lo
    return max(5,min(grams,cap))

@app.post('/api/crop-suggestion')
async def crop_suggestion(file:UploadFile=File(...)):
    """Sugere um recorte em volta da comida (frações 0-1 da foto) para o usuário ajustar.

    Usa as regiões da análise normal; se não houver nenhuma, devolve box=None.
    """
    if vision is None:
        raise HTTPException(503, model_error or 'Modelo local indisponível')
    raw=await file.read()
    if len(raw)>8*1024*1024:
        raise HTTPException(413,'Imagem deve ter no máximo 8 MB')
    image=load_image(raw)
    with model_lock:
        mask,_=vision.predict(image)
    box=food_crop_box(mask)
    return {'box':None if box is None else dict(zip(('x0','y0','x1','y1'),(round(v,4) for v in box)))}

@app.post('/api/analyze')
async def analyze(file:UploadFile=File(...),area_original:float=Form(1.0)):
    """Segmentação local, sem serviços externos, atalhos ou lista fixa de resultados.

    ``area_original``: quando o usuário recorta a foto em volta do prato, a fração da
    foto original que o recorte ocupa (0-1]. A área mínima de detecção vale sobre o
    recorte (o que a rede vê) e "% da foto" usa a foto ORIGINAL. As gramas usam a área no
    PRATO: o recorte, quando há; sem recorte, o retângulo em volta da comida detectada
    (o mesmo que /api/crop-suggestion sugere). Assim a distância da câmera não muda a porção.

    Um pixel pode ter muitas probabilidades independentes; o overlay escolhe só
    a melhor classe elegível. Os candidatos são calculados para CADA imagem.
    Os escores sigmoid NÃO são probabilidades calibradas de acerto: como a região
    só contém pixels acima do limiar da classe, o score médio é sempre >= limiar.
    """
    if vision is None:
        raise HTTPException(503, model_error or 'Modelo local indisponível')
    if not 0.01<=area_original<=1:
        raise HTTPException(422,'area_original deve estar entre 0,01 e 1')
    raw=await file.read()
    if len(raw)>8*1024*1024:
        raise HTTPException(413,'Imagem deve ter no máximo 8 MB')
    image=load_image(raw)
    with model_lock:
        mask,probs=vision.predict(image)

    plate_area=1.0
    if area_original>=1:
        box=food_crop_box(mask)
        if box:plate_area=max(0.05,(box[2]-box[0])*(box[3]-box[1]))
    detected=[]
    diagnostics=[]
    for class_id,record in FOOD_CLASS_MAP.items():
        region=mask==class_id
        frac=float(region.mean())
        score=float(probs[class_id-1][region].mean()) if region.any() else 0.0
        threshold=vision.thresholds.get(class_id,0.5)
        candidates=float((probs[class_id-1]>=threshold).mean())
        diagnostics.append({
            'class_id':class_id,'classe_visual':record['nome'],
            'regiao_percent':round(frac*100,2),
            'pixels_acima_limiar_percent':round(candidates*100,2),
            'score_medio_regiao':round(score,3),
            'limiar':threshold,
        })
        # Só a área mínima filtra. NÃO garante correção semântica.
        if frac<MIN_REGION_FRACTION:
            continue
        base={
            'class_id':class_id,'porcao_padrao_g':record['grams'],'origem':'modelo_local',
            'score_nao_calibrado':round(score,2),'cobertura_imagem_percent':round(frac*area_original*100,1),
            'cobertura_prato_percent':round(min(1.0,frac/plate_area)*100,1),
        }
        total=suggest_grams(record['grams'],min(1.0,frac/plate_area))
        if class_id in RECIPES:
            parts=RECIPES[class_id]
            recipe=', '.join(f"{round(f*100)}% {nome}" for nome,_,f in parts)
            for nome,taco_id,f in parts:
                detected.append({**base,'nome':f"{record['nome']} — {nome}",'taco_id':taco_id,
                    'grams':max(1,round(total*f)),
                    'note':(f"Parte do {record['nome'].lower()} ({total} g no total; receita aproximada: {recipe}). "
                            'A TACO não tem este preparo pronto; ajuste ou remova cada ingrediente. '
                            'Porção estimada pela área no prato; informe o peso real.')})
            continue
        detected.append({**base,'nome':record['nome'],'taco_id':record['taco_id'],'grams':total,
            'note':record['note']+'. Porção inicial estimada pela área no prato; informe o peso real.'})
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
