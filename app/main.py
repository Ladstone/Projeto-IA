"""PratoMetria - API local de segmentação visual e cálculo nutricional."""
from __future__ import annotations
from pathlib import Path
from io import BytesIO
import base64, json, os, sqlite3, threading
from datetime import datetime, date
from typing import Optional

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import FastAPI,File,Form,UploadFile,HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .nutrition import FOODS, FOOD_CLASS_MAP, DATA, META, details, recalculate
from .vision import VisionService, CombinedVisionService, ClassOverrideService, LABEL_COLORS, CLASS_LABELS, MIN_REGION_FRACTION, food_crop_box
from .unified_classes import RECIPES, VARIANTES, NOMES_EXIBIDOS, RECEITAS_UM_ITEM, RECEITA_ID_BASE
from .metas import calcular_metas, normaliza_objetivo, OBJETIVOS, ATIVIDADES
from . import llm

ROOT=Path(__file__).resolve().parents[1];STATIC=ROOT/'app'/'static';MODEL=ROOT/'models'/'unified_21class.pt';MODEL_NOVAS=ROOT/'models'/'unified_28class_v3.pt';MODEL_PURE=ROOT/'models'/'unified_28class_v3d.pt';DB=ROOT/'app'/'data'/'profile.sqlite3'
app=FastAPI(title='PratoMetria API',version='1.0.0',description='Análise de refeições por foto: segmentação local (PyTorch) + TACO + metas, histórico e feedback')
app.mount('/static',StaticFiles(directory=STATIC),name='static')
lock=threading.Lock();model_lock=threading.Lock()
DEFAULT={'objetivo':'hipertrofia','kcal':2400,'proteina_g':150,'carboidrato_g':290,'lipideos_g':75,'refeicoes':3}

def db_init():
    DB.parent.mkdir(exist_ok=True,parents=True)
    with sqlite3.connect(DB) as con:
        con.execute('create table if not exists profile (id integer primary key check(id=1), data text not null)')
        con.execute('insert or ignore into profile (id,data) values (1,?)',(json.dumps(DEFAULT),))
        # Histórico: cada prato analisado (com foto) vira um registro; correções atualizam o mesmo registro.
        con.execute('create table if not exists meals (id integer primary key autoincrement, criado_em text not null, '
                    'dia text not null, refeicao text not null, foto blob not null, itens text not null, totais text not null)')
        con.execute('create index if not exists meals_dia on meals(dia)')
        # Feedback do mês escrito pelo LLM: guardado para não gerar de novo a cada abertura.
        con.execute('create table if not exists feedback_mes (mes text primary key, criado_em text not null, '
                    'n_pratos integer not null, modelo text not null, texto text not null)')

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
class MealItem(FoodInput):
    nome_exibido:Optional[str]=None
class MealUpdate(BaseModel):
    items:list[MealItem]
class FeedbackPrato(BaseModel):
    items:list[MealItem]

def carregar_visao(env=os.environ):
    """Monta o serviço de visão. Padrão (sem variáveis de ambiente): a combinação, com base v2 (prato feito)
    + classes novas do v3 (CombinedVisionService) + purê do v3d (ClassOverrideService); ver app/vision.py
    e docs/RESULTADOS.md. PRATOMETRIA_MODEL=arquivo.pt sobe SÓ esse modelo (ex.: o v2 puro, de backup),
    a não ser que PRATOMETRIA_MODEL_NOVAS / PRATOMETRIA_MODEL_PURE também sejam informadas."""
    base=Path(env.get('PRATOMETRIA_MODEL') or MODEL)
    if not base.exists():raise FileNotFoundError('Arquivo do modelo não encontrado em models/.')
    padrao=not env.get('PRATOMETRIA_MODEL')
    novas=env.get('PRATOMETRIA_MODEL_NOVAS') or (str(MODEL_NOVAS) if padrao and MODEL_NOVAS.exists() else '')
    pure=env.get('PRATOMETRIA_MODEL_PURE') or (str(MODEL_PURE) if padrao and MODEL_PURE.exists() else '')
    service=VisionService(base)
    if novas:service=CombinedVisionService(service,VisionService(Path(novas)))
    if pure:service=ClassOverrideService(service,VisionService(Path(pure)))
    return service

vision=None
model_error=None
try:
    import torch
    torch.set_num_threads(3)
    vision=carregar_visao()
except FileNotFoundError as exc:
    model_error=str(exc)
except Exception as exc:
    model_error=f'Não foi possível carregar pesos: {exc}'
db_init()

@app.get('/')
def index():return FileResponse(STATIC/'index.html')

@app.get('/api/health')
def health():
    return {'ok':True,'model_ready':vision is not None,'model_error':model_error,'taco_records':len(DATA),'image_classes':(vision.n_class if vision else len(FOOD_CLASS_MAP)),'training_sources':training_sources(),'taco_fields':len(META),'mode':'local-only','model':model_description(),'model_label':(vision.metadata.get('label') if vision else None),'llm':llm.configurado(),'nutrition':'TACO 4ª edição — 2011'}

def training_sources():
    """Datasets do modelo ativo, lidos do .json (modelos antigos não registram e usaram só os dois primeiros)."""
    default=['MyFood Dataset','Brazilian Food Images']
    return (vision.metadata.get('datasets') if vision else None) or default

def model_description():
    """Descrição lida do .json do modelo ativo, para não ficar desatualizada ao trocar de modelo."""
    if vision is None:
        return None
    return vision.metadata.get('model') or f'{vision.architecture}, {vision.n_class} classes'

@app.get('/api/model/info')
def model_info():
    from .unified_classes import CLASS_LABELS
    return {'model':model_description(),
            'label':vision.metadata.get('label') if vision else None,
            'datasets':training_sources(),
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

# ---- Histórico de pratos -------------------------------------------------------------
THUMB_SIZE=256

def miniatura(image:Image.Image)->bytes:
    """Quadrado central reduzido (JPEG): é o que aparece nas bolinhas do calendário."""
    thumb=ImageOps.fit(image,(THUMB_SIZE,THUMB_SIZE),Image.Resampling.LANCZOS)
    buffer=BytesIO();thumb.save(buffer,format='JPEG',quality=80)
    return buffer.getvalue()

def meal_payload(items:list[dict])->tuple[str,str]:
    """Itens (nome para mostrar, gramas, id TACO) e totais da refeição, em JSON."""
    itens=[{'nome':it.get('nome_exibido') or it.get('nome') or '','grams':round(float(it['grams']),1),
            'taco_id':int(it['taco_id']),'class_id':it.get('class_id')} for it in items]
    total=recalculate(items,profile_get())['total'] if items else {}
    totais={k:round(float(total.get(k) or 0),1) for k in ('kcal','proteina_g','carboidrato_g','lipideos_g')}
    return json.dumps(itens,ensure_ascii=False),json.dumps(totais)

def meal_save(image:Image.Image,items:list[dict],meal_id:Optional[int]=None)->Optional[int]:
    """Cria o registro do prato (ou substitui o do mesmo prato, quando a pessoa refaz o recorte).

    Grava mesmo sem itens: a pessoa pode adicionar os alimentos à mão depois. Registro vazio não aparece no histórico."""
    itens,totais=meal_payload(items)
    foto=miniatura(image)
    with lock,sqlite3.connect(DB) as con:
        if meal_id and con.execute('select 1 from meals where id=?',(meal_id,)).fetchone():
            con.execute('update meals set foto=?,itens=?,totais=? where id=?',(foto,itens,totais,meal_id))
            return meal_id
        agora=datetime.now()
        cur=con.execute('insert into meals (criado_em,dia,refeicao,foto,itens,totais) values (?,?,?,?,?,?)',
                        (agora.isoformat(timespec='seconds'),agora.date().isoformat(),'',foto,itens,totais))  # refeicao: não usada
        return cur.lastrowid

@app.get('/api/meals')
def list_meals(mes:Optional[str]=None):
    """Pratos do mês (AAAA-MM; padrão: mês atual), com o resumo do calendário."""
    hoje=date.today();mes=mes or hoje.strftime('%Y-%m')
    try:datetime.strptime(mes,'%Y-%m')
    except ValueError:raise HTTPException(422,'mes deve ser AAAA-MM')
    with lock,sqlite3.connect(DB) as con:
        rows=con.execute("select id,criado_em,dia,itens,totais from meals where dia like ? and itens!='[]' order by criado_em",(mes+'-%',)).fetchall()
    meals=[{'id':i,'criado_em':c,'dia':d,'hora':c[11:16],'itens':json.loads(it),'totais':json.loads(t),
            'foto':f'/api/meals/{i}/foto'} for i,c,d,it,t in rows]
    return {'mes':mes,'hoje':hoje.isoformat(),'meta_kcal_dia':profile_get()['kcal'],'meals':meals,
            'resumo':{'dias':len({m['dia'] for m in meals}),'pratos':len(meals)}}

@app.get('/api/meals/{meal_id}/foto')
def meal_photo(meal_id:int):
    with lock,sqlite3.connect(DB) as con:row=con.execute('select foto from meals where id=?',(meal_id,)).fetchone()
    if not row:raise HTTPException(404,'Prato não encontrado')
    return Response(row[0],media_type='image/jpeg',headers={'Cache-Control':'no-cache'})

@app.put('/api/meals/{meal_id}')
def update_meal(meal_id:int,body:MealUpdate):
    """Correções do usuário (alimento ou gramas) atualizam o mesmo registro. Sem itens ele some do histórico,
    mas continua guardado: se a pessoa adicionar alimentos de novo, o prato volta com a mesma foto."""
    items=[b.model_dump() for b in body.items]
    try:itens,totais=meal_payload(items)
    except ValueError as e:raise HTTPException(422,str(e))
    with lock,sqlite3.connect(DB) as con:
        if not con.execute('select 1 from meals where id=?',(meal_id,)).fetchone():raise HTTPException(404,'Prato não encontrado')
        con.execute('update meals set itens=?,totais=? where id=?',(itens,totais,meal_id))
    return {'ok':True}

@app.delete('/api/meals/{meal_id}')
def delete_meal(meal_id:int):
    with lock,sqlite3.connect(DB) as con:n=con.execute('delete from meals where id=?',(meal_id,)).rowcount
    if not n:raise HTTPException(404,'Prato não encontrado')
    return {'ok':True}

# ---- Feedback escrito por LLM ------------------------------------------------------------
# O app calcula tudo (TACO, metas, médias); o LLM só transforma os números em texto. Se ele não responder,
# a resposta vem com 'texto': None e a página mostra as dicas por regra.
VERDURAS_LEGUMES={'Salada','Tomate','Cenoura','Abóbora','Beterraba','Brócolis','Cebola','Milho'}

def _objetivo(profile):
    return OBJETIVOS.get(profile.get('objetivo'),(profile.get('objetivo'),))[0]

@app.post('/api/feedback/prato')
def feedback_prato(body:FeedbackPrato):
    items=[b.model_dump() for b in body.items]
    if not items:raise HTTPException(422,'Prato sem alimentos')
    profile=profile_get()
    try:n=recalculate(items,profile)
    except ValueError as e:raise HTTPException(422,str(e))
    dados={'objetivo':_objetivo(profile),'refeicoes_por_dia':profile['refeicoes'],
           'alimentos':[{'nome':it.get('nome_exibido') or it['nome'],'gramas':round(it['grams'])} for it in items],
           'totais_do_prato':n['total'],'meta_desta_refeicao':n['referencia_refeicao'],
           'percentual_da_meta':{k:v['percentual'] for k,v in n['comparacao'].items()},
           'dicas_calculadas_pelo_app':[r['titulo']+': '+r['descricao'] for r in n['recomendacoes']]}
    try:texto,modelo=llm.feedback_prato(dados)
    except llm.LLMIndisponivel as e:return {'texto':None,'motivo':str(e)}
    return {'texto':texto,'modelo':modelo}

def estatisticas_mes(meals:list[dict],profile:dict)->dict:
    """Números do mês para o feedback (tudo calculado aqui, nada pelo LLM)."""
    dias=sorted({m['dia'] for m in meals});nd=max(1,len(dias))
    soma={k:sum(m['totais'].get(k,0) for m in meals) for k in ('kcal','proteina_g','carboidrato_g','lipideos_g','fibra_g')}
    meta_ref=profile['kcal']/max(1,profile['refeicoes'])
    freq={}
    for m in meals:
        for it in {i['nome'] for i in m['itens']}:freq[it]=freq.get(it,0)+1
    com_verdura=sum(any(i['nome'] in VERDURAS_LEGUMES or (i.get('class_id') and FOOD_CLASS_MAP.get(i['class_id'],{}).get('nome') in VERDURAS_LEGUMES)
                        for i in m['itens']) for m in meals)
    return {
        'objetivo':_objetivo(profile),
        'metas_diarias':{'kcal':profile['kcal'],'proteina_g':profile['proteina_g'],'carboidrato_g':profile['carboidrato_g'],'lipideos_g':profile['lipideos_g']},
        'refeicoes_planejadas_por_dia':profile['refeicoes'],
        'dias_com_registro':len(dias),'pratos_registrados':len(meals),
        'pratos_por_dia_registrado':round(len(meals)/nd,1),
        'media_por_dia_registrado':{k:round(v/nd,1) for k,v in soma.items()},
        'media_por_prato':{k:round(v/max(1,len(meals)),1) for k,v in soma.items()},
        'meta_kcal_por_refeicao':round(meta_ref),
        'pratos_acima_de_130_por_cento_da_meta_da_refeicao':sum(m['totais'].get('kcal',0)>1.3*meta_ref for m in meals),
        'pratos_abaixo_de_70_por_cento_da_meta_da_refeicao':sum(m['totais'].get('kcal',0)<0.7*meta_ref for m in meals),
        'pratos_com_verdura_ou_legume':com_verdura,
        'alimentos_mais_frequentes':[{'nome':k,'pratos':v} for k,v in sorted(freq.items(),key=lambda x:(-x[1],x[0]))[:8]],
        'alimentos_diferentes':len(freq),
    }

@app.get('/api/feedback/mes')
def get_feedback_mes(mes:str):
    with lock,sqlite3.connect(DB) as con:
        row=con.execute('select criado_em,n_pratos,modelo,texto from feedback_mes where mes=?',(mes,)).fetchone()
    if not row:return {'mes':mes,'texto':None}
    return {'mes':mes,'criado_em':row[0],'n_pratos':row[1],'modelo':row[2],'texto':json.loads(row[3])}

@app.post('/api/feedback/mes')
def post_feedback_mes(mes:str):
    meals=list_meals(mes)['meals']
    if not meals:raise HTTPException(422,'Nenhum prato registrado neste mês')
    dados={'mes':mes,**estatisticas_mes(meals,profile_get())}
    try:texto,modelo=llm.feedback_mes(dados)
    except llm.LLMIndisponivel as e:raise HTTPException(503,'Não foi possível gerar o feedback agora. Tente de novo em alguns minutos.') from e
    agora=datetime.now().isoformat(timespec='seconds')
    with lock,sqlite3.connect(DB) as con:
        con.execute('insert or replace into feedback_mes (mes,criado_em,n_pratos,modelo,texto) values (?,?,?,?,?)',
                    (mes,agora,len(meals),modelo,json.dumps(texto,ensure_ascii=False)))
    return {'mes':mes,'criado_em':agora,'n_pratos':len(meals),'modelo':modelo,'texto':texto}

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

DARK_LUMINANCE=60   # pixel com brilho médio (0-255) abaixo disto conta como escuro
DARK_SHARE=0.5      # fração de pixels escuros a partir da qual a variante é a "escura" (ex.: feijão preto)

def dark_fraction(image:Image.Image,mask:np.ndarray,class_id:int)->float:
    """Fração dos pixels da classe que são escuros, na foto reduzida ao tamanho da máscara."""
    small=np.asarray(image.resize((mask.shape[1],mask.shape[0]),Image.Resampling.BILINEAR),dtype=np.float32)
    px=small[mask==class_id]
    return float((px.mean(axis=1)<DARK_LUMINANCE).mean()) if len(px) else 0.0

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
async def analyze(file:UploadFile=File(...),area_original:float=Form(1.0),salvar:bool=Form(False),meal_id:Optional[int]=Form(None)):
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

    ``salvar``: o app manda 1 e o prato entra no histórico (``meal_id`` substitui o registro do mesmo
    prato quando a pessoa refaz o recorte). Sem o parâmetro (testes, scripts) nada é gravado.
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
        if class_id>probs.shape[0]:continue  # classe que o modelo ativo não conhece
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
        if class_id in RECIPES and class_id not in RECEITAS_UM_ITEM:
            parts=RECIPES[class_id]
            recipe=', '.join(f"{round(f*100)}% {nome}" for nome,_,f in parts)
            for nome,taco_id,f in parts:
                detected.append({**base,'nome':f"{record['nome']} — {nome}",'taco_id':taco_id,
                    'grams':max(1,round(total*f)),
                    'note':(f"Parte do {record['nome'].lower()} ({total} g no total; receita aproximada: {recipe}). "
                            'A TACO não tem este preparo pronto; ajuste ou remova cada ingrediente. '
                            'Porção estimada pela área no prato; informe o peso real.')})
            continue
        taco_id=RECEITA_ID_BASE+class_id if class_id in RECEITAS_UM_ITEM else record['taco_id']
        nome_curto=RECEITAS_UM_ITEM.get(class_id) or NOMES_EXIBIDOS.get(class_id)
        extra={'nome_exibido':nome_curto} if nome_curto else {}
        if class_id in VARIANTES:
            v=VARIANTES[class_id]
            taco_id=v['escuro'] if dark_fraction(image,mask,class_id)>=DARK_SHARE else v['claro']
            extra={'nome_exibido':v['nome_exibido']}
        note='Porção inicial estimada pela área no prato; informe o peso real.'
        detected.append({**base,**extra,'nome':record['nome'],'taco_id':taco_id,'grams':total,
            'note':f"{record['note']}. {note}" if record['note'] else note})
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
    saved=meal_save(image,detected,meal_id) if salvar else None
    return {
        'detected':detected,'unmatched':[],'preview_overlay':overlay,'meal_id':saved,
        'nutrition':recalculate(detected,profile_get()),
        'warning':warning,'mode':'local','analysis_quality':'experimental',
        'vision_debug':diagnostics,
    }
