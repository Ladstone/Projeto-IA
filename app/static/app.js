const $ = (id)=>document.getElementById(id);
const STATE={file:null,crop:null,foods:[],taco:[],tacoMap:new Map(),classes:{},segmentation:false,catalogPage:0,lastNutrition:null,unmatched:[],mealId:null};
function showStatus(id,message,error=false){const el=$(id);el.textContent=message;el.classList.toggle('error',error);}
const FIELD_LABELS={kcal:'Meta energética',proteina_g:'Proteínas',carboidrato_g:'Carboidratos',lipideos_g:'Gorduras',refeicoes:'Refeições por dia',objetivo:'Objetivo',sexo:'Sexo',idade:'Idade',peso_kg:'Peso',altura_cm:'Altura',atividade:'Nível de atividade',grams:'Gramas',taco_id:'Alimento TACO'};
// Erros 422 do FastAPI chegam como lista [{loc,msg}]: transforma em texto legível.
function formatApiError(detail){
 if(typeof detail==='string')return detail;
 if(Array.isArray(detail))return detail.map(d=>{const field=(d.loc||[]).filter(p=>typeof p==='string'&&p!=='body').pop();return `${FIELD_LABELS[field]||field||'Valor'}: ${d.msg}`;}).join('; ');
 return 'Erro inesperado no servidor.';
}
async function api(url,opts={}){let res=await fetch(url,opts);let body;try{body=await res.json();}catch{throw new Error(`Resposta inválida (${res.status})`)}if(!res.ok)throw new Error(formatApiError(body.detail));return body;}
const numeric=(v)=>Number(v||0);
const optNum=(v)=>v===''||v==null?null:Number(v);
const bodyForm=()=>({sexo:$('sex').value||null,idade:optNum($('age').value),peso_kg:optNum($('weight').value),altura_cm:optNum($('height').value),atividade:$('activity').value||null});
function getProfileForm(){return {...bodyForm(),objetivo:$('goal').value,kcal:numeric($('kcal').value),proteina_g:numeric($('protein').value),carboidrato_g:numeric($('carbs').value),lipideos_g:numeric($('fats').value),refeicoes:numeric($('meals').value)};}
function fillProfile(p){$('goal').value=p.objetivo;$('sex').value=p.sexo||'';$('age').value=p.idade??'';$('weight').value=p.peso_kg??'';$('height').value=p.altura_cm??'';$('activity').value=p.atividade||'';$('kcal').value=p.kcal;$('protein').value=p.proteina_g;$('carbs').value=p.carboidrato_g;$('fats').value=p.lipideos_g;$('meals').value=p.refeicoes;}
async function saveProfile(e){if(e)e.preventDefault();try{await api('/api/profile',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(getProfileForm())});showStatus('profileStatus','✓ Metas salvas e usadas no cálculo.');if(STATE.foods.length)await recalc();setTimeout(()=>{$('goalsDialog').close();showStatus('profileStatus','');},700);}catch(e){showStatus('profileStatus','Não foi possível salvar: '+e.message,true);}}
// Cores das classes vêm do CSS (--class-N), as mesmas do overlay gerado no backend.
const classColor=(id)=>getComputedStyle(document.documentElement).getPropertyValue(`--class-${id}`).trim()||'#9aa59f';
function selectFile(file){if(!file)return;if(!file.type.startsWith('image/')){showStatus('analysisStatus','Escolha um arquivo de imagem.',true);return}if(file.size>8*1024*1024){showStatus('analysisStatus','O arquivo excede 8 MB.',true);return}STATE.file=file;STATE.crop=null;STATE.mealId=null;STATE.foods=[];STATE.unmatched=[];renderUnmatched();renderFoods();renderEmptyStats();const url=URL.createObjectURL(file);if($('photo').dataset.objectUrl)URL.revokeObjectURL($('photo').dataset.objectUrl);$('photo').dataset.objectUrl=url;$('photo').src=url;$('dropArea').classList.add('has-photo');$('dropArea').classList.remove('show-overlay');$('overlay').removeAttribute('src');$('fileName').textContent=file.name;$('photoStatus').textContent='Foto pronta para análise';$('analyzeBtn').disabled=false;$('cropBtn').disabled=false;showStatus('analysisStatus','Ajuste o recorte e confirme para analisar.');openCrop();}
function normalize(s){return (s||'').normalize('NFD').replace(/[\u0300-\u036f]/g,'').toLowerCase().trim();}
// Nome exato primeiro; senão, entre os que contêm todas as palavras, prefere o que
// começa pela busca e depois o nome mais curto (ex.: "arroz" -> "Arroz, tipo 1, cozido").
function foodLookup(text){
 const query=normalize(text);if(!query)return undefined;
 const exact=STATE.taco.find(f=>normalize(f.nome)===query);if(exact)return exact;
 const terms=query.split(/[\s,]+/).filter(Boolean);
 const found=STATE.taco.filter(f=>terms.every(t=>normalize(f.nome).includes(t)));
 const rank=f=>[normalize(f.nome).startsWith(terms[0])?0:1,f.nome.length];
 return found.sort((a,b)=>{const ra=rank(a),rb=rank(b);return ra[0]-rb[0]||ra[1]-rb[1];})[0];
}
function renderFoods(){
 const list=$('foodList');list.textContent='';$('foodCount').textContent=`${STATE.foods.length} alimento(s)`;$('emptyMessage').style.display=STATE.foods.length?'none':'block';
 STATE.foods.forEach((it,idx)=>{
  const outer=document.createElement('div');outer.className='food-row';
  const top=document.createElement('div');top.className='food-top';const color=document.createElement('div');color.className='food-color';color.style.background=it.class_id?classColor(it.class_id):'var(--line)';top.append(color);
  const info=document.createElement('div');info.style.minWidth='0';info.style.flex='1';const name=document.createElement('div');name.className='food-name';name.textContent=it.nome_exibido||STATE.tacoMap.get(Number(it.taco_id))?.nome||it.nome;
  const sub=document.createElement('div');sub.className='food-sub';const badge=document.createElement('span');
  if(it.class_id){badge.className='badge badge-ai';badge.textContent=`⚠ Sugestão da IA · ${it.cobertura_prato_percent}% do prato`;}else{badge.className='badge badge-manual';badge.textContent='✓ Escolhido por você';}
  sub.append(badge);info.append(name,sub);top.append(info);outer.append(top);
  const controls=document.createElement('div');controls.className='food-controls';
  const gramsWrap=document.createElement('label');gramsWrap.className='grams';const gramsUnit=document.createElement('span');gramsUnit.textContent='g';
  const grams=document.createElement('input');grams.type='number';grams.inputMode='numeric';grams.value=it.grams;grams.min='1';grams.max='2500';grams.step='5';grams.title='Quantidade em gramas';grams.setAttribute('aria-label',`Gramas de ${it.nome}`);grams.addEventListener('change',()=>{STATE.foods[idx].grams=Math.max(1,Math.min(2500,Number(grams.value)||1));recalc();});
  gramsWrap.append(grams,gramsUnit);
  const selector=document.createElement('input');selector.type='search';selector.setAttribute('list','tacoOptions');selector.placeholder='Trocar por outro alimento da TACO';selector.setAttribute('aria-label',`Corrigir alimento pela TACO: ${it.nome}`);selector.title='Digite qualquer nome da TACO para corrigir a sugestão da IA';selector.value=it.nome_exibido||STATE.tacoMap.get(Number(it.taco_id))?.nome||it.nome;
  selector.addEventListener('change',()=>{const entry=foodLookup(selector.value);if(!entry){showStatus('analysisStatus','Não encontrei este alimento na TACO. Abra a tabela e pesquise.',true);return;}
    STATE.foods[idx].taco_id=entry.id;STATE.foods[idx].nome=entry.nome;STATE.foods[idx].nome_exibido=null;STATE.foods[idx].class_id=null;STATE.foods[idx].origem='manual';STATE.foods[idx].score_nao_calibrado=null;STATE.foods[idx].note='Alimento corrigido manualmente; confira o peso.';renderFoods();recalc();showStatus('analysisStatus',`Corrigido para ${entry.nome}.`);});
  const remove=document.createElement('button');remove.type='button';remove.className='remove-btn';remove.textContent='×';remove.title='Remover alimento';remove.setAttribute('aria-label',`Remover ${it.nome}`);remove.addEventListener('click',()=>{STATE.foods.splice(idx,1);renderFoods();recalc()});
  controls.append(gramsWrap,selector,remove);outer.append(controls);
  const note=document.createElement('div');note.className='food-note';note.textContent=it.note||'Confirme a massa e o preparo. A fotografia não mede gramas.';outer.append(note);
  const detailsBtn=document.createElement('button');detailsBtn.type='button';detailsBtn.className='link-btn';detailsBtn.textContent=Number(it.taco_id)>=9000?'Ver nutrientes · receita com itens da TACO':`Ver nutrientes · TACO #${it.taco_id}`;detailsBtn.addEventListener('click',()=>showFoodDetails(it.taco_id));outer.append(detailsBtn);list.append(outer);
 });
}
function renderEmptyStats(){cancelFeedback();[['Kcal','Kcal'],['Protein','Protein'],['Carbs','Carbs'],['Fats','Fats']].forEach(([stat,bar])=>{$('stat'+stat).textContent='—';$('bar'+bar).style.width='0%';$('bar'+bar).classList.remove('over');$('target'+stat).textContent='Referência da refeição: —'});$('mobileKcal').textContent='—';$('suggestions').replaceChildren();const d=document.createElement('div');d.className='reco';d.innerHTML='<div class="tiny">Próximo passo</div><h3>Envie uma foto</h3><p>Depois da análise, as sugestões comparam a refeição com as suas metas.</p>';$('suggestions').append(d);}
function renderStats(n){const stat=[['Kcal','kcal','kcal'],['Protein','proteina_g','g'],['Carbs','carboidrato_g','g'],['Fats','lipideos_g','g']];stat.forEach(([key,field,unit])=>{const cmp=n.comparacao[field];$('stat'+key).textContent=cmp.obtido.toLocaleString('pt-BR',{maximumFractionDigits:1});$('bar'+key).style.width=Math.min(100,Math.max(0,cmp.percentual||0))+'%';$('bar'+key).classList.toggle('over',(cmp.percentual||0)>135);$('target'+key).textContent=`Referência: ${cmp.alvo_refeicao.toLocaleString('pt-BR')} ${unit} (${cmp.percentual||0}%)`;});$('mobileKcal').textContent=n.comparacao.kcal.obtido.toLocaleString('pt-BR',{maximumFractionDigits:0});const parent=$('suggestions');parent.replaceChildren();n.recomendacoes.forEach(rec=>{const box=document.createElement('div');box.className='reco';const small=document.createElement('div');small.className='tiny';small.textContent=rec.tipo;const title=document.createElement('h3');title.textContent=rec.titulo;const p=document.createElement('p');p.textContent=rec.descricao;box.append(small,title,p);parent.append(box)});}
async function recalc(){try{const data=await api('/api/recalculate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({items:STATE.foods})});STATE.lastNutrition=data;renderStats(data);renderFullNutrition(data);syncMeal();requestFeedback(1500);}catch(err){showStatus('analysisStatus','Erro no cálculo: '+err.message,true)}}
// Correções (alimento, gramas, remover/adicionar) atualizam o mesmo registro do histórico.
async function syncMeal(){if(!STATE.mealId)return;try{await api(`/api/meals/${STATE.mealId}`,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({items:STATE.foods})});}catch(e){if(/não encontrado/i.test(e.message))STATE.mealId=null;}}
function renderUnmatched(){
 const el=$('unmatchedList');el.replaceChildren();
 if(!STATE.unmatched.length){el.classList.add('ai-hidden');return;}
 el.classList.remove('ai-hidden');const title=document.createElement('strong');
 title.textContent='A IA mencionou outros itens, mas não conseguiu associá-los com segurança à TACO:';el.append(title);
 STATE.unmatched.forEach(x=>{const d=document.createElement('div');d.className='unmatched-item';d.textContent=`${x.nome}${x.preparo?' ('+x.preparo+')':''} — ${x.motivo}`;el.append(d)});
}
function renderVisionDiagnostics(items){
 const holder=$('visionDiagnostics');if(!holder)return;holder.replaceChildren();
 if(!items||!items.length){holder.textContent='Sem dados de diagnóstico.';return;}
 const lines=items.filter(d=>d.regiao_percent>0.05).map(d=>
   `${d.classe_visual}: região ${d.regiao_percent.toFixed(1)}%, score não calibrado ${d.score_medio_regiao.toFixed(2)} (limiar ${d.limiar})`);
 holder.textContent=lines.length?lines.join('\n'):'Nenhuma região sugerida. A base de pesos está carregada, mas esta foto não produziu regiões válidas.';
}
async function analyze(){
 if(!STATE.file)return;
 const btn=$('analyzeBtn');btn.disabled=true;btn.textContent='Analisando imagem...';
 showStatus('analysisStatus','A rede local está segmentando a imagem. Os resultados podem conter erros.');
 const fd=new FormData();fd.append('file',STATE.crop?STATE.crop.file:STATE.file);if(STATE.crop)fd.append('area_original',String(STATE.crop.areaOriginal));
 fd.append('salvar','1');if(STATE.mealId)fd.append('meal_id',String(STATE.mealId));
 try{
   const res=await api('/api/analyze',{method:'POST',body:fd});
   STATE.foods=res.detected||[];STATE.unmatched=res.unmatched||[];STATE.mealId=res.meal_id??null;
   renderUnmatched();renderFoods();renderVisionDiagnostics(res.vision_debug);
   STATE.lastNutrition=res.nutrition;
   if(STATE.foods.length){renderStats(res.nutrition);renderFullNutrition(res.nutrition);requestFeedback(0);}
   else{renderEmptyStats();$('detailNutrients').replaceChildren();}
   if(res.preview_overlay){$('overlay').src=res.preview_overlay;$('dropArea').classList.toggle('show-overlay',$('overlaySwitch').checked);}
   $('photoStatus').textContent=`${STATE.foods.length} hipótese(s) visual(is)`;
   showStatus('analysisStatus',res.warning||'Análise concluída.');
 }catch(e){showStatus('analysisStatus','Não foi possível analisar: '+e.message,true)}
 finally{btn.disabled=false;btn.textContent='Analisar refeição'}
}
function addById(id){const f=STATE.tacoMap.get(Number(id));if(!f)return;STATE.foods.push({nome:f.nome,taco_id:f.id,grams:100,class_id:null,origem:'manual',note:'Selecionado manualmente na TACO. Confira a porção.'});renderFoods();recalc();showStatus('analysisStatus',`${f.nome} adicionado à refeição.`);if($('catalogDialog').open)showStatus('catalogStatus',`✓ ${f.nome} adicionado (100 g). Ajuste os gramas na lista.`);}
function addManual(){const entry=foodLookup($('foodSearch').value);if(!entry)return showStatus('analysisStatus','Não encontrei esse alimento. Use "Procurar na Tabela TACO completa".',true);addById(entry.id);$('foodSearch').value='';}
function openDialog(id,focusId){const d=$(id);if(!d.open)d.showModal();if(focusId)$(focusId).focus();}
function renderCatalog(){
 const query=normalize($('catalogSearch').value), terms=query.split(/\s+/).filter(Boolean);
 const found=STATE.taco.filter(f=>terms.every(t=>normalize(f.nome).includes(t)));
 const maxPage=Math.max(0,Math.ceil(found.length/12)-1);STATE.catalogPage=Math.min(STATE.catalogPage,maxPage);
 $('catalogCount').textContent=`${found.length} de ${STATE.taco.length} alimentos · página ${STATE.catalogPage+1}/${maxPage+1}`;
 const list=$('catalogList');list.replaceChildren();
 found.slice(STATE.catalogPage*12,STATE.catalogPage*12+12).forEach(f=>{
  const div=document.createElement('div');div.className='catalog-row';
  const text=document.createElement('div');text.className='catalog-info';const strong=document.createElement('strong');strong.textContent=f.nome;
  const small=document.createElement('small');small.textContent=`TACO #${f.id} · porção de referência: 100 g`;text.append(strong,small);
  const read=document.createElement('button');read.type='button';read.className='catalog-action';read.textContent='Nutrientes';read.addEventListener('click',()=>showFoodDetails(f.id));
  const add=document.createElement('button');add.type='button';add.className='catalog-add';add.textContent='+ Prato';add.addEventListener('click',()=>addById(f.id));
  div.append(text,read,add);list.append(div);
 });
 $('catalogPrev').disabled=STATE.catalogPage<=0;$('catalogNext').disabled=STATE.catalogPage>=maxPage;
}
async function showFoodDetails(id){
 const modal=$('tacoModal');$('tacoTitle').textContent='Carregando nutrientes...';$('tacoNutrients').replaceChildren();modal.showModal();
 try{
  const food=await api(`/api/taco/${id}`);$('tacoTitle').textContent=`#${food.id} · ${food.nome}`;
  $('tacoOrigin').textContent=`Fonte: ${food.fonte} · página ${food.pagina_pdf} do arquivo PDF · valores por 100 g`;
  food.nutrientes.forEach(n=>{const row=document.createElement('div');row.className='nutrient-line';const name=document.createElement('span');name.textContent=n.nome;const v=document.createElement('strong');v.textContent=n.valor===null?(n.observacao||'Não informado'):`${n.valor.toLocaleString('pt-BR',{maximumFractionDigits:3})} ${n.unidade}`;if(n.valor===null)v.className='nutrient-missing';row.append(name,v);$('tacoNutrients').append(row)});
 }catch(e){$('tacoTitle').textContent='Não foi possível consultar: '+e.message;}
}
function renderFullNutrition(data){const parent=$('detailNutrients');parent.replaceChildren();if(!data||!data.detalhamento){return};data.detalhamento.forEach(n=>{
 const row=document.createElement('div');row.className='nutrient-line';const name=document.createElement('span');name.textContent=n.nome;const num=document.createElement('strong');num.textContent=n.valor!==null?`${n.valor.toLocaleString('pt-BR',{maximumFractionDigits:2})} ${n.unidade}`:n.dados_ausentes_em.length?'Não calculável (dado ausente)':'—';if(n.valor===null)num.className='nutrient-missing';row.append(name,num);parent.append(row);
 });}
async function setup(){const [health,profile,foods,options]=await Promise.all([api('/api/health'),api('/api/profile'),api('/api/foods'),api('/api/profile/opcoes')]);fillSelect('goal',options.objetivos);fillSelect('activity',options.atividades,true);$('baseCount').textContent=`${health.taco_records} alimentos · ${health.taco_fields} nutrientes`;$('modelSummary').textContent=`Rede neural local com ${health.image_classes} categorias visuais, treinada com ${health.training_sources.join(', ')}`;fillProfile(profile);STATE.taco=foods.foods;STATE.tacoMap=new Map(STATE.taco.map(f=>[f.id,f]));STATE.classes=foods.class_map;const fragment=document.createDocumentFragment();STATE.taco.forEach(f=>{const op=document.createElement('option');op.value=f.nome;fragment.append(op)});$('tacoOptions').append(fragment);renderCatalog();}
$('profileForm').addEventListener('submit',saveProfile);$('file').addEventListener('change',e=>selectFile(e.target.files[0]));$('dropArea').addEventListener('click',()=>{$('file').click()});$('dropArea').addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();$('file').click()}});
['dragenter','dragover'].forEach(event=>$('dropArea').addEventListener(event,e=>{e.preventDefault();$('dropArea').classList.add('dragover')}));['dragleave','drop'].forEach(event=>$('dropArea').addEventListener(event,e=>{e.preventDefault();$('dropArea').classList.remove('dragover')}));$('dropArea').addEventListener('drop',e=>selectFile(e.dataTransfer.files[0]));
$('changeBtn').addEventListener('click',()=>$('file').click());$('analyzeBtn').addEventListener('click',analyze);$('overlaySwitch').addEventListener('change',e=>$('dropArea').classList.toggle('show-overlay',e.target.checked&&Boolean($('overlay').src)));
$('addFoodBtn').addEventListener('click',addManual);$('catalogSearch').addEventListener('input',()=>{STATE.catalogPage=0;renderCatalog()});$('catalogPrev').addEventListener('click',()=>{STATE.catalogPage--;renderCatalog()});$('catalogNext').addEventListener('click',()=>{STATE.catalogPage++;renderCatalog()});$('modalClose').addEventListener('click',()=>$('tacoModal').close());$('foodSearch').addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();addManual();}});
// Diálogos: metas (modal) e Tabela TACO (gaveta). Fecham pelo botão, por Esc ou clicando fora.
$('openGoals').addEventListener('click',()=>openDialog('goalsDialog','kcal'));$('editGoals').addEventListener('click',()=>openDialog('goalsDialog','kcal'));
$('openTaco').addEventListener('click',()=>openDialog('catalogDialog','catalogSearch'));$('openTacoInline').addEventListener('click',()=>openDialog('catalogDialog','catalogSearch'));
document.querySelectorAll('dialog').forEach(d=>{d.addEventListener('click',e=>{if(e.target===d)d.close();});d.querySelectorAll('[data-close]').forEach(b=>b.addEventListener('click',()=>d.close()));});
$('catalogDialog').addEventListener('close',()=>showStatus('catalogStatus',''));
renderEmptyStats();setup().catch(e=>showStatus('analysisStatus','Falha ao iniciar: '+e.message,true));

// Recorte manual: o usuário marca o prato; só o recorte vai para a IA.
// area_original (fração da foto que o recorte ocupa) mantém as gramas calculadas sobre a foto inteira.
const CROP={img:null,url:null,scale:1,rect:null,drag:null};
function showPreview(src,name,status){STATE.foods=[];STATE.unmatched=[];renderUnmatched();renderFoods();renderEmptyStats();$('photo').src=src;$('dropArea').classList.remove('show-overlay');$('overlay').removeAttribute('src');$('fileName').textContent=name;$('photoStatus').textContent=status;}
function openCrop(){
 if(!STATE.file)return;if(CROP.url)URL.revokeObjectURL(CROP.url);CROP.url=URL.createObjectURL(STATE.file);
 const img=new Image();img.onload=()=>{CROP.img=img;const cv=$('cropCanvas');CROP.scale=Math.min(1,1000/Math.max(img.naturalWidth,img.naturalHeight));
  cv.width=Math.round(img.naturalWidth*CROP.scale);cv.height=Math.round(img.naturalHeight*CROP.scale);CROP.rect=STATE.crop?{...STATE.crop.rect}:null;drawCrop();openDialog('cropDialog');
  if(CROP.rect)return showStatus('cropStatus','Recorte atual. Arraste por dentro para mover, pelos cantos para redimensionar, ou fora para marcar outro.');
  showStatus('cropStatus','Sugerindo um recorte em volta da comida…');suggestCrop();};
 img.src=CROP.url;
}
async function suggestCrop(){
 const W=CROP.img.naturalWidth,H=CROP.img.naturalHeight,fd=new FormData();fd.append('file',STATE.file);
 try{const res=await api('/api/crop-suggestion',{method:'POST',body:fd});if(CROP.rect||!$('cropDialog').open)return;
  if(!res.box)return showStatus('cropStatus','A IA não achou comida para sugerir um recorte. Arraste sobre a foto para marcar o prato.');
  const b=res.box;CROP.rect={x:b.x0*W,y:b.y0*H,w:(b.x1-b.x0)*W,h:(b.y1-b.y0)*H};drawCrop();
  showStatus('cropStatus','Recorte sugerido em volta da comida. Ajuste se precisar: arraste por dentro para mover e pelos cantos para redimensionar.');
 }catch(e){showStatus('cropStatus','Não foi possível sugerir um recorte ('+e.message+'). Arraste sobre a foto para marcar o prato.',true);}
}
const cropCorners=r=>[[r.x,r.y],[r.x+r.w,r.y],[r.x,r.y+r.h],[r.x+r.w,r.y+r.h]];
function drawCrop(){
 const cv=$('cropCanvas'),ctx=cv.getContext('2d'),s=CROP.scale;ctx.drawImage(CROP.img,0,0,cv.width,cv.height);
 if(!CROP.rect)return;const r={x:CROP.rect.x*s,y:CROP.rect.y*s,w:CROP.rect.w*s,h:CROP.rect.h*s};
 ctx.save();ctx.fillStyle='rgba(0,0,0,.55)';ctx.beginPath();ctx.rect(0,0,cv.width,cv.height);ctx.rect(r.x,r.y,r.w,r.h);ctx.fill('evenodd');
 ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.setLineDash([6,4]);ctx.strokeRect(r.x,r.y,r.w,r.h);ctx.setLineDash([]);
 ctx.fillStyle='#fff';const hs=Math.max(8,cv.width/90);cropCorners(r).forEach(([x,y])=>ctx.fillRect(x-hs/2,y-hs/2,hs,hs));ctx.restore();
}
function cropPoint(e){const cv=$('cropCanvas'),b=cv.getBoundingClientRect(),W=CROP.img.naturalWidth,H=CROP.img.naturalHeight;
 return {x:Math.max(0,Math.min(W,(e.clientX-b.left)/b.width*W)),y:Math.max(0,Math.min(H,(e.clientY-b.top)/b.height*H))};}
function cropFromPoints(a,b){return {x:Math.min(a.x,b.x),y:Math.min(a.y,b.y),w:Math.abs(a.x-b.x),h:Math.abs(a.y-b.y)};}
function applyCrop(){
 const r=CROP.rect,W=CROP.img?.naturalWidth,H=CROP.img?.naturalHeight;
 if(!r||r.w<0.1*W||r.h<0.1*H)return showStatus('cropStatus','Marque uma área maior em volta do prato.',true);
 const x=Math.round(r.x),y=Math.round(r.y),w=Math.round(r.w),h=Math.round(r.h);
 const c=document.createElement('canvas');c.width=w;c.height=h;c.getContext('2d').drawImage(CROP.img,x,y,w,h,0,0,w,h);
 c.toBlob(blob=>{if(!blob)return showStatus('cropStatus','Não foi possível recortar esta imagem.',true);
  const file=new File([blob],'recorte.jpg',{type:'image/jpeg'}),url=URL.createObjectURL(blob);
  if(STATE.crop?.url)URL.revokeObjectURL(STATE.crop.url);
  STATE.crop={file,url,rect:{x,y,w,h},areaOriginal:Math.max(0.01,Math.min(1,(w*h)/(W*H)))};
  showPreview(url,`${STATE.file.name} (recortada)`,'Recorte pronto para análise');$('cropDialog').close();analyze();},'image/jpeg',0.92);
}
function resetCrop(){if(STATE.crop?.url)URL.revokeObjectURL(STATE.crop.url);STATE.crop=null;CROP.rect=null;
 showPreview($('photo').dataset.objectUrl,STATE.file.name,'Foto pronta para análise');$('cropDialog').close();analyze();}
$('cropBtn').addEventListener('click',openCrop);$('cropDialog').addEventListener('close',()=>{if(STATE.file&&!STATE.foods.length&&$('analyzeBtn').textContent==='Analisar refeição')showStatus('analysisStatus','Foto selecionada. Use "Recortar no prato" ou "Analisar refeição" quando quiser.');});$('cropApply').addEventListener('click',applyCrop);$('cropReset').addEventListener('click',resetCrop);
// O que um ponto atinge: canto (redimensionar), interior (mover) ou fora (desenhar novo).
function cropHit(p){
 const r=CROP.rect;if(!r)return {mode:'draw'};const b=$('cropCanvas').getBoundingClientRect(),tol=16*CROP.img.naturalWidth/b.width;
 const corners=cropCorners(r);for(let i=0;i<4;i++){const [x,y]=corners[i];if(Math.abs(p.x-x)<=tol&&Math.abs(p.y-y)<=tol){const [ax,ay]=corners[3-i];return {mode:'resize',anchor:{x:ax,y:ay},cursor:i===0||i===3?'nwse-resize':'nesw-resize'};}}
 if(p.x>r.x&&p.x<r.x+r.w&&p.y>r.y&&p.y<r.y+r.h)return {mode:'move',cursor:'move'};
 return {mode:'draw'};
}
$('cropCanvas').addEventListener('pointerdown',e=>{if(!CROP.img)return;e.preventDefault();$('cropCanvas').setPointerCapture(e.pointerId);
 const p=cropPoint(e),hit=cropHit(p);
 if(hit.mode==='move')CROP.drag={mode:'move',off:{x:p.x-CROP.rect.x,y:p.y-CROP.rect.y}};
 else if(hit.mode==='resize')CROP.drag={mode:'resize',anchor:hit.anchor};
 else{CROP.drag={mode:'draw',anchor:p};CROP.rect=cropFromPoints(p,p);drawCrop();}});
$('cropCanvas').addEventListener('pointermove',e=>{if(!CROP.img)return;const p=cropPoint(e),d=CROP.drag;
 if(!d){$('cropCanvas').style.cursor=cropHit(p).cursor||'crosshair';return;}
 if(d.mode==='move'){const W=CROP.img.naturalWidth,H=CROP.img.naturalHeight,r=CROP.rect;
  r.x=Math.max(0,Math.min(W-r.w,p.x-d.off.x));r.y=Math.max(0,Math.min(H-r.h,p.y-d.off.y));}
 else CROP.rect=cropFromPoints(d.anchor,p);
 drawCrop();});
['pointerup','pointercancel'].forEach(ev=>$('cropCanvas').addEventListener(ev,()=>{if(!CROP.drag)return;CROP.drag=null;showStatus('cropStatus','Clique em "Analisar recorte" ou continue ajustando.');}));

// Metas pelas medidas: o backend calcula (Mifflin-St Jeor + objetivo) e a página só preenche os campos.
function fillSelect(id,items,keepBlank=false){const sel=$(id);const current=sel.value;if(!keepBlank)sel.replaceChildren();else sel.querySelectorAll('option:not([value=""])').forEach(o=>o.remove());
 items.forEach(it=>{const op=document.createElement('option');op.value=it.id;op.textContent=it.nome;sel.append(op);});if(current)sel.value=current;}
async function calcGoals(){
 const box=$('calcResult');box.classList.remove('error');box.replaceChildren();
 const body={...bodyForm(),objetivo:$('goal').value};
 const missing=[['sexo','sexo'],['idade','idade'],['peso_kg','peso'],['altura_cm','altura'],['atividade','nível de atividade']].filter(([k])=>body[k]==null).map(([,n])=>n);
 if(missing.length){box.classList.add('error');box.textContent='Preencha '+missing.join(', ')+' para calcular.';return;}
 try{
  const r=await api('/api/profile/calcular',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  $('kcal').value=r.kcal;$('protein').value=r.proteina_g;$('carbs').value=r.carboidrato_g;$('fats').value=r.lipideos_g;
  const head=document.createElement('strong');head.textContent=`${r.kcal} kcal · ${r.proteina_g} g proteína · ${r.carboidrato_g} g carboidrato · ${r.lipideos_g} g gordura por dia`;
  const ul=document.createElement('ul');r.explicacao.forEach(t=>{const li=document.createElement('li');li.textContent=t;ul.append(li);});
  box.append(head,ul);r.avisos.forEach(t=>{const w=document.createElement('div');w.className='warn';w.textContent='⚠ '+t;box.append(w);});
  const foot=document.createElement('div');foot.textContent='Os campos abaixo foram preenchidos. Ajuste se quiser e clique em "Salvar metas".';box.append(foot);
 }catch(e){box.classList.add('error');box.textContent='Não foi possível calcular: '+e.message;}
}
$('calcGoals').addEventListener('click',calcGoals);

// Histórico: calendário do mês; cada dia mostra até 3 fotos de pratos em bolinhas (+N quando tem mais).
const HIST={mes:null,data:null,sel:null};
const MESES=['janeiro','fevereiro','março','abril','maio','junho','julho','agosto','setembro','outubro','novembro','dezembro'];
const DIAS_SEMANA=['Domingo','Segunda','Terça','Quarta','Quinta','Sexta','Sábado'];
const parseDia=(iso)=>{const [y,m,d]=iso.split('-').map(Number);return new Date(y,m-1,d);};
const isoDia=(y,m,d)=>`${y}-${String(m).padStart(2,'0')}-${String(d).padStart(2,'0')}`;
const fmt=(v,dig=0)=>Number(v||0).toLocaleString('pt-BR',{maximumFractionDigits:dig});
function shiftMes(mes,delta){const [y,m]=mes.split('-').map(Number);const d=new Date(y,m-1+delta,1);return `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}`;}
async function loadHistory(mes){
 showStatus('histStatus','');
 try{
  const data=await api('/api/meals'+(mes?`?mes=${mes}`:''));HIST.data=data;HIST.mes=data.mes;
  if(!HIST.sel||!HIST.sel.startsWith(data.mes)){
   const dias=[...new Set(data.meals.map(m=>m.dia))];
   HIST.sel=data.hoje.startsWith(data.mes)?data.hoje:(dias.length?dias[dias.length-1]:`${data.mes}-01`);
  }
  renderHistory();loadMonthFeedback();
 }catch(e){showStatus('histStatus','Não foi possível carregar o histórico: '+e.message,true);}
}
function renderHistory(){
 const data=HIST.data,[y,m]=data.mes.split('-').map(Number);
 $('histDias').textContent=data.resumo.dias;$('histPratos').textContent=data.resumo.pratos;
 $('histMonth').textContent=`${MESES[m-1][0].toUpperCase()+MESES[m-1].slice(1)} ${y}`;
 $('histNext').disabled=data.mes>=data.hoje.slice(0,7);
 const porDia={};data.meals.forEach(ml=>{(porDia[ml.dia]=porDia[ml.dia]||[]).push(ml);});
 const grid=$('histGrid');grid.replaceChildren();
 ['D','S','T','Q','Q','S','S'].forEach((w,i)=>{const s=document.createElement('span');s.className='hist-wd';s.textContent=w;s.title=DIAS_SEMANA[i];grid.append(s);});
 const first=new Date(y,m-1,1).getDay(),n=new Date(y,m,0).getDate();
 for(let i=0;i<first;i++){const s=document.createElement('span');s.setAttribute('aria-hidden','true');grid.append(s);}
 for(let d=1;d<=n;d++){
  const iso=isoDia(y,m,d),meals=porDia[iso]||[],future=iso>data.hoje;
  const b=document.createElement('button');b.type='button';b.className='hist-cell'+(meals.length?' has':'')+(iso===data.hoje?' today':'');
  b.disabled=future;b.setAttribute('aria-pressed',String(iso===HIST.sel));
  b.setAttribute('aria-label',`${d} de ${MESES[m-1]}, ${meals.length?meals.length+(meals.length>1?' pratos':' prato'):'sem pratos'}`);
  const num=document.createElement('span');num.className='hist-num';num.textContent=d;b.append(num);
  const shown=meals.slice(0,3);
  shown.forEach((ml,i)=>{const img=document.createElement('img');img.className=`hist-dot p${shown.length}-${i}`;img.src=ml.foto;img.alt='';img.loading='lazy';img.style.zIndex=String(1+i);b.append(img);});
  if(meals.length>3){const more=document.createElement('span');more.className='hist-more';more.textContent='+'+(meals.length-3);b.append(more);}
  b.addEventListener('click',()=>{HIST.sel=iso;renderHistory();});
  grid.append(b);
 }
 renderHistoryDay(porDia[HIST.sel]||[]);
}
function renderHistoryDay(meals){
 const box=$('histDay'),data=HIST.data,dt=parseDia(HIST.sel);box.replaceChildren();
 const h=document.createElement('h3');h.textContent=`${DIAS_SEMANA[dt.getDay()]}, ${dt.getDate()} de ${MESES[dt.getMonth()]}`;
 const sub=document.createElement('span');sub.className='muted';sub.textContent=meals.length?`${meals.length} ${meals.length>1?'pratos registrados':'prato registrado'}`:'Sem registro';
 const head=document.createElement('div');head.append(h,sub);box.append(head);
 if(!meals.length){const p=document.createElement('p');p.className='muted';p.style.margin='0';p.textContent='Nenhum prato registrado neste dia.';box.append(p);return;}
 const tot=meals.reduce((a,ml)=>{['kcal','proteina_g','carboidrato_g','lipideos_g'].forEach(k=>a[k]+=ml.totais[k]||0);return a;},{kcal:0,proteina_g:0,carboidrato_g:0,lipideos_g:0});
 const pct=data.meta_kcal_dia?Math.round(tot.kcal/data.meta_kcal_dia*100):0;
 const total=document.createElement('div');total.className='hist-total';
 const tk=document.createElement('strong');tk.textContent=`${fmt(tot.kcal)} kcal`;
 const tm=document.createElement('span');tm.className='muted';tm.textContent=`${pct}% da meta do dia (${fmt(data.meta_kcal_dia)} kcal)`;total.append(tk,tm);
 const bar=document.createElement('div');bar.className='hist-bar';const fill=document.createElement('div');fill.style.width=Math.min(100,pct)+'%';fill.classList.toggle('over',pct>110);bar.append(fill);
 const mac=document.createElement('div');mac.className='hist-macros';
 [['p','P',tot.proteina_g],['c','C',tot.carboidrato_g],['g','G',tot.lipideos_g]].forEach(([cls,l,v])=>{const s=document.createElement('span');const b=document.createElement('b');b.className=cls;b.textContent=l;s.append(b,` ${fmt(v)} g`);mac.append(s);});
 box.append(total,bar,mac);
 meals.forEach(ml=>{
  const art=document.createElement('article');art.className='hist-meal';
  const img=document.createElement('img');img.src=ml.foto;img.alt=`Foto do prato das ${ml.hora}`;img.loading='lazy';
  const body=document.createElement('div');body.className='hist-meal-body';
  const top=document.createElement('div');top.className='hist-meal-top';const t=document.createElement('strong');t.textContent=ml.hora;top.append(t);
  const items=document.createElement('div');items.className='hist-meal-items';items.textContent=ml.itens.map(it=>`${it.nome} ${fmt(it.grams)} g`).join(' · ');
  const foot=document.createElement('div');foot.className='hist-meal-foot';const k=document.createElement('strong');k.textContent=`${fmt(ml.totais.kcal)} kcal`;
  const del=document.createElement('button');del.type='button';del.className='btn btn-sm hist-del';del.textContent='Apagar';del.setAttribute('aria-label',`Apagar prato das ${ml.hora}`);
  del.addEventListener('click',async()=>{if(!confirm('Apagar este prato do histórico?'))return;try{await api(`/api/meals/${ml.id}`,{method:'DELETE'});if(STATE.mealId===ml.id)STATE.mealId=null;await loadHistory(HIST.mes);}catch(e){showStatus('histStatus','Não foi possível apagar: '+e.message,true);}});
  foot.append(k,del);body.append(top,items,foot);art.append(img,body);box.append(art);
 });
}
$('openHistory').addEventListener('click',()=>{openDialog('historyDialog');loadHistory(HIST.mes);});
$('histPrev').addEventListener('click',()=>loadHistory(shiftMes(HIST.mes,-1)));
$('histNext').addEventListener('click',()=>loadHistory(shiftMes(HIST.mes,1)));

// Feedback escrito pelo LLM. Prato: automático depois da análise e refeito ~1,5 s depois da última correção.
// Se o serviço não responder, ficam as dicas por regra que renderStats já mostrou.
var FB={seq:0,ctrl:null,timer:null};  // var: renderEmptyStats roda antes desta linha na carga da página
function cancelFeedback(){if(!FB)return;clearTimeout(FB.timer);if(FB.ctrl)FB.ctrl.abort();FB.ctrl=null;FB.seq++;document.getElementById('aiLoading')?.remove();}
function requestFeedback(delay=0){cancelFeedback();if(!STATE.foods.length)return;FB.timer=setTimeout(runFeedback,delay);}
async function runFeedback(){
 const seq=++FB.seq,ctrl=new AbortController();FB.ctrl=ctrl;
 const parent=$('suggestions');document.getElementById('aiLoading')?.remove();
 const wait=document.createElement('div');wait.id='aiLoading';wait.className='reco reco-ai';wait.setAttribute('role','status');
 wait.innerHTML='<div class="tiny">Feedback do prato</div><p class="ai-wait">Escrevendo o feedback…</p>';parent.prepend(wait);
 const limite=setTimeout(()=>ctrl.abort(),30000);  // o servidor desiste em ~20 s; isto é só uma garantia
 // Sem texto da IA, o cartão de espera vira um aviso curto e as dicas por regra continuam abaixo.
 const semFeedback=()=>{
  wait.querySelector('.ai-wait').textContent='Feedback indisponível agora. Seguem as dicas calculadas.';
  const again=document.createElement('button');again.type='button';again.className='btn btn-sm ai-retry';again.textContent='Tentar de novo';
  again.addEventListener('click',()=>requestFeedback(0));wait.append(again);
 };
 try{
  const res=await api('/api/feedback/prato',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({items:STATE.foods}),signal:ctrl.signal});
  if(seq!==FB.seq)return;
  if(res.texto){wait.remove();renderAiFeedback(res.texto);}else semFeedback();
 }catch(e){if(seq===FB.seq)semFeedback();}
 finally{clearTimeout(limite);}
}
function aiList(title,items,cls){
 const box=document.createElement('div');box.className='ai-block '+(cls||'');
 const h=document.createElement('strong');h.textContent=title;const ul=document.createElement('ul');
 (items||[]).filter(Boolean).forEach(t=>{const li=document.createElement('li');li.textContent=t;ul.append(li);});
 box.append(h,ul);return box;
}
function renderAiFeedback(t){
 const card=document.createElement('div');card.className='reco reco-ai';
 const tiny=document.createElement('div');tiny.className='tiny';tiny.textContent='Feedback do prato';
 const resumo=document.createElement('p');resumo.className='ai-resumo';resumo.textContent=t.resumo||'';
 card.append(tiny,resumo);
 if(t.positivo)card.append(aiList('Ponto positivo',[t.positivo],'ai-pos'));
 if(t.sugestoes?.length)card.append(aiList('Sugestões',t.sugestoes));
 $('suggestions').replaceChildren(card);
}

// Feedback do mês: botão no Histórico; fica salvo e só é refeito quando a pessoa pede.
function renderMonthFeedback(fb,pratosNoMes){
 const box=$('histFb'),btn=$('histFbBtn');box.replaceChildren();
 btn.disabled=!pratosNoMes;btn.textContent=fb?.texto?'Gerar de novo':'Gerar feedback';
 if(!pratosNoMes){const p=document.createElement('p');p.className='muted ai-info';p.textContent='Sem pratos registrados neste mês.';box.append(p);return;}
 if(!fb?.texto)return;
 const t=fb.texto,resumo=document.createElement('p');resumo.className='ai-resumo';resumo.textContent=t.resumo||'';box.append(resumo);
 // Destaques, atenção e metas ficam recolhidos para o cartão não empurrar o calendário.
 const grid=document.createElement('div');grid.className='ai-grid';grid.id='histFbDetalhes';grid.hidden=true;
 grid.append(aiList('Destaques',t.destaques,'ai-pos'),aiList('Para prestar atenção',t.atencao),aiList('Metas para o próximo mês',t.metas,'ai-meta'));box.append(grid);
 const foot=document.createElement('div');foot.className='ai-foot';
 const tg=document.createElement('button');tg.type='button';tg.className='ai-toggle';tg.textContent='Ver destaques e metas';tg.setAttribute('aria-expanded','false');tg.setAttribute('aria-controls','histFbDetalhes');
 tg.addEventListener('click',()=>{const open=grid.hidden;grid.hidden=!open;tg.textContent=open?'Mostrar menos':'Ver destaques e metas';tg.setAttribute('aria-expanded',String(open));});
 const info=document.createElement('span');info.className='muted ai-info';const d=new Date(fb.criado_em);
 info.textContent=`Gerado em ${d.toLocaleDateString('pt-BR',{day:'2-digit',month:'2-digit'})} · ${fb.n_pratos} ${fb.n_pratos>1?'pratos':'prato'}`+(fb.n_pratos!==pratosNoMes?' · há pratos novos':'');
 foot.append(tg,info);box.append(foot);
}
async function loadMonthFeedback(){
 const mes=HIST.mes,n=HIST.data?.meals.length||0;
 try{const fb=await api(`/api/feedback/mes?mes=${mes}`);if(mes===HIST.mes)renderMonthFeedback(fb,n);}catch{renderMonthFeedback(null,n);}
}
$('histFbBtn').addEventListener('click',async()=>{
 const btn=$('histFbBtn'),mes=HIST.mes;btn.disabled=true;btn.textContent='Gerando…';showStatus('histStatus','');
 try{const fb=await api(`/api/feedback/mes?mes=${mes}`,{method:'POST'});if(mes===HIST.mes)renderMonthFeedback(fb,HIST.data.meals.length);}
 catch(e){showStatus('histStatus',e.message,true);loadMonthFeedback();}
});
