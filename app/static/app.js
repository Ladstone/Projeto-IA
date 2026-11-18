const $ = (id)=>document.getElementById(id);
const STATE={file:null,foods:[],taco:[],tacoMap:new Map(),classes:{},segmentation:false,catalogPage:0,lastNutrition:null,unmatched:[]};
function showStatus(id,message,error=false){const el=$(id);el.textContent=message;el.classList.toggle('error',error);}
const FIELD_LABELS={kcal:'Meta energética',proteina_g:'Proteínas',carboidrato_g:'Carboidratos',lipideos_g:'Gorduras',refeicoes:'Refeições por dia',objetivo:'Objetivo',grams:'Gramas',taco_id:'Alimento TACO'};
// Erros 422 do FastAPI chegam como lista [{loc,msg}]: transforma em texto legível.
function formatApiError(detail){
 if(typeof detail==='string')return detail;
 if(Array.isArray(detail))return detail.map(d=>{const field=(d.loc||[]).filter(p=>typeof p==='string'&&p!=='body').pop();return `${FIELD_LABELS[field]||field||'Valor'}: ${d.msg}`;}).join('; ');
 return 'Erro inesperado no servidor.';
}
async function api(url,opts={}){let res=await fetch(url,opts);let body;try{body=await res.json();}catch{throw new Error(`Resposta inválida (${res.status})`)}if(!res.ok)throw new Error(formatApiError(body.detail));return body;}
const numeric=(v)=>Number(v||0);
function getProfileForm(){return {objetivo:$('goal').value,kcal:numeric($('kcal').value),proteina_g:numeric($('protein').value),carboidrato_g:numeric($('carbs').value),lipideos_g:numeric($('fats').value),refeicoes:numeric($('meals').value)};}
function fillProfile(p){$('goal').value=p.objetivo;$('kcal').value=p.kcal;$('protein').value=p.proteina_g;$('carbs').value=p.carboidrato_g;$('fats').value=p.lipideos_g;$('meals').value=p.refeicoes;}
async function saveProfile(e){if(e)e.preventDefault();try{await api('/api/profile',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(getProfileForm())});showStatus('profileStatus','✓ Metas salvas e usadas no cálculo.');if(STATE.foods.length)await recalc();setTimeout(()=>{$('goalsDialog').close();showStatus('profileStatus','');},700);}catch(e){showStatus('profileStatus','Não foi possível salvar: '+e.message,true);}}
// Cores das classes vêm do CSS (--class-N), as mesmas do overlay gerado no backend.
const classColor=(id)=>getComputedStyle(document.documentElement).getPropertyValue(`--class-${id}`).trim()||'#9aa59f';
function selectFile(file){if(!file)return;if(!file.type.startsWith('image/')){showStatus('analysisStatus','Escolha um arquivo de imagem.',true);return}if(file.size>8*1024*1024){showStatus('analysisStatus','O arquivo excede 8 MB.',true);return}STATE.file=file;STATE.foods=[];STATE.unmatched=[];renderUnmatched();renderFoods();renderEmptyStats();const url=URL.createObjectURL(file);if($('photo').dataset.objectUrl)URL.revokeObjectURL($('photo').dataset.objectUrl);$('photo').dataset.objectUrl=url;$('photo').src=url;$('dropArea').classList.add('has-photo');$('dropArea').classList.remove('show-overlay');$('overlay').removeAttribute('src');$('fileName').textContent=file.name;$('photoStatus').textContent='Foto pronta para análise';$('analyzeBtn').disabled=false;showStatus('analysisStatus','Imagem selecionada. Clique em analisar.');}
async function selectExample(type){const resp=await fetch('/static/examples/'+type+'.jpg');if(!resp.ok)return showStatus('analysisStatus','Exemplo indisponível.',true);const blob=await resp.blob();selectFile(new File([blob],`${type}.jpg`,{type:'image/jpeg'}));}
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
  const info=document.createElement('div');info.style.minWidth='0';info.style.flex='1';const name=document.createElement('div');name.className='food-name';name.textContent=STATE.tacoMap.get(Number(it.taco_id))?.nome||it.nome;
  const sub=document.createElement('div');sub.className='food-sub';const badge=document.createElement('span');
  if(it.class_id){badge.className='badge badge-ai';badge.textContent=`⚠ Sugestão da IA · ${it.cobertura_imagem_percent}% da foto`;}else{badge.className='badge badge-manual';badge.textContent='✓ Escolhido por você';}
  sub.append(badge);info.append(name,sub);top.append(info);outer.append(top);
  const controls=document.createElement('div');controls.className='food-controls';
  const gramsWrap=document.createElement('label');gramsWrap.className='grams';const gramsUnit=document.createElement('span');gramsUnit.textContent='g';
  const grams=document.createElement('input');grams.type='number';grams.inputMode='numeric';grams.value=it.grams;grams.min='1';grams.max='2500';grams.step='5';grams.title='Quantidade em gramas';grams.setAttribute('aria-label',`Gramas de ${it.nome}`);grams.addEventListener('change',()=>{STATE.foods[idx].grams=Math.max(1,Math.min(2500,Number(grams.value)||1));recalc();});
  gramsWrap.append(grams,gramsUnit);
  const selector=document.createElement('input');selector.type='search';selector.setAttribute('list','tacoOptions');selector.placeholder='Trocar por outro alimento da TACO';selector.setAttribute('aria-label',`Corrigir alimento pela TACO: ${it.nome}`);selector.title='Digite qualquer nome da TACO para corrigir a sugestão da IA';selector.value=STATE.tacoMap.get(Number(it.taco_id))?.nome||it.nome;
  selector.addEventListener('change',()=>{const entry=foodLookup(selector.value);if(!entry){showStatus('analysisStatus','Não encontrei este alimento na TACO. Abra a tabela e pesquise.',true);return;}
    STATE.foods[idx].taco_id=entry.id;STATE.foods[idx].nome=entry.nome;STATE.foods[idx].class_id=null;STATE.foods[idx].origem='manual';STATE.foods[idx].score_nao_calibrado=null;STATE.foods[idx].note='Alimento corrigido manualmente; confira o peso.';renderFoods();recalc();showStatus('analysisStatus',`Corrigido para ${entry.nome}.`);});
  const remove=document.createElement('button');remove.type='button';remove.className='remove-btn';remove.textContent='×';remove.title='Remover alimento';remove.setAttribute('aria-label',`Remover ${it.nome}`);remove.addEventListener('click',()=>{STATE.foods.splice(idx,1);renderFoods();recalc()});
  controls.append(gramsWrap,selector,remove);outer.append(controls);
  const note=document.createElement('div');note.className='food-note';note.textContent=it.note||'Confirme a massa e o preparo. A fotografia não mede gramas.';outer.append(note);
  const detailsBtn=document.createElement('button');detailsBtn.type='button';detailsBtn.className='link-btn';detailsBtn.textContent=`Ver nutrientes · TACO #${it.taco_id}`;detailsBtn.addEventListener('click',()=>showFoodDetails(it.taco_id));outer.append(detailsBtn);list.append(outer);
 });
}
function renderEmptyStats(){[['Kcal','Kcal'],['Protein','Protein'],['Carbs','Carbs'],['Fats','Fats']].forEach(([stat,bar])=>{$('stat'+stat).textContent='—';$('bar'+bar).style.width='0%';$('bar'+bar).classList.remove('over');$('target'+stat).textContent='Referência da refeição: —'});$('mobileKcal').textContent='—';$('suggestions').replaceChildren();const d=document.createElement('div');d.className='reco';d.innerHTML='<div class="tiny">Próximo passo</div><h3>Envie uma foto</h3><p>Depois da análise, as sugestões comparam a refeição com as suas metas.</p>';$('suggestions').append(d);}
function renderStats(n){const stat=[['Kcal','kcal','kcal'],['Protein','proteina_g','g'],['Carbs','carboidrato_g','g'],['Fats','lipideos_g','g']];stat.forEach(([key,field,unit])=>{const cmp=n.comparacao[field];$('stat'+key).textContent=cmp.obtido.toLocaleString('pt-BR',{maximumFractionDigits:1});$('bar'+key).style.width=Math.min(100,Math.max(0,cmp.percentual||0))+'%';$('bar'+key).classList.toggle('over',(cmp.percentual||0)>135);$('target'+key).textContent=`Referência: ${cmp.alvo_refeicao.toLocaleString('pt-BR')} ${unit} (${cmp.percentual||0}%)`;});$('mobileKcal').textContent=n.comparacao.kcal.obtido.toLocaleString('pt-BR',{maximumFractionDigits:0});const parent=$('suggestions');parent.replaceChildren();n.recomendacoes.forEach(rec=>{const box=document.createElement('div');box.className='reco';const small=document.createElement('div');small.className='tiny';small.textContent=rec.tipo;const title=document.createElement('h3');title.textContent=rec.titulo;const p=document.createElement('p');p.textContent=rec.descricao;box.append(small,title,p);parent.append(box)});}
async function recalc(){try{const data=await api('/api/recalculate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({items:STATE.foods})});STATE.lastNutrition=data;renderStats(data);renderFullNutrition(data);}catch(err){showStatus('analysisStatus','Erro no cálculo: '+err.message,true)}}
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
 const fd=new FormData();fd.append('file',STATE.file);
 try{
   const res=await api('/api/analyze',{method:'POST',body:fd});
   STATE.foods=res.detected||[];STATE.unmatched=res.unmatched||[];
   renderUnmatched();renderFoods();renderVisionDiagnostics(res.vision_debug);
   STATE.lastNutrition=res.nutrition;
   if(STATE.foods.length){renderStats(res.nutrition);renderFullNutrition(res.nutrition);}
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
async function setup(){const [health,profile,foods]=await Promise.all([api('/api/health'),api('/api/profile'),api('/api/foods')]);$('baseCount').textContent=`${health.taco_records} alimentos · ${health.taco_fields} nutrientes`;fillProfile(profile);STATE.taco=foods.foods;STATE.tacoMap=new Map(STATE.taco.map(f=>[f.id,f]));STATE.classes=foods.class_map;const fragment=document.createDocumentFragment();STATE.taco.forEach(f=>{const op=document.createElement('option');op.value=f.nome;fragment.append(op)});$('tacoOptions').append(fragment);renderCatalog();}
$('profileForm').addEventListener('submit',saveProfile);$('file').addEventListener('change',e=>selectFile(e.target.files[0]));$('dropArea').addEventListener('click',()=>{$('file').click()});$('dropArea').addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();$('file').click()}});
['dragenter','dragover'].forEach(event=>$('dropArea').addEventListener(event,e=>{e.preventDefault();$('dropArea').classList.add('dragover')}));['dragleave','drop'].forEach(event=>$('dropArea').addEventListener(event,e=>{e.preventDefault();$('dropArea').classList.remove('dragover')}));$('dropArea').addEventListener('drop',e=>selectFile(e.dataTransfer.files[0]));
$('changeBtn').addEventListener('click',()=>$('file').click());$('analyzeBtn').addEventListener('click',analyze);$('overlaySwitch').addEventListener('change',e=>$('dropArea').classList.toggle('show-overlay',e.target.checked&&Boolean($('overlay').src)));
document.querySelectorAll('[data-example]').forEach(btn=>btn.addEventListener('click',()=>selectExample(btn.dataset.example)));$('addFoodBtn').addEventListener('click',addManual);$('catalogSearch').addEventListener('input',()=>{STATE.catalogPage=0;renderCatalog()});$('catalogPrev').addEventListener('click',()=>{STATE.catalogPage--;renderCatalog()});$('catalogNext').addEventListener('click',()=>{STATE.catalogPage++;renderCatalog()});$('modalClose').addEventListener('click',()=>$('tacoModal').close());$('foodSearch').addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();addManual();}});
// Diálogos: metas (modal) e Tabela TACO (gaveta). Fecham pelo botão, por Esc ou clicando fora.
$('openGoals').addEventListener('click',()=>openDialog('goalsDialog','kcal'));$('editGoals').addEventListener('click',()=>openDialog('goalsDialog','kcal'));
$('openTaco').addEventListener('click',()=>openDialog('catalogDialog','catalogSearch'));$('openTacoInline').addEventListener('click',()=>openDialog('catalogDialog','catalogSearch'));
document.querySelectorAll('dialog').forEach(d=>{d.addEventListener('click',e=>{if(e.target===d)d.close();});d.querySelectorAll('[data-close]').forEach(b=>b.addEventListener('click',()=>d.close()));});
$('catalogDialog').addEventListener('close',()=>showStatus('catalogStatus',''));
renderEmptyStats();setup().catch(e=>showStatus('analysisStatus','Falha ao iniciar o protótipo: '+e.message,true));
