# PratoMetria — resumo das mudanças

Comparação da versão original (commit `7a9a682`) com a versão atual da `main`. Atualizado em 25/11/2026. Para os resultados em uma página, ver `docs/RESULTADOS.md`.

## 1. Visão geral

- **O projeto:** protótipo acadêmico. O usuário envia a foto de um prato, uma rede neural local sugere os alimentos (28 categorias visuais), o usuário confere nomes e gramas, e a TACO (597 alimentos) dá os nutrientes. O resultado é comparado com as metas do usuário.
- **O que mudou:** foram três frentes.
  - **Fase A:** correções no pipeline.
  - **Frontend novo.**
  - **Fase B:** um novo modelo de IA (v2), treinado de outro jeito e avaliado em fotos separadas.
  - **Fase C:** o modelo v3, com um terceiro dataset (MyFoodRepo) e 7 categorias novas, 28 no total.
  - **Fase D (24 e 25/11/2026):** combinação de modelos, purê, histórico de pratos e feedback por IA.
- **Estado (25/11/2026):** o app usa por padrão a **combinação**: base v2 + categorias novas do v3 + purê do v3d. O v2 puro fica de backup (`models/unified_21class_v2.*`). Os testes passam (80 sem os datasets; os que leem os ZIPs são pulados).
- **Limite:** a identificação erra e as gramas são estimativas; o usuário confere e corrige.

## 2. O que mudou

### Fase A — correções sem retreinar
| O que | Por quê |
|---|---|
| A orientação EXIF é aplicada antes da IA | Fotos de celular chegavam deitadas, e a máscara não batia com a foto exibida |
| A porção inicial é proporcional à área (hoje: 12% do prato = porção padrão, limitada entre 25% e 200%; ver "Gramas pelo prato" abaixo) | Uma mancha de 1–2% da foto entrava com a porção cheia e inflava as kcal |
| Regiões com menos de 3% da foto são descartadas | Eram quase sempre falsos positivos |
| O filtro de "confiança" foi removido; o campo virou `score_nao_calibrado` | O filtro nunca descartava nada, e o valor sempre parecia alto |
| "Massa com molho vermelho" passou a usar a TACO #37 (massa cozida), com sugestão de somar a #159 (molho de tomate) | Antes usava a bolonhesa, que tem carne |
| Mensagens de erro legíveis e busca TACO ordenada por relevância | Usabilidade |
| "Purê" detectado vira três itens da TACO: batata cozida (#91, 80%), leite integral (#458, 16%) e manteiga com sal (#261, 4%) — `RECIPES` em `app/unified_classes.py` | A TACO não tem purê de batata; só batata cozida subestimava gordura e calorias |

**Arquivos:** `app/main.py`, `app/unified_classes.py`, `app/static/app.js`, `tests/test_analise_foto.py` (na época, `test_fase_a.py`).

### Frontend novo
O resumo nutricional ficava abaixo do catálogo de 597 alimentos e havia 4 avisos competindo entre si.

**O que mudou:**
- Agora a foto e os alimentos ficam à esquerda, e o resumo com as recomendações fica fixo à direita.
- As metas e a Tabela TACO viraram diálogos abertos por botões no cabeçalho.
- Cada alimento sugerido pela IA tem um selo "Sugestão da IA". O selo "IA local · experimental" do cabeçalho e o aviso amarelo "A identificação por IA é experimental" foram removidos em 17/11/2026.
- No celular a página vira uma coluna só, com uma barra de kcal fixa no rodapé.
- O redesenho em si não mudou o backend nem a API.
- Os botões "experimente uma foto dos datasets" foram removidos em 22/11/2026, para a página não parecer demonstração; as fotos foram para `tests/fixtures/`.
- Foi verificado com 20 checagens automáticas em navegador (desktop e celular).
- **Recorte no prato (21/11/2026):** ao escolher a foto, abre uma janela com um recorte já sugerido em volta da comida (`POST /api/crop-suggestion`). O usuário ajusta (move, redimensiona ou marca outro) e só então a análise roda, no recorte ou na foto inteira. Ajuda quando o prato ocupa pouco da foto: no teste com uma foto da internet, a carne só foi encontrada com o recorte. A área mínima de 3% vale sobre o recorte; a "% da foto" usa a foto original (`area_original`).
- **Gramas pelo prato (23/11/2026):** a porção passou a ser medida pela área do alimento **no prato** (o recorte; sem recorte, o retângulo em volta da comida), com 12% do prato = porção padrão e teto de 200%. Antes era a área na foto inteira (20% = porção padrão, teto de 150%), e uma foto real de almoço tirada de mais longe somava ~230 g para um prato de ~500 g. Com a nova regra, o mesmo prato dá ~495 g e não depende da distância da câmera. Continua sendo estimativa: o usuário deve corrigir os gramas.
- Uma análise automática em duas passadas (recortar e reanalisar a 320 px) foi testada e **descartada**: no conjunto de teste, as fotos do MyFood com algum alimento inventado subiram de 8% para 13,5–20%.
- **Metas pelas medidas (22/11/2026):** a janela "Minhas metas" ganhou cinco objetivos (emagrecer, definição, manter o peso, hipertrofia e esportes de resistência) e um cálculo opcional por sexo, idade, peso, altura e nível de atividade (`app/metas.py`, `POST /api/profile/calcular`). Usa a TMB de Mifflin-St Jeor × fator de atividade, ajustada pelo objetivo (−20% a +10%), proteína em g/kg (1,2 a 2,0) e gordura em % da energia; mostra a conta passo a passo e a pessoa pode editar antes de salvar. Travas: só adultos, piso de 1.200/1.500 kcal em déficit e proteína pelo peso de IMC 25 quando o IMC é 30 ou mais. Perfis com os objetivos antigos (ganho/perda/equilíbrio) são convertidos.

**Arquivos:** `app/static/index.html`, `app/static/styles.css` (cores e espaçamentos ficam em variáveis no topo), `app/static/app.js`.

### Fase B — modelo v2
O v1 errava de um jeito sistemático:
- Uma imagem marrom lisa virava "Feijão" com score 0,95, e ruído aleatório virava "Arroz".
- Ele não achava frango na foto de exemplo.

**Causa:** no treino, cada foto só ensinava a categoria anotada. Nada ensinava a rede o que *não* era cada alimento. Além disso, as máscaras do MyFood anotam todos os alimentos da foto, e o treino antigo ignorava isso.

**O que foi feito:**
- **Modelo pré-treinado:** MobileNetV3-Large (ImageNet) com cabeça LR-ASPP, em 224 px.
- **Supervisão negativa:** um pixel anotado como frango também ensina "não é arroz, não é feijão…".
- **Imagens sem comida:** cerca de 8% do treino são imagens sintéticas sem comida.
- **Separação dos dados:**
  - O teste brasileiro novo foi tirado da validação antiga, de modo que nenhum dos dois modelos treinou nele.
  - Fotos quase idênticas às do treino foram removidas da validação e do teste. Os próprios datasets têm a mesma foto repetida em divisões diferentes.
- **Limiares por classe** escolhidos **só na validação**. O teste foi usado uma única vez para cada modelo.
- **Treino** com checkpoint a cada época, log em arquivo e retomada.
- **Primeiro um piloto de 3 épocas** (16/11/2026); depois o **treino completo de 20 épocas** (21/11/2026, 99 min em CPU). A melhor época na validação foi a 16, e é ela que o app usa.

**Arquivos:**
- Modelo e predição: `app/vision.py`
- Dados, divisão e duplicatas: `scripts/data_v2.py`
- Treino: `scripts/train_v2.py`
- Exportação: `scripts/export_v2.py`
- Avaliação: `scripts/evaluate_models.py`
- Modelos: `models/` (`unified_21class.*` = base v2; `unified_21class_v2.*` = cópia de backup; os pesos do piloto e do v1 saíram do repositório em 25/11/2026 e estão no histórico do git)
- Testes: `tests/test_modelos.py`

### Fase C — modelo v3 com MyFoodRepo (23/11/2026)

**O problema:** o v2 só conhecia as 21 categorias dos dois datasets. Brócolis, pão, peixe e outras comidas comuns não eram reconhecidas, e em fotos variadas de celular ele quase não acertava.

**O que foi feito:**
- **Terceiro dataset:** MyFoodRepo v2.0, com 39.962 fotos de celular de refeições reais e 498 classes (detalhes em `docs/DATASETS.md`). Entra só uma amostra com teto de fotos por classe.
- **7 categorias novas (22 a 28), todas ligadas à TACO:** Brócolis (#100), Batata cozida (#91), Peixe (#301, merluza assada como aproximação), Pão (#53), Cebola (#107), Hambúrguer (sanduíche) e Lasanha. As duas últimas entram como **receita**: sanduíche = pão 40% + hambúrguer grelhado 60%; lasanha = massa 40% + carne moída 25% + molho de tomate 20% + muçarela 15%.
- **Reforço das antigas:** o MyFoodRepo também traz fotos de arroz, frango, salada, milho, purê etc.
- **Escolhas de classe:** coxa de frango (só 85 fotos) entrou em Frango; a carne de hambúrguer solta entrou em Carne bovina.
- **Feijão pelo tipo da cor (23/11/2026):** a rede só sabe que é feijão. A tela mostra "Feijão cozido", e a linha da TACO sai da cor da região: mais da metade dos pixels escuros usa o feijão preto (#567), senão o carioca (#561). Acertou o tipo nas 7 fotos reais de prato feito do mini teste.
- **Volta ao v2 (23/11/2026):** testado em fotos reais de prato feito, o v3 confundiu batata frita com macarrão e bife com frango, erros que o v2 não comete. O app voltou ao v2; o v3 fica em `models/unified_28class_v3.pt` para demonstração (`PRATOMETRIA_MODEL`, porta 8001). Antes de trocar de novo, testar também em fotos reais.
- **Combinação v2+v3 (24/11/2026, porta 8001, em teste):** `CombinedVisionService` em `app/vision.py`. Se o v3 vê uma categoria nova cobrindo 10% ou mais (era 3% até 24/11, ver AVALIACAO), a foto usa o v3; senão, o v2. No mini teste com fotos reais: prato feito igual ao v2 (29/35), "outros" igual ao v3 (5/8), 5 de 12 fotos perfeitas (v2: 2; v3: 4). Nas fotos brasileiras do teste é idêntico ao v2. Variável `PRATOMETRIA_MODEL_NOVAS`; testes em `tests/test_combinado.py`; métricas em `docs/METRICAS_COMBINADO.json`.
- **v3b (24/11/2026, descartado):** retreino sem massa e frango do MyFoodRepo e com peso maior nas fotos brasileiras. Trocou um erro por outro (perdeu frango). As opções ficaram no treino: `--mfr-skip-classes` e `--brazil-weight`.
- **Nomes curtos e purê como um item (24/11/2026):** a tela mostra "Arroz", "Frango" e "Carne bovina" em vez do nome completo da TACO, sem observação. O purê aparece como **um item só**, "Purê", calculado pela receita (batata 80%, leite em pó 2% + água 14%, manteiga 4%; id 9016, fora da lista de 597 da TACO). O leite integral fluido da TACO (#458) não tem calorias nem macros, por isso a receita usa o leite em pó diluído.
- **v3c e v3d (24/11/2026):** retreinos a partir dos pesos do v2 (`--init-from`) em que o MyFoodRepo só ensina as classes novas, cenoura e purê (`--mfr-old-classes`). O v3d recebe também 300 fotos de purê do images.cv com pseudo-rótulo (`scripts/pseudo_pure.py`, `--mashed`). Sozinhos, os dois ficam abaixo do v2 no prato feito; o v3d acerta purê em 90% de 196 fotos de teste (v2: 1%).
- **Purê do v3d na porta 8001 (24/11/2026, em teste):** `ClassOverrideService` em `app/vision.py`, variável `PRATOMETRIA_MODEL_PURE`. Onde o v3d marca purê, vale o purê; o resto continua com a combinação v2+v3. Números em `docs/AVALIACAO.md`.
- **Histórico de pratos (25/11/2026):** cada prato analisado no app é salvo sozinho (tabela `meals` no SQLite local, com miniatura de 256 px do recorte); correções atualizam o mesmo registro. Botão "Histórico" abre o calendário do mês com até 3 fotos por dia em bolinhas, dias registrados e pratos no mês; clicando num dia aparecem os pratos, o total contra a meta e a opção de apagar. API: `GET /api/meals?mes=AAAA-MM`, `GET /api/meals/{id}/foto`, `PUT` e `DELETE /api/meals/{id}`; `/api/analyze` só grava com `salvar=1`. Testes em `tests/test_historico.py`. Base para o feedback mensal.
- **Feedback por IA (25/11/2026):** `app/llm.py` chama a Gemini API (plano gratuito; chave no `.env`, modelo em `.env.example`). Feedback do prato automático após a análise e refeito 1,5 s depois da última correção (`POST /api/feedback/prato`); feedback do mês pelo botão no Histórico, salvo na tabela `feedback_mes` (`GET`/`POST /api/feedback/mes`). O app calcula todos os números (TACO, metas, médias do mês em `estatisticas_mes`); o LLM só redige, em JSON com esquema fixo. Ordem de modelos (medida no plano gratuito): Flash-Lite → 3.5 Flash-Lite → 3.1 Flash-Lite → Flash, no máximo 12 s por modelo e 20 s no total; modelo com 429/404 sai da fila por 10 min; pedido repetido reaproveita a resposta. Sem resposta, o card avisa, oferece "Tentar de novo" e valem as dicas por regra. Testes em `tests/test_feedback.py` (sem chamar a API).
- **Combinação vira o padrão (25/11/2026):** sem variáveis de ambiente, `carregar_visao` em `app/main.py` monta v2 + v3 + purê do v3d. `PRATOMETRIA_MODEL=models/unified_21class_v2.pt` sobe só o v2 (backup). 
- **Limpeza do repositório (25/11/2026):** saíram os pesos que o app não usa mais (`myfood_unet.pt`, v1 e v2 piloto); continuam no histórico do git (até o commit `c525370`) e as métricas deles seguem em `docs/METRICAS_*.json`. README reescrito e `docs/RESULTADOS.md` criado.
- **Teste independente (25/11/2026):** 21 fotos novas do Pinterest (12 de prato feito, 9 com purê), reunidas depois de todas as decisões. Combinação: 70/87 alimentos achados e 6 inventados; v2: 61/87 e 11. Purê achado em 9 de 9 fotos (v2: 0 de 9). Gabarito em `docs/FOTOS_TESTE_NOVO_GABARITO.json`, que não pode ser usado para ajustar nada. O script `avaliar_fotos_reais.py` ganhou `--gabarito` e passou a recortar exatamente como a página (arredondamento + JPEG 0,92).
- **Tomate e batata frita do v3 (25/11/2026):** a combinação padrão ganhou mais um passo (`CLASSES_DO_V3` em `app/main.py`): onde o v3 marca tomate ou batata frita, vale o v3. Escolhido num conjunto de desenvolvimento de 20 fotos (`docs/FOTOS_DEV_GABARITO.json`) e medido uma vez no teste independente: 74/87 alimentos achados (antes 70/87), 7 inventados (antes 6). O v3 roda uma vez só por foto (o `VisionService` guarda a última predição).
- **v4 (25/11/2026, experimento sem ganho):** retreino a partir do v2 com mais fotos do MyFoodRepo para tomate, batata frita, cenoura, brócolis e peixe (`--mfr-old-classes "6:250,10:400,11:300,16:200,22:300,24:300"`), purê e aumentos mais fortes (`--aug-forte`: prato longe, recorte mais fechado, luz). Sozinho fez 49/70 no desenvolvimento (combinação: 53/70) e não foi versionado. As opções ficaram em `scripts/train_v2.py` e `scripts/data_v2.py`. Nos scripts de avaliação, `...|modelo.pt:6,10` pega classes específicas de um especialista.
- **v5 (26/11/2026, experimento sem ganho):** receita do v3d + fotos do FoodSeg103 (arquivos `.parquet` do Hugging Face) para tomate, peixe, brócolis, cenoura, batata frita e ovo, filtradas foto a foto (`docs/FOODSEG103_EXCLUSOES.json`). Opções `--foodseg` e `--foodseg-exclusoes` em `scripts/train_v2.py`; carregador `load_foodseg` em `scripts/data_v2.py`. Como especialista de tomate, batata frita, ovo e cenoura fez 76/87 no teste independente (padrão: 74/87), mas com 11 alimentos inventados (padrão: 7). Não foi adotado nem versionado; detalhes na tabela 6 de `docs/RESULTADOS.md`.
- **Acabamento para a entrega (26/11/2026):** saiu `scripts/train_unified.py` (treino do v1, cujo modelo já não está no repositório); `scripts/verify_datasets.py` e os testes de dataset passaram a usar o carregador atual (`scripts/data_v2.py`); `tests/test_prototype.py` virou `tests/test_api.py` e `tests/test_fase_a.py` virou `tests/test_analise_foto.py`; `pyarrow` entrou em `requirements-training.txt`; `docs/DATASETS.md` e `docs/GUIA_EQUIPE.md` atualizados.
- **App:** aceita modelos de 21 ou 28 classes (lê `classes` do `.json`) e ganhou a variável `PRATOMETRIA_MODEL` para testar outro modelo sem trocar o ativo.
- **Treino:** 15 épocas, cerca de 12 min cada em CPU (~3 h). Interrompido uma vez por falta de RAM e retomado do checkpoint. Melhor época: 15.

**Arquivos:** `app/unified_classes.py`, `app/vision.py`, `app/main.py`, `scripts/data_v2.py`, `scripts/train_v2.py`, `scripts/export_v2.py`, `scripts/evaluate_models.py`, `tests/test_myfoodrepo.py`, `models/unified_28class_v3.*`, `scripts/pseudo_pure.py`, `models/unified_28class_v3d.*`, `docs/METRICAS_V3.json`.

## 3. Resultados no teste

### v2 × v3 (troca aprovada em 23/11/2026)

Mesmo teste para os dois. Fonte: `docs/METRICAS_V3.json`.

| Métrica | v2 | **v3** |
|---|---:|---:|
| MyFood (222 fotos): F1 de presença, macro de 8 classes | 0,94 | 0,93 |
| MyFood: falsos positivos por imagem | 0,10 | 0,11 |
| MyFood: pratos 100% corretos | 88% | 86% |
| MyFood: fundo marcado como comida | 5% | **3,5%** |
| Brasileiro (78 fotos): IoU médio da classe anotada | 0,74 | 0,70 |
| Brasileiro: pixels anotados na classe certa | 82% | 77% |
| Imagens sem comida (120 sintéticas): com detecção espúria | 0% | 3,3% |
| MyFoodRepo (294 fotos de celular): F1 de presença, macro | 0,12 | **0,56** |
| MyFoodRepo: falsos positivos por imagem | 1,04 | **0,75** |
| MyFoodRepo: pratos 100% corretos | 7,5% | **35%** |

F1 das categorias novas no teste do MyFoodRepo (o v2 não as conhece): Pão 0,80; Hambúrguer (sanduíche) 0,79; Lasanha 0,74; Brócolis 0,73; Batata cozida 0,64; Peixe 0,33; Cebola 0,17.

O v3 troca um pouco de acerto nos dois datasets antigos (Milho, Purê, Arroz e Salada no brasileiro foram as que mais caíram) por 7 categorias novas e por funcionar muito melhor em fotos variadas de celular. As classes mais fracas do v3 são Cebola, Peixe, Abóbora e Milho.

### v1 × v2 piloto × v2

Mesmo teste para os três, com a mesma regra de detecção do app. Fonte: `docs/METRICAS_V2_COMPLETO.json` (o relatório só do piloto continua em `docs/METRICAS_PILOTO_V2.json`).

| Métrica | v1 | v2 piloto | **v2 (ativo)** |
|---|---:|---:|---:|
| MyFood (222 fotos): F1 de presença | 0,35 | 0,89 | **0,94** |
| MyFood: falsos positivos por imagem | 2,84 | 0,19 | **0,10** |
| MyFood: imagens com algum falso positivo | 94% | 16% | **8%** |
| MyFood: pratos 100% corretos | 6% | 78% | **88%** |
| MyFood: fundo marcado como comida | 43% | 9% | **5%** |
| Brasileiro (78 fotos): IoU médio da classe anotada | 0,19 | 0,48 | **0,74** |
| Brasileiro: pixels anotados na classe certa | 35% | 65% | **82%** |
| Imagens sem comida (120 sintéticas): com detecção falsa | 89% | 0% | **0%** |

O v2 é melhor que o piloto em todas as métricas e em todas as classes. Destaques no conjunto brasileiro (IoU): Beterraba 0,06 → 0,64, Milho 0,03 → 0,44, Bife à milanesa 0,36 → 0,82, Tomate 0,49 → 0,85. Arroz, Salada e Massa com molho, que tinham piorado no piloto, voltaram ao nível do v1 ou acima.

**Ressalvas obrigatórias:**
- As fotos de teste vêm **dos próprios datasets** (MyFood e Brazilian Food Images). **Não há nenhuma avaliação com fotos reais de usuários.**
- No conjunto brasileiro, **todas as classes têm de 1 a 8 fotos de teste**, e no MyFood a Maçã tem 14. Nessas classes, uma foto muda muito o resultado.
- **As classes mais fracas do v2** são Milho (IoU 0,44) e Macarrão no conjunto brasileiro (0,60).
- Os limiares de Tomate (0,15) e Farofa (0,05) ficaram muito baixos, e vários outros ficaram em 0,90–0,95, nas pontas da faixa testada (0,05–0,95), em parte por falta de exemplos na validação.
- O IoU de 0,53 do v1 que aparece nos documentos antigos usava outra métrica, que ignorava falsos positivos. Ele não é comparável com esta tabela.

## 4. Como rodar do zero (PowerShell, Python 3.11+)

```powershell
git clone https://github.com/Ladstone/Projeto-IA.git
cd Projeto-IA
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
Abra http://127.0.0.1:8000.

**Testes:**
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```
O resultado esperado é **54 passam e 3 são pulados**. Os 3 pulados precisam dos ZIPs dos datasets (na pasta acima do repositório ou indicados por `PRATOMETRIA_DATASET_MYFOOD`, `PRATOMETRIA_DATASET_BRAZIL` e `PRATOMETRIA_DATASET_MYFOODREPO`); com eles, passam os 57.

## 5. Branches e como trocar de modelo

Só existe a branch **`main`**, no computador e no GitHub. As branches de trabalho (`fase-a-correcoes`, `frontend-redesign`, `fase-b-modelo`, `integracao`) já tinham sido incorporadas à `main` e foram apagadas em 17/11/2026.

**Para trocar de modelo** não é preciso copiar arquivos: use as variáveis de ambiente antes de subir o servidor.
```powershell
# padrão (sem variáveis): combinação v2 + categorias novas do v3 + purê do v3d
# só o v2 (backup)
$env:PRATOMETRIA_MODEL = "models/unified_21class_v2.pt"
# só o v3
$env:PRATOMETRIA_MODEL = "models/unified_28class_v3.pt"
# montar outra combinação
$env:PRATOMETRIA_MODEL = "models/unified_21class_v2.pt"
$env:PRATOMETRIA_MODEL_NOVAS = "models/unified_28class_v3.pt"
$env:PRATOMETRIA_MODEL_PURE = "models/unified_28class_v3d.pt"
```
Os pesos do v2 piloto e do v1 não estão mais no repositório (ver o histórico do git até o commit `c525370`).
Para só **testar** outro modelo, sem trocar o ativo: `$env:PRATOMETRIA_MODEL = "models/unified_21class_v2.pt"` antes de subir o app (de preferência em outra porta).
O app lê a arquitetura pelo `.json`, então não é preciso mudar código. Com outro modelo ativo, `tests/test_modelos.py` vai falhar, porque ele confere que o ativo é o v3 de 28 classes; com o v1, os testes de imagem marrom e de ruído também falham.

## 6. Como treinar de novo

**Datasets** (não estão no repositório):
- `MyFood Dataset.zip`: [MyFood Dataset](https://doi.org/10.5281/zenodo.4041488) (Zenodo, DOI 10.5281/zenodo.4041488). É o arquivo `MyFood Dataset.zip` do registro, idêntico ao usado aqui.
- `archive.zip` (MyFoodRepo v2.0, ~5,3 GB, só para o v3): [Kaggle](https://www.kaggle.com/datasets/sainikhileshreddy/food-recognition-2022), cópia do AIcrowd Food Recognition Benchmark 2022.
- `Brazilian food images.zip`: origem [brazilianfoodimage/dataset](https://github.com/brazilianfoodimage/dataset) (GitHub). A pasta `training_set` do ZIP usado é um clone desse repositório; a origem do `test_set` (160 fotos sem máscaras) não foi identificada.

Códigos de verificação (SHA-256) dos ZIPs usados: ver `docs/DATASETS.md`.

**Comandos:**
```powershell
.\.venv\Scripts\python.exe scripts/verify_datasets.py --myfood "<caminho>\MyFood Dataset.zip" --brazil "<caminho>\Brazilian food images.zip"
.\.venv\Scripts\python.exe scripts/train_v2.py --myfood "<caminho>\MyFood Dataset.zip" --brazil "<caminho>\Brazilian food images.zip" --epochs 20 --run-dir models/runs/v2
.\.venv\Scripts\python.exe scripts/export_v2.py --run-dir models/runs/v2 --out models/unified_21class_v2.pt
.\.venv\Scripts\python.exe scripts/evaluate_models.py --myfood "<...>" --brazil "<...>" --model v1=models/unified_21class_v1.pt --model piloto=models/unified_21class_v2_piloto.pt --model v2=models/unified_21class_v2.pt --out docs/METRICAS_V2_COMPLETO
# v3 (28 classes, com MyFoodRepo; ~3 h em CPU, feche programas pesados porque o treino usa vários GB de RAM)
.\.venv\Scripts\python.exe scripts/train_v2.py --myfood "<...>" --brazil "<...>" --myfoodrepo "<caminho>\archive.zip" --epochs 15 --run-dir models/runs/v3 --out models/unified_28class_v3.pt
.\.venv\Scripts\python.exe scripts/evaluate_models.py --myfood "<...>" --brazil "<...>" --myfoodrepo "<...>" --model v2=models/unified_21class_v2.pt --model v3=models/unified_28class_v3.pt --out docs/METRICAS_V3.json.json
```

**Tempo medido em CPU** (Ryzen 5 5500U, sem GPU): **cerca de 4,9 min por época**. O treino de 20 épocas levou **99 min**; a avaliação dos três modelos leva cerca de 1,5 min.

**Se o treino parar:** rode o mesmo comando, e ele retoma da última época concluída. Os checkpoints ficam em `models/runs/` (cerca de 300 MB), que não vai para o git.

**O `export_v2.py` não sobrescreve o modelo do app.** Trocar o modelo ativo é uma decisão da equipe, depois de comparar no mesmo teste. A troca para o v2 foi aprovada em 21/11/2026, e a troca para o v3 em 23/11/2026.

## 7. Próximas etapas e divisão sugerida

| Tarefa | Quem |
|---|---|
| Reforçar as classes fracas do v3 (Cebola, Peixe, Abóbora, Milho): teto maior no MyFoodRepo ou fotos próprias anotadas | IA + Dados |
| Avaliar novas categorias do MyFoodRepo que existem na TACO (espinafre, couve-flor, abobrinha, frutas, atum...) | IA |
| Montar um conjunto de **fotos reais** de pratos (com permissão) e anotá-lo para avaliar | Dados / Qualidade |
| Revisar os limiares que ficaram nas pontas (Tomate 0,15; Farofa 0,05; vários em 0,90–0,95) | IA |
| Conferir licenças e montar o roteiro da apresentação (seção 8) | Qualidade + Apresentação |

## 8. Avisos para a apresentação

**Não afirmar:**
- que a IA acerta qualquer refeição;
- que a foto mede gramas: a porção é uma estimativa pela área e deve ser corrigida;
- que a rede reconhece 597 alimentos: são 28 categorias visuais, e a TACO entra depois;
- que as métricas valem para fotos reais;
- que as recomendações são orientação clínica.

**Licenças:**
- **Fotos de exemplo:** os botões de exemplo foram removidos da página em 22/11/2026. As fotos eram imagens do conjunto de teste do MyFood (CC BY 4.0); cinco delas seguem em `tests/fixtures/`, usadas só pelos testes, com crédito em `tests/fixtures/CREDITOS.md`.
- **MyFood:** **CC BY 4.0** (verificado no Zenodo em 21/11/2026). Permite redistribuir com crédito. Citar: Freitas, C. N. C.; Cordeiro, F. R.; Macario, V. *MyFood: A Food Segmentation and Classification System to Aid Nutritional Monitoring*. SIBGRAPI 2020 ([arXiv:2012.03087](https://arxiv.org/abs/2012.03087)); dataset em [MyFood Dataset](https://doi.org/10.5281/zenodo.4041488) (Zenodo, DOI 10.5281/zenodo.4041488).
- **Brazilian Food Images:** o repositório de origem, [brazilianfoodimage/dataset](https://github.com/brazilianfoodimage/dataset) (GitHub), **não declara licença** nem autor (verificado em 21/11/2026). A informação anterior de que seria CC BY 4.0 não se confirmou. Sem licença, vale "todos os direitos reservados": usar para estudo é razoável, mas **não republicar** o dataset sem autorização do dono. Citar o repositório na apresentação.
- **TACO:** citar NEPA/UNICAMP, 4ª ed., 2011.
