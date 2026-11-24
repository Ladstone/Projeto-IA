# PratoMetria — protótipo acadêmico de IA nutricional

Protótipo executável de **interface + módulo de inteligência artificial + módulo de resposta**, pensado para entrega universitária e continuação por uma equipe. Não usa serviços externos ou chave de API.

> **Estado honesto:** o frontend e o backend funcionam. O app usa o modelo **v2** (21 categorias, dois datasets de imagens). Existe também um **v3** experimental com **28 categorias** e um terceiro dataset (MyFoodRepo), que reconhece brócolis, batata cozida, peixe, pão, cebola, hambúrguer e lasanha, mas **não é o ativo**: Em fotos reais de prato feito tiradas pela equipe, o v3 confundiu **batata frita com macarrão** (`prato feito 2`) e **bife com frango** (almoço de 23/11), erros que o v2 não comete. **A identificação continua experimental:**
> - **Milho é a classe mais fraca** do v2 (IoU de 0,44 no teste).
> - As métricas usam fotos dos próprios datasets; os testes com fotos reais ainda são poucos e informais.
>
> O projeto não deve ser apresentado como aplicativo nutricional preciso ou finalizado.

## O que já funciona

1. Enviar uma foto de refeição e executar **segmentação local** com uma rede PyTorch de 21 categorias visuais. O modelo ativo é o v2 (MobileNetV3 + LR-ASPP, 20 épocas); o v3 de 28 categorias, o v2 piloto e o v1 (U-Net) ficam como alternativas.
2. Ver regiões sugeridas, editar/retirar alimentos e acrescentar qualquer um dos **597 registros da TACO**.
3. Informar os gramas da porção (a foto **não mede peso**), consultar os **26 campos da Tabela 1 da TACO** e recalcular os totais.
4. Configurar metas pessoais de kcal/proteína/carboidratos/lipídios, comparar a refeição com essas metas e obter sugestões baseadas em regras.
5. Abrir **Diagnóstico do modelo** na interface para conferir regiões, scores e limiares calculados para cada fotografia.

**Não há lista fixa de itens retornados:** o backend calcula uma máscara nova para cada foto. Isso **não impede** que a rede erre repetidamente nas mesmas classes, pois o treinamento atual é limitado.

## Executar no Windows (PowerShell)

Pré-requisito: Python 3.11+ e internet **somente para instalar as dependências**. Entre na pasta `PratoMetria` (aquela que contém `app/`, `models/` e `requirements.txt`) e execute:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Abra **http://127.0.0.1:8000**. Para encerrar, `Ctrl + C`. Nenhum `.bat`, ativação de scripts ou chave externa é necessário. Se a porta 8000 estiver ocupada, tente `--port 8001` e abra `http://127.0.0.1:8001`.

Para editar com recarregamento automático, substitua a última linha por:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### Testar

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Testes que reabrem os ZIPs originais são ignorados se os datasets não estiverem localmente disponíveis. O modelo **já treinado** funciona **sem esses ZIPs**.

## Datasets de treinamento

- **MyFood Dataset.zip**: 750 imagens treino, 250 validação, 250 teste, com máscaras multiclasse. Fonte: [MyFood Dataset](https://doi.org/10.5281/zenodo.4041488) (Zenodo, DOI 10.5281/zenodo.4041488), licença CC BY 4.0.
- **Brazilian food images.zip**: 681 pares imagem/máscara em treino e 234 em validação, com 20 categorias; o `test_set` original tem 160 imagens sem máscaras. Fonte: [brazilianfoodimage/dataset](https://github.com/brazilianfoodimage/dataset) (GitHub), sem licença declarada.
- **archive.zip (MyFoodRepo v2.0)**: 39.962 fotos de refeições reais tiradas por usuários de um app suíço, com 498 classes anotadas por polígono (formato COCO). Fonte: AIcrowd Food Recognition Benchmark 2022, cópia do [Kaggle](https://www.kaggle.com/datasets/sainikhileshreddy/food-recognition-2022) (~5 GB). Só entra uma amostra: até 150 fotos por classe nova e 50 de reforço por classe antiga (1.491 fotos no treino).
- Os três conjuntos foram harmonizados em **28 categorias visuais** (21 dos dois primeiros + 7 do MyFoodRepo) e usados no treino dos pesos incluídos em `models/unified_21class.pt`.
- A tabela TACO é uma **base nutricional**, não um conjunto de treinamento de imagens.

Os ZIPs originais somam quase 6 GB e **não estão no repositório**. Para treinar outra vez, baixe-os nas fontes acima (ou peça à equipe); **não** são necessários para executar a demo. Consulte [Documentação dos datasets](docs/DATASETS.md) e [Avaliação e limitações](docs/AVALIACAO.md).

## Reproduzir/continuar treinamento

Com os dois ZIPs disponíveis, confira primeiro se eles estão íntegros:

```powershell
.\.venv\Scripts\python.exe scripts/verify_datasets.py --myfood "C:\caminho\MyFood Dataset.zip" --brazil "C:\caminho\Brazilian food images.zip"
```

Os comandos de treino e avaliação do v3 estão em **[A fazer](#a-fazer)**, item 1. O treino **não** substitui o modelo do app: ele grava em `models/runs/v3/` e no arquivo passado em `--out`. Uma nova execução não necessariamente reproduz o mesmo resultado. O script do v1 (`scripts/train_unified.py`) continua no repositório só como referência; ele **sobrescreve** `models/unified_21class.pt`, então não o use sem fazer uma cópia.

## Estrutura do código

```text
PratoMetria/
├── app/
│   ├── main.py               API FastAPI, upload, inferência, metas (SQLite local)
│   ├── vision.py             modelos v1 (U-Net) e v2/v3 (MobileNetV3 + LR-ASPP) e predição por foto
│   ├── unified_classes.py    28 categorias visuais, mapas dos datasets e hipóteses TACO
│   ├── nutrition.py          597 alimentos, porções, metas e sugestões
│   ├── static/               HTML, CSS e JavaScript
│   └── data/                 TACO CSV/JSON/PDF; SQLite local criado no primeiro uso
├── models/                   pesos treinados e metadados de validação
├── scripts/                  leitura e treinamento com os três datasets
├── tests/                    testes sem e com ZIPs originais
├── docs/                     handoff para equipe, arquitetura e limitações
├── requirements.txt
└── README.md
```

Endpoints principais: `GET /api/health`, `GET /api/model/info`, `POST /api/crop-suggestion`, `POST /api/analyze`, `POST /api/recalculate`, `GET /api/foods`, `GET /api/taco/{id}`, `GET/PUT /api/profile`, `GET /api/profile/opcoes`, `POST /api/profile/calcular`; documentação interativa em **http://127.0.0.1:8000/docs**.

## Próximas etapas

Situação em 23/11/2026. O app usa o **v2** (`models/unified_21class.pt`, cópia de `models/unified_21class_v2.pt`). O **v3** de 28 classes (`models/unified_28class_v3.(pt|json)`) foi ativo por algumas horas e voltou a ser alternativo: Em fotos reais de prato feito tiradas pela equipe, o v3 confundiu **batata frita com macarrão** (`prato feito 2`) e **bife com frango** (almoço de 23/11), erros que o v2 não comete. Ficam também o **v2 piloto** e o **v1**. Para voltar a um deles, copie os dois arquivos para `models/unified_21class.*`; o app aceita modelos de 21 ou 28 classes.

Para testar outro modelo sem trocar o ativo, use a variável `PRATOMETRIA_MODEL`:

```powershell
$env:PRATOMETRIA_MODEL = "models/unified_28class_v3.pt"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

Assim dá para mostrar o v2 em `http://127.0.0.1:8000` e o v3 em `http://127.0.0.1:8001` ao mesmo tempo.

### Combinação v2+v3 (em teste na porta 8001)

O v2 decide os pratos comuns e o v3 só entra quando uma das 7 categorias novas cobre 10% ou mais do prato (então a foto inteira usa o v3). No mini teste com fotos reais foi o melhor resultado até agora, e nas fotos brasileiras do teste fica igual ao v2. Números em `docs/AVALIACAO.md`. Ainda precisa ser testado com fotos novas antes de virar o modelo do app.

```powershell
$env:PRATOMETRIA_MODEL = "models/unified_21class_v2.pt"
$env:PRATOMETRIA_MODEL_NOVAS = "models/unified_28class_v3.pt"
$env:PRATOMETRIA_MODEL_PURE = "models/unified_28class_v3d.pt"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

`PRATOMETRIA_MODEL_PURE` (opcional) põe o **purê do v3d** por cima da combinação: o v2 chamava purê de arroz ou frango e via purê em só 1% das fotos de teste de purê; com o v3d, 90%. O v3d foi treinado com 300 fotos de purê do dataset "mashed potato" do images.cv, rotuladas automaticamente por `scripts/pseudo_pure.py` (ver `docs/AVALIACAO.md`). Nos scripts de avaliação: `base.pt+novas.pt|pure.pt`.

Os scripts de avaliação aceitam a combinação como `base.pt+novas.pt`, por exemplo `--model v2+v3=models/unified_21class_v2.pt+models/unified_28class_v3.pt`.

### Modelo v3 (alternativo)

Mesma rede do v2 (MobileNetV3-Large pré-treinada no ImageNet + LR-ASPP, 224 px, supervisão negativa por pixel e imagens sem comida), agora com 28 saídas. Categorias novas: **Brócolis, Batata cozida, Peixe, Pão, Cebola, Hambúrguer (sanduíche) e Lasanha**; as duas últimas entram na TACO como receita (vários ingredientes). Coxa de frango do MyFoodRepo foi somada a Frango. Treino de 15 épocas em CPU (~3 h, 12 min por época); a melhor época foi a 15 (F1 de presença 0,69 na validação, que agora inclui as fotos difíceis do MyFoodRepo). Época e limiares foram escolhidos **só na validação**.

Avaliação no conjunto de teste, com a mesma regra de detecção do app. Relatório completo em [`docs/METRICAS_V3.json`](docs/METRICAS_V3.json); o do v2 continua em [`docs/METRICAS_V2_COMPLETO.json`](docs/METRICAS_V2_COMPLETO.json).

| Teste | v1 | v2 | v3 |
|---|---:|---:|---:|
| MyFood (222 fotos): F1 de presença, macro de 8 classes | 0,35 | 0,94 | 0,93 |
| MyFood: falsos positivos por imagem | 2,84 | 0,10 | 0,11 |
| MyFood: pratos 100% corretos | 6% | 88% | 86% |
| MyFood: fundo marcado como comida | 43% | 5% | 3,5% |
| Brasileiro (78 fotos): pixels anotados na classe certa | 35% | 82% | 77% |
| Imagens sem comida (120 sintéticas): com detecção espúria | 89% | 0% | 3,3% |
| MyFoodRepo (294 fotos de celular): F1 de presença, macro | — | 0,12 | 0,56 |
| MyFoodRepo: pratos 100% corretos | — | 7,5% | 35% |

F1 das categorias novas no teste do MyFoodRepo: Pão 0,80; Hambúrguer 0,79; Lasanha 0,74; Brócolis 0,73; Batata cozida 0,64; Peixe 0,33; Cebola 0,17.

### Limitações conhecidas do v3

- **Regressões em fotos reais de prato feito:** batata frita vista como Macarrão e bife como Frango. O MyFoodRepo tem muita massa e frango assado, e o modelo puxou para essas classes. Por isso o v3 não é o ativo.
- **Classes fracas:** Cebola (F1 0,17) e Peixe (0,33) entre as novas; Abóbora (0,18 no MyFoodRepo) e Milho (IoU 0,26 no brasileiro, era 0,44 no v2) entre as antigas.
- **Pequena perda nos testes antigos:** no brasileiro, os pixels na classe certa caíram de 82% para 77%; Purê, Arroz e Salada foram as que mais perderam.
- **Detecção espúria em imagens sem comida:** 3,3% das imagens sintéticas (o v2 tinha 0%).
- **Não há avaliação com fotos reais de usuários.** As fotos do MyFoodRepo são as mais próximas disso, mas vêm da mesma fonte do treino.
- No conjunto brasileiro, todas as classes têm de 1 a 8 fotos de teste, então os números por classe são frágeis.

### A fazer

1. **Retreinar o v3**, se necessário (~3 h em CPU, Ryzen 5 5500U; feche programas pesados, porque o treino usa vários GB de RAM):
   ```powershell
   .\.venv\Scripts\python.exe scripts/train_v2.py --myfood "C:\caminho\MyFood Dataset.zip" --brazil "C:\caminho\Brazilian food images.zip" --myfoodrepo "C:\caminho\archive.zip" --epochs 15 --run-dir models/runs/v3 --out models/unified_28class_v3.pt
   .\.venv\Scripts\python.exe scripts/evaluate_models.py --myfood ... --brazil ... --myfoodrepo ... --model v2=models/unified_21class_v2.pt --model v3=models/unified_28class_v3.pt --out docs/METRICAS_V3.json
   ```
   O treino grava checkpoint a cada época e log em `models/runs/v3/train.log`. Se for interrompido, rode **o mesmo comando** para retomar. Trocar o modelo ativo exige aprovação explícita da equipe, depois de comparar no mesmo teste.
2. **Reforçar classes fracas** (Cebola, Peixe, Abóbora, Milho): aumentar o teto de fotos do MyFoodRepo para elas ou anotar fotos próprias.
3. **Novas categorias:** o MyFoodRepo tem outras comidas que existem na TACO (espinafre, couve-flor, abobrinha, frutas, atum, camarão...). O mapa fica em `MYFOODREPO_LABELS`, em `app/unified_classes.py`.
4. **Avaliar com fotos reais** de pratos, tiradas com permissão e anotadas, que não venham dos datasets.

## Antes de entregar ao professor

Leia [Guia da equipe](docs/GUIA_EQUIPE.md). **Não** envie `.venv`, dados pessoais, banco SQLite gerado, chaves de API nem logs. O arquivo de pesos `.pt` e os 597 registros da TACO já estão incluídos. A equipe deve conferir licenças/atribuições dos datasets e das imagens de demonstração antes de publicar o repositório publicamente.

Fonte nutricional: NEPA/UNICAMP, *Tabela Brasileira de Composição de Alimentos (TACO)*, 4ª ed. rev. e ampl., 2011. Os registros e o PDF preservam essa atribuição.
