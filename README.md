# PratoMetria — protótipo acadêmico de IA nutricional

Protótipo executável de **interface + módulo de inteligência artificial + módulo de resposta**, pensado para entrega universitária e continuação por uma equipe. Não usa serviços externos ou chave de API.

> **Estado honesto:** o frontend e o backend funcionam. O app usa o modelo **v2**, treinado por 20 épocas com **dois datasets de imagens**. No conjunto de teste ele é bem melhor que o v1 e que o piloto, mas **a identificação continua experimental**:
> - **Milho é a classe mais fraca** (IoU de 0,44 no teste, com poucas fotos).
> - **Não há avaliação com fotos reais de usuários.** Todas as métricas usam fotos dos próprios datasets.
>
> O projeto não deve ser apresentado como aplicativo nutricional preciso ou finalizado.

## O que já funciona

1. Enviar uma foto de refeição e executar **segmentação local** com uma rede PyTorch de 21 categorias visuais. O modelo ativo é o v2 (MobileNetV3 + LR-ASPP, 20 épocas); o v2 piloto e o v1 (U-Net) ficam como backup.
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

- **MyFood Dataset.zip**: 750 imagens treino, 250 validação, 250 teste, com máscaras multiclasse.
- **Brazilian food images.zip**: 681 pares imagem/máscara em treino e 234 em validação, com 20 categorias; o `test_set` original tem 160 imagens sem máscaras.
- Os dois conjuntos foram harmonizados em **21 categorias visuais** e usados no treino dos pesos incluídos em `models/unified_21class.pt`.
- A tabela TACO é uma **base nutricional**, não um conjunto de treinamento de imagens.

Os dois ZIPs originais somam centenas de MB e **não estão no repositório**. É necessário obtê-los separadamente com a equipe para treinar outra vez; **não** são necessários para executar a demo. Consulte [Documentação dos datasets](docs/DATASETS.md) e [Avaliação e limitações](docs/AVALIACAO.md).

## Reproduzir/continuar treinamento

Com os dois ZIPs disponíveis, confira primeiro se eles estão íntegros:

```powershell
.\.venv\Scripts\python.exe scripts/verify_datasets.py --myfood "C:\caminho\MyFood Dataset.zip" --brazil "C:\caminho\Brazilian food images.zip"
```

Os comandos de treino, exportação e avaliação do v2 estão em **[A fazer](#a-fazer)**, item 1. O treino **não** substitui o modelo do app: ele grava em `models/runs/v2/`, e `export_v2.py` gera `models/unified_21class_v2.pt`. Uma nova execução não necessariamente reproduz o mesmo resultado. O script do v1 (`scripts/train_unified.py`) continua no repositório só como referência; ele **sobrescreve** `models/unified_21class.pt`, então não o use sem fazer uma cópia.

## Estrutura do código

```text
PratoMetria/
├── app/
│   ├── main.py               API FastAPI, upload, inferência, metas (SQLite local)
│   ├── vision.py             modelos v1 (U-Net) e v2 (MobileNetV3 + LR-ASPP) e predição por foto
│   ├── unified_classes.py    21 categorias visuais e hipóteses de mapeamento TACO
│   ├── nutrition.py          597 alimentos, porções, metas e sugestões
│   ├── static/               HTML, CSS, JavaScript e exemplos
│   └── data/                 TACO CSV/JSON/PDF; SQLite local criado no primeiro uso
├── models/                   pesos treinados e metadados de validação
├── scripts/                  leitura e treinamento com os dois datasets
├── tests/                    testes sem e com ZIPs originais
├── docs/                     handoff para equipe, arquitetura e limitações
├── requirements.txt
└── README.md
```

Endpoints principais: `GET /api/health`, `GET /api/model/info`, `POST /api/crop-suggestion`, `POST /api/analyze`, `POST /api/recalculate`, `GET /api/foods`, `GET /api/taco/{id}`, `GET/PUT /api/profile`; documentação interativa em **http://127.0.0.1:8000/docs**.

## Próximas etapas

Situação em 21/11/2026. O app usa o **v2** (`models/unified_21class.pt`, cópia de `models/unified_21class_v2.pt`), treinado por 20 épocas. Ficam como backup o **v2 piloto** de 3 épocas (`models/unified_21class_v2_piloto.(pt|json)`) e o **v1**, U-Net de 160 px (`models/unified_21class_v1.(pt|json)`). Para voltar a um deles, copie os dois arquivos para `models/unified_21class.*`.

### Modelo v2

MobileNetV3-Large pré-treinada no ImageNet + LR-ASPP, 224 px, supervisão negativa por pixel e imagens sem comida. Treino de 20 épocas em CPU (99 min); a melhor época foi a 16 (F1 de presença 0,97 na validação). Época e limiares foram escolhidos **só na validação**.

### Limitações conhecidas do v2

- **Milho é a classe mais fraca** (IoU 0,44 no teste); Macarrão no conjunto brasileiro fica em 0,60.
- **Não há avaliação com fotos reais de usuários.** Todas as métricas usam fotos dos datasets MyFood e Brazilian Food Images, que vêm das mesmas fontes do treino.
- No conjunto brasileiro, todas as classes têm de 1 a 8 fotos de teste, então os números por classe são frágeis.

Avaliação única no conjunto de teste, com a mesma regra de detecção do app. Quase duplicatas do treino foram removidas do teste. Relatório completo em [`docs/METRICAS_V2_COMPLETO.json`](docs/METRICAS_V2_COMPLETO.json) (o do piloto continua em [`docs/METRICAS_PILOTO_V2.json`](docs/METRICAS_PILOTO_V2.json)).

| Teste | v1 | v2 piloto | v2 |
|---|---:|---:|---:|
| MyFood (222 fotos): F1 de presença, macro de 8 classes | 0,35 | 0,89 | 0,94 |
| MyFood: falsos positivos por imagem | 2,84 | 0,19 | 0,10 |
| MyFood: pratos 100% corretos | 6% | 78% | 88% |
| MyFood: fundo marcado como comida | 43% | 9% | 5% |
| Brasileiro (78 fotos): pixels anotados na classe certa | 35% | 65% | 82% |
| Imagens sem comida (120 sintéticas): com detecção espúria | 89% | 0% | 0% |

Cuidados ao interpretar:
- No conjunto brasileiro, **todas as classes têm 1 a 8 fotos de teste**: os números por classe são indicativos, não conclusivos.
- As fotos de teste vêm das mesmas fontes do treino. **Não há ainda avaliação com fotos reais de usuários.**
- O IoU de 0,53 relatado antes para o v1 usava outra métrica. Ela avaliava só a classe anotada de cada foto e ignorava falsos positivos de outras classes, então não é comparável com a tabela acima.

### A fazer

1. **Retreinar o v2**, se necessário (~100 min em CPU, Ryzen 5 5500U):
   ```powershell
   .\.venv\Scripts\python.exe scripts/train_v2.py --myfood "C:\caminho\MyFood Dataset.zip" --brazil "C:\caminho\Brazilian food images.zip" --epochs 20 --run-dir models/runs/v2
   .\.venv\Scripts\python.exe scripts/export_v2.py --run-dir models/runs/v2 --out models/unified_21class_v2.pt
   .\.venv\Scripts\python.exe scripts/evaluate_models.py --myfood ... --brazil ... --model v1=models/unified_21class_v1.pt --model v2=models/unified_21class_v2.pt
   ```
   O treino grava checkpoint a cada época e log em `models/runs/v2/train.log`. Se for interrompido, rode **o mesmo comando** para retomar. Trocar o modelo ativo exige aprovação explícita da equipe, depois de comparar no mesmo teste.
2. **Reforçar classes com poucos exemplos** (Milho, Salada com maionese, Mandioca, Cenoura, Batata frita, entre outras). Elas precisam de mais fotos anotadas.
3. **Revisar limiares extremos.** No v2, os limiares escolhidos na validação ficaram muito baixos para Tomate (0,15) e Farofa (0,05), e em 0,90–0,95 para várias outras classes, em parte por falta de exemplos.
4. **Avaliar com fotos reais** de pratos, tiradas com permissão e anotadas, que não venham dos datasets.

## Antes de entregar ao professor

Leia [Guia da equipe](docs/GUIA_EQUIPE.md). **Não** envie `.venv`, dados pessoais, banco SQLite gerado, chaves de API nem logs. O arquivo de pesos `.pt` e os 597 registros da TACO já estão incluídos. A equipe deve conferir licenças/atribuições dos datasets e das imagens de demonstração antes de publicar o repositório publicamente.

Fonte nutricional: NEPA/UNICAMP, *Tabela Brasileira de Composição de Alimentos (TACO)*, 4ª ed. rev. e ampl., 2011. Os registros e o PDF preservam essa atribuição.
