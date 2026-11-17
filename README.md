# PratoMetria — protótipo acadêmico de IA nutricional

Protótipo executável de **interface + módulo de inteligência artificial + módulo de resposta**, pensado para entrega universitária e continuação por uma equipe. Não usa serviços externos ou chave de API.

> **Estado honesto:** o frontend e o backend funcionam. O app usa o modelo **v2 preliminar**, um piloto de 3 épocas treinado com **dois datasets de imagens**. No conjunto de teste ele é bem melhor que o v1, mas **a identificação continua experimental**:
> - **Milho e Beterraba ainda não funcionam** (IoU de 0,03 e 0,06 no teste).
> - **Não há avaliação com fotos reais de usuários.** Todas as métricas usam fotos dos próprios datasets.
>
> O projeto não deve ser apresentado como aplicativo nutricional preciso ou finalizado.

## O que já funciona

1. Enviar uma foto de refeição e executar **segmentação local** com uma rede PyTorch de 21 categorias visuais. O modelo ativo é o v2 preliminar (MobileNetV3 + LR-ASPP); o v1 (U-Net) fica como backup.
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

Com os dois ZIPs disponíveis:

```powershell
.\.venv\Scripts\python.exe scripts/verify_datasets.py --myfood "C:\caminho\MyFood Dataset.zip" --brazil "C:\caminho\Brazilian food images.zip"
.\.venv\Scripts\python.exe scripts/train_unified.py --myfood "C:\caminho\MyFood Dataset.zip" --brazil "C:\caminho\Brazilian food images.zip" --epochs 9 --dimension 160
```

A segunda linha **substitui o checkpoint** `models/unified_21class.pt`; faça uma cópia antes de treinar. O peso que acompanha este ZIP foi escolhido por validação em um treino de 5 épocas, não em 9 épocas completas. Uma nova execução não necessariamente reproduz o mesmo resultado.

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

Endpoints principais: `GET /api/health`, `GET /api/model/info`, `POST /api/analyze`, `POST /api/recalculate`, `GET /api/foods`, `GET /api/taco/{id}`, `GET/PUT /api/profile`; documentação interativa em **http://127.0.0.1:8000/docs**.

## Próximas etapas

Situação em 16/11/2026. Na branch `fase-b-modelo`, o app usa o **v2 preliminar** (`models/unified_21class.pt`, cópia de `models/unified_21class_v2_piloto.pt`). O v1 (U-Net de 160 px) está guardado em `models/unified_21class_v1.(pt|json)`. Para voltar a ele, copie esses dois arquivos para `models/unified_21class.*`.

### Modelo v2 preliminar

É um **piloto de 3 épocas**: MobileNetV3-Large pré-treinada no ImageNet + LR-ASPP, 224 px, supervisão negativa por pixel e imagens sem comida. Época e limiares foram escolhidos **só na validação**.

### Limitações conhecidas do v2 preliminar

- **Milho e Beterraba ainda não funcionam.** O IoU no teste foi de 0,03 e 0,06, com 1 e 2 fotos de teste.
- **Não há avaliação com fotos reais de usuários.** Todas as métricas usam fotos dos datasets MyFood e Brazilian Food Images, que vêm das mesmas fontes do treino.
- No conjunto brasileiro, todas as classes têm de 1 a 8 fotos de teste, então os números por classe são frágeis. Arroz e Salada pioraram um pouco nesse conjunto em relação ao v1.
- É um piloto curto: as métricas de validação ainda subiam na última época.

Avaliação única no conjunto de teste, com a mesma regra de detecção do app. Quase duplicatas do treino foram removidas do teste. Relatório completo em [`docs/METRICAS_PILOTO_V2.json`](docs/METRICAS_PILOTO_V2.json).

| Teste | v1 | v2 piloto |
|---|---:|---:|
| MyFood (222 fotos): F1 de presença, macro de 8 classes | 0,35 | 0,89 |
| MyFood: falsos positivos por imagem | 2,84 | 0,19 |
| MyFood: pratos 100% corretos | 6% | 78% |
| MyFood: fundo marcado como comida | 43% | 9% |
| Brasileiro (78 fotos): pixels anotados na classe certa | 35% | 65% |
| Imagens sem comida (120 sintéticas): com detecção espúria | 89% | 0% |

Cuidados ao interpretar:
- No conjunto brasileiro, **todas as classes têm 1 a 8 fotos de teste**: os números por classe são indicativos, não conclusivos.
- As fotos de teste vêm das mesmas fontes do treino. **Não há ainda avaliação com fotos reais de usuários.**
- O IoU de 0,53 relatado antes para o v1 usava outra métrica. Ela avaliava só a classe anotada de cada foto e ignorava falsos positivos de outras classes, então não é comparável com a tabela acima.

### A fazer

1. **Treino completo do v2** (planejado para a semana de 23/11/2026). São ~20 épocas a ~6,3 min cada, cerca de **2 h 10 min** em CPU (Ryzen 5 5500U):
   ```powershell
   .\.venv\Scripts\python.exe scripts/train_v2.py --myfood "C:\caminho\MyFood Dataset.zip" --brazil "C:\caminho\Brazilian food images.zip" --epochs 20 --run-dir models/runs/v2
   .\.venv\Scripts\python.exe scripts/export_v2.py --run-dir models/runs/v2 --out models/unified_21class_v2.pt
   .\.venv\Scripts\python.exe scripts/evaluate_models.py --myfood ... --brazil ... --model v1=models/unified_21class_v1.pt --model v2=models/unified_21class_v2.pt
   ```
   O treino grava checkpoint a cada época e log em `models/runs/v2/train.log`. Se for interrompido, rode **o mesmo comando** para retomar.
2. **Decidir se o v2 completo substitui o v2 preliminar.** Isso exige aprovação explícita: compare os dois com `evaluate_models.py` no mesmo teste antes de trocar.
3. **Fazer Milho e Beterraba funcionarem e reforçar classes com poucos exemplos** (Salada com maionese, Mandioca, Cenoura, Batata frita, entre outras). Elas precisam de mais fotos anotadas. Alguns limiares escolhidos na validação ficaram nos extremos (Tomate 0,15; Farofa 0,20; Maçã 0,95) por falta de exemplos.
4. **Avaliar com fotos reais** de pratos, tiradas com permissão e anotadas, que não venham dos datasets.
5. **Frontend novo** (branch `frontend-redesign`): verificado com 20 checagens automáticas e prints. Falta revisão e merge.
6. **Pequenos ajustes de texto:** falta um ponto antes de "Porção inicial estimada…" na nota de cada alimento (`app/main.py`).

## Antes de entregar ao professor

Leia [Guia da equipe](docs/GUIA_EQUIPE.md). **Não** envie `.venv`, dados pessoais, banco SQLite gerado, chaves de API nem logs. O arquivo de pesos `.pt` e os 597 registros da TACO já estão incluídos. A equipe deve conferir licenças/atribuições dos datasets e das imagens de demonstração antes de publicar o repositório publicamente.

Fonte nutricional: NEPA/UNICAMP, *Tabela Brasileira de Composição de Alimentos (TACO)*, 4ª ed. rev. e ampl., 2011. Os registros e o PDF preservam essa atribuição.
