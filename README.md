# PratoMetria — protótipo acadêmico de IA nutricional

Protótipo executável de **interface + módulo de inteligência artificial + módulo de resposta**, pensado para entrega universitária e continuação por uma equipe. Não usa serviços externos ou chave de API.

> **Estado honesto:** o frontend e backend funcionam; o modelo foi treinado com **dois datasets de imagens**, mas **a identificação ainda é experimental e frequentemente incorreta** em fotos novas. O projeto não deve ser apresentado como aplicativo nutricional preciso ou finalizado.

## O que já funciona

1. Enviar uma foto de refeição e executar **segmentação local** com uma rede U-Net/PyTorch de 21 categorias visuais.
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
│   ├── vision.py             U-Net e predição por foto
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

## Antes de entregar ao professor

Leia [Guia da equipe](docs/GUIA_EQUIPE.md). **Não** envie `.venv`, dados pessoais, banco SQLite gerado, chaves de API nem logs. O arquivo de pesos `.pt` e os 597 registros da TACO já estão incluídos. A equipe deve conferir licenças/atribuições dos datasets e das imagens de demonstração antes de publicar o repositório publicamente.

Fonte nutricional: NEPA/UNICAMP, *Tabela Brasileira de Composição de Alimentos (TACO)*, 4ª ed. rev. e ampl., 2011. Os registros e o PDF preservam essa atribuição.
