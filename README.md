# PratoMetria

Aplicativo acadêmico que analisa a **foto de uma refeição**: uma rede neural identifica os alimentos e onde eles estão no prato, a **Tabela TACO** calcula os nutrientes, e o app compara o resultado com as metas da pessoa. Projeto da disciplina de Inteligência Artificial, com os três módulos do enunciado: **entrada** (foto e recorte), **IA** (segmentação) e **saída** (nutrientes, metas, histórico e feedback).

## O que o app faz

1. **Analisa a foto no próprio computador.** Sugere um recorte em volta do prato e segmenta a imagem em **28 categorias** de alimentos (arroz, feijão, frango, carne, salada, purê, macarrão, lasanha, hambúrguer…).
2. **Estima as gramas** pela área que cada alimento ocupa no prato. A foto não mede peso: é um ponto de partida que a pessoa corrige.
3. **Calcula os nutrientes pela TACO** (597 alimentos, 26 campos). Dá para trocar qualquer alimento, ajustar gramas e adicionar itens pela busca.
4. **Compara com as metas** de calorias, proteína, carboidrato e gordura. As metas podem ser calculadas pelas medidas da pessoa (fórmula de Mifflin-St Jeor + objetivo).
5. **Guarda o histórico:** cada prato analisado fica salvo e aparece num calendário do mês, com as fotos.
6. **Escreve um feedback** sobre cada prato e sobre o mês (opcional, usa a Gemini API gratuita). O app calcula todos os números; a IA de texto só redige.

## Como rodar (Windows, PowerShell)

Pré-requisito: Python 3.11 ou mais novo. Na pasta do projeto:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Abra **http://127.0.0.1:8000**. Para encerrar, `Ctrl + C`.

### Feedback por IA (opcional)

Copie `.env.example` para `.env` e cole em `GEMINI_API_KEY` uma chave gratuita de https://aistudio.google.com ("Get API key"). O `.env` não vai para o git.

- Só a lista de alimentos, gramas, totais e metas é enviada. **A foto não sai do computador.**
- Sem chave, o app funciona normalmente e mostra dicas calculadas por regra.
- O plano gratuito às vezes está ocupado: o app tenta outros modelos e, se nenhum responder, mostra as dicas por regra e um botão "Tentar de novo".

### Testes

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Os testes não chamam a Gemini API. Os que leem os datasets originais são pulados quando os ZIPs não estão no computador.

## O modelo de IA

A rede é uma **MobileNetV3-Large + LR-ASPP** (segmentação: classifica cada pixel), treinada em PyTorch, em CPU. O app usa uma **combinação de três modelos**, porque cada um é melhor numa coisa:

| Modelo | Arquivo | Papel na combinação |
|---|---|---|
| v2 (21 categorias) | `models/unified_21class.pt` | base: decide o prato feito comum |
| v3 (28 categorias) | `models/unified_28class_v3.pt` | traz 7 categorias novas (brócolis, batata cozida, peixe, pão, cebola, hambúrguer e lasanha) e o tomate e a batata frita, que o v2 perde no prato misto |
| v3d (28 categorias) | `models/unified_28class_v3d.pt` | só o purê, que o v2 confundia com arroz |

`models/unified_21class_v2.pt` é uma cópia do v2, guardada como **backup**. Para rodar só o v2:

```powershell
$env:PRATOMETRIA_MODEL = "models/unified_21class_v2.pt"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

Resultados resumidos (detalhes em [docs/RESULTADOS.md](docs/RESULTADOS.md)):

| Medida | v1 | v2 | Combinação |
|---|---:|---:|---:|
| MyFood: pratos 100% corretos no teste | 6% | 88% | — |
| Fotos reais da equipe, pratos fora do comum: alimentos achados | — | 1/8 | 5/8 |
| Fotos de purê em que o purê foi achado | — | 1% | 90% |
| Teste independente (21 fotos novas): alimentos achados | — | 70% | 85% |

## Limitações

- **A identificação erra.** As categorias mais fracas são milho, cebola e peixe. O app sempre deixa a pessoa corrigir.
- **A foto não mede peso**, óleo, molho nem modo de preparo.
- **Os testes com fotos reais são pequenos.** O teste independente tem 21 fotos, de Pinterest; fotos de celular no dia a dia tendem a ser mais difíceis.
- **Não é orientação nutricional.** As metas e os feedbacks são estimativas educacionais e não substituem um nutricionista.
- **Um usuário só.** O histórico e as metas ficam num banco local (SQLite), sem login.

## Datasets

| Dataset | Uso | Fonte |
|---|---|---|
| MyFood | 1.250 fotos com máscaras | [Zenodo](https://doi.org/10.5281/zenodo.4041488), CC BY 4.0 |
| Brazilian Food Images | 915 fotos, um alimento por foto | [GitHub](https://github.com/brazilianfoodimage/dataset) |
| MyFoodRepo v2.0 | amostra de ~1.500 fotos de celular | [Kaggle](https://www.kaggle.com/datasets/sainikhileshreddy/food-recognition-2022) |
| Mashed potato (images.cv) | 300 fotos de purê | images.cv |

Os ZIPs somam quase 6 GB e **não estão no repositório**; não são necessários para rodar o app. A TACO (NEPA/UNICAMP, 4ª ed., 2011) é a base nutricional, não um dataset de imagens. Detalhes em [docs/DATASETS.md](docs/DATASETS.md).

## Estrutura

```text
app/
  main.py             API (FastAPI): análise, metas, histórico, feedback
  vision.py           modelos e a combinação
  unified_classes.py  as 28 categorias e a ligação com a TACO
  nutrition.py        cálculo pela TACO e dicas por regra
  metas.py            metas pelas medidas da pessoa
  llm.py              feedback escrito pela Gemini API
  static/             página (HTML, CSS, JavaScript)
  data/               TACO; o banco SQLite é criado no primeiro uso
models/               pesos treinados (.pt) e metadados (.json)
scripts/              treino, avaliação e preparo dos dados
tests/                testes automáticos
docs/                 documentação
```

## Documentação

- [RESULTADOS.md](docs/RESULTADOS.md): a evolução do modelo em uma página.
- [AVALIACAO.md](docs/AVALIACAO.md): todas as medidas e decisões, com as limitações.
- [ARQUITETURA.md](docs/ARQUITETURA.md): fluxo dos dados, endpoints e onde alterar cada coisa.
- [DATASETS.md](docs/DATASETS.md): origem, conteúdo e verificação dos datasets.
- [RESUMO_MUDANCAS.md](docs/RESUMO_MUDANCAS.md): histórico do projeto e comandos de treino.
- [GUIA_EQUIPE.md](docs/GUIA_EQUIPE.md): primeiro acesso e roteiro de demonstração.
