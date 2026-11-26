# Integração e rastreabilidade dos datasets

## Resumo conferido pela execução do decodificador

O comando `python scripts/verify_datasets.py --myfood ... --brazil ... --myfoodrepo ...` lê os ZIPs com o mesmo carregador do treino (`scripts/data_v2.py`). Resultado em 26/11/2026:

| Origem | Treino | Validação | Teste |
|---|---:|---:|---:|
| MyFood | 750 | 250 | 250 |
| Brazilian Food Images | 681 | 119 | 115 |
| MyFoodRepo v2.0 (amostra com teto) | 1.491 | 292 | 294 |
| **Total** | **2.922** | **661** | **659** |

Os números são de antes da remoção de quase duplicatas na validação e no teste. O `test_set` do ZIP brasileiro (160 fotos) não tem máscaras e não é usado. O purê do images.cv entra à parte (seção própria, abaixo).

- **MyFood:** imagens `train/*.png`, `val/*.png`, `test/*.png` com anotações em `train_ann`, `val_ann`, `test_ann`. O código converte os rótulos 2 a 10 para IDs comuns; o rótulo 1 indica região ignorada.
- **Brazilian Food Images:** pastas como `training_set/chicken/` com pares de arquivos imagem `chicken0000.jpg` e máscara `chicken0000_.jpg`. As máscaras são binárias e anotam apenas **a classe daquela pasta** em um prato que pode conter outros alimentos.
- **Normalização:** `app/unified_classes.py` faz o mapeamento das classes para 28 categorias visuais (21 destes dois datasets + 7 do MyFoodRepo, ver abaixo), unificando, por exemplo, ovo frito/cozido na classe `Ovo`.
- **Supervisão por pixel (`scripts/data_v2.py`):**
  - MyFood: as máscaras anotam todos os alimentos da foto. Cada pixel anotado é positivo para sua classe e negativo para as outras 20; o fundo é negativo para as 21.
  - Brasileiro: só um alimento é anotado. Pixels da máscara são positivos para a classe e negativos para as outras 20; pixels fora da máscara são negativos **só para a classe anotada** e desconhecidos para as demais. Isso evita ensinar que outro alimento visível, mas não anotado, é fundo.
  - Imagens sintéticas sem comida (cerca de 8% por época): negativas para todas as classes.
- **Split:** no conjunto brasileiro, pares de treino/validação são particionados deterministicamente por SHA-1 do conteúdo da foto. No v2, a validação antiga (234 pares) é dividida ao meio, por outro trecho do hash, em **validação nova** e **teste novo**, para que nenhum modelo tenha treinado nas fotos de teste. Depois da remoção de quase duplicatas, restam 75 fotos brasileiras na validação e 78 no teste. Fotos quase duplicadas das de treino são removidas da validação e do teste (os próprios datasets repetem fotos entre divisões).
- **Treinamento:** todos os modelos do app são MobileNetV3-Large + LR-ASPP com saídas sigmoid, treinados por `scripts/train_v2.py`, que lê as fontes direto dos ZIPs em uma única execução. O **v2** (21 classes, `models/unified_21class.pt`) usa MyFood e Brazilian Food Images; o **v3** (28 classes) acrescenta o MyFoodRepo (`--myfoodrepo`); o **v3d** acrescenta o purê do images.cv (`--mashed`). O app usa os três juntos (ver `docs/RESULTADOS.md`). O script do v1 (`scripts/train_unified.py`, rede `FoodUNet`) saiu do repositório em 26/11/2026 e está no histórico do git.

### Classes brasileiras anotadas no treino (quantidade de pares)

```text
beans 57; beef 60; beet 25; cabbage 13; carrot 54; cassava 12;
chicken 45; corn 14; egg 21; farofa 29; fries 42; mashed 23;
mayo(potato salad) 14; milanesesteak 27; noodles 17;
pastaredsauce 27; pumpkin 27; rice 57; salad 69; tomato 48.
```

## MyFoodRepo v2.0 (desde o v3, 23/11/2026)

- **Arquivo:** `archive.zip` (~5,3 GB), cópia do Kaggle ([food-recognition-2022](https://www.kaggle.com/datasets/sainikhileshreddy/food-recognition-2022)) do dataset do AIcrowd Food Recognition Benchmark 2022. O download oficial no AIcrowd estava dando erro de servidor. Artigo: Mohanty et al., *The Food Recognition Benchmark: Using Deep Learning to Recognize Food in Images*, Frontiers in Nutrition, 2022.
- **Conteúdo usado:** `raw_data/public_training_set_release_2.0/` — 39.962 fotos, 76.491 polígonos, 498 classes, anotações COCO em `annotations.json` (o campo `segmentation` vem como texto JSON). A validação oficial (1.000 fotos) e o teste sem rótulos não são usados.
- **Divisão:** SHA-1 do nome do arquivo `% 10`: 0 = teste, 1 = validação, resto = treino. Determinística e sem sorteio.
- **Amostra com teto:** de cada divisão entram fotos até cada classe atingir o teto (`MFR_CAPS` em `scripts/data_v2.py`): 150 por classe nova e 50 por classe antiga no treino; 30 e 10 na validação e no teste. Uma foto conta para todas as classes que contém. Resultado: 1.491 fotos no treino, 292 na validação, 294 no teste.
- **Mapa de classes:** `MYFOODREPO_LABELS` em `app/unified_classes.py`. Polígonos de classes fora do mapa viram "outro alimento" (ignorados no treino, como o código 1 do MyFood). Decisões: `chicken-leg` (coxa, 85 fotos) entra em Frango; `hamburger` é só a carne e entra em Carne bovina; `hamburger-bun` entra em Pão; o sanduíche inteiro é `hamburger-bread-meat-ketchup`.
- **Fotos de treino por categoria nova (total no dataset):** Pão 5.611 (somando os tipos de pão), Batata cozida 644, Peixe 423 (fish + salmon), Brócolis 421, Cebola 285, Hambúrguer (sanduíche) 154, Lasanha 104.
- **SHA-256:** `00afdf97a392e6baaf6e1bd13c5487917227b8e4030847f804c79ab165aa7879`

## Purê de batata do images.cv (desde o v3d, 24/11/2026)

`mashed p.zip`: 1.298 fotos de purê de batata (classe `mashed_potato`) em 256 × 256, já divididas em train 908 / val 194 / test 196, baixadas do images.cv. Só tem a etiqueta "a foto tem purê", sem máscara. `scripts/pseudo_pure.py` gera a máscara com o v3c (foto de um alimento só: toda a comida; senão, onde o v3c diz purê) e usa no máximo 300 fotos de treino e 60 de validação com purê em pelo menos 10% da foto. A divisão `test` não entra no treino: é o teste independente de purê. A versão de 32 × 32 (`mashed.zip`) foi descartada por ser pequena demais. SHA-256 de `mashed p.zip`: `04bb3734605e22ccdf56d1688f3cefd00c6ba87a96d3105f070ddd782078af20`.

## FoodSeg103 (só no experimento v5, 26/11/2026)

Dataset de segmentação com 7.118 fotos e 103 ingredientes ([artigo](https://arxiv.org/abs/2105.05409); cópia usada: `EduardoPacheco/FoodSeg103` no Hugging Face, em arquivos `.parquet`). Foi testado para reforçar tomate, peixe, brócolis, cenoura, batata frita e ovo: entram só fotos em que a classe ocupa 3% ou mais, e 859 fotos de peixe, batata frita e ovo foram revisadas uma a uma (`docs/FOODSEG103_EXCLUSOES.json` lista as que saíram: sushi, peixe inteiro, nachos, chips, ovo mexido, máscara errada). O modelo treinado com ele (v5) **não entrou no app** (tabela 6 de `docs/RESULTADOS.md`). O carregador ficou em `load_foodseg` (`scripts/data_v2.py`, opção `--foodseg` do treino, que precisa do `pyarrow`).

## Fontes, licenças e verificação

| Dataset | Fonte | Licença (verificada em 21/11/2026) |
|---|---|---|
| MyFood | [MyFood Dataset](https://doi.org/10.5281/zenodo.4041488) (Zenodo, DOI 10.5281/zenodo.4041488) · artigo SIBGRAPI 2020, [arXiv:2012.03087](https://arxiv.org/abs/2012.03087) · Freitas, Cordeiro e Macario | **CC BY 4.0**: redistribuição permitida com crédito |
| Brazilian Food Images | [brazilianfoodimage/dataset](https://github.com/brazilianfoodimage/dataset) (GitHub) | **Não declarada** (sem LICENSE, README sem autor): não republicar sem autorização |

Códigos de verificação dos ZIPs usados no treino (confira com `sha256sum` ou `Get-FileHash`):

```text
MyFood Dataset.zip           SHA-256 870210634b83c0452a8db612232d4cae4476587e1fa2d918aefe04b027a04954
                             MD5     552de1d0596f8f11ab481517790441f9 (igual ao informado pelo Zenodo)
Brazilian food images.zip    SHA-256 393d01c0952b87debebe5397953e15564d824a402553686eb9f98946fe175f85
```

O ZIP do MyFood é o próprio arquivo do Zenodo. No ZIP brasileiro, a pasta `training_set` é um clone do repositório acima; a origem do `test_set` não foi identificada.

## Como a TACO se relaciona

Os dados TACO **não são imagens** e não podem, por si, treinar reconhecimento de alimentos. O modelo prevê categorias como `Frango`, que têm correspondências iniciais **editáveis** em `app/unified_classes.py`: por exemplo, `Frango` aponta provisoriamente para `Frango, peito, sem pele, grelhado` (TACO #410). A tabela TACO, íntegra como 597 registros/26 campos estruturados da **Tabela 1**, está em `app/data/taco_tabela1_completa.csv` e `app/data/taco.json`; as Tabelas 2 e 3 permanecem consultáveis no PDF. O peso da porção é informado/corrigido pelo usuário.

## O que os dados ainda não cobrem

Quatro retreinos com mais fotos estrangeiras (v3c, v3d, v4 e v5) pioraram quando usados sozinhos nas fotos reais de prato feito. O que falta é **foto brasileira de prato misto**, tirada de celular, com anotação: principalmente tomate em rodelas ou picado, arroz parcialmente coberto e peixe. Um próximo passo viável é juntar essas fotos anotando só a lista de alimentos de cada uma, sem máscara, como foi feito com o purê. As fotos do teste independente (`docs/FOTOS_TESTE_NOVO_GABARITO.json`) não devem ser usadas para ajustar nada.
