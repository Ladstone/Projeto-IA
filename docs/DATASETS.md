# Integração e rastreabilidade dos dois datasets

## Resumo conferido pela execução do decodificador

O comando `python scripts/verify_datasets.py --myfood ... --brazil ...` foi executado sobre os dois arquivos ZIP enviados. Resultado:

| Origem | Treino | Validação | Teste |
|---|---:|---:|---:|
| MyFood | 750 | 250 | 250 (máscaras) |
| Brazilian Food Images | 681 | 234 | 160 (sem máscaras no ZIP) |
| **Treino conjunto** | **1.431** | **484** | **250 com máscaras para pontuação** |

- **MyFood:** imagens `train/*.png`, `val/*.png`, `test/*.png` com anotações em `train_ann`, `val_ann`, `test_ann`. O código converte os rótulos 2 a 10 para IDs comuns; o rótulo 1 indica região ignorada.
- **Brazilian Food Images:** pastas como `training_set/chicken/` com pares de arquivos imagem `chicken0000.jpg` e máscara `chicken0000_.jpg`. As máscaras são binárias e anotam apenas **a classe daquela pasta** em um prato que pode conter outros alimentos.
- **Normalização:** `app/unified_classes.py` faz o mapeamento das classes para 21 categorias visuais, unificando, por exemplo, ovo frito/cozido na classe `Ovo`.
- **Supervisão por pixel (v2, `scripts/data_v2.py`):**
  - MyFood: as máscaras anotam todos os alimentos da foto. Cada pixel anotado é positivo para sua classe e negativo para as outras 20; o fundo é negativo para as 21.
  - Brasileiro: só um alimento é anotado. Pixels da máscara são positivos para a classe e negativos para as outras 20; pixels fora da máscara são negativos **só para a classe anotada** e desconhecidos para as demais. Isso evita ensinar que outro alimento visível, mas não anotado, é fundo.
  - Imagens sintéticas sem comida (cerca de 8% por época): negativas para as 21 classes.
- **Split:** no conjunto brasileiro, pares de treino/validação são particionados deterministicamente por SHA-1 do conteúdo da foto. No v2, a validação antiga (234 pares) é dividida ao meio, por outro trecho do hash, em **validação nova** e **teste novo**, para que nenhum modelo tenha treinado nas fotos de teste. Depois da remoção de quase duplicatas, restam 75 fotos brasileiras na validação e 78 no teste. Fotos quase duplicadas das de treino são removidas da validação e do teste (os próprios datasets repetem fotos entre divisões).
- **Treinamento real:** o modelo ativo (v2) é MobileNetV3-Large + LR-ASPP com 21 saídas sigmoid, treinado por `scripts/train_v2.py`, que lê **ambas** as fontes direto dos ZIPs em uma única execução. Os pesos ficam em `models/unified_21class.pt` e a inferência é `VisionService.predict` em `app/vision.py`. O script antigo do v1 (`scripts/train_unified.py`, rede `FoodUNet`) continua no repositório só como referência.

### Classes brasileiras anotadas no treino (quantidade de pares)

```text
beans 57; beef 60; beet 25; cabbage 13; carrot 54; cassava 12;
chicken 45; corn 14; egg 21; farofa 29; fries 42; mashed 23;
mayo(potato salad) 14; milanesesteak 27; noodles 17;
pastaredsauce 27; pumpkin 27; rice 57; salad 69; tomato 48.
```

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

## Recomendações para o treinamento seguinte

Criar um **conjunto de avaliação anotado de pratos completos** com classes e pesos reais, aumentar diversidade (iluminação/pratos/preparos), testar segmentadores pré-treinados mais robustos e calibrar previsões por classe em validação não vista. Não usar imagens de teste para ajustar hiperparâmetros. Retreinar com logs reprodutíveis e comparar precisão/recall de detecção de alimentos, não apenas IoU supervisionado por classe.
