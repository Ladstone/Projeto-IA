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
- **Perda supervisionada parcial:** em fotos brasileiras, apenas a cabeça da classe anotada contribui para BCE/Dice. Isso evita ensinar que outro alimento visível, mas não anotado, é necessariamente fundo.
- **Split:** no conjunto brasileiro, pares de treino/validação são particionados deterministicamente por SHA1 do conteúdo da foto. Isso protege contra arquivos idênticos nos dois grupos; **não prova** ausência de fotos quase duplicadas.
- **Treinamento real:** rede `FoodUNet` com 21 saídas sigmoid, pesos salvos em `models/unified_21class.pt`. O método de inferência é `VisionService.predict` em `app/vision.py`. `scripts/train_unified.py` lê e processa **ambas** as fontes em uma única execução de treinamento.

### Classes brasileiras anotadas no treino (quantidade de pares)

```text
beans 57; beef 60; beet 25; cabbage 13; carrot 54; cassava 12;
chicken 45; corn 14; egg 21; farofa 29; fries 42; mashed 23;
mayo(potato salad) 14; milanesesteak 27; noodles 17;
pastaredsauce 27; pumpkin 27; rice 57; salad 69; tomato 48.
```

## Como a TACO se relaciona

Os dados TACO **não são imagens** e não podem, por si, treinar reconhecimento de alimentos. O modelo prevê categorias como `Frango`, que têm correspondências iniciais **editáveis** em `app/unified_classes.py`: por exemplo, `Frango` aponta provisoriamente para `Frango, peito, sem pele, grelhado` (TACO #410). A tabela TACO, íntegra como 597 registros/26 campos estruturados da **Tabela 1**, está em `app/data/taco_tabela1_completa.csv` e `app/data/taco.json`; as Tabelas 2 e 3 permanecem consultáveis no PDF. O peso da porção é informado/corrigido pelo usuário.

## Recomendações para o treinamento seguinte

Criar um **conjunto de avaliação anotado de pratos completos** com classes e pesos reais, aumentar diversidade (iluminação/pratos/preparos), testar segmentadores pré-treinados mais robustos e calibrar previsões por classe em validação não vista. Não usar imagens de teste para ajustar hiperparâmetros. Retreinar com logs reprodutíveis e comparar precisão/recall de detecção de alimentos, não apenas IoU supervisionado por classe.
