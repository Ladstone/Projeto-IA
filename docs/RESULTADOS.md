# Resultados: a evolução do modelo

Resumo em uma página. Os detalhes de cada medida estão em [AVALIACAO.md](AVALIACAO.md) e nos arquivos `METRICAS_*.json`.

## As versões

| Versão | O que é | Situação |
|---|---|---|
| v1 | U-Net pequena, 160 px, 21 categorias | substituída (arquivo fora do repositório; ver histórico do git) |
| v2 | MobileNetV3-Large + LR-ASPP, 224 px, 21 categorias, 2 datasets | **base do app** e backup (`models/unified_21class_v2.pt`) |
| v3 | mesma rede, 28 categorias, + MyFoodRepo | usada só para as 7 categorias novas |
| v3d | v3 retreinado a partir do v2, + 300 fotos de purê | usada só para o purê |
| **Combinação** | v2 + categorias novas do v3 + purê do v3d + tomate e batata frita do v3 | **padrão do app desde 25/11/2026** |
| v4 | retreino a partir do v2 com mais fotos das classes fracas e aumentos mais fortes | experimento sem ganho; não está no repositório |
| v5 | receita do v3d + fotos filtradas do FoodSeg103 (tomate, peixe, brócolis, cenoura, batata frita, ovo) | experimento sem ganho; não está no repositório |

## 1. Teste dos datasets (fotos separadas, nunca usadas no treino)

| Medida | v1 | v2 | v3 |
|---|---:|---:|---:|
| MyFood (222 fotos): pratos 100% corretos | 6% | **88%** | 86% |
| MyFood: alimentos inventados por foto | 2,84 | **0,10** | 0,11 |
| MyFood: fundo marcado como comida | 43% | 5% | **3,5%** |
| Brasileiro (78 fotos): pixels na classe certa | 35% | **82%** | 77% |
| Fotos sem comida (120) com detecção | 89% | **0%** | 3,3% |
| MyFoodRepo (294 fotos de celular): pratos 100% corretos | — | 7,5% | **35%** |

**Leitura:** o v2 resolveu o problema principal do v1 (inventar comida). O v3 conhece mais comidas, mas piorou um pouco no prato brasileiro. Por isso nenhum dos dois sozinho é o padrão.

## 2. Mini teste com 12 fotos reais da equipe

Fotos de celular e da internet, fora dos datasets (`docs/FOTOS_REAIS_GABARITO.json`, `scripts/avaliar_fotos_reais.py`).

| Medida | v2 | v3 | v2 + v3 | **Combinação (padrão)** |
|---|---:|---:|---:|---:|
| Prato feito (7 fotos): alimentos achados | 29/35 | 26/35 | 29/35 | **29/35** |
| Outros pratos (5 fotos): alimentos achados | 1/8 | 5/8 | 4/8 | **5/8** |
| Alimentos inventados | 7 | 4 | 4 | 5 |
| Fotos sem nenhum erro | 2/12 | 4/12 | 4/12 | 4/12 |

**Leitura:** a combinação fica com o melhor dos dois: o prato feito do v2 e as comidas novas do v3.

## 3. Purê: o caso que motivou o v3d

O v2 chamava purê de arroz ou frango. Teste com 196 fotos de purê que ficaram fora do treino (dataset "mashed potato" do images.cv).

| Medida | v2 | v2 + v3 | **Combinação (padrão)** |
|---|---:|---:|---:|
| Fotos de purê em que o purê foi achado | 1% | 5% | **90%** |
| Purê inventado em 366 fotos sem purê | 10 | 10 | 15 |

**Leitura:** o purê passou de quase nunca achado para 90%, ao custo de 5 fotos a mais com purê inventado em 366.

## 4. Teste independente: 21 fotos novas

Fotos do Pinterest reunidas em 25/11/2026, **depois** de todas as decisões sobre o modelo: 12 de prato feito e 9 de pratos com purê. Nenhuma foi usada para ajustar modelo, limiar ou regra (`docs/FOTOS_TESTE_NOVO_GABARITO.json`, resultado completo em `docs/METRICAS_TESTE_NOVO.json`).

| Medida | v2 | **Combinação (padrão)** |
|---|---:|---:|
| Todas (21 fotos): alimentos achados | 61/87 (70%) | **70/87 (80%)** |
| Todas: alimentos inventados | 11 | **6** |
| Prato feito (12 fotos): alimentos achados | 45/55 | 45/55 |
| Com purê (9 fotos): purê achado | 0 de 9 | **9 de 9** |
| Com purê: alimentos achados | 16/32 (50%) | **25/32 (78%)** |
| Com purê: alimentos inventados | 8 | **3** |

**Leitura:** no prato feito comum a combinação é igual ao v2; com purê ela acha o purê em todas as fotos e para de inventar arroz no lugar dele. O acerto de 80% fica perto dos 79% do mini teste (34/43), então o mini teste era só um pouco otimista.

**Erros que se repetem na combinação:** tomate em rodelas, picado ou no vinagrete (6 fotos), batata frita (2), arroz parcialmente coberto (4), brócolis e cenoura em cubos.

## 5. Tomate e batata frita do v3 (25/11/2026)

O teste independente mostrou que a combinação perdia tomate no prato misto (o v2 só viu tomate sozinho) e, às vezes, a batata frita. Duas tentativas, comparadas num conjunto de desenvolvimento de 20 fotos (`docs/FOTOS_DEV_GABARITO.json`: o mini teste + 8 fotos da equipe):

| Variante | Desenvolvimento: achados | Inventados |
|---|---:|---:|
| Combinação da tabela 4 | 53/70 | 8 |
| **+ tomate e batata frita do v3** | **56/70** | 8 |
| v4 sozinho | 49/70 | 7 |
| v2 + v4 no lugar do v3 e do v3d | 51/70 | 7 |
| + tomate e batata frita do v4 | 56/70 | 8 |

Empate entre a segunda e a última; ficou a que usa o v3, por inventar menos nas fotos de validação dos datasets (tomate em 60 de 355 contra 65; batata frita em 20 de 364 contra 22) e por não precisar de modelo novo. Só depois da escolha ela foi medida, uma vez, no teste independente:

| Teste independente (21 fotos) | Combinação da tabela 4 | **Com tomate e batata frita do v3 (padrão)** |
|---|---:|---:|
| Alimentos achados | 70/87 (80%) | **74/87 (85%)** |
| Alimentos inventados | 6 | 7 |
| Fotos sem nenhum erro | 4/21 | **7/21** |

**Leitura:** tomate achado em 3 fotos a mais e batata frita em 2; em uma foto o recorte mudou e a carne se perdeu. O **v4** (2 h de treino, mais fotos suíças de tomate, peixe, brócolis, cenoura e batata frita, aumentos mais fortes) não superou a combinação: retreinar a partir do v2 continua perdendo acertos no prato feito, e o peixe ficou em F1 0,37. O que falta é foto brasileira de prato misto com máscara.

## 6. v5 com o FoodSeg103 (26/11/2026, sem ganho)

Mais uma tentativa de reforçar as classes fracas: a receita do v3d com fotos do FoodSeg103, revisadas uma a uma (859 fotos vistas; saíram sushi, peixe inteiro, nachos, chips, ovo mexido e máscaras erradas; lista em `docs/FOODSEG103_EXCLUSOES.json`). 8 épocas, F1 de presença na validação 0,830; peixe subiu de 0,40 para 0,54.

| Variante | Desenvolvimento: achados | Inventados | Sem erro |
|---|---:|---:|---:|
| Padrão (tabela 5) | 56/70 | 8 | 7/20 |
| v5 sozinho | 51/70 | 13 | 6/20 |
| v2 + v5 no lugar do v3 e do v3d | 53/70 | 10 | 7/20 |
| Padrão + tomate e batata frita do v5 | 57/70 | 8 | 9/20 |
| Padrão + tomate, batata frita, ovo e cenoura do v5 | 58/70 | 7 | 10/20 |

A última foi a escolhida e medida uma vez no teste independente:

| Teste independente (21 fotos) | **Padrão** | Com o v5 |
|---|---:|---:|
| Alimentos achados | 74/87 (85%) | 76/87 (87%) |
| Alimentos inventados | **7** | 11 |
| Fotos sem nenhum erro | 7/21 | 7/21 |

**Leitura:** o v5 achou cenoura em 2 fotos a mais, mas inventou cenoura em 3 e ovo em 2; em tomate e batata frita empatou com o v3. O ganho visto no desenvolvimento não se repetiu, e o padrão não mudou. Subir o limiar da cenoura para 0,99 corta as invenções na validação (47 para 18 em 888 fotos, perdendo 11 de 124 acertos), mas isso não foi confirmado em fotos novas; no ovo o limiar não resolve. É o quarto retreino (v3c, v3d, v4, v5) que piora sozinho nas fotos reais: o limite está nos dados, não na duração do treino.

## Como ler estes números

- **As tabelas 2 a 4 medem a combinação antes do ajuste da tabela 5** (sem o tomate e a batata frita do v3).
- **O mini teste (tabela 2) foi usado para decidir** as regras da combinação, então é um pouco otimista. O teste independente (tabela 4) não foi usado para decidir nada, mas tem só 21 fotos, de Pinterest: bem iluminadas e vistas de cima. O gabarito das duas listas foi feito à mão, pela presença de cada alimento.
- **A regra dos 10% é instável perto do limite.** Na foto 21 do teste novo, o peixe visto pelo v3 vai de 8,9% a 10,3% conforme 1 pixel do recorte e a compressão JPEG, e isso decide se o peixe aparece. O script de teste recorta exatamente como a página (`recorte_como_no_app`).
- **Os testes dos datasets usam fotos das mesmas fontes do treino.** Não garantem o desempenho em qualquer foto de celular.
- **A tabela 1 não inclui a combinação.** Ela foi medida nos datasets só na primeira versão (`METRICAS_COMBINADO.json`: igual ao v2 nas fotos brasileiras), antes da regra dos 10% e do purê; a medida não foi refeita.
- **A foto não mede peso.** As gramas são uma estimativa pela área no prato, e o usuário deve corrigir.
- Casos ruins conhecidos: o "almoço caio" (frango sai como carne quando há purê) e purê inventado em batata frita ou pão.
