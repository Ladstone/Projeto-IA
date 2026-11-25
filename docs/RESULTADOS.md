# Resultados: a evolução do modelo

Resumo em uma página. Os detalhes de cada medida estão em [AVALIACAO.md](AVALIACAO.md) e nos arquivos `METRICAS_*.json`.

## As versões

| Versão | O que é | Situação |
|---|---|---|
| v1 | U-Net pequena, 160 px, 21 categorias | substituída (arquivo fora do repositório; ver histórico do git) |
| v2 | MobileNetV3-Large + LR-ASPP, 224 px, 21 categorias, 2 datasets | **base do app** e backup (`models/unified_21class_v2.pt`) |
| v3 | mesma rede, 28 categorias, + MyFoodRepo | usada só para as 7 categorias novas |
| v3d | v3 retreinado a partir do v2, + 300 fotos de purê | usada só para o purê |
| **Combinação** | v2 + categorias novas do v3 + purê do v3d | **padrão do app desde 25/11/2026** |

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

## Como ler estes números

- **O mini teste é pequeno e foi usado para decidir** as regras da combinação, então os números dele são otimistas. Falta um teste com fotos novas que ninguém usou para decidir.
- **Os testes dos datasets usam fotos das mesmas fontes do treino.** Não garantem o desempenho em qualquer foto de celular.
- **A tabela 1 não inclui a combinação.** Ela foi medida nos datasets só na primeira versão (`METRICAS_COMBINADO.json`: igual ao v2 nas fotos brasileiras), antes da regra dos 10% e do purê; a medida não foi refeita.
- **A foto não mede peso.** As gramas são uma estimativa pela área no prato, e o usuário deve corrigir.
- Casos ruins conhecidos: o "almoço caio" (frango sai como carne quando há purê) e purê inventado em batata frita ou pão.
