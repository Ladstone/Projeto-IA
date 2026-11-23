# Avaliação e limitações verificadas

### Modelo incluído (v3, ativo desde 23/11/2026)

Mesma arquitetura do v2 (MobileNetV3-Large + LR-ASPP, 224 × 224), com **28 saídas sigmoid**. Treino de 15 épocas com três fontes de imagens: MyFood, Brazilian Food Images e uma amostra do **MyFoodRepo v2.0** (fotos de celular de refeições reais). Categorias novas: Brócolis, Batata cozida, Peixe, Pão, Cebola, Hambúrguer (sanduíche) e Lasanha. Época e limiares escolhidos só na validação. Relatório: `docs/METRICAS_V3.json`.

| Medida no teste (mesma regra de detecção do app) | v2 | **v3** |
|---|---:|---:|
| MyFood (222 fotos): F1 de presença, macro de 8 classes | 0,94 | 0,93 |
| MyFood: falsos positivos por imagem | 0,10 | 0,11 |
| MyFood: pratos 100% corretos | 88% | 86% |
| MyFood: fundo marcado como comida | 5% | **3,5%** |
| Brasileiro (78 fotos): IoU médio da classe anotada | 0,74 | 0,70 |
| Brasileiro: pixels anotados na classe certa | 82% | 77% |
| Imagens sem comida (120 sintéticas): com detecção espúria | 0% | 3,3% |
| MyFoodRepo (294 fotos de celular): F1 de presença, macro | 0,12 | **0,56** |
| MyFoodRepo: falsos positivos por imagem | 1,04 | **0,75** |
| MyFoodRepo: pratos 100% corretos | 7,5% | **35%** |

F1 das categorias novas no teste do MyFoodRepo (o v2 não as conhece): Pão 0,80; Hambúrguer (sanduíche) 0,79; Lasanha 0,74; Brócolis 0,73; Batata cozida 0,64; Peixe 0,33; Cebola 0,17.

**Interpretação do v3:**
- Nos dois testes antigos, o v3 fica **um pouco abaixo do v2**; as maiores perdas são Milho (IoU 0,44 → 0,26), Purê (0,82 → 0,68), Arroz e Salada no conjunto brasileiro.
- Em fotos variadas de celular (MyFoodRepo), o v3 é **muito melhor**, e é o único que reconhece as 7 categorias novas.
- As classes mais fracas do v3 são Cebola, Abóbora, Peixe e Milho.
- O teste do MyFoodRepo usa fotos separadas por hash do nome do arquivo, nunca usadas no treino, mas da mesma fonte. **Continua sem avaliação com fotos reais dos usuários do app.**

### Modelo anterior (v2, ativo de 21/11 a 23/11/2026; backup em `models/unified_21class_v2.*`)

Arquitetura: MobileNetV3-Large pré-treinada no ImageNet + cabeça LR-ASPP, 21 saídas sigmoid, inferência em 224 × 224. Treino de 20 épocas com as duas fontes de imagens, supervisão negativa por pixel e cerca de 8% de imagens sintéticas sem comida. A época (16) e os limiares por classe foram escolhidos **só na validação**; o teste foi usado uma única vez. Detalhes em `docs/RESUMO_MUDANCAS.md` e relatório completo em `docs/METRICAS_V2_COMPLETO.json`.

| Medida no teste (mesma regra de detecção do app) | v1 | v2 piloto | v2 |
|---|---:|---:|---:|
| MyFood (222 fotos): F1 de presença, macro de 8 classes | 0,35 | 0,89 | **0,94** |
| MyFood: falsos positivos por imagem | 2,84 | 0,19 | **0,10** |
| MyFood: pratos 100% corretos | 6% | 78% | **88%** |
| MyFood: fundo marcado como comida | 43% | 9% | **5%** |
| Brasileiro (78 fotos): IoU médio da classe anotada | 0,19 | 0,48 | **0,74** |
| Brasileiro: pixels anotados na classe certa | 35% | 65% | **82%** |
| Imagens sem comida (120 sintéticas): com detecção espúria | 89% | 0% | **0%** |

**Interpretação obrigatória:**
- As fotos de teste vêm das **mesmas fontes do treino** (quase duplicatas removidas). **Não há avaliação com fotos reais de usuários**, então esses números não garantem o desempenho em fotos de celular.
- No conjunto brasileiro, cada classe tem de **1 a 8 fotos de teste**: os números por classe são indicativos, não conclusivos. As classes mais fracas são Milho (IoU 0,44) e Macarrão (0,60).
- "Pratos 100% corretos" mede presença/ausência de cada categoria, não a quantidade nem o tipo de preparo.
- Os scores são saídas sigmoid **não calibradas**; não devem ser lidos como probabilidade de acerto.

#### Histórico: métrica antiga do v1

O v1 (U-Net pequena, 160 × 160, `models/unified_21class_v1.pt`) foi descrito originalmente com IoU médio de 0,53 na validação e no teste MyFood. Essa métrica avaliava só a classe anotada de cada foto e **ignorava falsos positivos** de outras classes, por isso não é comparável com a tabela acima. Na métrica atual o v1 marca alimentos inexistentes em 94% das fotos do MyFood e classificava uma foto de peito de frango como arroz/feijão.

### Entradas diferentes, previsões diferentes — mas não necessariamente corretas

O endpoint `POST /api/analyze` executa o modelo para cada foto, escolhe classes a partir de mapas de previsão e devolve `vision_debug` com regiões, scores e limiares. **Não** possui roteiro que devolve sempre arroz, feijão etc. Um teste automatizado verifica que duas fotografias de teste (`tests/fixtures/`) produzem máscaras diferentes. Isso não é evidência de precisão nutricional; é só evidência de inferência não-fixa.

### Limites dos nutrientes

- A TACO fornece nutrientes **por 100 g**, e o backend multiplica por `gramas/100`.
- Uma imagem isolada não mede massa, absorção de óleo, tempero, caldo oculto nem método de preparo.
- As sugestões comparativas são **regras**, não orientação clínica individual.
- Valores TACO ausentes não são arbitrariamente transformados em zero no detalhamento.

### Melhorias para o grupo

Prioridade 1: avaliar com **fotos reais** de pratos completos, tiradas com permissão e anotadas. Prioridade 2: mais fotos anotadas para as classes com poucos exemplos (Milho, Macarrão, Salada com maionese, Mandioca, Cenoura, Batata frita) e revisão dos limiares que ficaram nas pontas da faixa (Tomate 0,15; Farofa 0,05; vários em 0,90–0,95). Prioridade 3: identificação do modo de preparo e validação da porção (hoje estimada pela área da foto). Prioridade 4: calibração dos scores e comunicação da incerteza.
