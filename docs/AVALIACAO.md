# Avaliação e limitações verificadas

### Combinação v2+v3 (porta 8001, em teste desde 24/11/2026)

O app passa a foto nos dois modelos. Se o v3 acha alguma **categoria nova** (Brócolis, Batata cozida, Peixe, Pão, Cebola, Hambúrguer ou Lasanha) com a área mínima de 3%, a foto inteira usa o v3; senão, vale o v2. Assim um prato feito comum fica com o resultado do v2, e o v3 não consegue trocar batata frita por macarrão nem carne por frango. Código: `CombinedVisionService` em `app/vision.py`; ligado pela variável `PRATOMETRIA_MODEL_NOVAS`. O app ativo (porta 8000) **continua só com o v2**.

Mini teste com 12 fotos reais de pratos (`docs/FOTOS_REAIS_GABARITO.json`, `scripts/avaliar_fotos_reais.py`):

| | v2 | v3 | v3b | **v2+v3** |
|---|---|---|---|---|
| Prato feito (7 fotos): alimentos achados | 29/35 | 26/35 | 27/35 | **29/35** |
| Prato feito: alimentos inventados | 1 | 2 | 0 | 1 |
| Outros (5 fotos com comidas novas): achados | 1/8 | 5/8 | 4/8 | **5/8** |
| Outros: inventados | 6 | 2 | 2 | 3 |
| Fotos perfeitas, no total | 2/12 | 4/12 | 4/12 | **5/12** |

Teste dos datasets (`docs/METRICAS_COMBINADO.json`):

| Medida no teste | v2 | v3 | **v2+v3** |
|---|---:|---:|---:|
| Brasileiro: IoU médio da classe anotada | 0,742 | 0,702 | **0,742** |
| Brasileiro: recall de presença | 0,935 | 0,922 | **0,935** |
| MyFood: pratos 100% corretos | 88,3% | 86,5% | 87,4% |
| MyFood: imagens com algum falso positivo | 8,1% | 9,5% | 9,0% |
| Imagens sem comida com detecção espúria | 0% | 3,3% | **0%** |
| MyFoodRepo: F1 de presença, macro | 0,115 | 0,556 | 0,427 |
| MyFoodRepo: pratos 100% corretos | 7,5% | 34,7% | 27,2% |

**Leitura:** nas fotos brasileiras é idêntico ao v2; no MyFood perde cerca de 1 ponto; no MyFoodRepo fica entre os dois (fotos suíças sem categoria nova vão para o v2, que não conhece aquela comida). **Cuidado:** a regra foi escolhida olhando as mesmas 12 fotos do mini teste, então o resultado nelas é otimista. Antes de virar o modelo do app, testar com fotos novas que ninguém usou para decidir.

**v3b (descartado):** retreino do v3 sem as fotos de massa e frango do MyFoodRepo e com peso 2 nas fotos brasileiras (`--mfr-skip-classes 8,3 --brazil-weight 2` em `scripts/train_v2.py`). Corrigiu batata frita → macarrão e carne → frango, mas passou a perder o frango e inventou abóbora no "almoço leo". Não foi commitado.

**Área mínima de 3%:** testada também com 2,5% e 2%. Com o v2 não acha nada a mais no mini teste e piora um pouco o MyFood (imagens com falso positivo 8,1% → 9,0% com 2,5%). Ficou em 3%.

### Modelo v3 (alternativo; foi ativo por algumas horas em 23/11/2026)

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
