# Avaliação e limitações verificadas

### Modelo incluído (v2, ativo desde 21/11/2026)

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

O endpoint `POST /api/analyze` executa o modelo para cada foto, escolhe classes a partir de mapas de previsão e devolve `vision_debug` com regiões, scores e limiares. **Não** possui roteiro que devolve sempre arroz, feijão etc. Um teste automatizado verifica que duas fotografias de exemplo produzem máscaras diferentes. Isso não é evidência de precisão nutricional; é só evidência de inferência não-fixa.

### Limites dos nutrientes

- A TACO fornece nutrientes **por 100 g**, e o backend multiplica por `gramas/100`.
- Uma imagem isolada não mede massa, absorção de óleo, tempero, caldo oculto nem método de preparo.
- As sugestões comparativas são **regras**, não orientação clínica individual.
- Valores TACO ausentes não são arbitrariamente transformados em zero no detalhamento.

### Melhorias para o grupo

Prioridade 1: avaliar com **fotos reais** de pratos completos, tiradas com permissão e anotadas. Prioridade 2: mais fotos anotadas para as classes com poucos exemplos (Milho, Macarrão, Salada com maionese, Mandioca, Cenoura, Batata frita) e revisão dos limiares que ficaram nas pontas da faixa (Tomate 0,15; Farofa 0,05; vários em 0,90–0,95). Prioridade 3: identificação do modo de preparo e validação da porção (hoje estimada pela área da foto). Prioridade 4: calibração dos scores e comunicação da incerteza.
