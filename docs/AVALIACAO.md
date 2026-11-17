# Avaliação e limitações verificadas

### Modelo incluído

Arquitetura: U-Net pequena com saídas sigmoid (21 classes). Resolução de inferência 160 × 160. Treinamento com duas fontes de imagens, inicialização parcial dos pesos de MyFood e melhor checkpoint selecionado por validação.

| Medida | Resultado reportado no treinamento anterior |
|---|---:|
| IoU médio em validação conjunta, apenas classes com máscara disponível | **0,5305** |
| IoU médio no teste MyFood, 8 categorias unificadas presentes | **0,5275** |
| IoU no teste MyFood para frango | **0,353** |
| Fotos do conjunto brasileiro de teste anotadas para métrica | **0** |

**Interpretação obrigatória:** IoU médio 0,53 NÃO significa 53% de acerto das refeições, nem demonstra que os 21 alimentos possam ser identificados confiavelmente em qualquer fotografia. A validação brasileira só avalia a classe explicitamente anotada por par; a taxa de falsos positivos de outras classes **não foi medida de modo abrangente**. A rede produz falsos positivos e confusões em pratos desconhecidos. No protótipo, uma foto de peito de frango já foi classificada como arroz/feijão; não foi comprovada correção desse problema.

### Entradas diferentes, previsões diferentes — mas não necessariamente corretas

O endpoint `POST /api/analyze` executa o modelo para cada foto, escolhe classes a partir de mapas de previsão e devolve `vision_debug` com regiões, scores e limiares. **Não** possui roteiro que devolve sempre arroz, feijão etc. Um teste automatizado verifica que duas fotografias de exemplo produzem máscaras diferentes. Isso não é evidência de precisão nutricional; é só evidência de inferência não-fixa.

### Limites dos nutrientes

- A TACO fornece nutrientes **por 100 g**, e o backend multiplica por `gramas/100`.
- Uma imagem isolada não mede massa, absorção de óleo, tempero, caldo oculto nem método de preparo.
- As sugestões comparativas são **regras**, não orientação clínica individual.
- Valores TACO ausentes não são arbitrariamente transformados em zero no detalhamento.

### Melhorias para o grupo

Prioridade 1: avaliações de falso positivo por refeição completa. Prioridade 2: segmentação de melhor arquitetura/resolução e anotação de classes ausentes. Prioridade 3: identificação do modo de preparo e validação da porção. Prioridade 4: documentação de métricas e abordagem de incerteza.
