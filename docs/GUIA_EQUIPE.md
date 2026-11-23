# Guia de repasse para a equipe

## Objetivo da entrega

Atender ao enunciado acadêmico: **módulo de interface, módulo de IA, módulo de resposta**, apresentação do funcionamento e link para o repositório GitHub.

**O que NÃO afirmar:** que todas as refeições são identificadas corretamente, que a rede reconhece 597 classes, ou que a foto mede peso de alimentos. A rede prevê 28 categorias; a TACO fornece informações de 597 registros após seleção do preparo.

## Primeiro acesso

1. Clonar o repositório: `git clone https://github.com/Ladstone/Projeto-IA.git`.
2. Abrir o PowerShell nessa pasta. Seguir os três comandos do `README.md`.
3. Abrir `http://127.0.0.1:8000/api/health`: verificar `model_ready: true`, `taco_records: 597` e `mode: local-only`.
4. Abrir `/`: salvar metas, selecionar uma foto, analisar e revisar alimentos.
5. Expandir **Diagnóstico do modelo (para desenvolvedores)**. A saída é calculada para cada imagem. Se o modelo confundir alimentos, registrar o caso e guardar a foto com permissão.
6. Corrigir os alimentos pela busca TACO, informar as quantidades em gramas e confirmar as recomendações.

## Responsabilidades recomendadas

- **Frontend:** `app/static/index.html` + `app/static/app.js`, responsividade, mensagens de erro, facilidade para editar rótulos, testes manuais.
- **Backend e nutrição:** `app/main.py`, `app/nutrition.py`, revisão das correspondências de preparo, tratamento de erros e persistência de perfil.
- **IA:** `app/vision.py`, `app/unified_classes.py`, `scripts/data_v2.py`, `scripts/train_v2.py`, `scripts/evaluate_models.py`, análise dos falsos positivos; re-treinamento/validação.
- **Qualidade e entrega:** `tests/`, `docs/`, avaliação com fotografias não usadas no treino, GitHub e apresentação.

## Pontos prioritários de correção

1. **Falsos positivos:** o v2 reduziu bastante o problema (8% das fotos de teste do MyFood com algum alimento inexistente, contra 94% no v1), e o v3 ficou parecido (9,5%), mas ainda acontece. As classes mais fracas do v3 são Cebola, Peixe, Abóbora e Milho.
2. **Separação treino/avaliação:** o conjunto brasileiro possui máscaras individuais por classe; é necessário avaliar fotos novas de pratos completos com anotações multiclasse antes de afirmar acurácia geral.
3. **Estimativa de peso:** a porção inicial é a porção padrão da categoria escalada pela área que o alimento ocupa **no prato**: o recorte ou, sem recorte, o retângulo em volta da comida (12% do prato = porção padrão; limitada a 25%–200%). Assim a distância da câmera não muda a porção. É uma heurística, **não uma medição de massa**. O usuário deve informar os gramas.
4. **Tipo de preparo:** "frango" visual é mapeado provisoriamente para peito grelhado; o usuário precisa confirmar ou substituir.
5. **Recomendações:** atualmente são regras e comparações com metas por refeição. Não é orientação clínica.

## Como demonstrar com transparência

1. Configurar metas.
2. Enviar a foto de um prato, ajustar o recorte sugerido e observar as sugestões. Use fotos tiradas pela equipe; a página não tem mais fotos de exemplo.
3. Se um item estiver errado, abrir a busca TACO e substituí-lo por um preparo correto; mostrar o recálculo de nutrientes.
4. Mostrar que o módulo visual chama uma rede local (`app/vision.py`) e usa pesos de `models/unified_21class.pt`.
5. Mostrar que o treinamento lê **MyFood, Brazilian Food Images e MyFoodRepo** (`scripts/train_v2.py`), explicar a supervisão parcial das máscaras brasileiras e a comparação v2 × v3 (`docs/RESUMO_MUDANCAS.md`, seção 3).
6. Exibir os testes (`python -m pytest -q`) e ser claro sobre erros da IA.

## GitHub

O repositório é https://github.com/Ladstone/Projeto-IA (branch `main`). Antes de cada envio, conferir `git status` e o conteúdo que será publicado. **Evitar subir `.venv`, arquivos ZIP brutos dos datasets, `models/runs/` (checkpoints de treino), `profile.sqlite3`, imagens privadas e informações pessoais.** O `.gitignore` já exclui esses itens.

## Limitações de redistribuição

Os arquivos ZIP originais de imagens não estão no repositório por tamanho; as fontes e licenças estão em `docs/DATASETS.md` (MyFood: CC BY 4.0; Brazilian Food Images: sem licença declarada, não republicar). As cinco fotos em `tests/fixtures/` são do MyFood, usadas só pelos testes, com crédito em `tests/fixtures/CREDITOS.md`. A fonte TACO deve permanecer citada.
