# Guia de repasse para a equipe

## Objetivo da entrega

Atender ao enunciado acadêmico: **módulo de interface, módulo de IA, módulo de resposta**, apresentação do funcionamento e link para o repositório GitHub.

**O que NÃO afirmar:** que todas as refeições são identificadas corretamente, que a rede reconhece 597 classes, ou que a foto mede peso de alimentos. A rede prevê 21 categorias; a TACO fornece informações de 597 registros após seleção do preparo.

## Primeiro acesso

1. Baixar este ZIP e extrair a pasta **PratoMetria**.
2. Abrir o PowerShell nessa pasta. Seguir os três comandos do `README.md`.
3. Abrir `http://127.0.0.1:8000/api/health`: verificar `model_ready: true`, `taco_records: 597` e `mode: local-only`.
4. Abrir `/`: salvar metas, selecionar uma foto, analisar e revisar alimentos.
5. Expandir **Diagnóstico do modelo (para desenvolvedores)**. A saída é calculada para cada imagem. Se o modelo confundir frango com arroz, registrar o caso e guardar a foto com permissão.
6. Corrigir os alimentos pela busca TACO, informar as quantidades em gramas e confirmar as recomendações.

## Responsabilidades recomendadas

- **Frontend:** `app/static/index.html` + `app/static/app.js`, responsividade, mensagens de erro, facilidade para editar rótulos, testes manuais.
- **Backend e nutrição:** `app/main.py`, `app/nutrition.py`, revisão das correspondências de preparo, tratamento de erros e persistência de perfil.
- **IA:** `app/vision.py`, `app/unified_classes.py`, `scripts/train_unified.py`, análise dos falsos positivos; re-treinamento/validação.
- **Qualidade e entrega:** `tests/`, `docs/`, avaliação com fotografias não usadas no treino, GitHub e apresentação.

## Pontos prioritários de correção

1. **Falsos positivos repetidos:** a rede por vezes identifica classes ausentes. Ajustar estratégia de detecção, dataset e calibração sem inventar rótulos corretos.
2. **Separação treino/avaliação:** o conjunto brasileiro possui máscaras individuais por classe; é necessário avaliar fotos novas de pratos completos com anotações multiclasse antes de afirmar acurácia geral.
3. **Estimativa de peso:** a porção inicial é a porção padrão da categoria escalada pela área visível (20% da foto = porção padrão; limitada a 25%–150%). É uma heurística, **não uma medição de massa**. O usuário deve informar os gramas.
4. **Tipo de preparo:** "frango" visual é mapeado provisoriamente para peito grelhado; o usuário precisa confirmar ou substituir.
5. **Recomendações:** atualmente são regras e comparações com metas por refeição. Não é orientação clínica.

## Como demonstrar com transparência

1. Configurar metas.
2. Enviar a foto de demonstração `app/static/examples/lunch.jpg` e observar as sugestões.
3. Se um item estiver errado, abrir a busca TACO e substituí-lo por um preparo correto; mostrar o recálculo de nutrientes.
4. Mostrar que o módulo visual chama uma rede local (`app/vision.py`) e usa pesos de `models/unified_21class.pt`.
5. Mostrar que o treinamento lê **MyFood e Brazilian Food Images** (`scripts/train_unified.py`) e explicar a supervisão parcial das máscaras brasileiras.
6. Exibir os testes (`python -m pytest -q`) e ser claro sobre erros da IA.

## GitHub

Criar repositório GitHub vazio (por exemplo `pratometria-prototipo`) e, **na pasta `PratoMetria`**:

```powershell
git init
git add .
git commit -m "Protótipo PratoMetria com IA local e TACO"
git branch -M main
git remote add origin https://github.com/ORGANIZACAO/pratometria-prototipo.git
git push -u origin main
```

Substituir o URL por aquele criado pela equipe; não existe link de GitHub criado automaticamente neste pacote. Conferir `git status` e o conteúdo que será publicado. **Evitar subir `.venv`, arquivos ZIP brutos dos datasets, `profile.sqlite3`, imagens privadas e informações pessoais.**

## Limitações de redistribuição

Os arquivos ZIP originais de imagens não foram incorporados ao repositório por tamanho e por ausência de verificação das respectivas licenças. As seis fotografias de exemplo incluídas precisam ter sua permissão/licença conferida pela equipe **antes de publicar um repositório público**. Em caso de dúvida, removê-las e também os botões de exemplo do frontend. A fonte TACO deve permanecer citada.
