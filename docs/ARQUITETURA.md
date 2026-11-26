# Arquitetura técnica

```text
Upload da foto (HTML/JS)
       |
       v
Recorte no prato (sugerido por POST /api/crop-suggestion, editável; opcional)
       |
       v
POST /api/analyze (FastAPI)
       |
       v
Visão (app/vision.py), padrão = combinação de três modelos MobileNetV3 + LR-ASPP:
  VisionService          v2, 21 classes: decide o prato feito comum
  CombinedVisionService  + 7 classes novas do v3 (foto inteira vai para o v3 se uma delas cobre >= 10%)
  ClassOverrideService   + purê do v3d por cima do resultado
  ClassOverrideService   + tomate e batata frita do v3 (o v3 roda uma vez só por foto)
       |
       +--> Máscaras e diagnóstico por classe (hipóteses)
       |
       v
Mapeamento provisório classe -> registro TACO (editável)
       |
       v
Seleção/correção de alimento e gramas no frontend
       |
       v
POST /api/recalculate -> TACO 597 / 26 campos -> recomendações por regra
       |                                         ^
       |                                         |
       |                      GET/PUT /api/profile -> SQLite local (metas diárias)
       v
Histórico: o prato analisado é salvo no SQLite (tabela meals, com miniatura da foto);
           correções atualizam o mesmo registro -> GET /api/meals?mes=AAAA-MM (calendário)
       |
       v
Feedback (opcional): app/llm.py envia SÓ alimentos, gramas, totais e metas para a Gemini API,
           que devolve o texto. POST /api/feedback/prato (automático) e /api/feedback/mes (botão)
```

- **Interface:** `app/static/index.html` e `app/static/app.js`, sem framework, consumindo JSON REST. A única ligação externa é opcional: o texto do feedback (Gemini API), feita pelo backend; a foto nunca é enviada.
- **Backend:** FastAPI `app/main.py`, upload limitado a 8 MB, JPEG/PNG/WebP. Servidor local em `127.0.0.1` por padrão.
- **IA:** PyTorch + torchvision `app/vision.py`. Padrão: `models/unified_21class.pt` (v2) + `unified_28class_v3.pt` + `unified_28class_v3d.pt`, montados por `carregar_visao` em `app/main.py`. Arquitetura e número de classes são lidos do `.json` de cada modelo. `PRATOMETRIA_MODEL` sobe um modelo só (ex.: o v2 de backup); `PRATOMETRIA_MODEL_NOVAS` e `PRATOMETRIA_MODEL_PURE` montam outra combinação, rótulos em `app/unified_classes.py`. Limiar por classe nos metadados `.json`; diagnóstico disponibilizado em `vision_debug` para depuração.
- **Nutrição:** `app/nutrition.py`, fonte TACO 2011. O ID da classe de rede NÃO equivale ao código do alimento TACO.
- **Persistência:** SQLite local em `app/data/profile.sqlite3`: um perfil (metas), os pratos do histórico (`meals`) e o feedback de cada mês (`feedback_mes`). Não há múltiplos usuários.
- **Feedback por IA:** `app/llm.py`, Gemini API (plano gratuito), chave em `.env`. O app calcula os números e o modelo só redige, em JSON com esquema fixo. Fila de modelos com limite de tempo; sem resposta, valem as recomendações por regra.
- **Modelagem:** 28 classes são categorias visuais; 597 alimentos correspondem a descrições nutricionais, incluindo variações de preparo não distinguíveis visualmente.

### Pontos de alteração

- Alterar rótulos/correspondência TACO: `app/unified_classes.py` (sem mudar IDs do modelo inadvertidamente).
- Alterar decisões/limiares: `app/vision.py` e `models/unified_21class.json`; revalidar após ajustes.
- Alterar área mínima de região: `MIN_REGION_FRACTION` em `app/vision.py` (usada pelo app e por `scripts/evaluate_models.py`); sugestão de gramas: `suggest_grams` em `app/main.py`.
- Alterar sugestão por metas: `app/nutrition.py`.
- Alterar o cálculo das metas pelas medidas (objetivos, fatores de atividade, g/kg de proteína, % de gordura, pisos): `app/metas.py`.
- Alterar interface: `app/static/index.html` e `app/static/app.js`.
- Alterar o tom ou o formato do feedback, ou a fila de modelos: `app/llm.py`. Estatísticas do mês: `estatisticas_mes` em `app/main.py`.
- Alterar as regras da combinação (10%, classes do especialista): `CombinedVisionService` e `ClassOverrideService` em `app/vision.py`.
- Novo treino: `scripts/train_v2.py` → `scripts/export_v2.py` → `scripts/evaluate_models.py`, com os três ZIPs originais, sendo o MyFoodRepo opcional via `--myfoodrepo` (comandos em `docs/RESUMO_MUDANCAS.md`).

### Endpoints

`GET /`, `GET /api/health`, `GET /api/model/info`, `POST /api/crop-suggestion`, `POST /api/analyze`, `GET/PUT /api/profile`, `GET /api/profile/opcoes`, `POST /api/profile/calcular`, `POST /api/recalculate`, `GET /api/foods`, `GET /api/taco/metadata`, `GET /api/taco/{id}`, `GET /api/taco/csv`, `GET /api/taco/pdf`, `GET /api/meals`, `GET /api/meals/{id}/foto`, `PUT/DELETE /api/meals/{id}`, `POST /api/feedback/prato`, `GET/POST /api/feedback/mes`.

### Privacidade e deploy

Não há login/autenticação nem banco multiusuário. O endpoint para salvar metas sobrescreve um único perfil, e o histórico guarda miniaturas das fotos no computador. Use **somente localmente** para demonstração; não publique em um servidor acessível pela internet sem autenticação, controles de dados e adequação à LGPD.
