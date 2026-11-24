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
VisionService.predict -> pesos PyTorch unified_21class.pt (v2: MobileNetV3 + LR-ASPP, 21 classes; v3 de 28 classes opcional;
                         CombinedVisionService v2+v3 com PRATOMETRIA_MODEL_NOVAS e purê do v3d
                         com PRATOMETRIA_MODEL_PURE / ClassOverrideService, em teste na porta 8001)
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
POST /api/recalculate -> TACO 597 / 26 campos -> recomendações
       ^
       |
GET/PUT /api/profile -> SQLite local (metas diárias)
```

- **Interface:** `app/static/index.html` e `app/static/app.js`, sem framework, consumindo JSON REST. Não há ligação com serviços externos de IA.
- **Backend:** FastAPI `app/main.py`, upload limitado a 8 MB, JPEG/PNG/WebP. Servidor local em `127.0.0.1` por padrão.
- **IA:** PyTorch + torchvision `app/vision.py`, `models/unified_21class.pt` (v2, MobileNetV3-Large + LR-ASPP, 224 px, 21 classes; arquitetura e número de classes são lidos do `.json`, então o v3 de 28 classes e o v1 U-Net também carregam; a variável `PRATOMETRIA_MODEL` aponta outro arquivo para teste), rótulos em `app/unified_classes.py`. Limiar por classe nos metadados `.json`; diagnóstico disponibilizado em `vision_debug` para depuração.
- **Nutrição:** `app/nutrition.py`, fonte TACO 2011. O ID da classe de rede NÃO equivale ao código do alimento TACO.
- **Persistência:** SQLite de um único perfil local em `app/data/profile.sqlite3`; não adequado ainda para múltiplos usuários.
- **Modelagem:** 28 classes são categorias visuais; 597 alimentos correspondem a descrições nutricionais, incluindo variações de preparo não distinguíveis visualmente.

### Pontos de alteração

- Alterar rótulos/correspondência TACO: `app/unified_classes.py` (sem mudar IDs do modelo inadvertidamente).
- Alterar decisões/limiares: `app/vision.py` e `models/unified_21class.json`; revalidar após ajustes.
- Alterar área mínima de região: `MIN_REGION_FRACTION` em `app/vision.py` (usada pelo app e por `scripts/evaluate_models.py`); sugestão de gramas: `suggest_grams` em `app/main.py`.
- Alterar sugestão por metas: `app/nutrition.py`.
- Alterar o cálculo das metas pelas medidas (objetivos, fatores de atividade, g/kg de proteína, % de gordura, pisos): `app/metas.py`.
- Alterar interface: `app/static/index.html` e `app/static/app.js`.
- Novo treino: `scripts/train_v2.py` → `scripts/export_v2.py` → `scripts/evaluate_models.py`, com os três ZIPs originais, sendo o MyFoodRepo opcional via `--myfoodrepo` (comandos em `docs/RESUMO_MUDANCAS.md`).

### Endpoints

`GET /`, `GET /api/health`, `GET /api/model/info`, `POST /api/crop-suggestion`, `POST /api/analyze`, `GET/PUT /api/profile`, `GET /api/profile/opcoes`, `POST /api/profile/calcular`, `POST /api/recalculate`, `GET /api/foods`, `GET /api/taco/metadata`, `GET /api/taco/{id}`, `GET /api/taco/csv`, `GET /api/taco/pdf`.

### Privacidade e deploy

Não há login/autenticação nem banco multiusuário. O endpoint para salvar metas sobrescreve um único perfil de exemplo. Use **somente localmente** para demonstração; não publique em um servidor acessível pela internet sem autenticação, controles de dados e adequação à LGPD.
