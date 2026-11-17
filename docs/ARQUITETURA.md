# Arquitetura técnica

```text
Upload da foto (HTML/JS)
       |
       v
POST /api/analyze (FastAPI)
       |
       v
VisionService.predict -> pesos PyTorch unified_21class.pt
       |
       +--> Máscaras e diagnóstico de 21 classes (hipóteses)
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
- **IA:** PyTorch `app/vision.py`, `models/unified_21class.pt`, rótulos em `app/unified_classes.py`. Limiar por classe nos metadados `.json`; diagnóstico disponibilizado em `vision_debug` para depuração.
- **Nutrição:** `app/nutrition.py`, fonte TACO 2011. O ID da classe de rede NÃO equivale ao código do alimento TACO.
- **Persistência:** SQLite de um único perfil local em `app/data/profile.sqlite3`; não adequado ainda para múltiplos usuários.
- **Modelagem:** 21 classes são categorias visuais; 597 alimentos correspondem a descrições nutricionais, incluindo variações de preparo não distinguíveis visualmente.

### Pontos de alteração

- Alterar rótulos/correspondência TACO: `app/unified_classes.py` (sem mudar IDs do modelo inadvertidamente).
- Alterar decisões/limiares: `app/vision.py` e `models/unified_21class.json`; revalidar após ajustes.
- Alterar quantidade mínima de região/score: `app/main.py` em `/api/analyze`.
- Alterar sugestão por metas: `app/nutrition.py`.
- Alterar interface: `app/static/index.html` e `app/static/app.js`.
- Novo treino: `scripts/train_unified.py` com os dois ZIPs originais.

### Endpoints

`GET /`, `GET /api/health`, `GET /api/model/info`, `POST /api/analyze`, `GET/PUT /api/profile`, `POST /api/recalculate`, `GET /api/foods`, `GET /api/taco/metadata`, `GET /api/taco/{id}`, `GET /api/taco/csv`, `GET /api/taco/pdf`.

### Privacidade e deploy

Não há login/autenticação nem banco multiusuário. O endpoint para salvar metas sobrescreve um único perfil de exemplo. Use **somente localmente** para demonstração; não publique em um servidor acessível pela internet sem autenticação, controles de dados e adequação à LGPD.
