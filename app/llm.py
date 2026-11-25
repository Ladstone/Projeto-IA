"""Texto de feedback escrito por um LLM (Gemini API, plano gratuito).

O LLM só REDIGE: todos os números (gramas, kcal, metas, médias do mês) são calculados pelo app
com a TACO e entram prontos no pedido. Sem chave, sem internet ou com o serviço ocupado, as funções
levantam ``LLMIndisponivel`` e o app mostra as dicas por regra de sempre.

Configuração no arquivo ``.env`` (modelo em ``.env.example``): ``GEMINI_API_KEY`` e, opcional,
``GEMINI_MODEL``. Só a lista de alimentos, gramas, totais e metas sai do computador; a foto não.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = 'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'
# Plano gratuito, medido em 25/11/2026: os Flash-Lite respondem em 1-7 s; o Flash comum esgota a cota (429)
# ou leva mais de 20 s; os 2.5 não existem mais para chaves novas (404). Por isso o Lite vem primeiro.
PADRAO = 'gemini-flash-lite-latest'
RESERVAS = ['gemini-3.5-flash-lite', 'gemini-3.1-flash-lite', 'gemini-flash-latest']
TIMEOUT = 12          # por modelo: um modelo lento não segura a tela
ORCAMENTO = 20        # tempo total por pedido, somando as tentativas
PAUSA_COTA = 600      # modelo com 429/404 fica 10 min fora da fila
_pausados: dict[str, float] = {}
_cache: dict[str, tuple[dict, str]] = {}   # mesmo pedido -> mesma resposta (corrigir e desfazer não gasta cota)
log = logging.getLogger('uvicorn.error')


class LLMIndisponivel(RuntimeError):
    """Sem chave, sem rede, cota esgotada ou resposta inválida: usar as dicas por regra."""


def carregar_env(path: Path = ROOT / '.env'):
    """Lê KEY=valor do .env sem sobrescrever variáveis que já existem no ambiente."""
    if not path.exists():
        return
    for linha in path.read_text(encoding='utf-8').splitlines():
        linha = linha.strip()
        if not linha or linha.startswith('#') or '=' not in linha:
            continue
        k, v = linha.split('=', 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


carregar_env()


def configurado() -> bool:
    return bool(os.getenv('GEMINI_API_KEY', '').strip())


def modelos() -> list[str]:
    principal = os.getenv('GEMINI_MODEL', '').strip() or PADRAO
    return [principal] + [m for m in RESERVAS if m != principal]


def _chamar(model: str, system: str, prompt: str, schema: dict, max_tokens: int, timeout: float = TIMEOUT) -> dict:
    body = {
        'systemInstruction': {'parts': [{'text': system}]},
        'contents': [{'role': 'user', 'parts': [{'text': prompt}]}],
        'generationConfig': {'temperature': 0.6, 'maxOutputTokens': max_tokens,
                             'responseMimeType': 'application/json', 'responseSchema': schema},
    }
    req = urllib.request.Request(API.format(model=model), data=json.dumps(body).encode('utf-8'),
                                 headers={'x-goog-api-key': os.environ['GEMINI_API_KEY'].strip(),
                                          'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    cand = (data.get('candidates') or [{}])[0]
    if cand.get('finishReason') not in (None, 'STOP'):
        raise LLMIndisponivel(f'resposta incompleta ({cand.get("finishReason")})')
    texto = ''.join(p.get('text', '') for p in cand.get('content', {}).get('parts', []) if not p.get('thought'))
    try:
        return json.loads(texto)
    except json.JSONDecodeError as exc:
        raise LLMIndisponivel('resposta fora do formato') from exc


def gerar_json(system: str, prompt: str, schema: dict, max_tokens: int = 2048) -> tuple[dict, str]:
    """Devolve (json da resposta, modelo usado). Tenta o modelo principal e depois os de reserva."""
    if not configurado():
        raise LLMIndisponivel('GEMINI_API_KEY não configurada')
    chave = json.dumps([system, prompt, schema, max_tokens], ensure_ascii=False, sort_keys=True)
    if chave in _cache:
        return _cache[chave]
    inicio, erros = time.monotonic(), []
    fila = [m for m in modelos() if _pausados.get(m, 0) < time.monotonic()] or modelos()
    for model in fila:
        resta = ORCAMENTO - (time.monotonic() - inicio)
        if resta < 2:
            erros.append('tempo esgotado')
            break
        try:
            resultado = (_chamar(model, system, prompt, schema, max_tokens, min(TIMEOUT, resta)), model)
            log.info('feedback LLM: %s em %.1f s%s', model, time.monotonic() - inicio, f' (antes: {"; ".join(erros)})' if erros else '')
            if len(_cache) > 200:
                _cache.clear()
            _cache[chave] = resultado
            return resultado
        except urllib.error.HTTPError as exc:
            erros.append(f'{model}: HTTP {exc.code}')
            if exc.code in (400, 401, 403):  # chave ou pedido inválido: o outro modelo não resolve
                break
            if exc.code in (404, 429):       # modelo inexistente ou cota esgotada: tira da fila por um tempo
                _pausados[model] = time.monotonic() + PAUSA_COTA
        except (urllib.error.URLError, TimeoutError, OSError, LLMIndisponivel) as exc:
            erros.append(f'{model}: {exc}')
    log.warning('feedback LLM indisponível em %.1f s: %s', time.monotonic() - inicio, '; '.join(erros))
    raise LLMIndisponivel('; '.join(erros))


# ---- Prompts -------------------------------------------------------------------------------------
TOM = ('Você escreve o feedback do PratoMetria, um app brasileiro que analisa fotos de refeições e calcula '
       'nutrientes pela Tabela TACO. Escreva em português do Brasil, com tom acolhedor, direto e prático, '
       'como um amigo que entende de nutrição. Sem culpa, sem julgamento e sem linguagem de dieta restritiva. '
       'Use SOMENTE os números que vierem nos dados; não invente valores, não calcule novos números e não cite '
       'nutrientes que não estão nos dados. As gramas são estimativas feitas pela foto. Não faça diagnóstico nem '
       'recomendação médica. Sugira trocas e ajustes com alimentos comuns no Brasil, em medidas caseiras '
       '(uma concha, um filé pequeno, meia xícara), sem inventar gramas ou nutrientes para elas. Escreva números '
       'no formato brasileiro: vírgula decimal, kcal sem casas decimais e gramas arredondadas.')

SCHEMA_PRATO = {
    'type': 'OBJECT',
    'properties': {
        'resumo': {'type': 'STRING', 'description': 'Uma ou duas frases sobre o prato em relação à meta da refeição.'},
        'positivo': {'type': 'STRING', 'description': 'Um ponto positivo concreto do prato.'},
        'sugestoes': {'type': 'ARRAY', 'items': {'type': 'STRING'},
                      'description': 'Uma ou duas sugestões práticas e curtas para este prato ou o próximo.'},
    },
    'required': ['resumo', 'positivo', 'sugestoes'],
}

SCHEMA_MES = {
    'type': 'OBJECT',
    'properties': {
        'resumo': {'type': 'STRING', 'description': 'Dois a quatro frases sobre o mês, pelos pratos registrados.'},
        'destaques': {'type': 'ARRAY', 'items': {'type': 'STRING'}, 'description': 'Dois ou três pontos positivos.'},
        'atencao': {'type': 'ARRAY', 'items': {'type': 'STRING'}, 'description': 'Um a três padrões para melhorar.'},
        'metas': {'type': 'ARRAY', 'items': {'type': 'STRING'},
                  'description': 'Duas ou três metas simples e mensuráveis para o próximo mês.'},
    },
    'required': ['resumo', 'destaques', 'atencao', 'metas'],
}


def feedback_prato(dados: dict) -> tuple[dict, str]:
    prompt = ('Escreva o feedback deste prato falando diretamente com a pessoa ("você"). Seja breve: cada campo com '
              'no máximo duas frases, e cite no máximo dois números.\n\n'
              'Dados (JSON):\n' + json.dumps(dados, ensure_ascii=False, indent=1))
    return gerar_json(TOM, prompt, SCHEMA_PRATO, max_tokens=1024)


def feedback_mes(dados: dict) -> tuple[dict, str]:
    prompt = ('Escreva o feedback do mês a partir do histórico de pratos, falando diretamente com a pessoa ("você"). '
              'A pessoa pode não ter registrado todas as refeições: diga isso UMA vez, no resumo, e não conclua quanto '
              'ela comeu no dia inteiro quando houver poucos pratos por dia. Não repita "pratos registrados" em todo item. '
              'Não despeje os números: escolha os dois ou três mais importantes para o objetivo dela e explique o que '
              'significam na prática. Destaques e pontos de atenção devem ser padrões concretos (alimentos, porções, '
              'variedade), não listas de dados. Cada item com no máximo duas frases.\n\n'
              'Dados do mês (JSON):\n' + json.dumps(dados, ensure_ascii=False, indent=1))
    return gerar_json(TOM, prompt, SCHEMA_MES, max_tokens=2048)
