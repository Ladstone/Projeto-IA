"""Metas diárias calculadas pelas medidas da pessoa (estimativa educacional, não prescrição).

Passos:
1. Taxa metabólica basal (TMB) pela equação de Mifflin-St Jeor (1990):
   10 x peso(kg) + 6,25 x altura(cm) - 5 x idade + 5 (masculino) ou - 161 (feminino).
2. Gasto energético total (GET) = TMB x fator de atividade.
3. Meta de energia = GET ajustado pelo objetivo, nunca abaixo de um piso de segurança.
4. Proteína em g por kg de peso (faixas da ISSN, 1,2 a 2,2 g/kg conforme o objetivo).
   Com IMC >= 30, usa o peso de referência de IMC 25 para não superestimar a proteína.
5. Gordura como % da energia (faixa AMDR de 20 a 35%); carboidrato fica com o restante.
"""

ATIVIDADES = {
    'sedentario': (1.2, 'Sedentário (pouco ou nenhum exercício)'),
    'leve': (1.375, 'Leve (exercício 1 a 3 dias por semana)'),
    'moderado': (1.55, 'Moderado (exercício 3 a 5 dias por semana)'),
    'intenso': (1.725, 'Intenso (exercício 6 a 7 dias por semana)'),
    'muito_intenso': (1.9, 'Muito intenso (treino 2x por dia ou trabalho físico pesado)'),
}

# id: (rótulo, ajuste de energia sobre o GET, proteína g/kg, fração da energia em gordura, explicação)
OBJETIVOS = {
    'emagrecimento': ('Emagrecer (perder gordura)', -0.20, 1.8, 0.25,
                      'déficit de 20% para perder gordura; proteína alta ajuda a preservar músculo'),
    'definicao': ('Definição (perder gordura mantendo músculo)', -0.10, 2.0, 0.25,
                  'déficit moderado de 10% com proteína bem alta, para quem já treina'),
    'manutencao': ('Manter o peso', 0.0, 1.2, 0.30,
                   'energia igual ao gasto estimado'),
    'hipertrofia': ('Hipertrofia (ganhar massa muscular)', 0.10, 1.8, 0.25,
                    'superávit de 10% para ganhar massa com pouco ganho de gordura; depende de treino de força'),
    'resistencia': ('Desempenho em esportes de resistência', 0.0, 1.4, 0.25,
                    'energia igual ao gasto, com mais carboidrato para treinos longos'),
}

# Versões antigas do perfil usavam outros ids.
OBJETIVOS_ANTIGOS = {'ganho': 'hipertrofia', 'perda': 'emagrecimento', 'equilibrio': 'manutencao'}
DEFICIT = {'emagrecimento', 'definicao'}
PISO_KCAL = {'feminino': 1200, 'masculino': 1500}


def normaliza_objetivo(objetivo):
    return OBJETIVOS_ANTIGOS.get(objetivo, objetivo)


def calcular_metas(sexo, idade, peso_kg, altura_cm, atividade, objetivo):
    """Devolve metas diárias (kcal, proteína, carboidrato, gordura) e a explicação de cada passo."""
    objetivo = normaliza_objetivo(objetivo)
    if sexo not in PISO_KCAL:
        raise ValueError('Sexo deve ser "feminino" ou "masculino" (a equação usa o sexo biológico)')
    if atividade not in ATIVIDADES:
        raise ValueError('Nível de atividade inválido')
    if objetivo not in OBJETIVOS:
        raise ValueError('Objetivo inválido')
    if not 18 <= idade <= 100:
        raise ValueError('O cálculo automático é só para adultos (18 a 100 anos). '
                         'Para menores, as metas devem vir de um profissional de saúde.')
    if not 30 <= peso_kg <= 300 or not 120 <= altura_cm <= 230:
        raise ValueError('Confira o peso (30 a 300 kg) e a altura (120 a 230 cm)')

    fator, nome_atividade = ATIVIDADES[atividade]
    nome_objetivo, ajuste, prot_kg, frac_gordura, motivo = OBJETIVOS[objetivo]
    tmb = 10 * peso_kg + 6.25 * altura_cm - 5 * idade + (5 if sexo == 'masculino' else -161)
    get = tmb * fator
    kcal = get * (1 + ajuste)
    avisos = []
    if objetivo in DEFICIT and kcal < PISO_KCAL[sexo]:
        kcal = PISO_KCAL[sexo]
        avisos.append(f'A meta foi limitada ao mínimo de {PISO_KCAL[sexo]} kcal/dia. '
                      'Dietas abaixo disso precisam de acompanhamento profissional.')
    kcal = round(kcal / 10) * 10

    altura_m = altura_cm / 100
    imc = peso_kg / altura_m ** 2
    peso_proteina = peso_kg
    if imc >= 30:
        peso_proteina = 25 * altura_m ** 2
        avisos.append(f'Com IMC {imc:.1f}, a proteína foi calculada sobre um peso de referência de '
                      f'{peso_proteina:.0f} kg (IMC 25), para não superestimá-la.')
    proteina = round(prot_kg * peso_proteina)
    gordura = round(kcal * frac_gordura / 9)
    carbo = round(max(0, kcal - 4 * proteina - 9 * gordura) / 4)

    explicacao = [
        f'Gasto em repouso (TMB, Mifflin-St Jeor): {tmb:.0f} kcal/dia.',
        f'Com atividade "{nome_atividade.split(" (")[0].lower()}" (x{fator}): gasto total de {get:.0f} kcal/dia.',
        f'Objetivo "{nome_objetivo}": {motivo} -> {kcal} kcal/dia.',
        f'Proteína: {prot_kg} g por kg -> {proteina} g. Gordura: {frac_gordura:.0%} da energia -> {gordura} g. '
        f'Carboidrato: o restante -> {carbo} g.',
    ]
    return {
        'objetivo': objetivo, 'kcal': kcal, 'proteina_g': proteina, 'carboidrato_g': carbo, 'lipideos_g': gordura,
        'detalhes': {'tmb': round(tmb), 'gasto_total': round(get), 'fator_atividade': fator,
                     'ajuste_objetivo_percent': round(ajuste * 100), 'imc': round(imc, 1)},
        'explicacao': explicacao, 'avisos': avisos,
    }
