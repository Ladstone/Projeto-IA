"""Avaliação final no TESTE (nunca usado no treino, na escolha de época ou de limiares).

Compara modelos no mesmo teste novo, com a MESMA regra do app (VisionService.predict:
máscara exclusiva com os limiares do .json de cada modelo + área mínima de detecção).

  python scripts/evaluate_models.py --myfood "...zip" --brazil "...zip" \
      --model v1=models/unified_21class_v1.pt --model v2=models/unified_21class_v2.pt

Grupos reportados SEPARADAMENTE:
  1. MyFood test (fotos reais, máscaras multiclasse): IoU, presença P/R/F1, falsos positivos.
  2. Brasileiro test novo (fotos reais, 1 alimento anotado/foto): IoU e recall da classe anotada.
  3. Imagens sintéticas sem comida (não são fotos de pratos): taxa de detecção espúria.
"""
import argparse, io, json, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.unified_classes import N_CLASS, CLASS_LABELS
from app.vision import VisionService, detected_classes, MIN_REGION_FRACTION
from scripts.data_v2 import (load_records, drop_near_duplicates, presence_status, synthetic_eval_set,
                             TRAIN_SYNTHETIC_KINDS, MYFOOD_ONLY_CLASSES, IGNORE)


def predict_mask(service, pil, dim):
    mask, _ = service.predict(pil)
    return np.asarray(Image.fromarray(mask).resize((dim, dim), Image.Resampling.NEAREST))


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return p, r, f


def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def eval_myfood(service, records, dim):
    inter, union = np.zeros(N_CLASS + 1), np.zeros(N_CLASS + 1)
    tp, fp, fn = np.zeros(N_CLASS + 1), np.zeros(N_CLASS + 1), np.zeros(N_CLASS + 1)
    confusion = np.zeros((N_CLASS + 1, N_CLASS + 1), dtype=np.int64)
    fp_per_image, perfect, bg_fp, bg_total, brazil_only_px, food_px = [], 0, 0, 0, 0, 0
    for r in records:
        mask = predict_mask(service, Image.open(io.BytesIO(r['raw'])).convert('RGB'), dim)
        gt, valid = r['label'], r['label'] != IGNORE
        for c in range(1, N_CLASS + 1):
            p, g = (mask == c) & valid, (gt == c) & valid
            inter[c] += (p & g).sum(); union[c] += (p | g).sum()
        np.add.at(confusion, (gt[valid].astype(int), mask[valid].astype(int)), 1)
        bg = gt == 0; bg_fp += int((mask[bg] > 0).sum()); bg_total += int(bg.sum())
        brazil_only_px += int(np.isin(mask[valid], [c for c in range(1, N_CLASS + 1) if c not in MYFOOD_ONLY_CLASSES]).sum())
        food_px += int(valid.sum())
        status = presence_status('myfood', gt, None, r['has_unknown_food'])
        det = detected_classes(mask)
        n_fp = n_fn = 0
        for c in range(1, N_CLASS + 1):
            s, d = status[c - 1], c in det
            if s == 1 and d: tp[c] += 1
            elif s == 1: fn[c] += 1; n_fn += 1
            elif s == 0 and d: fp[c] += 1; n_fp += 1
        fp_per_image.append(n_fp); perfect += int(n_fp == 0 and n_fn == 0)
    per_class = {}
    for c in MYFOOD_ONLY_CLASSES:
        p, rc, f = prf(tp[c], fp[c], fn[c])
        per_class[CLASS_LABELS[c]] = dict(iou=float(inter[c] / union[c]) if union[c] else None, precisao=p, recall=rc, f1=f,
                                          imagens_positivas=int(tp[c] + fn[c]))
    fp_other = {CLASS_LABELS[c]: int(fp[c]) for c in range(1, N_CLASS + 1) if c not in MYFOOD_ONLY_CLASSES and fp[c]}
    T, FP_, FN_ = tp.sum(), fp.sum(), fn.sum()
    return dict(
        imagens=len(records),
        iou_medio_classes_myfood=mean([v['iou'] for v in per_class.values()]),
        f1_presenca_macro=mean([v['f1'] for v in per_class.values()]),
        precisao_presenca_micro=prf(T, FP_, FN_)[0], recall_presenca_micro=prf(T, FP_, FN_)[1],
        falsos_positivos_por_imagem=float(np.mean(fp_per_image)),
        imagens_com_algum_falso_positivo_pct=100 * float(np.mean(np.array(fp_per_image) > 0)),
        pratos_perfeitos_pct=100 * perfect / len(records),
        fundo_marcado_como_comida_pct=100 * bg_fp / max(bg_total, 1),
        pixels_em_classes_so_brasileiras_pct=100 * brazil_only_px / max(food_px, 1),
        falsos_positivos_em_classes_so_brasileiras=fp_other,
        por_classe=per_class,
        matriz_confusao_pixels={'linhas_gt_colunas_pred_0_fundo': confusion.tolist()},
    )


def eval_brazil(service, records, dim):
    by = defaultdict(lambda: dict(inter=0, union=0, tp=0, n=0, ok=0, outro=0, nada=0, px=0))
    for r in records:
        mask = predict_mask(service, Image.open(io.BytesIO(r['raw'])).convert('RGB'), dim)
        c, m = r['cls'], r['label'].astype(bool)
        s = by[c]
        s['inter'] += int(((mask == c) & m).sum()); s['union'] += int(((mask == c) | m).sum())
        s['px'] += int(m.sum()); s['ok'] += int((mask[m] == c).sum()); s['nada'] += int((mask[m] == 0).sum())
        s['outro'] += int(((mask[m] != c) & (mask[m] != 0)).sum())
        if presence_status('brazil', r['label'], c, True)[c - 1] == 1:
            s['n'] += 1; s['tp'] += int(c in detected_classes(mask))
    per_class = {CLASS_LABELS[c]: dict(imagens=s['n'], iou=s['inter'] / s['union'] if s['union'] else None,
                                       recall_presenca=s['tp'] / s['n'] if s['n'] else None,
                                       pixels_anotados_classe_certa_pct=100 * s['ok'] / max(s['px'], 1),
                                       pixels_anotados_outra_classe_pct=100 * s['outro'] / max(s['px'], 1),
                                       pixels_anotados_sem_classe_pct=100 * s['nada'] / max(s['px'], 1))
                 for c, s in sorted(by.items())}
    tot = {k: sum(s[k] for s in by.values()) for k in ('tp', 'n', 'ok', 'outro', 'nada', 'px')}
    return dict(imagens=len(records), iou_medio=mean([v['iou'] for v in per_class.values()]),
                recall_presenca_micro=tot['tp'] / max(tot['n'], 1),
                pixels_anotados_classe_certa_pct=100 * tot['ok'] / max(tot['px'], 1),
                pixels_anotados_outra_classe_pct=100 * tot['outro'] / max(tot['px'], 1),
                pixels_anotados_sem_classe_pct=100 * tot['nada'] / max(tot['px'], 1),
                por_classe=per_class,
                observacao='Só um alimento é anotado por foto: precisão e falsos positivos de OUTRAS classes não são mensuráveis aqui.')


def eval_synthetic(service, images):
    by = defaultdict(list)
    for kind, pil in images:
        mask, _ = service.predict(pil)
        by[kind].append(len(detected_classes(mask)))
    per_kind = {k: dict(imagens=len(v), com_deteccao_pct=100 * float(np.mean(np.array(v) > 0)), deteccoes_media=float(np.mean(v)),
                        tipo_visto_no_treino_v2=k in TRAIN_SYNTHETIC_KINDS) for k, v in by.items()}
    allv = [x for v in by.values() for x in v]
    unseen = [x for k, v in by.items() if k not in TRAIN_SYNTHETIC_KINDS for x in v]
    return dict(imagens=len(allv), com_deteccao_pct=100 * float(np.mean(np.array(allv) > 0)),
                tipos_nao_vistos_com_deteccao_pct=100 * float(np.mean(np.array(unseen) > 0)), por_tipo=per_kind)


def fmt(v, pct=False):
    if v is None: return '—'
    return f'{v:.1f}%' if pct else f'{v:.3f}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--myfood', required=True); p.add_argument('--brazil', required=True)
    p.add_argument('--model', action='append', required=True, help='nome=caminho.pt (repita)')
    p.add_argument('--dimension', type=int, default=224, help='resolução da verdade de campo na comparação')
    p.add_argument('--out', default=str(ROOT / 'docs' / 'METRICAS_FASE_B.json'))
    a = p.parse_args()
    t0 = time.time()
    train_records = load_records(a.myfood, a.brazil, 'train', a.dimension)
    test_all = load_records(a.myfood, a.brazil, 'test', a.dimension, keep_raw=True)
    test, removed = drop_near_duplicates(test_all, train_records)
    myfood = [r for r in test if r['src'] == 'myfood']; brazil = [r for r in test if r['src'] == 'brazil']
    synthetic = synthetic_eval_set()
    print(f'TESTE MyFood {len(myfood)} | Brasileiro {len(brazil)} | sintéticas {len(synthetic)} | '
          f'quase-duplicatas do treino removidas {removed} ({time.time() - t0:.0f}s)', flush=True)
    report = dict(teste=dict(myfood=len(myfood), brasileiro=len(brazil), sinteticas=len(synthetic),
                             quase_duplicatas_removidas=removed, area_minima_deteccao=MIN_REGION_FRACTION,
                             regra='VisionService.predict + detected_classes, idêntica ao app'),
                  notas=['O teste brasileiro novo é metade da validação antiga do v1: nenhum dos modelos treinou nessas fotos, '
                         'mas o v1 escolheu limiares na validação antiga, o que o favorece levemente.',
                         'Imagens sintéticas não são fotos de pratos e são reportadas à parte; o v2 treinou com 4 dos 6 tipos, o v1 com nenhum.',
                         'O test_set brasileiro original (160 fotos sem máscaras) não é pontuável.'],
                  modelos={})
    for spec in a.model:
        name, path = spec.split('=', 1)
        s = time.time(); service = VisionService(ROOT / path if not Path(path).is_absolute() else Path(path))
        res = dict(arquivo=path, arquitetura=service.architecture, resolucao=service.resolution,
                   myfood=eval_myfood(service, myfood, a.dimension), brasileiro=eval_brazil(service, brazil, a.dimension),
                   sem_comida_sinteticas=eval_synthetic(service, synthetic))
        res['segundos'] = round(time.time() - s)
        report['modelos'][name] = res
        print(f'{name} avaliado em {res["segundos"]}s', flush=True)
    Path(a.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')

    names = list(report['modelos'])
    rows = [('MyFood: IoU médio (8 classes)', 'myfood', 'iou_medio_classes_myfood', False),
            ('MyFood: F1 de presença (macro)', 'myfood', 'f1_presenca_macro', False),
            ('MyFood: precisão de presença', 'myfood', 'precisao_presenca_micro', False),
            ('MyFood: recall de presença', 'myfood', 'recall_presenca_micro', False),
            ('MyFood: falsos positivos por imagem', 'myfood', 'falsos_positivos_por_imagem', False),
            ('MyFood: imagens com algum falso positivo', 'myfood', 'imagens_com_algum_falso_positivo_pct', True),
            ('MyFood: pratos 100% corretos', 'myfood', 'pratos_perfeitos_pct', True),
            ('MyFood: fundo marcado como comida', 'myfood', 'fundo_marcado_como_comida_pct', True),
            ('Brasileiro: IoU médio da classe anotada', 'brasileiro', 'iou_medio', False),
            ('Brasileiro: recall de presença', 'brasileiro', 'recall_presenca_micro', False),
            ('Brasileiro: pixels anotados na classe certa', 'brasileiro', 'pixels_anotados_classe_certa_pct', True),
            ('Sintéticas: imagens com detecção espúria', 'sem_comida_sinteticas', 'com_deteccao_pct', True),
            ('Sintéticas (tipos não vistos): com detecção', 'sem_comida_sinteticas', 'tipos_nao_vistos_com_deteccao_pct', True)]
    print('\n| Métrica | ' + ' | '.join(names) + ' |\n|---|' + '---:|' * len(names))
    for label, group, key, pct in rows:
        print(f'| {label} | ' + ' | '.join(fmt(report['modelos'][n][group][key], pct) for n in names) + ' |')
    for group, key in (('myfood', 'f1'), ('brasileiro', 'iou')):
        print(f'\nPor classe ({group}, {key}):')
        for cls in report['modelos'][names[0]][group]['por_classe']:
            print(f'  {cls}: ' + ' | '.join(f'{n} {fmt(report["modelos"][n][group]["por_classe"][cls][key])}' for n in names))
    print('\nRelatório completo em', a.out)


if __name__ == '__main__':
    main()
