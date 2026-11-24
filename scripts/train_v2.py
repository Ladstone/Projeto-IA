"""Treino v2: MobileNetV3-Large (ImageNet) + LR-ASPP, N_CLASS classes, supervisão negativa.

Uso (piloto e treino completo usam pastas de execução diferentes):
  python scripts/train_v2.py --myfood "...MyFood Dataset.zip" --brazil "...Brazilian food images.zip" --epochs 3 --run-dir models/runs/pilot
  python scripts/train_v2.py ... --epochs 20 --run-dir models/runs/v2 --out models/unified_21class_v2.pt
  Com MyFoodRepo (classes 22+): acrescente --myfoodrepo "...archive.zip"

Retomar: rode o MESMO comando. Se ``<run-dir>/last.pt`` existir, o treino continua da
última época concluída (modelo, otimizador, agenda de LR e estados aleatórios).

Este script NUNCA lê o conjunto de teste: época e limiares são escolhidos só na validação.
"""
import argparse, json, math, os, random, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.unified_classes import N_CLASS, CLASS_LABELS
from app.vision import build_model, MIN_REGION_FRACTION
from scripts.data_v2 import load_records, drop_near_duplicates, FoodDataset, sample_weights


def datasets_used(myfoodrepo):
    return ['MyFood Dataset', 'Brazilian Food Images'] + (['MyFoodRepo v2.0'] if myfoodrepo else [])

THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.05), 2)
MIN_VAL_POSITIVES = 3  # classe com menos imagens positivas na validação não entra na média de F1


class Logger:
    def __init__(self, path):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, *parts):
        line = time.strftime('%Y-%m-%d %H:%M:%S ') + ' '.join(str(p) for p in parts)
        print(line, flush=True)
        with self.path.open('a', encoding='utf-8') as f:
            f.write(line + '\n')


def masked_loss(logits, target, known):
    """BCE só em pixels conhecidos (positivos com peso 5) + Dice nos pares (imagem, classe) com positivos."""
    weights = known * (1 + 4 * target)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction='none')
    bce = (bce * weights).sum() / weights.sum().clamp(min=1)
    probs = torch.sigmoid(logits) * known
    has_pos = (target.sum((2, 3)) > 0).float()
    dice = 1 - (2 * (probs * target).sum((2, 3)) + 1) / (probs.sum((2, 3)) + target.sum((2, 3)) + 1)
    dice = (dice * has_pos).sum() / has_pos.sum().clamp(min=1)
    return bce + .5 * dice


@torch.no_grad()
def validate(model, loader):
    """Perda, IoU de pixel por (limiar, classe) e frações previstas por imagem para o F1 de presença.

    Aproximação documentada: aqui cada classe é avaliada de forma independente (score >= limiar);
    o app ainda aplica a exclusividade por pixel. A avaliação final usa a regra exata do app.
    """
    model.eval()
    thr = torch.tensor(THRESHOLDS, dtype=torch.float32)[:, None, None, None, None]
    inter = torch.zeros(len(THRESHOLDS), N_CLASS, dtype=torch.float64); union = torch.zeros_like(inter)
    fracs, statuses, loss_sum, n = [], [], 0.0, 0
    for x, target, known, status in loader:
        logits = model(x)
        loss_sum += masked_loss(logits, target, known).item() * len(x); n += len(x)
        pred = torch.sigmoid(logits)[None] >= thr                     # (T,B,C,H,W)
        t, k = target.bool()[None], known.bool()[None]
        inter += (pred & t & k).sum((1, 3, 4)).double(); union += ((pred | t) & k).sum((1, 3, 4)).double()
        hw = pred.shape[-1] * pred.shape[-2]
        fracs.append((pred.sum((3, 4)).double() / hw).permute(1, 0, 2).numpy())  # (B,T,C)
        statuses.append(status.numpy())
    iou = (inter / union.clamp(min=1)).numpy()
    return loss_sum / max(n, 1), iou, np.concatenate(fracs), np.concatenate(statuses)


def presence_scores(fracs, statuses):
    """F1 de presença por (limiar, classe) com a regra de área mínima; retorna f1, n_positivos e contagens."""
    pred = fracs >= MIN_REGION_FRACTION                     # (N,T,C)
    pos, neg = (statuses == 1)[:, None, :], (statuses == 0)[:, None, :]
    tp = (pred & pos).sum(0); fp = (pred & neg).sum(0); fn = (~pred & pos).sum(0)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    return f1, (statuses == 1).sum(0), tp, fp, fn


def choose_thresholds(iou, fracs, statuses):
    """Limiar por classe escolhido SÓ na validação: melhor F1 de presença (empate -> limiar maior).
    Sem positivos suficientes: melhor IoU de pixel; sem nenhum dado: 0,5."""
    f1, npos, *_ = presence_scores(fracs, statuses)
    chosen, method, sel_f1, sel_iou = {}, {}, {}, {}
    for c in range(N_CLASS):
        if npos[c] >= MIN_VAL_POSITIVES:
            j = int(np.flatnonzero(f1[:, c] == f1[:, c].max()).max()); method[c + 1] = 'f1_presenca_val'
        elif iou[:, c].max() > 0:
            j = int(np.flatnonzero(iou[:, c] == iou[:, c].max()).max()); method[c + 1] = 'iou_pixel_val (poucos positivos)'
        else:
            j = int(np.argmin(np.abs(THRESHOLDS - .5))); method[c + 1] = 'padrao_0.5 (sem dados na val)'
        chosen[c + 1] = float(THRESHOLDS[j]); sel_f1[c + 1] = float(f1[j, c]); sel_iou[c + 1] = float(iou[j, c])
    eligible = [c for c in range(N_CLASS) if npos[c] >= MIN_VAL_POSITIVES]
    score = float(np.mean([sel_f1[c + 1] for c in eligible])) if eligible else 0.0
    return chosen, method, sel_f1, sel_iou, score, npos


def lr_lambda_factory(total_steps, warmup_steps):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return f


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--myfood', required=True); p.add_argument('--brazil', required=True)
    p.add_argument('--myfoodrepo', default='', help='archive.zip do MyFoodRepo v2.0 (opcional)')
    p.add_argument('--epochs', type=int, default=20); p.add_argument('--batch', type=int, default=12)
    p.add_argument('--lr', type=float, default=5e-4); p.add_argument('--dimension', type=int, default=224)
    p.add_argument('--synthetic-frac', type=float, default=0.08)
    p.add_argument('--threads', type=int, default=min(6, os.cpu_count() or 6))
    p.add_argument('--run-dir', default=str(ROOT / 'models' / 'runs' / 'v2'))
    p.add_argument('--out', default='', help='Se informado, grava o melhor modelo final (.pt + .json) aqui')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--mfr-skip-classes', default='', help='ids comuns separados por vírgula; fotos de TREINO do MyFoodRepo com essas classes ficam de fora (ex.: 8,3 = Macarrão, Frango)')
    p.add_argument('--brazil-weight', type=float, default=1.0, help='multiplica a chance de sortear fotos do dataset brasileiro')
    a = p.parse_args()
    run = Path(a.run_dir); run.mkdir(parents=True, exist_ok=True)
    log = Logger(run / 'train.log')
    torch.set_num_threads(a.threads)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

    t0 = time.time()
    skip = [int(x) for x in a.mfr_skip_classes.split(',') if x.strip()]
    train = load_records(a.myfood, a.brazil, 'train', a.dimension, myfoodrepo_zip=a.myfoodrepo or None, mfr_skip_classes=skip)
    val, removed = drop_near_duplicates(load_records(a.myfood, a.brazil, 'val', a.dimension, myfoodrepo_zip=a.myfoodrepo or None), train)
    n_syn = int(round(a.synthetic_frac * len(train)))
    log('DADOS treino', len(train), 'validacao', len(val), 'quase-duplicatas removidas da val', removed,
        'sinteticas/epoca', n_syn, f'({time.time() - t0:.0f}s)')
    per_src = lambda recs: dict(Counter(r['src'] for r in recs))
    log('  por fonte: treino', per_src(train), 'validacao', per_src(val), '| classes', N_CLASS)
    log('  opcoes: MyFoodRepo sem classes', skip, '| peso brasileiro', a.brazil_weight)
    (run / 'config.json').write_text(json.dumps(dict(classes=N_CLASS, datasets=datasets_used(a.myfoodrepo), mfr_skip_classes=skip, brazil_weight=a.brazil_weight), ensure_ascii=False), encoding='utf-8')

    tr_ds = FoodDataset(train, a.dimension, augment=True, n_synthetic=n_syn)
    tr_ds.py_rng.seed(a.seed)
    sampler = WeightedRandomSampler(sample_weights(train, n_syn, {'brazil': a.brazil_weight}), num_samples=len(tr_ds), replacement=True)
    tr_loader = DataLoader(tr_ds, batch_size=a.batch, sampler=sampler, num_workers=0, drop_last=True)
    va_loader = DataLoader(FoodDataset(val, a.dimension), batch_size=a.batch, shuffle=False, num_workers=0)

    model = build_model('lraspp_mobilenet_v3', pretrained_backbone=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = len(tr_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(a.epochs * steps, steps))
    start, best, history = 0, -1.0, []
    last = run / 'last.pt'
    if last.exists():
        ck = torch.load(last, map_location='cpu', weights_only=False)
        if ck['epochs_total'] != a.epochs:
            raise SystemExit(f'last.pt foi criado com --epochs {ck["epochs_total"]}; use o mesmo valor ou outra --run-dir')
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched'])
        start, best, history = ck['epoch'], ck['best'], ck['history']
        random.setstate(ck['py_rng']); np.random.set_state(ck['np_rng']); torch.set_rng_state(ck['torch_rng'])
        tr_ds.py_rng.setstate(ck['ds_rng'])
        log('RETOMANDO após a época', start, 'melhor F1 val', round(best, 4))
    else:
        log('INICIO', vars(a))

    for epoch in range(start, a.epochs):
        s = time.time(); model.train(); total = 0.0
        for i, (x, target, known, _) in enumerate(tr_loader):
            opt.zero_grad(set_to_none=True)
            loss = masked_loss(model(x), target, known)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 3.0); opt.step(); sched.step()
            total += loss.item()
            if i % 20 == 0:
                log(f'  epoca {epoch + 1} passo {i}/{steps} perda {loss.item():.4f} {(time.time() - s) / (i + 1):.2f}s/passo')
        train_s = time.time() - s
        val_loss, iou, fracs, statuses = validate(model, va_loader)
        thr, method, f1c, iouc, score, npos = choose_thresholds(iou, fracs, statuses)
        miou = float(np.mean(list(iouc.values())))
        rec = dict(epoch=epoch + 1, train_loss=total / steps, val_loss=val_loss, val_presence_macro_f1=score,
                   val_mean_iou_at_thresholds=miou, train_seconds=round(train_s), total_seconds=round(time.time() - s))
        history.append(rec)
        torch.save(model.state_dict(), run / f'epoch_{epoch + 1:02d}.pt')
        improved = score > best
        if improved:
            best = score
            torch.save(model.state_dict(), run / 'best.pt')
            (run / 'best.json').write_text(json.dumps(dict(thresholds=thr, threshold_method=method, val_f1_per_class=f1c,
                                                           val_iou_per_class=iouc, val_positive_images=npos.tolist(), **rec),
                                                      indent=2, ensure_ascii=False), encoding='utf-8')
        # Grava primeiro em arquivo temporário: uma queda durante a gravação não corrompe o last.pt anterior.
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=epoch + 1,
                        epochs_total=a.epochs, best=best, history=history, py_rng=random.getstate(),
                        np_rng=np.random.get_state(), torch_rng=torch.get_rng_state(), ds_rng=tr_ds.py_rng.getstate()),
                   run / 'last.tmp')
        os.replace(run / 'last.tmp', last)
        (run / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        log(f'EPOCA {epoch + 1}/{a.epochs} perda_treino {rec["train_loss"]:.4f} perda_val {val_loss:.4f} '
            f'F1_presenca_val {score:.4f} IoU_val {miou:.4f} {"*melhor*" if improved else ""} '
            f'treino {train_s / 60:.1f}min total {rec["total_seconds"] / 60:.1f}min')
        log('  F1 por classe:', {CLASS_LABELS[c]: round(v, 2) for c, v in f1c.items() if npos[c - 1] >= MIN_VAL_POSITIVES})

    log('FIM do treino em', f'{(time.time() - t0) / 60:.1f} min; melhor F1 val', round(best, 4))
    if a.out:
        info = json.loads((run / 'best.json').read_text(encoding='utf-8'))
        out = Path(a.out)
        torch.save(torch.load(run / 'best.pt', weights_only=True), out)
        meta = dict(model=f'MobileNetV3-Large (ImageNet) + LR-ASPP, {N_CLASS} saídas sigmoid', architecture='lraspp_mobilenet_v3',
                    normalize='imagenet', classes=N_CLASS, resolution=a.dimension, datasets=datasets_used(a.myfoodrepo),
                    thresholds={str(k): v for k, v in info['thresholds'].items()}, threshold_method=info['threshold_method'],
                    min_region_fraction=MIN_REGION_FRACTION, best_epoch=info['epoch'], epochs_trained=len(history),
                    val_presence_macro_f1=info['val_presence_macro_f1'], val_mean_iou_at_thresholds=info['val_mean_iou_at_thresholds'],
                    mfr_skip_classes=skip, brazil_weight=a.brazil_weight,
                    train_size=len(train), val_size=len(val), synthetic_per_epoch=n_syn, val_near_duplicates_removed=removed,
                    selection='época e limiares escolhidos SÓ na validação; teste nunca usado no treino')
        out.with_suffix('.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
        log('MODELO FINAL gravado em', out, 'da época', info['epoch'])


if __name__ == '__main__':
    main()
