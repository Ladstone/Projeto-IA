"""Exporta o melhor checkpoint de uma execução de train_v2.py como modelo utilizável pelo app.

  python scripts/export_v2.py --run-dir models/runs/pilot --out models/unified_21class_v2_piloto.pt

Usa SOMENTE o que o treino gravou (best.pt, best.json, history.json, train.log): época e
limiares vêm da validação. Não lê o conjunto de teste. Não substitui o modelo ativo do app
(models/unified_21class.pt) a menos que --out aponte para ele.
"""
import argparse, json, re, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.vision import MIN_REGION_FRACTION, build_model


def data_info_from_log(log_path):
    """Lê do train.log a linha DADOS (tamanhos, duplicatas removidas, sintéticas)."""
    for line in Path(log_path).read_text(encoding='utf-8').splitlines():
        m = re.search(r"DADOS treino (\d+) validacao (\d+) quase-duplicatas removidas da val (\{.*?\}) sinteticas/epoca (\d+)", line)
        if m:
            return dict(train_size=int(m[1]), val_size=int(m[2]), val_near_duplicates_removed=eval(m[3], {}),
                        synthetic_per_epoch=int(m[4]))
    return {}


def export(run_dir, out, dimension=224, label=''):
    run, out = Path(run_dir), Path(out)
    info = json.loads((run / 'best.json').read_text(encoding='utf-8'))
    history = json.loads((run / 'history.json').read_text(encoding='utf-8'))
    state = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
    # Execuções antigas não têm config.json: eram 21 classes, só MyFood + brasileiro.
    cfg_path = run / 'config.json'
    cfg = json.loads(cfg_path.read_text(encoding='utf-8')) if cfg_path.exists() else dict(classes=21, datasets=['MyFood Dataset', 'Brazilian Food Images'])
    build_model('lraspp_mobilenet_v3', n_class=cfg['classes']).load_state_dict(state)  # falha cedo se o checkpoint não bater
    torch.save(state, out)
    meta = dict(model=f'MobileNetV3-Large (ImageNet) + LR-ASPP, {cfg["classes"]} saídas sigmoid', architecture='lraspp_mobilenet_v3',
                normalize='imagenet', classes=cfg['classes'], resolution=dimension, datasets=cfg['datasets'],
                label=label, thresholds={str(k): v for k, v in info['thresholds'].items()}, threshold_method=info['threshold_method'],
                val_positive_images=info['val_positive_images'], min_region_fraction=MIN_REGION_FRACTION,
                best_epoch=info['epoch'], epochs_trained=len(history),
                val_presence_macro_f1=info['val_presence_macro_f1'], val_mean_iou_at_thresholds=info['val_mean_iou_at_thresholds'],
                selection='época e limiares escolhidos SÓ na validação; teste nunca usado no treino',
                source_run=str(run.relative_to(ROOT)) if run.is_relative_to(ROOT) else str(run),
                **data_info_from_log(run / 'train.log'))
    out.with_suffix('.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    return meta


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True); p.add_argument('--out', required=True)
    p.add_argument('--dimension', type=int, default=224); p.add_argument('--label', default='')
    a = p.parse_args()
    if Path(a.out).resolve() == (ROOT / 'models' / 'unified_21class.pt').resolve():
        raise SystemExit('Recusado: --out é o modelo ativo do app. Troque o modelo do app só com aprovação explícita.')
    m = export(a.run_dir, a.out, a.dimension, a.label)
    print('Exportado', a.out, '| época', m['best_epoch'], 'de', m['epochs_trained'], '| F1 val', round(m['val_presence_macro_f1'], 4))
