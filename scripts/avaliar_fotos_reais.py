"""Mini teste com fotos reais (fora dos datasets): compara modelos no fluxo do app.

Para cada foto: sugere o recorte em volta da comida (como a tela faz), analisa o recorte e
aplica a mesma área mínima do app. Compara com o gabarito de presença:
- ``tem``: alimentos que estão no prato e devem ser achados;
- ``aceito``: itens pequenos ou ambíguos (potinho ao lado, cenoura no arroz). Achar ou não, tanto faz.
Qualquer outra classe detectada conta como alimento inventado (falso positivo).

Uso:
  python scripts/avaliar_fotos_reais.py --dir "C:/Users/.../Downloads" \
      --model v2=models/unified_21class_v2.pt --model v3=models/unified_28class_v3.pt
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.vision import VisionService, MIN_REGION_FRACTION, food_crop_box  # noqa: E402
from app.unified_classes import FOOD_CLASS_MAP  # noqa: E402

NAME_TO_ID = {v['nome']: k for k, v in FOOD_CLASS_MAP.items()}
GABARITO = ROOT / 'docs' / 'FOTOS_REAIS_GABARITO.json'


def detect(service, image, crop=True):
    mask, _ = service.predict(image)
    if crop:
        box = food_crop_box(mask)
        if box:
            w, h = image.size
            image = image.crop((int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)))
            mask, _ = service.predict(image)
    return {int(c) for c in set(mask.flatten().tolist()) - {0} if (mask == c).mean() >= MIN_REGION_FRACTION}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', required=True, help='pasta com as fotos do gabarito')
    p.add_argument('--model', action='append', required=True, help='nome=caminho.pt (pode repetir)')
    p.add_argument('--sem-recorte', action='store_true', help='analisa a foto inteira em vez do recorte sugerido')
    p.add_argument('--out', default='', help='grava o resultado em JSON')
    a = p.parse_args()
    fotos = json.loads(GABARITO.read_text(encoding='utf-8'))['fotos']
    for f in fotos:
        for k in ('tem', 'aceito'):
            unknown = [n for n in f.get(k, []) if n not in NAME_TO_ID]
            assert not unknown, (f['arquivo'], unknown)
    images = {f['arquivo']: ImageOps.exif_transpose(Image.open(Path(a.dir) / f['arquivo'])).convert('RGB') for f in fotos}
    result = {}
    for spec in a.model:
        name, path = spec.split('=', 1)
        service = VisionService(ROOT / path)
        n_class = service.n_class
        rows = []
        print(f'\n== {name} ({path}, {n_class} classes)')
        for f in fotos:
            tem = {NAME_TO_ID[n] for n in f['tem']}
            aceito = {NAME_TO_ID[n] for n in f.get('aceito', [])}
            got = detect(service, images[f['arquivo']], crop=not a.sem_recorte)
            # Classes que o modelo nem tem (ex.: Brócolis no v2) contam como faltando, de propósito.
            faltou, inventou = tem - got, got - tem - aceito
            nomes = lambda s: ', '.join(sorted(FOOD_CLASS_MAP[c]['nome'] for c in s)) or '-'
            print(f"  {f['arquivo']:<38} faltou: {nomes(faltou):<35} inventou: {nomes(inventou)}")
            rows.append(dict(arquivo=f['arquivo'], grupo=f.get('grupo', 'outros'), n_tem=len(tem), n_achou=len(tem & got),
                             detectado=sorted(FOOD_CLASS_MAP[c]['nome'] for c in got),
                             faltou=sorted(FOOD_CLASS_MAP[c]['nome'] for c in faltou),
                             inventou=sorted(FOOD_CLASS_MAP[c]['nome'] for c in inventou)))
        resumo = {}
        for grupo in ['prato feito', 'outros', 'todas']:
            rs = [r for r in rows if grupo == 'todas' or r['grupo'] == grupo]
            if not rs:
                continue
            achados, total = sum(r['n_achou'] for r in rs), sum(r['n_tem'] for r in rs)
            resumo[grupo] = dict(alimentos_achados=f'{achados}/{total}', recall=round(achados / total, 3),
                                 alimentos_inventados=sum(len(r['inventou']) for r in rs),
                                 fotos_perfeitas=f"{sum(not r['faltou'] and not r['inventou'] for r in rs)}/{len(rs)}")
            print(f'  RESUMO {grupo}:', resumo[grupo])
        result[name] = dict(modelo=path, resumo=resumo, fotos=rows)
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
