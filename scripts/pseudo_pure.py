"""Gera rótulos de purê (pseudo-rótulos) para o dataset "mashed potato" do images.cv.

O dataset só diz "esta foto tem purê", não onde. Cada foto vira uma máscara de purê assim:
- foto com UM alimento só (uma classe ocupa >= 85% da comida que o modelo viu): toda a comida é purê,
  mesmo que o modelo tenha chamado de arroz. É justamente o erro que queremos corrigir;
- senão: só os pixels em que o modelo de apoio (v3c) já diz purê.
Os outros pixels ficam "desconhecidos" no treino (não ensinam nada).

Fotos com menos de 10% de purê ficam de fora. Teto: 300 no treino, 60 na validação; o "test" do zip
NÃO é usado aqui (fica como teste independente de purê).

Uso: python scripts/pseudo_pure.py "C:/.../mashed p.zip" models/unified_28class_v3c.pt models/runs/mashed_pure.npz
"""
import io
import random
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.vision import load_vision  # noqa: E402

PURE = 16
DIM = 224
CAPS = {'train': 300, 'val': 60}
MIN_PURE = 0.10
UM_ALIMENTO = 0.85


def pseudo_label(mask):
    food = mask > 0
    if not food.any():
        return None, 'sem comida'
    dom, n = Counter(mask[food].tolist()).most_common(1)[0]
    if n / food.sum() >= UM_ALIMENTO:
        return food.astype(np.uint8), 'um alimento' if dom == PURE else f'um alimento (modelo disse {dom})'
    return (mask == PURE).astype(np.uint8), 'misturado'


def main():
    zip_path, model, out = sys.argv[1:4]
    service = load_vision(model, ROOT)
    rng = random.Random(42)
    data = {}
    with zipfile.ZipFile(zip_path) as z:
        for part in ('train', 'val'):
            names = sorted(n for n in z.namelist() if f'/data/{part}/' in n and n.endswith('.jpg'))
            ok, why = [], Counter()
            for name in names:
                im = Image.open(io.BytesIO(z.read(name))).convert('RGB').resize((DIM, DIM), Image.Resampling.BILINEAR)
                mask, _ = service.predict(im)
                mask = np.array(Image.fromarray(mask.astype(np.uint8)).resize((DIM, DIM), Image.Resampling.NEAREST))
                label, kind = pseudo_label(mask)
                if label is None or label.mean() < MIN_PURE:
                    why['descartada'] += 1
                    continue
                why['um alimento, modelo disse outra coisa' if '(' in kind else kind] += 1
                ok.append((name, np.asarray(im, dtype=np.uint8), label))
            rng.shuffle(ok)
            ok = ok[:CAPS[part]]
            print(part, len(names), 'fotos ->', len(ok), 'usadas |', dict(why))
            data[f'{part}_names'] = np.array([n for n, _, _ in ok])
            data[f'{part}_img'] = np.stack([i for _, i, _ in ok])
            data[f'{part}_label'] = np.stack([l for _, _, l in ok])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    print('gravado', out)


if __name__ == '__main__':
    main()
