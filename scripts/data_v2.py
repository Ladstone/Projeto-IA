"""Dados da Fase B: divisões sem vazamento, supervisão negativa e imagens sem comida.

Divisões
--------
- MyFood: pastas oficiais train / val / test (test tem máscaras multiclasse).
- MyFoodRepo v2.0 (opcional, ``archive.zip`` do Kaggle): só o conjunto de treino oficial é usado,
  dividido por SHA-1 do nome do arquivo (``% 10``: 0 = teste, 1 = validação, resto = treino).
  De cada divisão entra só uma amostra com teto de imagens por classe (ver ``MFR_CAPS``).
- Brazilian Food Images (training_set): mesma regra de hash do treino v1 para o treino
  (SHA-1 dos bytes, ``hash[:8] % 5 != 0``). A validação antiga (``% 5 == 0``) é dividida
  ao meio por outro trecho do hash em VALIDAÇÃO nova e TESTE novo. Assim nem o modelo v1
  nem o v2 treinaram nas fotos de teste e a comparação entre eles é justa.
  (O v1 escolheu limiares na validação antiga, que inclui o teste novo: leve vantagem ao v1.)

Supervisão por pixel (``target`` e ``known``, ambos N_CLASS x H x W)
--------------------------------------------------------------
- MyFood: as máscaras anotam TODOS os alimentos da foto (código 1 = outro alimento,
  ignorado). Cada pixel conhecido é positivo para sua classe e negativo para as outras 20;
  o fundo (código 0) é negativo para as 21.
- MyFoodRepo: igual ao MyFood. Polígonos de categorias mapeadas viram a classe comum; os
  de categorias fora do mapa viram "outro alimento" (ignorado); o resto é fundo.
- Brasileiro: só um alimento é anotado. Pixels anotados: positivos para a classe e
  negativos para as outras 20 (um pixel é um alimento só). Pixels fora da máscara:
  negativos apenas para a classe anotada; desconhecidos para as demais.
- Sintéticas (sem comida): negativas para as 21 classes em todos os pixels.
"""
import hashlib, io, json, random, zipfile
from collections import Counter
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
from torch.utils.data import Dataset

from app.unified_classes import N_CLASS, BRAZIL_LABELS, MYFOOD_LABELS, MYFOODREPO_LABELS, NEW_CLASSES
from app.vision import preprocess, MIN_REGION_FRACTION

IGNORE = 255
MYFOOD_ONLY_CLASSES = sorted(set(MYFOOD_LABELS.values()))  # classes com rótulo no MyFood
MYFOODREPO_CLASSES = sorted(set(MYFOODREPO_LABELS.values()))  # classes com rótulo no MyFoodRepo
# Fontes que anotam TODOS os alimentos da foto -> classes que, se ausentes do rótulo, estão ausentes da foto.
FULLY_ANNOTATED = {'myfood': MYFOOD_ONLY_CLASSES, 'myfoodrepo': MYFOODREPO_CLASSES}
MFR_PREFIX = 'raw_data/public_training_set_release_2.0/'
# Teto de imagens por classe em cada divisão: (classes novas 22+, classes antigas de reforço).
MFR_CAPS = {'train': (150, 50), 'val': (30, 10), 'test': (30, 10)}


def brazil_split(raw: bytes) -> str:
    h = hashlib.sha1(raw).hexdigest()
    if int(h[:8], 16) % 5 != 0:
        return 'train'
    return 'test' if int(h[8:16], 16) % 2 == 0 else 'val'


def dhash(img: Image.Image) -> int:
    """Hash de diferença 64 bits: imagens quase idênticas têm distância de Hamming pequena."""
    g = np.asarray(img.convert('L').resize((9, 8), Image.Resampling.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int(''.join('1' if b else '0' for b in bits), 2)


def thumb(arr):
    """Miniatura 32x32 em cinza, padronizada (média 0, desvio 1): correlação insensível a brilho/contraste."""
    g = np.asarray(Image.fromarray(arr).convert('L').resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float32).ravel()
    return (g - g.mean()) / (g.std() + 1e-6)


# Calibrado inspecionando pares val x treino: correlações <= 0,92 eram fotos diferentes
# (ex.: maçãs distintas em fundo branco; recortes em fundo preto), >= 0,94 a mesma foto
# (inclusive com cor/recorte levemente alterados). O hash sozinho falhava no fundo preto.
DUP_MAX_HASH_DISTANCE = 12
DUP_MIN_CORRELATION = 0.93


def myfoodrepo_split(file_name: str) -> str:
    v = int(hashlib.sha1(file_name.encode()).hexdigest()[:8], 16) % 10
    return 'test' if v == 0 else 'val' if v == 1 else 'train'


def polygons_to_label(anns, width, height, dim):
    """Rasteriza os polígonos COCO direto em dim x dim: classe comum, IGNORE (outro alimento) ou 0."""
    canvas = Image.new('L', (dim, dim), 0)
    draw = ImageDraw.Draw(canvas)
    sx, sy = dim / width, dim / height
    for cls, seg in anns:
        polys = json.loads(seg) if isinstance(seg, str) else seg
        for poly in polys:
            if len(poly) >= 6:
                draw.polygon([(poly[i] * sx, poly[i + 1] * sy) for i in range(0, len(poly) - 1, 2)],
                             fill=IGNORE if cls is None else cls)
    return np.array(canvas)


def select_myfoodrepo(images, anns_by_image, part, caps):
    """Escolhe imagens da divisão até cada classe atingir o teto; ordem fixa (hash), sem sorteio.

    Uma imagem entra se tiver alguma classe ainda abaixo do teto; todas as classes dela contam.
    Classes novas são preferidas como "categoria" da imagem (usada no equilíbrio do sorteio)."""
    cap_new, cap_old = caps
    count, chosen = Counter(), []
    for img in sorted(images, key=lambda i: hashlib.sha1(i['file_name'].encode()).hexdigest()):
        if myfoodrepo_split(img['file_name']) != part:
            continue
        classes = {c for c, _ in anns_by_image.get(img['id'], ()) if c is not None}
        open_ = [c for c in classes if count[c] < (cap_new if c in NEW_CLASSES else cap_old)]
        if not open_:
            continue
        count.update(classes)
        chosen.append((img, max(open_, key=lambda c: (c in NEW_CLASSES, -count[c]))))
    return chosen, count


def load_myfoodrepo(zip_path, part, dim=224, keep_raw=False, caps=None, skip_classes=()):
    """``skip_classes``: ids comuns; imagens que tenham qualquer uma delas ficam de fora
    (ex.: massa e frango, que no v3 puxavam batata frita e bife para essas classes)."""
    caps = caps or MFR_CAPS[part]
    skip = set(skip_classes)
    with zipfile.ZipFile(zip_path) as z:
        data = json.load(z.open(MFR_PREFIX + 'annotations.json'))
        cat_name = {c['id']: c['name'] for c in data['categories']}
        anns_by_image = {}
        for a in data['annotations']:
            anns_by_image.setdefault(a['image_id'], []).append((MYFOODREPO_LABELS.get(cat_name[a['category_id']]), a['segmentation']))
        if skip:
            anns_by_image = {k: v for k, v in anns_by_image.items() if not skip & {c for c, _ in v}}
        chosen, _ = select_myfoodrepo(data['images'], anns_by_image, part, caps)
        records = []
        for img, primary in chosen:
            raw = z.read(MFR_PREFIX + 'images/' + img['file_name'])
            im = Image.open(io.BytesIO(raw)).convert('RGB')
            if im.size != (img['width'], img['height']):
                continue  # anotação feita em outra orientação/tamanho: descarta em vez de errar a máscara
            anns = anns_by_image[img['id']]
            records.append(dict(src='myfoodrepo', category=f'mfr:{primary}', name=img['file_name'], cls=None,
                                img=np.asarray(im.resize((dim, dim), Image.Resampling.BILINEAR), dtype=np.uint8),
                                label=polygons_to_label(anns, *im.size, dim),
                                has_unknown_food=any(c is None for c, _ in anns),
                                hash=dhash(im), raw=raw if keep_raw else None))
    return records


def load_records(myfood_zip, brazil_zip, part, dim=224, keep_raw=False, myfoodrepo_zip=None, mfr_skip_classes=()):
    """Lê uma divisão. Cada registro: img (dim x dim x 3), label (dim x dim), metadados."""
    records = []
    with zipfile.ZipFile(myfood_zip) as z:
        names = set(z.namelist())
        for path in sorted(p for p in names if p.startswith(f'MyFood Dataset/{part}/') and p.endswith('.png')):
            fname = path.split('/')[-1]
            mpath = f'MyFood Dataset/{part}_ann/{fname}'
            if mpath not in names:
                continue
            raw = z.read(path)
            im = Image.open(io.BytesIO(raw)).convert('RGB')
            codes = np.array(Image.open(io.BytesIO(z.read(mpath))).convert('L').resize((dim, dim), Image.Resampling.NEAREST))
            label = np.zeros_like(codes)
            label[codes == 1] = IGNORE
            for code, unified in MYFOOD_LABELS.items():
                label[codes == code] = unified
            records.append(dict(src='myfood', category=fname.rsplit('_', 1)[0], name=path, cls=None,
                                img=np.asarray(im.resize((dim, dim), Image.Resampling.BILINEAR), dtype=np.uint8),
                                label=label, has_unknown_food=bool((codes == 1).any()),
                                hash=dhash(im), raw=raw if keep_raw else None))
    with zipfile.ZipFile(brazil_zip) as z:
        names = set(z.namelist())
        for path in sorted(names):
            pieces = path.split('/')
            if (not path.startswith('Brazilian food images/training_set/') or not path.endswith('.jpg')
                    or path.endswith('_.jpg') or len(pieces) != 4 or pieces[2] not in BRAZIL_LABELS):
                continue
            mpath = path[:-4] + '_.jpg'
            if mpath not in names:
                continue
            raw = z.read(path)
            if brazil_split(raw) != part:
                continue
            m = (np.array(Image.open(io.BytesIO(z.read(mpath))).convert('L').resize((dim, dim), Image.Resampling.NEAREST)) > 127).astype(np.uint8)
            if m.mean() < .001:
                continue
            im = Image.open(io.BytesIO(raw)).convert('RGB')
            records.append(dict(src='brazil', category=pieces[2], name=path, cls=BRAZIL_LABELS[pieces[2]],
                                img=np.asarray(im.resize((dim, dim), Image.Resampling.BILINEAR), dtype=np.uint8),
                                label=m, has_unknown_food=True, hash=dhash(im), raw=raw if keep_raw else None))
    if myfoodrepo_zip:
        records += load_myfoodrepo(myfoodrepo_zip, part, dim, keep_raw, skip_classes=mfr_skip_classes)
    if not records:
        raise ValueError(f'Nenhuma imagem na divisão {part}')
    return records


def is_near_duplicate(record, train_hashes, train_thumbs):
    """Duas etapas: hash de diferença seleciona candidatos; correlação de pixels confirma."""
    d = np.unpackbits(np.bitwise_xor(train_hashes, np.uint64(record['hash'])).view(np.uint8)).reshape(len(train_hashes), 64).sum(1)
    cand = np.flatnonzero(d <= DUP_MAX_HASH_DISTANCE)
    if not len(cand):
        return False
    corr = train_thumbs[cand] @ thumb(record['img']) / train_thumbs.shape[1]
    return bool(corr.max() >= DUP_MIN_CORRELATION)


def drop_near_duplicates(records, train_records):
    """Remove de val/test imagens que são (quase) a mesma foto de alguma imagem de treino."""
    hashes = np.array([r['hash'] for r in train_records], dtype=np.uint64)
    thumbs = np.stack([thumb(r['img']) for r in train_records])
    flags = [is_near_duplicate(r, hashes, thumbs) for r in records]
    kept = [r for r, f in zip(records, flags) if not f]
    removed = Counter(r['src'] for r, f in zip(records, flags) if f)
    return kept, dict(removed)


def targets_for(src, label, cls):
    """Constrói target/known (21 x H x W) a partir do rótulo já aumentado."""
    h, w = label.shape
    target = np.zeros((N_CLASS, h, w), dtype=np.float32)
    known = np.zeros((N_CLASS, h, w), dtype=np.float32)
    if src in FULLY_ANNOTATED:
        valid = label != IGNORE
        known[:] = valid
        for c in range(1, N_CLASS + 1):
            target[c - 1] = label == c
    elif src == 'brazil':
        m = label.astype(bool)
        known[:] = m                # pixels anotados: negativos para as outras classes
        known[cls - 1] = 1          # a classe anotada é conhecida em toda a imagem
        target[cls - 1] = m
    else:                           # sintética, sem comida
        known[:] = 1
    return target, known


def presence_status(src, label, cls, has_unknown_food):
    """Por classe: 1 = presente (>= área mínima), 0 = ausente com certeza, -1 = não avaliável."""
    status = np.full(N_CLASS, -1, dtype=np.int8)
    if src == 'synthetic':
        status[:] = 0
    elif src in FULLY_ANNOTATED:
        for c in range(1, N_CLASS + 1):
            frac = float((label == c).mean())
            if frac >= MIN_REGION_FRACTION:
                status[c - 1] = 1
            elif frac == 0 and (c in FULLY_ANNOTATED[src] or not has_unknown_food):
                status[c - 1] = 0   # classe que a fonte não rotula só é ausente se não há "outro alimento"
    else:
        if float(label.mean()) >= MIN_REGION_FRACTION:
            status[cls - 1] = 1
    return status


# ---------------------------------------------------------------- imagens sem comida
TRAIN_SYNTHETIC_KINDS = ['solid', 'gradient', 'uniform_noise', 'stripes']
EVAL_SYNTHETIC_KINDS = TRAIN_SYNTHETIC_KINDS + ['blobs', 'gaussian_noise']  # 2 tipos nunca vistos no treino


def synthetic_image(kind, rng: np.random.Generator, size=(320, 240)):
    w, h = size
    if kind == 'solid':
        arr = np.broadcast_to(rng.integers(0, 256, 3, dtype=np.uint8), (h, w, 3)).copy()
    elif kind == 'gradient':
        a, b = rng.integers(0, 256, 3), rng.integers(0, 256, 3)
        t = np.linspace(0, 1, w)[None, :, None] if rng.random() < .5 else np.linspace(0, 1, h)[:, None, None]
        arr = np.broadcast_to(a * (1 - t) + b * t, (h, w, 3)).astype(np.uint8)
    elif kind == 'uniform_noise':
        arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    elif kind == 'stripes':
        period = int(rng.integers(6, 40)); a, b = rng.integers(0, 256, 3), rng.integers(0, 256, 3)
        stripe = ((np.arange(w) // period) % 2)[None, :, None]
        arr = np.broadcast_to(np.where(stripe, a, b), (h, w, 3)).astype(np.uint8)
    elif kind == 'blobs':
        low = rng.integers(0, 256, (4, 5, 3), dtype=np.uint8)
        return Image.fromarray(low).resize(size, Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(6))
    elif kind == 'gaussian_noise':
        base = rng.integers(40, 216, 3)
        arr = np.clip(base + rng.normal(0, 40, (h, w, 3)), 0, 255).astype(np.uint8)
    else:
        raise ValueError(kind)
    return Image.fromarray(arr)


def synthetic_eval_set(per_kind=20, seed=1_000_003):
    rng = np.random.default_rng(seed)
    return [(kind, synthetic_image(kind, rng)) for kind in EVAL_SYNTHETIC_KINDS for _ in range(per_kind)]


# ---------------------------------------------------------------- datasets PyTorch
def _augment(img, label, py_rng):
    if py_rng.random() < .5:
        img, label = img[:, ::-1], label[:, ::-1]
    if py_rng.random() < .15:
        img, label = img[::-1], label[::-1]
    if py_rng.random() < .25:
        k = py_rng.randrange(4); img, label = np.rot90(img, k), np.rot90(label, k)
    dim = img.shape[0]
    if py_rng.random() < .5:  # recorte aleatório de 60% a 100% do lado
        side = int(dim * py_rng.uniform(.6, 1.0)); x, y = py_rng.randrange(dim - side + 1), py_rng.randrange(dim - side + 1)
        img = np.asarray(Image.fromarray(np.ascontiguousarray(img[y:y + side, x:x + side])).resize((dim, dim), Image.Resampling.BILINEAR))
        label = np.asarray(Image.fromarray(np.ascontiguousarray(label[y:y + side, x:x + side])).resize((dim, dim), Image.Resampling.NEAREST))
    img = img.astype(np.float32)
    if py_rng.random() < .5:
        gray = img.mean(2, keepdims=True); img = gray + (img - gray) * py_rng.uniform(.8, 1.2)
    if py_rng.random() < .5:
        img = img * py_rng.uniform(.8, 1.2) + py_rng.uniform(-10, 10)
    return np.clip(img, 0, 255).astype(np.uint8), np.ascontiguousarray(label)


class FoodDataset(Dataset):
    """Registros reais + ``n_synthetic`` imagens sem comida geradas na hora (só no treino)."""

    def __init__(self, records, dim=224, augment=False, n_synthetic=0, normalize='imagenet'):
        self.records, self.dim, self.augment, self.n_synthetic, self.normalize = records, dim, augment, n_synthetic, normalize
        self.py_rng = random.Random()

    def __len__(self): return len(self.records) + self.n_synthetic

    def __getitem__(self, idx):
        if idx >= len(self.records):
            rng = np.random.default_rng(self.py_rng.randrange(2 ** 32))
            kind = TRAIN_SYNTHETIC_KINDS[rng.integers(len(TRAIN_SYNTHETIC_KINDS))]
            img = np.asarray(synthetic_image(kind, rng).resize((self.dim, self.dim)), dtype=np.uint8)
            src, label, cls, unknown = 'synthetic', np.zeros((self.dim, self.dim), np.uint8), None, False
        else:
            r = self.records[idx]
            img, label, src, cls, unknown = r['img'], r['label'], r['src'], r['cls'], r['has_unknown_food']
        if self.augment:
            img, label = _augment(img, label, self.py_rng)
        target, known = targets_for(src, np.ascontiguousarray(label), cls)
        status = presence_status(src, label, cls, unknown)
        return preprocess(np.ascontiguousarray(img), self.normalize), torch.from_numpy(target), torch.from_numpy(known), torch.from_numpy(status)


def sample_weights(records, n_synthetic, src_weight=None):
    """Equilibra classes (1/sqrt(frequência)); as sintéticas recebem n_synthetic/len(records) da massa total.

    ``src_weight`` multiplica o peso de cada fonte (ex.: {'brazil': 2.0} sorteia o dobro de fotos brasileiras)."""
    counts = Counter(r['category'] for r in records)
    src_weight = src_weight or {}
    w = np.array([src_weight.get(r['src'], 1.0) / counts[r['category']] ** .5 for r in records], dtype=np.float64)
    w /= w.sum()
    if n_synthetic:
        share = n_synthetic / len(records)
        w = np.concatenate([w * (1 - share), np.full(n_synthetic, share / n_synthetic)])
    return w.tolist()
