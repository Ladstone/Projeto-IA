"""Segmentação multialimento local. Treinamento conjunto com MyFood e Brazilian Food Images.

Duas arquiteturas são suportadas, escolhidas pelo campo ``architecture`` do .json do modelo:
- ``foodunet`` (v1): U-Net pequena treinada do zero, 160 px, sem normalização.
- ``lraspp_mobilenet_v3`` (v2): MobileNetV3-Large pré-treinada no ImageNet + cabeça LR-ASPP,
  224 px, normalização ImageNet. Na inferência NÃO há download: os pesos completos estão no .pt.
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from .unified_classes import N_CLASS, CLASS_LABELS, LABEL_COLORS

# Regiões menores que isto foram, nos testes com as fotos de exemplo, quase sempre
# falsos positivos. Heurística usada igualmente pelo app e pela avaliação.
MIN_REGION_FRACTION = 0.03
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ConvBlock(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.seq = nn.Sequential(nn.Conv2d(inp, out, 3, padding=1), nn.BatchNorm2d(out), nn.ReLU(inplace=True),
                                 nn.Conv2d(out, out, 3, padding=1), nn.BatchNorm2d(out), nn.ReLU(inplace=True))

    def forward(self, x): return self.seq(x)


class FoodUNet(nn.Module):
    def __init__(self, n_class=N_CLASS):
        super().__init__()
        self.enc1 = ConvBlock(3, 12); self.enc2 = ConvBlock(12, 24); self.enc3 = ConvBlock(24, 48)
        self.bottleneck = ConvBlock(48, 72)
        self.dec3 = ConvBlock(72 + 48, 48); self.dec2 = ConvBlock(48 + 24, 24); self.dec1 = ConvBlock(24 + 12, 12)
        self.head = nn.Conv2d(12, n_class, 1)

    def forward(self, x):
        a = self.enc1(x); b = self.enc2(F.max_pool2d(a, 2)); c = self.enc3(F.max_pool2d(b, 2)); h = self.bottleneck(F.max_pool2d(c, 2))
        h = self.dec3(torch.cat([F.interpolate(h, size=c.shape[-2:], mode='bilinear', align_corners=False), c], 1))
        h = self.dec2(torch.cat([F.interpolate(h, size=b.shape[-2:], mode='bilinear', align_corners=False), b], 1))
        h = self.dec1(torch.cat([F.interpolate(h, size=a.shape[-2:], mode='bilinear', align_corners=False), a], 1))
        return self.head(h)


class LRASPPFood(nn.Module):
    """MobileNetV3-Large + LR-ASPP com uma saída sigmoid independente por classe (retorna logits)."""

    def __init__(self, n_class=N_CLASS, pretrained_backbone=False):
        super().__init__()
        from torchvision.models import MobileNet_V3_Large_Weights
        from torchvision.models.segmentation import lraspp_mobilenet_v3_large
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        self.net = lraspp_mobilenet_v3_large(weights=None, weights_backbone=weights, num_classes=n_class)

    def forward(self, x): return self.net(x)['out']


def build_model(architecture='foodunet', pretrained_backbone=False, n_class=N_CLASS):
    if architecture == 'foodunet':
        return FoodUNet(n_class)
    if architecture == 'lraspp_mobilenet_v3':
        return LRASPPFood(n_class, pretrained_backbone=pretrained_backbone)
    raise ValueError(f'Arquitetura desconhecida: {architecture}')


def preprocess(arr_uint8, normalize):
    """HxWx3 uint8 -> tensor 3xHxW float, com normalização ImageNet opcional."""
    arr = arr_uint8.astype(np.float32) / 255.0
    if normalize == 'imagenet':
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))


def exclusive_mask(probs, thresholds):
    """Cada pixel recebe a classe elegível (score >= limiar) de maior score; 0 = nenhuma."""
    thr = np.array([thresholds.get(i + 1, .5) for i in range(probs.shape[0])], dtype=np.float32)[:, None, None]
    eligible = probs >= thr
    winners = np.where(eligible, probs, -1).argmax(0) + 1
    return np.where(eligible.any(0), winners, 0).astype('uint8')


def detected_classes(mask, min_fraction=MIN_REGION_FRACTION):
    """{class_id: fração da imagem} das classes cuja região atinge a área mínima (regra do app)."""
    out = {}
    for class_id in range(1, N_CLASS + 1):
        frac = float((mask == class_id).mean())
        if frac >= min_fraction:
            out[class_id] = frac
    return out


def food_crop_box(mask, margin=0.05):
    """Retângulo (x0, y0, x1, y1) em frações da foto em volta das regiões de comida; None se não há.

    Usado só para SUGERIR um recorte ao usuário, que pode ajustá-lo antes da análise.
    """
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    gh, gw = mask.shape
    x0, x1, y0, y1 = xs.min() / gw, (xs.max() + 1) / gw, ys.min() / gh, (ys.max() + 1) / gh
    pad = margin * max(x1 - x0, y1 - y0)
    return (max(0.0, x0 - pad), max(0.0, y0 - pad), min(1.0, x1 + pad), min(1.0, y1 + pad))


class VisionService:
    def __init__(self, weight_path: Path, device='cpu'):
        self.device = device
        meta_path = Path(weight_path).with_suffix('.json')
        self.metadata = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
        self.architecture = self.metadata.get('architecture', 'foodunet')
        self.normalize = self.metadata.get('normalize', 'none')
        self.resolution = int(self.metadata.get('resolution', 160))
        # Modelos antigos têm 21 saídas; as classes 22+ ficam de fora (o app não as detecta com eles).
        self.n_class = int(self.metadata.get('classes', 21))
        self.model = build_model(self.architecture, n_class=self.n_class)
        state = torch.load(weight_path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval().to(device)
        self.thresholds = {int(k): float(v) for k, v in self.metadata.get('thresholds', {}).items()}
        self._ultima = None  # (imagem, dimensão, resultado): o mesmo modelo pode ser consultado 2x na combinação

    @torch.no_grad()
    def predict(self, img: Image.Image, dimension=None):
        dimension = dimension or self.resolution
        if self._ultima and self._ultima[0] is img and self._ultima[1] == dimension:
            return self._ultima[2]
        resultado = self._predict(img, dimension)
        self._ultima = (img, dimension, resultado)
        return resultado

    def _predict(self, img, dimension):
        arr = np.asarray(img.convert('RGB').resize((dimension, dimension), Image.Resampling.BILINEAR), dtype=np.uint8)
        data = preprocess(arr, self.normalize).unsqueeze(0).to(self.device)
        probs = torch.sigmoid(self.model(data))[0].cpu().numpy()  # (n_class,H,W)
        return exclusive_mask(probs, self.thresholds), probs


class CombinedVisionService:
    """Combina dois modelos: ``base`` (ex.: v2, 21 classes, melhor no prato feito) e ``novas``
    (ex.: v3, 28 classes, que conhece brócolis, lasanha, hambúrguer etc.).

    modo ``'foto'`` (padrão):
    - se o modelo novo acha uma classe nova (22+) cobrindo pelo menos ``troca_minima`` (10%) da imagem,
      a foto inteira usa o resultado dele;
    - senão vale o modelo base; pixels de classe nova só entram onde o base não viu comida nenhuma
      (assim a sugestão de recorte inclui um sanduíche que o v2 não conhece, sem apagar o que o v2 viu).
    Com 3% (a área mínima), uma "cebola" de 4% ou uma "lasanha" de 8% inventadas pelo v3 faziam a foto
    inteira trocar de modelo e perder o frango ou a carne que o v2 tinha acertado (fotos reais, 24/11/2026).
    modo ``'pixel'``: vale a máscara do base, e só os pixels que o modelo novo marca como classe nova são
    trocados (o base continua decidindo o resto da foto, inclusive onde inventaria frango numa lasanha).
    Mesma interface de ``VisionService`` (predict, thresholds, metadata, n_class).
    """

    def __init__(self, base: VisionService, novas: VisionService, modo='foto', troca_minima=0.10):
        if modo not in ('foto', 'pixel'):
            raise ValueError(f'modo desconhecido: {modo}')
        self.modo, self.troca_minima = modo, troca_minima
        if novas.n_class <= base.n_class:
            raise ValueError('o modelo de classes novas precisa ter mais classes que o modelo base')
        self.base, self.novas = base, novas
        self.n_class, self.n_base = novas.n_class, base.n_class
        self.resolution, self.architecture = base.resolution, base.architecture
        self.thresholds = {**{k: v for k, v in novas.thresholds.items() if k > self.n_base},
                           **{k: v for k, v in base.thresholds.items() if k <= self.n_base}}
        self.metadata = {**base.metadata, 'classes': self.n_class,
                         'datasets': novas.metadata.get('datasets') or base.metadata.get('datasets'),
                         'label': f"{base.metadata.get('label') or 'base'} + classes {self.n_base + 1}-{self.n_class} do outro modelo",
                         'model': f"{base.metadata.get('model', base.architecture)}; classes novas de "
                                  f"{novas.metadata.get('model', novas.architecture)}"}

    @torch.no_grad()
    def predict(self, img: Image.Image, dimension=None):
        mask_b, probs_b = self.base.predict(img, dimension)
        mask_n, probs_n = self.novas.predict(img, dimension or mask_b.shape[0])
        if mask_n.shape != mask_b.shape:  # resoluções diferentes: alinha ao modelo base
            size = (mask_b.shape[1], mask_b.shape[0])
            mask_n = np.asarray(Image.fromarray(mask_n).resize(size, Image.Resampling.NEAREST))
            probs_n = np.stack([np.asarray(Image.fromarray(p).resize(size, Image.Resampling.BILINEAR)) for p in probs_n])
        nova = mask_n > self.n_base
        if self.modo == 'foto':
            maior = max((mask_n == c).mean() for c in range(self.n_base + 1, self.n_class + 1))
            if maior >= self.troca_minima:
                return mask_n, probs_n  # comida nova grande: vale o modelo novo na foto inteira
            nova &= mask_b == 0  # comida nova pequena: só onde o base não viu comida
        mask = np.where(nova, mask_n, mask_b).astype('uint8')
        probs = np.concatenate([probs_b, probs_n[self.n_base:]], axis=0)
        return mask, probs


class ClassOverrideService:
    """Pega algumas classes de um modelo especialista por cima de outro serviço.

    Ex.: v2 (+v3) chama purê de arroz ou frango; o v3d acerta purê em 90% das fotos de teste. Onde o
    especialista marca uma das ``classes`` (padrão: 16 = Purê), o pixel vira essa classe; o resto continua
    como o serviço principal decidiu."""

    def __init__(self, main, especialista: VisionService, classes=(16,)):
        self.main, self.especialista, self.classes = main, especialista, tuple(classes)
        self.n_class, self.resolution, self.architecture = main.n_class, main.resolution, main.architecture
        self.thresholds = {**main.thresholds, **{c: especialista.thresholds[c] for c in self.classes if c in especialista.thresholds}}
        self.metadata = {**main.metadata, 'label': f"{main.metadata.get('label') or 'modelo'} + "
                                                   f"{', '.join(str(c) for c in self.classes)} de outro modelo"}

    @torch.no_grad()
    def predict(self, img: Image.Image, dimension=None):
        mask, probs = self.main.predict(img, dimension)
        mask_e, probs_e = self.especialista.predict(img, dimension or mask.shape[0])
        if mask_e.shape != mask.shape:
            size = (mask.shape[1], mask.shape[0])
            mask_e = np.asarray(Image.fromarray(mask_e).resize(size, Image.Resampling.NEAREST))
            probs_e = np.stack([np.asarray(Image.fromarray(p).resize(size, Image.Resampling.BILINEAR)) for p in probs_e])
        mask, probs = mask.copy(), probs.copy()
        for c in self.classes:
            mask[mask_e == c] = c
            probs[c - 1] = probs_e[c - 1]
        return mask, probs


def load_vision(spec, root=None):
    """``"a.pt"`` -> VisionService; ``"a.pt+b.pt"`` -> CombinedVisionService(base=a, novas=b);
    ``"...|c.pt"`` -> purê (classe 16) vem do modelo c por cima do resto."""
    def resolve(p):
        p = Path(p)
        return p if p.is_absolute() or root is None else Path(root) / p
    spec, _, especialista = str(spec).rpartition('|')
    if spec:  # "...|c.pt" (purê) ou "...|c.pt:6,10" (outras classes); pode encadear vários especialistas
        arquivo, _, classes = especialista.rpartition(':')
        if not arquivo or not classes.replace(',', '').isdigit():  # sem ":ids" (ou é o ":" de "C:\...")
            arquivo, classes = especialista, '16'
        ids = tuple(int(c) for c in classes.split(','))
        return ClassOverrideService(load_vision(spec, root), VisionService(resolve(arquivo)), ids)
    spec = especialista
    spec, _, modo = spec.partition('@')
    parts = spec.split('+')
    if len(parts) == 1:
        return VisionService(resolve(parts[0]))
    base, novas = parts
    return CombinedVisionService(VisionService(resolve(base)), VisionService(resolve(novas)), modo or 'foto')
