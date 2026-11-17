"""Segmentação multialimento local. Treinamento conjunto com MyFood e Brazilian Food Images.

Máscaras brasileiras: supervisão PARCIAL de apenas uma classe por fotografia.
Não se presume que todo outro alimento na fotografia seja fundo.
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from .unified_classes import N_CLASS, CLASS_LABELS, LABEL_COLORS

class ConvBlock(nn.Module):
    def __init__(self, inp,out):
        super().__init__()
        self.seq=nn.Sequential(nn.Conv2d(inp,out,3,padding=1),nn.BatchNorm2d(out),nn.ReLU(inplace=True),
            nn.Conv2d(out,out,3,padding=1),nn.BatchNorm2d(out),nn.ReLU(inplace=True))
    def forward(self,x):return self.seq(x)

class FoodUNet(nn.Module):
    def __init__(self,n_class=N_CLASS):
        super().__init__()
        self.enc1=ConvBlock(3,12);self.enc2=ConvBlock(12,24);self.enc3=ConvBlock(24,48)
        self.bottleneck=ConvBlock(48,72)
        self.dec3=ConvBlock(72+48,48);self.dec2=ConvBlock(48+24,24);self.dec1=ConvBlock(24+12,12)
        self.head=nn.Conv2d(12,n_class,1)
    def forward(self,x):
        a=self.enc1(x);b=self.enc2(F.max_pool2d(a,2));c=self.enc3(F.max_pool2d(b,2));h=self.bottleneck(F.max_pool2d(c,2))
        h=self.dec3(torch.cat([F.interpolate(h,size=c.shape[-2:],mode='bilinear',align_corners=False),c],1))
        h=self.dec2(torch.cat([F.interpolate(h,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
        h=self.dec1(torch.cat([F.interpolate(h,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1))
        return self.head(h)

class VisionService:
    def __init__(self,weight_path:Path,device='cpu'):
        self.device=device
        self.model=FoodUNet()
        state=torch.load(weight_path,map_location='cpu',weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval().to(device)
        meta_path=weight_path.with_suffix('.json')
        self.metadata=json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
        self.thresholds={int(k):float(v) for k,v in self.metadata.get('thresholds',{}).items()}
    @torch.no_grad()
    def predict(self,img:Image.Image,dimension=160):
        original=img.convert('RGB')
        arr=np.asarray(original.resize((dimension,dimension),Image.Resampling.BILINEAR),dtype=np.float32)/255.0
        data=torch.from_numpy(arr.transpose(2,0,1)).unsqueeze(0).to(self.device)
        outputs=self.model(data)
        probs=torch.sigmoid(outputs)[0].cpu().numpy()  # (21,H,W)
        thresholds=np.array([self.thresholds.get(i+1,.5) for i in range(N_CLASS)],dtype=np.float32)[:,None,None]
        eligible=probs>=thresholds
        # Cada pixel tem somente uma classe de alimento para desenhar o overlay.
        # A detecção de presença usa os mapas originais por classe antes da exclusividade.
        scores=np.where(eligible,probs,-1)
        winners=scores.argmax(0)+1
        mask=np.where(eligible.any(0),winners,0).astype('uint8')
        return mask,probs
