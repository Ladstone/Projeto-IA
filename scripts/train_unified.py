"""Train a 21-class multilabel segmentation model on BOTH supplied image datasets.

Requirements: torch, numpy, Pillow. No internet and no cloud account required.
Examples:
 python scripts/train_unified.py --myfood "MyFood Dataset.zip" --brazil "Brazilian food images.zip" --epochs 10

Brazilian Food Images has one binary mask per (food class, photograph). Other foods
are NOT annotated, so loss applies only to the annotated head, preventing false
"background" supervision for foods visible elsewhere in the same photograph.
"""
import argparse, hashlib, io, json, random, os, sys, time, zipfile
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.vision import FoodUNet
from app.unified_classes import N_CLASS, BRAZIL_LABELS, MYFOOD_LABELS, MYFOOD_NAME_LABELS, CLASS_LABELS

class Combined(Dataset):
    def __init__(self,myfood,brazil,part='train',dim=160,augment=False,seed=42):
        self.records=[];self.dim=dim;self.augment=augment
        r=random.Random(seed)
        with zipfile.ZipFile(myfood) as z:
            folder='test' if part=='test' else ('val' if part=='val' else 'train')
            files=sorted([p for p in z.namelist() if p.startswith(f'MyFood Dataset/{folder}/') and p.endswith('.png')])
            for path in files:
                nm=path.split('/')[-1]
                category=nm.rsplit('_',1)[0]
                mpath=f'MyFood Dataset/{folder}_ann/{nm}'
                if mpath not in z.namelist():continue
                im=Image.open(io.BytesIO(z.read(path))).convert('RGB').resize((dim,dim),Image.Resampling.BILINEAR)
                mask=Image.open(io.BytesIO(z.read(mpath))).convert('L').resize((dim,dim),Image.Resampling.NEAREST)
                a=np.array(im,dtype=np.uint8);y=np.array(mask,dtype=np.uint8)
                # Mixed lunch: all classes 2..10 are annotated; 1 is unknown/ignored.
                if category=='lunch': heads=list(MYFOOD_NAME_LABELS.values())
                else: heads=[MYFOOD_NAME_LABELS[category]]
                self.records.append((a,y,tuple(sorted(set(heads))), 'myfood',category))
        if part!='test':
            with zipfile.ZipFile(brazil) as z:
                names=set(z.namelist());groups={}
                for path in sorted(names):
                    if not path.startswith('Brazilian food images/training_set/') or not path.endswith('.jpg') or path.endswith('_.jpg'):continue
                    pieces=path.split('/')
                    if len(pieces)!=4 or pieces[2] not in BRAZIL_LABELS:continue
                    mpath=path[:-4]+'_.jpg'
                    if mpath not in names:continue
                    raw=z.read(path);hashval=hashlib.sha1(raw).hexdigest()
                    # Group identical images to avoid direct duplicates leaking across splits.
                    # Deterministic 20% holdout by image bytes. No pixel-level test labels
                    # are available for the supplied Brazilian test_set.
                    is_val=int(hashval[:8],16)%5==0
                    if (part=='val') != is_val:continue
                    im=Image.open(io.BytesIO(raw)).convert('RGB').resize((dim,dim),Image.Resampling.BILINEAR)
                    mask=Image.open(io.BytesIO(z.read(mpath))).convert('L').resize((dim,dim),Image.Resampling.NEAREST)
                    y=(np.array(mask,dtype=np.uint8)>127).astype('uint8')
                    if y.mean()<.001:continue
                    label=BRAZIL_LABELS[pieces[2]]
                    self.records.append((np.array(im,dtype=np.uint8),y,(label,),'brazil',pieces[2]))
        if not self.records:raise ValueError(f'No images in split {part}')
    def __len__(self):return len(self.records)
    def __getitem__(self,idx):
        arr,y,heads,src,cat=self.records[idx]
        if self.augment:
            if random.random()<.5:arr=arr[:,::-1];y=y[:,::-1]
            if random.random()<.15:arr=arr[::-1];y=y[::-1]
            if random.random()<.25:
                k=random.randrange(4);arr=np.rot90(arr,k);y=np.rot90(y,k)
            if random.random()<.5:
                a=np.asarray(ImageEnhance.Color(Image.fromarray(np.ascontiguousarray(arr))).enhance(random.uniform(.85,1.15)),dtype=np.uint8)
                arr=a
            if random.random()<.5:
                arr=np.clip(arr.astype(np.float32)*random.uniform(.83,1.18)+random.uniform(-8,8),0,255).astype('uint8')
        arr=np.ascontiguousarray(arr);y=np.ascontiguousarray(y)
        target=np.zeros((N_CLASS,self.dim,self.dim),dtype=np.uint8)
        supervised=np.zeros((N_CLASS,),dtype=np.float32)
        valid=np.ones((self.dim,self.dim),dtype=np.float32)
        for c in heads:supervised[c-1]=1
        if src=='myfood':
            valid[y==1]=0 # Unlabeled pixels in the MyFood dataset.
            for mcode,unified_id in MYFOOD_LABELS.items():
                if unified_id in heads:target[unified_id-1] |= (y==mcode)
        else:target[heads[0]-1]=y
        return torch.from_numpy(np.ascontiguousarray(arr.transpose(2,0,1).copy())).float()/255.0,torch.from_numpy(target).float(),torch.from_numpy(supervised),torch.from_numpy(valid)

def supervised_loss(logits,targets,supervision,valid,loss_mode="partial"):
    # Only annotated (image, class) head influences gradients.
    mask=supervision[:,:,None,None]*valid[:,None,:,:]
    weights=1+5*targets
    base=nn.functional.binary_cross_entropy_with_logits(logits,targets,reduction='none')
    bce=(base*weights*mask).sum()/(mask*weights).sum().clamp(min=1)
    probs=torch.sigmoid(logits)*valid[:,None,:,:]
    numerator=2*(probs*targets).sum((2,3))+1
    denominator=(probs+targets*valid[:,None,:,:]).sum((2,3))+1
    dice=1-(numerator/denominator)
    dice=(dice*supervision).sum()/supervision.sum().clamp(min=1)
    # Positive pixels have a known label: make it compete against other classes.
    # This avoids declaring unannotated Brazilian foods to be background while
    # still discouraging 10 simultaneous class predictions on the same pixels.
    if loss_mode=="partial":
        return bce+.7*dice
    known_positive=targets*mask
    positive_ce=-(nn.functional.log_softmax(logits,dim=1)*known_positive).sum()/known_positive.sum().clamp(min=1)
    multi_conflict=(torch.sigmoid(logits).sum(dim=1)-1).clamp(min=0).square().mean()
    return bce+.7*dice+.65*positive_ce+.06*multi_conflict

@torch.no_grad()
def evaluate(model,loader,thresholds=(.3,.4,.5,.6)):
    model.eval()
    inter=np.zeros((len(thresholds),N_CLASS),dtype=np.float64)
    union=np.zeros_like(inter)
    positives=np.zeros((N_CLASS,),dtype=np.float64)
    for x,y,superv,valid in loader:
        scores=torch.sigmoid(model(x))
        valid=valid[:,None,:,:].bool()
        eval_mask=superv[:,:,None,None].bool()&valid
        truth=(y>0.5)&valid
        positives+=(truth&eval_mask).sum((0,2,3)).numpy()
        for k,t in enumerate(thresholds):
            pred=(scores>t)&valid
            inter[k]+=((truth&pred)&eval_mask).sum((0,2,3)).numpy()
            union[k]+=(((truth|pred)&eval_mask)).sum((0,2,3)).numpy()
    iou=inter/np.maximum(union,1)
    best_idx=np.argmax(iou,axis=0)
    best=iou[best_idx,np.arange(N_CLASS)]
    threshold_map={str(k+1):float(thresholds[j]) for k,j in enumerate(best_idx)}
    return float(best.mean()),iou,threshold_map,best,positives

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--myfood',required=True);parser.add_argument('--brazil',required=True)
    parser.add_argument('--epochs',type=int,default=10);parser.add_argument('--dimension',type=int,default=160)
    parser.add_argument('--batch',type=int,default=12);parser.add_argument('--lr',type=float,default=.0015);parser.add_argument('--resume',default='');parser.add_argument('--loss-mode',choices=['partial','competition'],default='partial')
    parser.add_argument('--out',default=str(ROOT/'models'/'unified_21class.pt'))
    args=parser.parse_args();torch.manual_seed(42);random.seed(42);np.random.seed(42);torch.set_num_threads(min(5,os.cpu_count() or 5))
    t0=time.time(); print('Loading masks and images from BOTH ZIPs...',flush=True)
    tr=Combined(args.myfood,args.brazil,part='train',dim=args.dimension,augment=True)
    va=Combined(args.myfood,args.brazil,part='val',dim=args.dimension,augment=False)
    print('LOSS_MODE',args.loss_mode,flush=True)
    print('TRAIN',len(tr),'VALIDATION',len(va), 'train_source',dict(Counter(x[3] for x in tr.records)),'val_source',dict(Counter(x[3] for x in va.records)),'class_count',N_CLASS,'secs',round(time.time()-t0,1),flush=True)
    cc=Counter(record[2][0] for record in tr.records)
    weights=[1/(cc[row[2][0]]**.5) for row in tr.records]
    sampler=WeightedRandomSampler(weights,num_samples=len(tr),replacement=True)
    train_loader=DataLoader(tr,batch_size=args.batch,sampler=sampler,num_workers=0)
    val_loader=DataLoader(va,batch_size=args.batch,shuffle=False,num_workers=0)
    model=FoodUNet(N_CLASS)
    # Reuse convolutional features learned from the original MyFood run.
    prev=ROOT/'models'/'myfood_unet.pt'
    if prev.exists():
        state=torch.load(prev,map_location='cpu',weights_only=True)
        state={k:v for k,v in state.items() if not k.startswith('head.')}
        result=model.load_state_dict(state,strict=False)
        print('Initialized shared encoder/decoder from MyFood model; new multi-label head has 21 channels.',flush=True)
    if args.resume:
        model.load_state_dict(torch.load(args.resume,map_location='cpu',weights_only=True))
        print('Fine-tuning resumed from',args.resume,flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.0001)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs,eta_min=.0001)
    best=-1
    if args.resume and Path(args.out).exists() and Path(args.out).with_suffix('.json').exists():
        best=float(json.loads(Path(args.out).with_suffix('.json').read_text())['val_macro_iou'])
        print('Keeping existing best validation IoU',round(best,4),'until improved',flush=True)
    for epoch in range(args.epochs):
        s=time.time();model.train();loss_total=0
        for x,y,superv,valid in train_loader:
            opt.zero_grad(set_to_none=True)
            logits=model(x)
            loss=supervised_loss(logits,y,superv,valid,loss_mode=args.loss_mode)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),max_norm=3.0);opt.step();loss_total+=loss.item()
        score,ious,thr,per,positive=evaluate(model,val_loader)
        sched.step()
        if score>best:
            best=score
            Path(args.out).parent.mkdir(parents=True,exist_ok=True)
            torch.save(model.state_dict(),args.out)
            Path(args.out).with_suffix('.json').write_text(json.dumps({'model':'FoodUNet sigmoid independent heads','datasets':['MyFood Dataset','Brazilian Food Images'],'classes':N_CLASS,'resolution':args.dimension,'thresholds':thr,'val_macro_iou':score,'epochs_trained':epoch+1,'train_size':len(tr),'val_size':len(va)},indent=2),encoding='utf-8')
        print('epoch',epoch+1,'loss',round(loss_total/len(train_loader),4),'val_macro_iou',round(score,4),'best',round(best,4),'s',round(time.time()-s,1),'perclass',{CLASS_LABELS[j+1]:round(float(per[j]),2) for j in range(N_CLASS)},flush=True)
    print('DONE in',round(time.time()-t0,1),'secs',flush=True)
    model.load_state_dict(torch.load(args.out,weights_only=True,map_location='cpu'))
    test=Combined(args.myfood,args.brazil,part='test',dim=args.dimension)
    tscore,tiou,tthreshold,tper,counts=evaluate(model,DataLoader(test,batch_size=args.batch),thresholds=(.5,))
    print('HOLDOUT MyFood TEST (default threshold .5) macro IoU',round(tscore,4),'perclass',{CLASS_LABELS[j+1]:round(float(tper[j]),3) for j in range(N_CLASS) if counts[j]>0},flush=True)
    # Test-score for classes actually represented; zeros for new brazil-only classes aren't included.
    relevant=[j for j in range(N_CLASS) if counts[j]>0]
    report={'train':len(tr),'validation':len(va),'test_myfood':len(test),'val_macro_iou':best,'myfood_test_present_classes_macro_iou':float(np.mean([tper[j] for j in relevant])),'test_perclass':{CLASS_LABELS[j+1]:round(float(tper[j]),3) for j in relevant},'notes':['Brazilian test_set consists only of unlabeled photographs: it has no pixel-wise ground truth and is not scored.','Brazilian labeled images were held out deterministically using SHA-1 grouping; 20% expected.','The MyFood test contains no ground-truth masks for Brazil-only classes; do not report 21-class test score on it.','Potential duplicate or near-duplicate photographs across datasets were not exhaustively checked.']}
    (ROOT/'MODEL_EVALUATION_V4.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')

if __name__=='__main__':main()
