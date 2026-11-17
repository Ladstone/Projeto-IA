"""Verify that BOTH uploaded ZIPs can be decoded by training loader.
Usage: python scripts/verify_datasets.py --myfood path.zip --brazil path.zip
"""
import argparse,collections,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.train_unified import Combined

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--myfood',required=True);p.add_argument('--brazil',required=True)
 a=p.parse_args()
 for split in ['train','val','test']:
  ds=Combined(a.myfood,a.brazil,part=split,dim=64)
  print(split,'total:',len(ds),'por fonte:',dict(collections.Counter(r[3] for r in ds.records)))
  if split=='train':
   print('classes brasileiras:',dict(sorted(collections.Counter(r[4] for r in ds.records if r[3]=='brazil').items())))
   assert sum(r[3]=='myfood' for r in ds.records)>0
   assert sum(r[3]=='brazil' for r in ds.records)>0
  assert len(ds)>0
 print('OK: ambas as fontes foram lidas, e suas máscaras estão disponíveis para o treinamento.')
if __name__=='__main__':main()
