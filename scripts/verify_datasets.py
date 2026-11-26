"""Confere que os ZIPs dos datasets são lidos pelo carregador do treino (``scripts/data_v2.py``).

Uso: python scripts/verify_datasets.py --myfood "MyFood Dataset.zip" --brazil "Brazilian food images.zip" [--myfoodrepo archive.zip]
"""
import argparse
import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.data_v2 import load_records  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--myfood', required=True)
    p.add_argument('--brazil', required=True)
    p.add_argument('--myfoodrepo', default='', help='opcional: archive.zip do MyFoodRepo v2.0')
    a = p.parse_args()
    for part in ('train', 'val', 'test'):
        records = load_records(a.myfood, a.brazil, part, dim=64, myfoodrepo_zip=a.myfoodrepo or None)
        fontes = collections.Counter(r['src'] for r in records)
        print(part, 'total:', len(records), 'por fonte:', dict(fontes))
        if part == 'train':
            print('classes brasileiras:', dict(sorted(collections.Counter(r['category'] for r in records if r['src'] == 'brazil').items())))
        assert fontes['myfood'] > 0 and fontes['brazil'] > 0, f'faltou uma fonte na divisão {part}'
    print('OK: as fontes foram lidas e as máscaras estão disponíveis para o treino.')


if __name__ == '__main__':
    main()
