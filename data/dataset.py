"""Independent parameter splits with reproducible seeds and saved metadata."""

import hashlib
import json
from pathlib import Path

from data.generator import generate_sample


def create_splits(output_path, *, train=800, val=100, test=100, seed=42):
    if min(train, val, test) < 1:
        raise ValueError('all split sizes must be positive')
    counts = {'train': train, 'val': val, 'test': test}
    cursor = seed
    splits = {}
    for name, count in counts.items():
        rows = []
        for sample_seed in range(cursor, cursor + count):
            sample = generate_sample(seed=sample_seed)
            rows.append({'seed': sample_seed, **{key: sample[key] for key in ('A','gamma','omega','b')}})
        splits[name] = rows
        cursor += count
    encoded = json.dumps(splits, sort_keys=True).encode()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return {'path': str(path), 'sha256': hashlib.sha256(encoded).hexdigest(), 'counts': counts}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/splits/parameters.json')
    parser.add_argument('--train', type=int, default=800)
    parser.add_argument('--val', type=int, default=100)
    parser.add_argument('--test', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    print(create_splits(args.output, train=args.train, val=args.val, test=args.test, seed=args.seed))
