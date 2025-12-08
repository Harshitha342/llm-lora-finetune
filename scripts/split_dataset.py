#!/usr/bin/env python3
import json
import argparse
from sklearn.model_selection import train_test_split
import os

def read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(l) for l in f]

def write_jsonl(path, items):
    with open(path, 'w', encoding='utf-8') as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--train_frac', type=float, default=0.8)
    parser.add_argument('--val_frac', type=float, default=0.1)
    parser.add_argument('--test_frac', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    items = read_jsonl(args.input)

    # Check fractions sum to 1
    assert abs(args.train_frac + args.val_frac + args.test_frac - 1.0) < 1e-6, \
        "train_frac + val_frac + test_frac must sum to 1.0"

    # Split
    train_val, test = train_test_split(items, test_size=args.test_frac, random_state=args.seed)
    val_ratio = args.val_frac / (args.train_frac + args.val_frac)
    train, val = train_test_split(train_val, test_size=val_ratio, random_state=args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    write_jsonl(f"{args.out_dir}/train.jsonl", train)
    write_jsonl(f"{args.out_dir}/val.jsonl", val)
    write_jsonl(f"{args.out_dir}/test.jsonl", test)

    print(f"Wrote train={len(train)}, val={len(val)}, test={len(test)} to {args.out_dir}")
