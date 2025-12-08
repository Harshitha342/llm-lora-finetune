#!/usr/bin/env python3
import json
import argparse
import os
from transformers import AutoTokenizer

def clean_text(t: str) -> str:
    return ' '.join(t.split())

def prepare_jsonl(input_path, output_path, model_name, text_key='response', prompt_key='instruction', max_length=2048):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    with open(input_path, 'r', encoding='utf-8') as f:
        items = [json.loads(line) for line in f]

    processed = []
    for item in items:
        prompt = clean_text(item.get(prompt_key, ''))
        response = clean_text(item.get(text_key, ''))
        if not prompt or not response:
            continue
        full = f"### Instruction:\\n{prompt}\\n\\n### Response:\\n{response}"
        tokenized = tokenizer(full, truncation=True, max_length=max_length)
        if len(tokenized['input_ids']) < 4:
            continue
        processed.append({'text': full})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as out:
        for p in processed:
            out.write(json.dumps(p, ensure_ascii=False) + '\\n')

    print(f'Wrote {len(processed)} cleaned examples to {output_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--prompt-key', default='instruction')
    parser.add_argument('--text-key', default='response')
    parser.add_argument('--max-length', type=int, default=2048)
    args = parser.parse_args()
    prepare_jsonl(args.input, args.output, args.model, text_key=args.text_key, prompt_key=args.prompt_key, max_length=args.max_length)
