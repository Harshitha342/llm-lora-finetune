#!/usr/bin/env python3
import json
import argparse
from faker import Faker
import random

fake = Faker()

PROMPT_TEMPLATES = [
    "Summarize the following passage:\n{doc}",
    "Explain in simple terms:\n{doc}",
    "Translate to formal English:\n{doc}",
    "Given the context, write a short answer to: {question}\nContext: {doc}",
    "Rewrite the following to be more concise:\n{doc}"
]

def make_example():
    doc = ' '.join(fake.paragraphs(nb=random.randint(1,3)))
    template = random.choice(PROMPT_TEMPLATES)
    if '{question}' in template:
        q = fake.sentence(nb_words=6)
        prompt = template.format(question=q, doc=doc)
        response = fake.sentence(nb_words=20)
    else:
        prompt = template.format(doc=doc)
        response = fake.paragraph(nb_sentences=2)
    return {'instruction': prompt, 'response': response}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--n', type=int, default=1000)
    args = parser.parse_args()

    with open(args.out, 'w', encoding='utf-8') as f:
        for _ in range(args.n):
            f.write(json.dumps(make_example(), ensure_ascii=False) + '\n')
    print(f'Wrote {args.n} examples to {args.out}')
