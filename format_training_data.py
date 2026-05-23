"""
Convert train.json / validation.json into a chat-message format consumable
by TRL's SFTTrainer (and by RapidFire AI's RFSFTConfig wrapper, which
delegates to TRL).

Output JSONL, one record per line:
    {"messages": [{"role": "system",    "content": "..."},
                  {"role": "user",      "content": "..."},
                  {"role": "assistant", "content": "<JSON>"}]}

SFTTrainer applies the model's chat template automatically and masks out
everything before the assistant turn -- so the model is only supervised on
producing the JSON, not on rote-copying the schema.

CRITICAL: the system+user messages are produced by `prompt.build_messages`,
the same function main.py uses at inference. Changing one side without the
other is the #1 source of fine-tuned-model regressions.
"""
import argparse
import json
import os
from typing import Dict, List

from prompt import build_messages, target_string
from schema_utils import load_schema


def format_split(input_path: str, schemas_dir: str, output_path: str) -> None:
    with open(input_path) as f:
        items = json.load(f)

    schema_cache: Dict[str, Dict] = {}
    n_written = 0
    max_chars = 0
    with open(output_path, 'w') as fout:
        for ex in items:
            db_id = ex['db_id']
            if db_id not in schema_cache:
                schema_cache[db_id] = load_schema(schemas_dir, db_id)
            sch = schema_cache[db_id]

            # Pass gold_links so the filtered serializer oracle-includes any
            # gold columns that BM25 missed. Inference passes gold_links=None.
            msgs: List[Dict[str, str]] = build_messages(
                db_id, ex['question'], sch, gold_links=ex['schema_links'],
            )
            msgs.append({'role': 'assistant', 'content': target_string(ex['schema_links'])})

            line = json.dumps({'messages': msgs, 'db_id': db_id, 'question_id': ex['question_id']})
            fout.write(line + '\n')
            n_written += 1
            total_chars = sum(len(m['content']) for m in msgs)
            if total_chars > max_chars:
                max_chars = total_chars

    print(f"Wrote {n_written} examples to {output_path}")
    print(f"Max example char-length: {max_chars}  (rough ~{max_chars // 4} tokens)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input',  required=True, help='train.json or validation.json')
    ap.add_argument('--output', required=True, help='Output JSONL path')
    ap.add_argument('--schemas_dir', default='./schemas')
    args = ap.parse_args()
    format_split(args.input, args.schemas_dir, args.output)


if __name__ == '__main__':
    main()
