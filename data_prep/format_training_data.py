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
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompt import build_messages, target_string
from schema_utils import load_schema


def format_split(input_path: str, schemas_dir: str, output_path: str, style: str = 'compact',
                 use_cot: bool = False) -> None:
    with open(input_path) as f:
        items = json.load(f)

    schema_cache: Dict[str, Dict] = {}
    n_written = 0
    n_skipped_no_cot = 0
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
                db_id, ex['question'], sch, gold_links=ex['schema_links'], style=style,
            )

            json_str = target_string(ex['schema_links'])
            if use_cot:
                reasoning = ex.get('reasoning')
                if not reasoning:
                    n_skipped_no_cot += 1
                    continue
                # Assistant turn: reasoning sentence then the canonical JSON.
                # parse_model_output picks up the FIRST balanced JSON object so
                # any preceding reasoning text is harmless at inference.
                assistant = f"{reasoning.strip()}\n{json_str}"
            else:
                assistant = json_str
            msgs.append({'role': 'assistant', 'content': assistant})

            line = json.dumps({'messages': msgs, 'db_id': db_id, 'question_id': ex['question_id']})
            fout.write(line + '\n')
            n_written += 1
            total_chars = sum(len(m['content']) for m in msgs)
            if total_chars > max_chars:
                max_chars = total_chars

    cot_note = f"  (cot, skipped {n_skipped_no_cot} without reasoning)" if use_cot else ""
    print(f"Wrote {n_written} examples to {output_path}  (style={style}){cot_note}")
    print(f"Max example char-length: {max_chars}  (rough ~{max_chars // 4} tokens)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input',  required=True, help='train.json or validation.json')
    ap.add_argument('--output', required=True, help='Output JSONL path')
    ap.add_argument('--schemas_dir', default='./schemas')
    ap.add_argument('--prompt_style', default='compact',
                    choices=['compact', 'types', 'keys', 'types_keys'],
                    help='Schema serialization style. MUST match the style main.py uses at inference.')
    ap.add_argument('--cot', action='store_true',
                    help='If set, expect a "reasoning" field on each input example and prepend '
                         'it to the assistant target. Use with data produced by generate_cot_traces.py.')
    args = ap.parse_args()
    format_split(args.input, args.schemas_dir, args.output, style=args.prompt_style, use_cot=args.cot)


if __name__ == '__main__':
    main()
