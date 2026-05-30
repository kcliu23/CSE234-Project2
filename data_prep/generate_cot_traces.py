"""
generate_cot_traces.py -- For each example in train.json (or train_augmented.json),
generate a short chain-of-thought reasoning trace via the Claude API, paired with
the existing gold schema_links JSON.

Output format (training-side):
    {"reasoning": "<1-2 sentence rationale>", "schema_links": {...gold...}}

At training time the assistant turn becomes:
    <reasoning text>
    <gold JSON>

So the SFT model learns to emit a brief rationale before the JSON. At inference,
the parser (parse_model_output) ignores everything before the first balanced JSON
block, so any leading reasoning text is harmless if the model adopts the same
format.

Why short? Each extra output token costs ~30ms at inference. The submitted
ensemble already takes ~11 min on 101 questions; we have ~4 min budget headroom
before hitting the 15-min grading cap. Capping reasoning at ~50 tokens keeps
the 3-adapter ensemble inference under budget.

We use Claude Haiku 4.5 (same as augment_data.py). Each call is given the
question, the relevant schema (BM25-filtered to top-20 tables -- matches what
the trained SLM will see at inference), and the gold schema_links, and asked
to write a 1-2 sentence rationale that *explains* (not derives -- we already
have the answer) how the linked tables/columns answer the question.

Cost ~$1-2 in Haiku API spend for 301-517 examples.

Run:
  python generate_cot_traces.py \
      --input data/train.json \
      --output data/train_cot.json \
      --audit data/train_cot_audit.json
"""
import argparse
import json
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schema_utils import load_schema, serialize_schema_filtered


PROMPT_TEMPLATE = """You are explaining how a SQL query's schema-links map to a question, for use as
training data for a small schema-linking model.

DATABASE: {db_id}

SCHEMA (top-20 BM25-retrieved tables -- exactly what the small model will see):
{schema_text}

QUESTION: {question}

GOLD SQL: {gold_sql}

GOLD schema_links: {gold_json}

YOUR TASK: write a single sentence (max ~50 tokens) explaining WHY those
specific tables and columns are the right answer for this question. Refer to
the table/column names from the schema. Don't restate the JSON. Be terse and
information-dense -- this is teaching signal for a 1.5B model, not prose.

Output JUST the explanation sentence. No JSON, no preamble, no quotes.
"""


def call_claude(prompt: str, model: str = 'claude-haiku-4-5-20251001', max_tokens: int = 200) -> str:
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='data/train.json')
    ap.add_argument('--schemas_dir', default='schemas')
    ap.add_argument('--output', default='data/train_cot.json',
                    help='Augmented train file with per-example reasoning traces.')
    ap.add_argument('--audit', default='data/train_cot_audit.json',
                    help='Per-call audit log (prompt, raw model output, included flag).')
    ap.add_argument('--model', default='claude-haiku-4-5-20251001')
    ap.add_argument('--max_examples', type=int, default=0,
                    help='Cap on examples to process (0 = all). Useful for a smoke test.')
    ap.add_argument('--max_tables', type=int, default=20,
                    help='BM25 keep-cap for the schema text shown to Claude. Must match '
                         'main.py inference value so the trained model sees the same view.')
    args = ap.parse_args()

    if not os.environ.get('ANTHROPIC_API_KEY'):
        sys.exit("ERROR: ANTHROPIC_API_KEY not set in env.")

    examples = json.load(open(args.input))
    if args.max_examples > 0:
        examples = examples[:args.max_examples]
    schema_cache: Dict[str, Dict] = {}

    out: List[Dict] = []
    audit: List[Dict] = []
    for i, ex in enumerate(examples, 1):
        db_id = ex['db_id']
        if db_id not in schema_cache:
            schema_cache[db_id] = load_schema(args.schemas_dir, db_id)
        sch = schema_cache[db_id]
        # Same BM25 view the SLM will see at inference. gold_links forces gold
        # tables in (matches our training-time oracle augmentation).
        schema_text = serialize_schema_filtered(
            sch, ex['question'], gold_links=ex.get('schema_links'),
            max_tables=args.max_tables, threshold_cols=500, style='compact',
            retrieval='bm25',
        )
        prompt = PROMPT_TEMPLATE.format(
            db_id=db_id, schema_text=schema_text,
            question=ex['question'],
            gold_sql=ex.get('gold_sql', '(not provided)'),
            gold_json=json.dumps(ex['schema_links'], separators=(',', ': ')),
        )
        try:
            reasoning = call_claude(prompt, model=args.model)
        except Exception as e:
            print(f"[cot] q{ex['question_id']}: API error: {e}", file=sys.stderr)
            audit.append({'question_id': ex['question_id'], 'db_id': db_id,
                          'status': f'api_error: {e}'})
            time.sleep(1)
            continue
        # Sanity check: reasoning should be non-empty, should not be huge, and
        # should reference at least one of the gold tables (otherwise it's not
        # actually grounded).
        gold_tables_lc = {t.lower() for t in ex['schema_links']}
        words = reasoning.lower()
        ok = (
            bool(reasoning)
            and len(reasoning.split()) <= 80
            and any(t in words for t in gold_tables_lc)
        )
        if not ok:
            audit.append({'question_id': ex['question_id'], 'db_id': db_id,
                          'reasoning': reasoning, 'status': 'failed_validation'})
            continue
        out_ex = dict(ex)
        out_ex['reasoning'] = reasoning
        out.append(out_ex)
        audit.append({'question_id': ex['question_id'], 'db_id': db_id,
                      'reasoning': reasoning, 'status': 'ok'})
        if i % 25 == 0 or i == len(examples):
            print(f"[cot] {i}/{len(examples)} processed, {len(out)} kept", file=sys.stderr)
            # Atomic save in case we get killed mid-run.
            tmp = args.output + '.tmp'
            json.dump(out, open(tmp, 'w'), indent=2)
            os.replace(tmp, args.output)
            tmp_audit = args.audit + '.tmp'
            json.dump(audit, open(tmp_audit, 'w'), indent=2)
            os.replace(tmp_audit, args.audit)

    print(f"[cot] DONE: {len(out)}/{len(examples)} examples with traces; "
          f"{len(examples) - len(out)} dropped. Output: {args.output}, audit: {args.audit}")


if __name__ == '__main__':
    main()
