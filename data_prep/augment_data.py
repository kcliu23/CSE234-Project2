"""
augment_data.py -- Generate additional (question, gold_sql) training examples
via the Claude API for under-represented schemas, then label each with the
project's `sql_to_schema_links.py` extractor.

Why: train.json has 301 examples across 17 schemas, but the 9 SBO modules
have only 6-12 examples each (vs 24+ for the parks/NTSB/NYSED domains).
That class imbalance is visible in our val errors -- the SBO modules
account for 55% of "missing-gold-table" misses despite being only ~30% of
val questions. The rubric (project2-statement.pdf S3) explicitly allows
using a frontier API offline for training data augmentation:
"You may use such APIs offline for training data augmentation if you
document it."

What the script does:
  1. For each target schema (default: all 9 SBO modules), prompt Claude
     with the schema + 3-5 few-shot examples from existing train data.
  2. Ask Claude to produce N new (question, SQL) pairs in T-SQL dialect
     using only identifiers that appear in the provided schema.
  3. Validate each generated SQL by running it through
     sql_to_schema_links.extract_schema_links. Drop:
       - SQL that fails to parse (sqlglot error)
       - SQL that touches zero recognized schema tables (likely hallucinated)
       - Duplicates of existing-train questions
  4. Append survivors to data/train_augmented.json with stable
     question_ids starting at 1000 (existing max is 301).
  5. Write a per-example audit log so the report can document what was
     generated, what survived validation, and the failure modes.

Run:
  # 1) set your key once (the script defaults to Anthropic, falls back to OpenAI):
  export ANTHROPIC_API_KEY=sk-ant-...
  # 2) generate augmentation (will cost roughly $0.30-$1 in Haiku API spend):
  python augment_data.py --target_per_schema 24 \
      --output data/train_augmented.json \
      --audit data/train_augmented_audit.json
"""
import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Tuple

# Re-use the project's own SQL-to-links extractor for validation.
from sql_to_schema_links import load_schema as load_sql_schema, extract_schema_links


SBO_SCHEMAS = [
    'SBODemoUS-Banking',
    'SBODemoUS-Business Partners',
    'SBODemoUS-Finance',
    'SBODemoUS-General',
    'SBODemoUS-Human Resources',
    'SBODemoUS-Inventory and Production',
    'SBODemoUS-Reports',
    'SBODemoUS-Sales Opportunities',
    'SBODemoUS-Service',
]


def serialize_schema_compact(schema: Dict[str, Dict[str, str]]) -> str:
    """One line per table: `TableName: col1, col2, ...`. Matches the prompt
    serializer used at training time so the model conditions on the same view."""
    lines = []
    for t, cols in schema.items():
        lines.append(f"{t}: {', '.join(cols.keys())}")
    return '\n'.join(lines)


GEN_PROMPT_TEMPLATE = """You generate SYNTHETIC training examples for a schema-linking SFT dataset.

DATABASE: {db_id}

SCHEMA (table: col1, col2, ...):
{schema_text}

EXISTING EXAMPLES from this database (for tone + difficulty calibration):
{fewshot}

YOUR TASK: produce {n} NEW (natural-language question, SQL query) pairs against the schema above.

HARD CONSTRAINTS:
1. SQL dialect: T-SQL (Microsoft SQL Server). The corpus is built from SNAILS gold SQL which is T-SQL.
2. Use ONLY table and column identifiers that appear in the SCHEMA above. Do not invent names.
3. Match identifier CASING from the schema.
4. Each query should reference 1-4 tables. Mix single-table queries with multi-table JOINs.
5. Mix query types: counts, lists, aggregations, filters, group-by, simple joins.
6. Questions should sound natural -- something an analyst or end-user would actually ask. Avoid robotic phrasing.
7. Do NOT duplicate or trivially paraphrase the EXISTING EXAMPLES.

OUTPUT FORMAT: a single JSON array. Each element is an object with exactly:
  {{"question": "<the natural-language question>",
    "gold_sql": "<the T-SQL query>"}}

Output ONLY the JSON array. No prose before or after, no markdown code fences.
"""


def build_fewshot(existing: List[Dict], db_id: str, k: int = 5) -> str:
    """Return up to k existing (question, sql) pairs from this db (or related)
    as a string for in-context demos."""
    same = [ex for ex in existing if ex['db_id'] == db_id]
    if len(same) < k:
        # SBO siblings if needed
        related = [ex for ex in existing if ex['db_id'].startswith('SBODemoUS-') and ex['db_id'] != db_id]
        same = same + random.sample(related, min(k - len(same), len(related)))
    sample = random.sample(same, min(k, len(same)))
    out = []
    for ex in sample:
        out.append(f"  Q: {ex['question']}\n  SQL: {ex['gold_sql']}")
    return '\n\n'.join(out)


def call_claude(prompt: str, model: str, max_tokens: int = 4096) -> str:
    """One Anthropic API call. Raises on error."""
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def call_openai(prompt: str, model: str, max_tokens: int = 4096) -> str:
    """One OpenAI API call (fallback if no Anthropic key)."""
    from openai import OpenAI
    client = OpenAI()  # reads OPENAI_API_KEY from env
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def parse_json_array(text: str) -> List[Dict]:
    """Strip code fences if present, json.loads, expect a list of dicts."""
    s = text.strip()
    if s.startswith('```'):
        s = s.split('```', 2)[1]
        if s.startswith('json'):
            s = s[4:]
        s = s.rsplit('```', 1)[0]
    s = s.strip()
    return json.loads(s)


def validate_example(ex: Dict, schema: Dict, seen_questions: set) -> Tuple[bool, str, Dict]:
    """Returns (ok, reason, ex_with_links). ex_with_links has 'schema_links' filled."""
    q = (ex.get('question') or '').strip()
    sql = (ex.get('gold_sql') or '').strip()
    if not q or not sql:
        return False, 'empty_question_or_sql', ex
    if q.lower() in seen_questions:
        return False, 'duplicate_question', ex
    links, err = extract_schema_links(sql, schema, dialect='tsql')
    if err:
        return False, f'parse_error: {err[:120]}', ex
    if not links:
        return False, 'zero_recognized_tables', ex
    ex_out = dict(ex)
    ex_out['schema_links'] = links
    return True, 'ok', ex_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--existing_train', default='data/train.json',
                    help='Current train.json -- used for few-shot and dedup.')
    ap.add_argument('--schemas_dir', default='schemas')
    ap.add_argument('--output', default='data/train_augmented.json',
                    help='Output: existing + validated new examples merged.')
    ap.add_argument('--audit', default='data/train_augmented_audit.json',
                    help='Per-example audit log (raw model output + validation result).')
    ap.add_argument('--target_per_schema', type=int, default=24,
                    help='How many NEW (valid) examples to add per target schema.')
    ap.add_argument('--batch_size', type=int, default=8,
                    help='Examples requested per API call. Smaller = better quality but more cost.')
    ap.add_argument('--focus_schemas', default=','.join(SBO_SCHEMAS),
                    help='Comma-separated db_ids to augment. Default: all 9 SBO modules.')
    ap.add_argument('--model', default='claude-haiku-4-5-20251001',
                    help='Anthropic model (default) or OpenAI model (use --use_openai).')
    ap.add_argument('--use_openai', action='store_true',
                    help='Use OPENAI_API_KEY and an OpenAI model instead of Anthropic.')
    ap.add_argument('--max_attempts_per_schema', type=int, default=10,
                    help='Cap on API-call rounds per schema before giving up.')
    ap.add_argument('--dry_run', action='store_true',
                    help='Print prompts and exit without API calls.')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    # Validate API key early -- nothing worse than failing 200 calls in.
    if not args.dry_run:
        if args.use_openai:
            if not os.environ.get('OPENAI_API_KEY'):
                sys.exit("ERROR: --use_openai but OPENAI_API_KEY not set in env.")
        else:
            if not os.environ.get('ANTHROPIC_API_KEY'):
                sys.exit("ERROR: ANTHROPIC_API_KEY not set in env. (Or pass --use_openai with OPENAI_API_KEY.)")

    with open(args.existing_train) as f:
        existing = json.load(f)
    seen_questions = {ex['question'].strip().lower() for ex in existing}
    next_qid = max(ex['question_id'] for ex in existing) + 700  # gap above any current id

    target_schemas = [s.strip() for s in args.focus_schemas.split(',') if s.strip()]
    print(f"[augment] existing train: {len(existing)} examples, target schemas: {len(target_schemas)}")

    all_new: List[Dict] = []
    audit: List[Dict] = []

    for db_id in target_schemas:
        try:
            schema = load_sql_schema(args.schemas_dir, db_id)
        except FileNotFoundError as e:
            print(f"[augment] SKIP {db_id!r}: schema file not found ({e})")
            continue
        schema_text = serialize_schema_compact(schema)

        kept_for_this_schema: List[Dict] = []
        attempts = 0
        while len(kept_for_this_schema) < args.target_per_schema and attempts < args.max_attempts_per_schema:
            attempts += 1
            n_remaining = args.target_per_schema - len(kept_for_this_schema)
            n_request = min(args.batch_size, n_remaining + 2)  # over-request a bit
            fewshot = build_fewshot(existing, db_id, k=5)
            prompt = GEN_PROMPT_TEMPLATE.format(
                db_id=db_id, schema_text=schema_text, fewshot=fewshot, n=n_request,
            )

            if args.dry_run:
                print(f"\n=== {db_id} | attempt {attempts} ===\n{prompt[:1500]}\n[...]")
                break

            try:
                raw = (call_openai if args.use_openai else call_claude)(prompt, args.model)
            except Exception as e:
                print(f"[augment] {db_id} attempt {attempts}: API error: {e}")
                time.sleep(2)
                continue

            try:
                batch = parse_json_array(raw)
                if not isinstance(batch, list):
                    raise ValueError('not a list')
            except Exception as e:
                print(f"[augment] {db_id} attempt {attempts}: JSON parse failed: {e}")
                audit.append({'db_id': db_id, 'attempt': attempts, 'status': 'json_parse_failed',
                              'raw_excerpt': raw[:500]})
                continue

            n_kept_this_batch = 0
            for ex_in in batch:
                ex_in.setdefault('db_id', db_id)
                ok, reason, ex_out = validate_example(ex_in, schema, seen_questions)
                audit.append({'db_id': db_id, 'attempt': attempts,
                              'question': ex_in.get('question'), 'sql': ex_in.get('gold_sql'),
                              'status': reason})
                if not ok:
                    continue
                ex_out['question_id'] = next_qid
                ex_out['db_id'] = db_id
                next_qid += 1
                seen_questions.add(ex_out['question'].strip().lower())
                kept_for_this_schema.append(ex_out)
                all_new.append(ex_out)
                n_kept_this_batch += 1
                if len(kept_for_this_schema) >= args.target_per_schema:
                    break

            print(f"[augment] {db_id} attempt {attempts}: kept {n_kept_this_batch}/{len(batch)}  "
                  f"(running total {len(kept_for_this_schema)}/{args.target_per_schema})")

        print(f"[augment] {db_id}: FINAL kept = {len(kept_for_this_schema)} new examples")

    if args.dry_run:
        return

    # Merge and write.
    merged = existing + all_new
    with open(args.output, 'w') as f:
        json.dump(merged, f, indent=2)
    with open(args.audit, 'w') as f:
        json.dump(audit, f, indent=2)

    # Summary.
    n_attempts = len(audit)
    n_kept = len(all_new)
    n_drop = n_attempts - n_kept
    from collections import Counter
    drop_reasons = Counter(a['status'] for a in audit if a['status'] != 'ok')
    print()
    print(f"[augment] SUMMARY: generated {n_attempts} candidates, kept {n_kept}, dropped {n_drop}")
    for reason, n in drop_reasons.most_common():
        print(f"  dropped {n:>4}  {reason}")
    print(f"[augment] merged train: {len(existing)} (existing) + {n_kept} (new) = {len(merged)}")
    print(f"[augment] wrote {args.output} and {args.audit}")


if __name__ == '__main__':
    main()
