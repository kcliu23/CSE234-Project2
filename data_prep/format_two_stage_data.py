"""
format_two_stage_data.py -- Split each train example into two-stage SFT
training data:

  Stage A (table prediction): given question + the TABLE NAMES of the
    schema (no columns), predict the set of tables the SQL touches.
    Output: {"tables": ["T1", "T2", ...]}

  Stage B (column prediction): given question + the columns of ONE
    specific table, predict the columns of THAT table the SQL touches.
    Output: {"columns": ["c1", "c2", ...]}  (possibly empty for
    wildcard tables like SELECT COUNT(*) FROM t)

Each train.json example produces:
  - 1 stage-A example
  - K stage-B examples (one per gold table)

Inference (in main.py --two_stage):
  - Stage A predicts the table set.
  - For each predicted table, stage B predicts the columns.
  - Merge into the final {"<Table>": [<cols>]} contract.

Run:
  python format_two_stage_data.py \
      --input data/train.json \
      --schemas_dir schemas \
      --output_a data/train_stageA.jsonl \
      --output_b data/train_stageB.jsonl

  python format_two_stage_data.py \
      --input data/validation.json \
      --schemas_dir schemas \
      --output_a data/validation_stageA.jsonl \
      --output_b data/validation_stageB.jsonl
"""
import argparse
import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schema_utils import load_schema, tokenize_identifier, _bm25_score_tables


STAGE_A_SYSTEM = (
    "You are a schema-linking assistant. Given a database's TABLE NAMES and a "
    "natural-language question, identify the tables the underlying SQL would "
    "reference. Reply with a single JSON object on one line and nothing else, "
    "of the form: {\"tables\": [\"Table1\", \"Table2\"]}. Use only table names "
    "that appear in the provided list -- do not invent names."
)

STAGE_B_SYSTEM = (
    "You are a schema-linking assistant. Given a database table's COLUMN names "
    "and a natural-language question, identify the columns of that table the "
    "underlying SQL would reference. Reply with a single JSON object on one "
    "line and nothing else, of the form: {\"columns\": [\"Col1\", \"Col2\"]}. "
    "A table referenced with no specific columns (e.g. `SELECT COUNT(*) FROM t`) "
    "should produce {\"columns\": []}. Use only column names that appear in the "
    "provided list -- do not invent names."
)


def build_stageA_user(db_id: str, question: str, schema: Dict, max_tables: int = 40) -> str:
    """For stage A we only show table names (no columns). With table names only
    being very cheap (~1 line per table), we can afford a bigger max_tables
    cap than the joint serializer's default of 20.
    """
    tables = schema['tables']
    if len(tables) > max_tables:
        # Same BM25 filter as the joint prompt, just on table-name docs.
        scores = _bm25_score_tables(schema, question)
        kept = {t for t, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:max_tables]}
        tables = [t for t in tables if t in kept]
    table_list = '\n'.join(f"- {t}" for t in tables)
    return (
        f"Database: {db_id}\n"
        f"Tables:\n{table_list}\n\n"
        f"Question: {question}\n\n"
        f"Output the tables JSON now."
    )


def build_stageB_user(db_id: str, question: str, table_name: str, columns: List[str]) -> str:
    col_list = ', '.join(columns)
    return (
        f"Database: {db_id}\n"
        f"Table: {table_name}\n"
        f"Columns: {col_list}\n\n"
        f"Question: {question}\n\n"
        f"Output the columns JSON now."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='train.json or validation.json')
    ap.add_argument('--schemas_dir', default='schemas')
    ap.add_argument('--output_a', required=True, help='Stage A JSONL output path')
    ap.add_argument('--output_b', required=True, help='Stage B JSONL output path')
    ap.add_argument('--stage_a_max_tables', type=int, default=40,
                    help='Cap on tables shown in stage-A prompts; BM25-filtered if exceeded.')
    args = ap.parse_args()

    with open(args.input) as f:
        items = json.load(f)

    schema_cache: Dict[str, Dict] = {}
    def get_schema(db_id):
        if db_id not in schema_cache:
            schema_cache[db_id] = load_schema(args.schemas_dir, db_id)
        return schema_cache[db_id]

    n_a, n_b = 0, 0
    with open(args.output_a, 'w') as fa, open(args.output_b, 'w') as fb:
        for ex in items:
            db_id = ex['db_id']
            sch = get_schema(db_id)
            gold = ex.get('schema_links', {})

            # --- Stage A ---
            user_a = build_stageA_user(db_id, ex['question'], sch, args.stage_a_max_tables)
            tgt_a = json.dumps({'tables': sorted(gold.keys())}, separators=(',', ': '))
            msgs_a = [
                {'role': 'system',    'content': STAGE_A_SYSTEM},
                {'role': 'user',      'content': user_a},
                {'role': 'assistant', 'content': tgt_a},
            ]
            fa.write(json.dumps({'messages': msgs_a, 'db_id': db_id, 'question_id': ex['question_id']}) + '\n')
            n_a += 1

            # --- Stage B (one example per gold table) ---
            # Case-insensitive lookup of gold tables in schema.
            lc_tables = {t.lower(): t for t in sch['tables']}
            for gtbl, gcols in gold.items():
                tlc = str(gtbl).lower()
                if tlc not in lc_tables:
                    continue  # skip gold tables not in schema (shouldn't happen)
                canon_t = lc_tables[tlc]
                cols_all = sch['columns'][canon_t]
                user_b = build_stageB_user(db_id, ex['question'], canon_t, cols_all)
                # gold cols in original casing
                lc_cols = {c.lower(): c for c in cols_all}
                gold_cols = sorted({lc_cols[str(c).lower()] for c in (gcols or [])
                                    if str(c).lower() in lc_cols})
                tgt_b = json.dumps({'columns': gold_cols}, separators=(',', ': '))
                msgs_b = [
                    {'role': 'system',    'content': STAGE_B_SYSTEM},
                    {'role': 'user',      'content': user_b},
                    {'role': 'assistant', 'content': tgt_b},
                ]
                fb.write(json.dumps({'messages': msgs_b, 'db_id': db_id,
                                     'question_id': ex['question_id'], 'table': canon_t}) + '\n')
                n_b += 1

    print(f"Wrote {n_a} stage-A examples to {args.output_a}")
    print(f"Wrote {n_b} stage-B examples to {args.output_b}")


if __name__ == '__main__':
    main()
