"""
expand_columns.py -- Post-processing pass that adds high-confidence columns
the model missed, based on keyword overlap between the question and each
unpredicted column's name tokens.

Rationale: column score is our weakest axis (0.63 vs table 0.76 on the champion
ensemble). Inspection shows the model sometimes misses columns that are
LITERALLY named in the question (e.g., gold col `AssignDate` when question
says "the assigned date"). This script catches that specific failure mode
without any model retraining.

Conservative rule: a candidate column C is added to the prediction for table T
only if every "meaningful" token of C (length >= 3, non-stopword) appears in
the question. This is a high-precision rule -- it won't add `CASEID` just
because the question mentions some unrelated case, but it will add `ResolDate`
when the question literally says "resolved date".

Run:
    python expand_columns.py \
        --input predictions/preds_CHAMPION_v2_embed.json \
        --questions data/validation_input.json \
        --schemas_dir schemas \
        --output predictions/preds_CHAMPION_v2_embed_expanded.json
"""
import argparse
import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schema_utils import load_schema, tokenize_identifier


STOPWORDS = {
    # English filler
    'the', 'a', 'an', 'of', 'and', 'or', 'in', 'on', 'at', 'to', 'for',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'has', 'have', 'had',
    'do', 'does', 'did', 'with', 'from', 'by', 'into', 'onto', 'as',
    'me', 'us', 'you', 'i', 'we', 'they', 'it', 'them', 'their', 'our',
    # Query-y words
    'show', 'list', 'find', 'get', 'give', 'tell', 'display', 'return',
    'what', 'when', 'where', 'who', 'which', 'why', 'how', 'whose',
    'many', 'much', 'some', 'any', 'no', 'not', 'all', 'each', 'every',
    'count', 'sum', 'avg', 'min', 'max', 'group', 'order', 'sort',
    'that', 'this', 'these', 'those', 'there', 'than', 'then', 'over',
    'most', 'least', 'top', 'bottom', 'first', 'last', 'next', 'previous',
    'one', 'two', 'three', 'between', 'including', 'only', 'also',
}


def expand_one(question: str, schema_cols: List[str], predicted_cols: List[str], min_tokens: int = 2) -> List[str]:
    """Return predicted_cols PLUS any extra columns the keyword rule wants to add.

    Rule (high-precision): a column C is added iff
        (1) C has at least `min_tokens` "meaningful" tokens (len >= 3, non-stopword)
        (2) ALL of C's meaningful tokens appear (case-insensitive) in the question.

    Single-token columns like "ID", "Date", "Code" are never added even if their
    one token appears in the question -- that one-token rule was too aggressive
    in the v1 pass (added 73 cols, net -0.009 leaderboard from precision loss).
    """
    q_tokens = set(tokenize_identifier(question)) - STOPWORDS
    q_tokens = {t for t in q_tokens if len(t) >= 3}
    if not q_tokens:
        return list(predicted_cols)

    predicted_lc = {c.lower() for c in predicted_cols}
    extras = []
    for c in schema_cols:
        if c.lower() in predicted_lc:
            continue
        c_tokens = set(tokenize_identifier(c))
        meaningful = {t for t in c_tokens - STOPWORDS if len(t) >= 3}
        if len(meaningful) < min_tokens:
            continue  # skip generic / single-token columns
        overlap = meaningful & q_tokens
        if overlap == meaningful:
            extras.append(c)
    return sorted(set(predicted_cols) | set(extras))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input',  required=True, help='Predictions JSON to expand.')
    ap.add_argument('--questions', default='data/validation_input.json',
                    help='Question file with question_id, db_id, question.')
    ap.add_argument('--schemas_dir', default='schemas')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    preds = json.load(open(args.input))
    questions = {q['question_id']: q for q in json.load(open(args.questions))}
    schema_cache: Dict[str, Dict] = {}

    n_added_cols = 0
    out_preds = []
    for entry in preds:
        qid = entry['question_id']
        q = questions.get(qid)
        if q is None:
            out_preds.append(entry); continue
        db_id = q['db_id']
        if db_id not in schema_cache:
            schema_cache[db_id] = load_schema(args.schemas_dir, db_id)
        sch = schema_cache[db_id]
        sl_in = entry.get('schema_links', {}) or {}
        sl_out = {}
        for tbl, cols in sl_in.items():
            schema_cols_for_tbl = sch['columns'].get(tbl, [])
            new_cols = expand_one(q['question'], schema_cols_for_tbl, cols or [])
            n_added_cols += len(new_cols) - len(cols or [])
            sl_out[tbl] = new_cols
        out_preds.append({'question_id': qid, 'schema_links': sl_out})

    json.dump(out_preds, open(args.output, 'w'), indent=2)
    print(f"[expand] wrote {len(out_preds)} predictions to {args.output}")
    print(f"[expand] added {n_added_cols} columns total across all (qid, table) pairs")


if __name__ == '__main__':
    main()
