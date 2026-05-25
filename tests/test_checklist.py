"""
Verifies our pipeline against the README's quick-start checklist.

Run with:
    cd CSE234-Project2 && python tests/test_checklist.py
"""
import json
import os
import subprocess
import sys
import tempfile

# Allow `import schema_utils` from one level up.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompt import build_messages, parse_model_output, target_string
from schema_utils import (
    canonicalize_prediction,
    db_id_to_filename,
    load_schema,
    serialize_schema_compact,
)


def section(title):
    print(f"\n=== {title} ===")


def assert_eq(actual, expected, msg):
    ok = actual == expected
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {msg}")
    if not ok:
        print(f"    expected: {expected!r}")
        print(f"    actual  : {actual!r}")
    return ok


def main():
    all_ok = True

    # ---- 1. question_id is per-file, not globally unique ----
    section("1. question_id per-file (no cross-file dedup)")
    train = json.load(open('data/train.json'))
    val = json.load(open('data/validation_input.json'))
    train_ids = sorted({x['question_id'] for x in train})
    val_ids = sorted({x['question_id'] for x in val})
    all_ok &= assert_eq(train_ids[:3], [1, 2, 3], "train IDs start at 1")
    all_ok &= assert_eq(val_ids[:3], [1, 2, 3], "val IDs also start at 1")
    overlap = set(train_ids) & set(val_ids)
    print(f"  (note) train/val ID overlap size = {len(overlap)} -- our code never collides on these because main.py uses per-file IDs only")

    # Verify main.py round-trips question_ids 1:1 in --mock mode.
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
        out_path = f.name
    subprocess.run(
        [sys.executable, 'main.py', '--mock', '--input', 'data/validation_input.json', '--output', out_path],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    preds = json.load(open(out_path))
    pred_ids = sorted(p['question_id'] for p in preds)
    all_ok &= assert_eq(pred_ids, val_ids, "main.py emits one prediction per input question_id")
    all_ok &= assert_eq(type(preds[0]['question_id']).__name__, 'int', "question_id stays an int through the pipeline")
    os.unlink(out_path)

    # ---- 2. db_ids with spaces map to underscored filenames ----
    section("2. db_id with spaces -> underscored filename")
    space_dbs = [
        'SBODemoUS-Business Partners',
        'SBODemoUS-Human Resources',
        'SBODemoUS-Inventory and Production',
        'SBODemoUS-Sales Opportunities',
    ]
    for db in space_dbs:
        fname = db_id_to_filename(db)
        expect = db.replace(' ', '_') + '.json'
        all_ok &= assert_eq(fname, expect, f"{db!r} -> {expect}")
        # Loader actually opens the file successfully:
        sch = load_schema('schemas/', db)
        all_ok &= assert_eq(sch['db_id'], db, f"loader returns matching db_id for {db!r}")
        all_ok &= (len(sch['tables']) > 0)

    # ---- 3. Spider format: [-1, "*"] is skipped ----
    section("3. Spider format: synthetic [-1, '*'] is dropped")
    sch = load_schema('schemas/', 'NTSB')
    all_ok &= assert_eq('*' in [c for cols in sch['columns'].values() for c in cols], False,
                       "no '*' column appears in any table's column list")
    # And the raw file does have that synthetic entry, to be sure we'd otherwise pick it up:
    raw = json.load(open('schemas/NTSB.json'))
    has_star = any(tidx == -1 and cname == '*' for tidx, cname in raw['column_names_original'])
    all_ok &= assert_eq(has_star, True, "raw schema does include [-1,'*'] (so we're really filtering it)")

    # ---- 4. Wildcards: table with no columns is {"t": []}, not omitted ----
    section("4. Wildcard: {\"t\": []} preserved through prediction pipeline")
    sch_ntsb = load_schema('schemas/', 'NTSB')
    # canonicalize_prediction should preserve an empty-column table:
    canon = canonicalize_prediction({'INJURY': []}, sch_ntsb)
    all_ok &= assert_eq(canon, {'INJURY': []}, "canonicalize_prediction keeps {'INJURY': []}")
    # target_string (training target) should preserve an empty-column table:
    tgt = target_string({'INJURY': []})
    all_ok &= assert_eq(tgt, '{"INJURY": []}', "target_string emits {\"INJURY\": []} as-is")
    # And we explicitly tell the model about this rule in the system prompt:
    sys_msg = build_messages('NTSB', 'q', sch_ntsb)[0]['content']
    all_ok &= assert_eq('"t": []' in sys_msg, True, "system prompt explicitly instructs the model on the empty-list rule")
    # Real train.json examples with empty col lists round-trip correctly:
    train_wildcards = [ex for ex in json.load(open('data/train.json'))
                       if any(not v for v in ex['schema_links'].values())]
    print(f"  (note) {len(train_wildcards)} training examples have at least one empty-column table")
    for ex in train_wildcards:
        sch = load_schema('schemas/', ex['db_id'])
        canon = canonicalize_prediction(ex['schema_links'], sch)
        # Tables with empty lists should still be present after canonicalization:
        for tbl, cols in ex['schema_links'].items():
            if not cols:
                all_ok &= assert_eq(tbl in canon and canon[tbl] == [], True,
                                    f"q{ex['question_id']} ({ex['db_id']}): empty-col table {tbl!r} survives")

    # ---- 5. Hallucination filtering ----
    section("5. Predictions: hallucinated identifiers are dropped")
    sch = load_schema('schemas/', 'NTSB')
    pred_with_garbage = {
        'INJURY': ['AIS', 'REGION', 'TOTALLY_FAKE_COL'],
        'TableThatDoesNotExist': ['x', 'y'],
        'TIRE': ['CASEID'],   # legit
    }
    canon = canonicalize_prediction(pred_with_garbage, sch)
    all_ok &= assert_eq(set(canon.keys()), {'INJURY', 'TIRE'},
                       "fake table dropped, legit tables kept")
    all_ok &= assert_eq(set(canon['INJURY']), {'AIS', 'REGION'},
                       "fake column dropped from valid table")

    # ---- 6. Casing: rewrite to schema casing ----
    section("6. Casing: predictions are re-cased to match the schema")
    canon = canonicalize_prediction({'injury': ['ais', 'region']}, sch)
    all_ok &= assert_eq(canon, {'INJURY': ['AIS', 'REGION']},
                       "lower-case identifiers re-cased to schema casing")
    # Duplicate-cased keys are merged (same logical table emitted twice):
    canon = canonicalize_prediction({'injury': ['ais'], 'INJURY': ['region']}, sch)
    all_ok &= assert_eq(canon, {'INJURY': ['AIS', 'REGION']},
                       "duplicate-cased keys merged, not overwritten")

    # ---- 7. Output cardinality: 1:1 with input, order-independent ----
    section("7. Output: exactly one entry per input question_id")
    n_in = len(json.load(open('data/validation_input.json')))
    n_out = len(json.load(open('/tmp/preds_mock.json'))) if os.path.exists('/tmp/preds_mock.json') else None
    if n_out is None:
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
            out_path = f.name
        subprocess.run([sys.executable, 'main.py', '--mock',
                       '--input', 'data/validation_input.json', '--output', out_path],
                      check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        n_out = len(json.load(open(out_path)))
        os.unlink(out_path)
    all_ok &= assert_eq(n_out, n_in, f"output count == input count ({n_in})")

    # ---- 8. schemas/ available at ./schemas relative to working dir ----
    section("8. schemas/ folder exists at the expected relative path")
    all_ok &= assert_eq(os.path.isdir('schemas'), True, "./schemas/ exists")
    all_ok &= assert_eq(os.path.isfile('schemas/_index.json'), True, "./schemas/_index.json exists")
    all_ok &= assert_eq(len([f for f in os.listdir('schemas') if f.endswith('.json') and f != '_index.json']),
                       17, "17 schema files present")

    print()
    print("==========================================================")
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    print("==========================================================")
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
