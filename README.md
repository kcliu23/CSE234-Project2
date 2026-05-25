# CSE/DSC 234 Project 2 -- Submission

Schema-linking pipeline built on `Qwen2.5-Coder-1.5B-Instruct` + LoRA, trained via
RapidFire AI. **Validation leaderboard: 0.7002** (Table 0.7648 / Column 0.6356).

## How to run (TA grading workflow)

```bash
python3 main.py --input <input.json> --output <pred.json>
```

That's it. With no other flags this runs the submitted **3-way LoRA ensemble**
with embedding-based table retrieval, producing the 0.7002 result. On the
~24 GB MIG slice it completes the 101-question validation set in **~11 minutes**,
well under the 15-minute grading budget.

The model artifact is loaded as follows:

- Base model: `Qwen/Qwen2.5-Coder-1.5B-Instruct` (pulled from HF Hub via
  `AutoModelForCausalLM.from_pretrained`).
- `./adapter/` -- the single best LoRA adapter (sweep 13). Per the rubric this
  folder is committed at the repo root.
- `./adapter_ensemble/sweep{13,15,18}/` -- the three LoRA adapters that the
  default ensemble path loads sequentially (same base, different LR / training
  data). Each is ~150 MB. `main.py` swaps them with `PeftModel.from_pretrained`
  between passes and unions the per-(table, column) predictions.

After each adapter completes, partial-union predictions are atomically written
to `--output`, so even a mid-third-adapter timeout returns a 2-way ensemble
(strictly better than a single adapter).

To use a single adapter (faster, lower score), pass `--single` -- this loads
just `./adapter/`.

## Required Python dependencies

```
rapidfireai      # used by train_rapidfire.py for training (NOT inference)
transformers
trl
peft
torch
datasets
sentence-transformers   # for BAAI/bge-small-en-v1.5 retriever in main.py
sqlglot                 # for sql_to_schema_links.py
PyPDF2 / pypdf          # only if you re-read the project statement
anthropic               # only if you re-run augment_data.py
```

All standard, no auth-gated models. The BGE retriever auto-downloads (~30 MB).

## Repo layout (our additions)

```
main.py                     inference entry (ensemble default, --single fallback)
prompt.py                   shared prompt + parsing utilities
schema_utils.py             schema loading, serialization (compact/types/keys),
                            BM25 / embed / hybrid table retrievers
train_rapidfire.py          training entry (RapidFire AI RFGridSearch)
train_single.py             plain TRL SFTTrainer fallback (used during early dev)
format_training_data.py     train.json / validation.json -> JSONL
format_two_stage_data.py    splits each example into stage-A / stage-B SFT pairs
augment_data.py             generates additional SBO training data via Claude API
expand_columns.py           keyword-based column-expansion post-processor (not in final)
reproduce_champion.py       rebuilds preds_CHAMPION_v2_embed.json from per-model preds
adapter/                    SUBMITTED single LoRA (sweep 13, rubric-required path)
adapter_ensemble/           SUBMITTED 3 LoRA adapters used by the ensemble default
adapter_v2/                 EARLY artifact (train_single.py path), kept for reference
data/                       train.json, validation.json, *.jsonl preprocessed,
                            train_augmented.json (Claude-generated SBO aug)
predictions/                all per-model + ensemble validation outputs, kept for the report
logs/                       per-sweep rapidfire.log, training.log, metrics.json
                            (extracted from the rapidfireai mlflow DB)
docs/                       course-provided project statement + sample_main.py
schemas/                    17 Spider-format DB schemas (rubric-required path)
tests/                      smoke tests
```

The release-packet documentation below (file formats, gotchas, etc.) is the
upstream README we started from and is preserved verbatim for reference.

---

# Project 2 Release Packet -- CSE/DSC 234, Spring 2026

This packet accompanies the Project 2 statement PDF. **Read the statement first**;
this README only documents the files, commands, and a handful of likely gotchas.

## Quick-start checklist (read before writing code)

The following are easy to get wrong and will cost you score or break your runs:

- [ ] **`question_id` is per-file, not globally unique.** `train.json` has IDs
      1..301, `validation_input.json` has 1..101, the hidden test has 1..N
      starting at 1 again. Do not assume globally unique IDs across splits.
- [ ] **Four `db_id` values contain spaces** and map to underscored filenames:
      `"SBODemoUS-Business Partners"` → `schemas/SBODemoUS-Business_Partners.json`
      (also `Human Resources`, `Inventory and Production`, `Sales Opportunities`).
- [ ] **Schemas use Spider format.** Their `column_names_original` field is
      a list of `[table_index, column_name]` pairs, plus a synthetic
      `[-1, "*"]` entry at index 0 that you should ignore. See the loader
      snippet below.
- [ ] **A referenced table with no columns is `{"t": []}`, not omitted.**
      `select count(*) from t` and `select * from t` both produce
      `{"t": []}` for the table-only case.
- [ ] **Hallucinated identifiers (not in schema) count as false positives.**
      They lower your precision. Post-process your model's output to drop
      identifiers absent from the target schema.
- [ ] **Identifier casing in the output is matched case-insensitively** during
      grading, but emit the schema's casing for easy debugging.
- [ ] **Output order doesn't matter** (graded by `question_id`), but you must
      output exactly one entry per input `question_id`.
- [ ] **Commit the `schemas/` folder into your repo** so that `main.py` can
      find it at `./schemas/` when graders run it.

## Loading a schema in Python

```python
import json
def load_schema_as_dict(db_id, schemas_dir='./schemas'):
    fname = db_id.replace(' ', '_').replace('/', '_') + '.json'
    with open(f'{schemas_dir}/{fname}') as f:
        s = json.load(f)
    schema = {t: [] for t in s['table_names_original']}
    for tidx, cname in s['column_names_original']:
        if tidx == -1:       # skip the synthetic '*' entry
            continue
        schema[s['table_names_original'][tidx]].append(cname)
    return schema  # {table_name: [col, col, ...]}
```

## Contents

```
release_packet/
├── README.md                            (this file)
├── schemas/                             one Spider-format JSON per database (17 dbs)
│   ├── _index.json                      summary of all schemas
│   ├── ASIS_20161108_HerpInv_Database.json
│   ├── ATBI.json
│   ├── ... (15 more)
├── train.json                           301 training examples (q, db_id, gold_sql, schema_links)
├── validation.json                      101 validation examples (same fields as train)
├── validation_input.json                Same 101 questions in input-only format (what your main.py sees)
├── validation_gold_schema_links.json    Parallel gold answers for the validation split
├── eval.py                              Grader: computes Table & Column P/R/F1 + leaderboard score
├── sql_to_schema_links.py               Helper: extract schema_links from any SQL (for data aug)
└── sample_main.py                       Stub illustrating the expected CLI / I/O contract for main.py
```

## File formats

### Schema file (Spider format)

```json
{
  "db_id": "NTSB",
  "table_names_original": ["AIRBAG", "CHILDSEAT", "CRASH", "CDC", "EVENT", ...],
  "column_names_original": [
      [-1, "*"],
      [0, "CASEID"], [0, "PSU"], [0, "CASENO"], [0, "BAGDEPLOY"], ...,
      [1, "CASEID"], [1, "SEATTYPE"], ...
  ],
  "column_types": ["int", "int", "int", "varchar", ...],
  "primary_keys": [...],
  "foreign_keys": [...]
}
```

Each `column_names_original` entry is `[table_index, column_name]`. The first
entry `[-1, "*"]` is a synthetic wildcard you should ignore for schema
linking. See the loader snippet at the top of this README for a 6-line
function that converts this into a `{table: [columns]}` Python dict.

### Training example

```json
{
  "question_id": 1,
  "db_id": "NTSB",
  "question": "Show a count of injuries by body region where the injury severity is critical. The lookup code for critical injury is 5.",
  "gold_sql": "SELECT REGION, COUNT(*) INJCOUNT FROM INJURY WHERE AIS = 5 GROUP BY REGION",
  "schema_links": {
    "INJURY": ["AIS", "REGION"]
  }
}
```

### Input to `main.py` (what TA grader feeds in)

```json
[
  {"question_id": 1, "db_id": "<DB_NAME>", "question": "<NL question>"},
  ...
]
```

### Output from `main.py` (what your model must produce)

```json
[
  {"question_id": 1,
   "schema_links": {"<Table1>": ["<Col1>", "<Col2>"], "<Table2>": []}},
  ...
]
```

A table referenced with no columns (e.g. `select count(*) from t`) MUST appear with
an empty list, not be omitted. Identifier casing should match the schema (case-insensitive
matching is applied during grading, but matching casing helps you debug).

### Boundary cases for wildcards (`*`)

The gold ground-truth treats SQL `*` as a syntactic wildcard, NOT as "all columns".
Concretely:

| Gold SQL                                  | Gold `schema_links`         |
|-------------------------------------------|-----------------------------|
| `select count(*) from t`                  | `{"t": []}`                 |
| `select * from t`                         | `{"t": []}`                 |
| `select * from t where x=1`               | `{"t": ["x"]}`              |
| `select count(*), x from t group by x`    | `{"t": ["x"]}`              |
| `select a from t1 join t2 on t1.k=t2.k`   | `{"t1": ["a","k"], "t2": ["k"]}` |

Note that the table appears in the output even when no specific columns are named.
This matches what `sql_to_schema_links.py` produces from any SQL.

## Commands

### Run the sample stub end-to-end and grade it

```bash
python sample_main.py \
    --input  validation_input.json \
    --output preds.json \
    --schemas_dir schemas/

python eval.py \
    --predictions preds.json \
    --gold        validation_gold_schema_links.json \
    --schemas_dir schemas/ \
    --questions_input validation_input.json \
    --per_question_out per_q.csv
```

### Generate schema_links for an SQL query (e.g. for augmentation)

```bash
python sql_to_schema_links.py --schemas_dir schemas/ \
    --db_id NTSB \
    --sql "select count(*) from AIRBAG where BAGDEPLOY = 'YES'"
```

Batch mode (over a list of `{db_id, gold_sql, ...}` records):

```bash
python sql_to_schema_links.py --schemas_dir schemas/ \
    --batch_in  my_aug_queries.json \
    --batch_out my_aug_queries_with_links.json
```

## Dataset provenance

The NL questions, gold SQL queries, and database schemas are drawn from SNAILS
[Luoma & Kumar, SIGMOD 2025], an artifact suite developed at UCSD ADALab.
We extract per-question schema-link ground truth automatically from each gold
SQL using `sqlglot` parsing + schema-aware column qualification (see
`sql_to_schema_links.py`). The same extractor was used to build the training
and validation splits in this packet.

The hidden test set is drawn from the same source distribution and graded
with the same `eval.py` script.

## Required Python dependencies

The grader (`eval.py`) uses only the Python standard library and has no
third-party dependencies. The helper (`sql_to_schema_links.py`), which you
will need only if you generate augmented training data, requires:

```
sqlglot>=23.0
```

(Your own training/inference pipeline will additionally need `rapidfireai`,
`transformers`, `trl`,
`peft`, etc.; install those as part of your environment setup.)
# CSE234-Project2
