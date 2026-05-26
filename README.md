# CSE/DSC 234 Project 2 — Schema Linking

A schema-linking pipeline built on `Qwen2.5-Coder-1.5B-Instruct` fine-tuned with LoRA adapters via RapidFire AI.

**Validation leaderboard score: 0.7006** (Table F1: 0.7648 / Column F1: 0.6356)

---

## Inference

```bash
python3 main.py --input <input.json> --output <pred.json>
```

By default, this runs the submitted **3-way LoRA ensemble** with embedding-based table retrieval, reproducing the 0.7006 validation result. On a ~24 GB MIG slice, inference over the 101-question validation set completes in approximately **11 minutes** — well within the 15-minute grading budget.

To use a single adapter (faster, slightly lower score):

```bash
python3 main.py --input <input.json> --output <pred.json> --single
```

### Model artifacts

| Path | Description |
|---|---|
| `./adapter/` | Single best LoRA adapter (sweep 13) — required by rubric |
| `./adapter_ensemble/sweep{13,15,18}/` | Three LoRA adapters used by the default ensemble path |

The base model (`Qwen/Qwen2.5-Coder-1.5B-Instruct`) is pulled from Hugging Face Hub via `AutoModelForCausalLM.from_pretrained`. Each ensemble adapter is ~150 MB and is loaded sequentially using `PeftModel.from_pretrained`.

After each adapter completes, partial-union predictions are atomically written to `--output`. A mid-third-adapter timeout will still return a 2-way ensemble result, which is strictly better than a single-adapter result.

---

## Repository Layout

```
main.py                     Inference entry point (ensemble default; --single for single adapter)
prompt.py                   Shared prompt construction and output parsing utilities
schema_utils.py             Schema loading, serialization (compact/types/keys),
                            BM25 / embedding / hybrid table retrievers
train_rapidfire.py          Training entry point (RapidFire AI RFGridSearch)
train_single.py             Plain TRL SFTTrainer fallback (used during early development)
format_training_data.py     Converts train.json / validation.json to JSONL format
format_two_stage_data.py    Splits each example into stage-A / stage-B SFT pairs
augment_data.py             Generates additional SBO training examples via the Claude API
expand_columns.py           Keyword-based column-expansion post-processor (not in final submission)
reproduce_champion.py       Rebuilds preds_CHAMPION_v2_embed.json from per-model predictions
adapter/                    Submitted single LoRA adapter (sweep 13; rubric-required path)
adapter_ensemble/           Submitted 3-adapter ensemble used by the default inference path
adapter_v2/                 Early artifact from the train_single.py path (kept for reference)
data/                       train.json, validation.json, preprocessed JSONL files,
                            and train_augmented.json (Claude-generated SBO augmentations)
predictions/                Per-model and ensemble validation outputs (kept for the report)
logs/                       Per-sweep rapidfire.log, training.log, and metrics.json
                            (extracted from the RapidFire AI MLflow database)
docs/                       Course-provided project statement and sample_main.py
schemas/                    17 Spider-format database schemas (rubric-required path)
tests/                      Smoke tests
```

---

## Python Dependencies

```
transformers
trl
peft
torch
datasets
sentence-transformers   # BAAI/bge-small-en-v1.5 retriever used in main.py
sqlglot                 # used by sql_to_schema_links.py
rapidfireai             # used by train_rapidfire.py (training only, not inference)
anthropic               # used by augment_data.py (data augmentation only)
PyPDF2 / pypdf          # only needed if re-reading the project statement PDF
```

All packages are publicly available. The BGE retriever model auto-downloads (~30 MB) on first run.

---

## Release Packet Reference

The sections below document the file formats, commands, and grading contract from the upstream course release packet. They are preserved verbatim for reference.

### Quick-Start Checklist

The following are easy to get wrong and will cost score or break runs:

- [ ] **`question_id` is per-file, not globally unique.** `train.json` has IDs 1–301, `validation_input.json` has 1–101, and the hidden test set starts at 1 again. Do not assume globally unique IDs across splits.
- [ ] **Four `db_id` values contain spaces** and map to underscored filenames: `"SBODemoUS-Business Partners"` → `schemas/SBODemoUS-Business_Partners.json` (also `Human Resources`, `Inventory and Production`, `Sales Opportunities`).
- [ ] **Schemas use Spider format.** The `column_names_original` field is a list of `[table_index, column_name]` pairs, plus a synthetic `[-1, "*"]` entry at index 0 that should be ignored.
- [ ] **A referenced table with no columns must be `{"t": []}`, not omitted.** Both `select count(*) from t` and `select * from t` produce `{"t": []}`.
- [ ] **Hallucinated identifiers count as false positives.** Post-process model output to drop any identifier absent from the target schema.
- [ ] **Identifier casing is matched case-insensitively** during grading, but emit schema casing for easier debugging.
- [ ] **Output order does not matter** (graded by `question_id`), but exactly one entry per input `question_id` is required.
- [ ] **Commit the `schemas/` folder** so that `main.py` can find it at `./schemas/` when run by graders.

### Loading a Schema in Python

```python
import json

def load_schema_as_dict(db_id, schemas_dir='./schemas'):
    fname = db_id.replace(' ', '_').replace('/', '_') + '.json'
    with open(f'{schemas_dir}/{fname}') as f:
        s = json.load(f)
    schema = {t: [] for t in s['table_names_original']}
    for tidx, cname in s['column_names_original']:
        if tidx == -1:  # skip the synthetic '*' entry
            continue
        schema[s['table_names_original'][tidx]].append(cname)
    return schema  # {table_name: [col, col, ...]}
```

### File Formats

**Schema file (Spider format)**

```json
{
  "db_id": "NTSB",
  "table_names_original": ["AIRBAG", "CHILDSEAT", "CRASH", "CDC", "EVENT"],
  "column_names_original": [
      [-1, "*"],
      [0, "CASEID"], [0, "PSU"], [0, "CASENO"], [0, "BAGDEPLOY"],
      [1, "CASEID"], [1, "SEATTYPE"]
  ],
  "column_types": ["int", "int", "int", "varchar"],
  "primary_keys": [],
  "foreign_keys": []
}
```

**Training example**

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

**Input to `main.py`**

```json
[
  {"question_id": 1, "db_id": "<DB_NAME>", "question": "<NL question>"},
  ...
]
```

**Output from `main.py`**

```json
[
  {
    "question_id": 1,
    "schema_links": {"<Table1>": ["<Col1>", "<Col2>"], "<Table2>": []}
  },
  ...
]
```

A table referenced with no columns must appear with an empty list, not be omitted. Identifier casing should match the schema (case-insensitive matching is applied during grading).

**Wildcard boundary cases**

| Gold SQL | Gold `schema_links` |
|---|---|
| `select count(*) from t` | `{"t": []}` |
| `select * from t` | `{"t": []}` |
| `select * from t where x=1` | `{"t": ["x"]}` |
| `select count(*), x from t group by x` | `{"t": ["x"]}` |
| `select a from t1 join t2 on t1.k=t2.k` | `{"t1": ["a","k"], "t2": ["k"]}` |

### Commands

**Run the sample stub and grade it**

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

**Generate schema links from a SQL query**

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

### Dataset Provenance

NL questions, gold SQL queries, and database schemas are drawn from SNAILS [Luoma & Kumar, SIGMOD 2025], an artifact suite developed at UCSD ADALab. Per-question schema-link ground truth is extracted automatically from each gold SQL using `sqlglot` parsing with schema-aware column qualification (see `sql_to_schema_links.py`). The hidden test set is drawn from the same source distribution and graded with the same `eval.py` script.

### Grader Dependencies

The grader (`eval.py`) uses only the Python standard library. The schema-link extractor (`sql_to_schema_links.py`) requires:

```
sqlglot>=23.0
```
