# CSE/DSC 234 Project 2 — Schema Linking

A schema-linking pipeline built on a **mixed-base, mixed-retriever 4-way LoRA ensemble** with majority-2 aggregation, across `Qwen2.5-Coder-1.5B-Instruct` (sweeps 13 and 15) and `Qwen3-1.7B` (sweep 22), all fine-tuned with LoRA adapters via RapidFire AI.

**Validation leaderboard score: 0.7270** (Table Score: 0.7911 / Column Score: 0.6630)

---

## Inference

```bash
python3 main.py --input <input.json> --output <pred.json>
```

By default, this runs the submitted **4-way LoRA ensemble** (sweep 22 + embed, sweep 13 + embed, sweep 15 + BM25, sweep 13 + hybrid) with **majority-2 aggregation** — an identifier survives only if it appears in at least 2 of the 4 per-(adapter, retriever) predictions. On a ~24 GB MIG slice, inference over the 101-question validation set completes in approximately **13 minutes** — within the 15-minute grading budget.

The ensemble groups adapters by their base model (read from each adapter's `adapter_config.json`) so each base is loaded only once. The default `--ensemble_dirs` order is deliberate: Qwen3 leads with sweep 22, then the Coder base loads for the three Qwen-Coder adapters. This guarantees that after the first 3 adapters complete, the on-disk partial-union is the same 3-way config (sweep 22 / sweep 13 / sweep 15) as the previous locked champion (0.7107) — so a mid-adapter-4 kill ships the locked baseline rather than a degraded 3-way of all-Coder adapters. After adapter 4 completes, the partial-write switches to majority-2 (the 0.7270 result).

The 4th entry reuses sweep 13's weights with the hybrid retriever — pure post-hoc diversity, no extra training. The `--ensemble_dirs` flag accepts a `path[:retriever]` spec per adapter; `--aggregation` accepts `union` (legacy 3-way behavior), `maj2` (the default), or `maj2_or_lead` (maj2 union'd with the first adapter's outputs).

To use a single adapter (faster, slightly lower score):

```bash
python3 main.py --input <input.json> --output <pred.json> --single
```

### Model artifacts

| Path | Used at inference as |
|---|---|
| `./adapter/` | Single best LoRA adapter (sweep 13) — required by rubric |
| `./adapter_ensemble/sweep22/` | Pass 1: Qwen3-1.7B base, embedding retrieval |
| `./adapter_ensemble/sweep13/` | Pass 2: Qwen-Coder-1.5B base, embedding retrieval |
| `./adapter_ensemble/sweep15/` | Pass 3: Qwen-Coder-1.5B base, BM25 retrieval |
| `./adapter_ensemble/sweep13/` (reused) | Pass 4: Qwen-Coder-1.5B base, hybrid retrieval |

Both base models (`Qwen/Qwen2.5-Coder-1.5B-Instruct` and `Qwen/Qwen3-1.7B`) are pulled from Hugging Face Hub via `AutoModelForCausalLM.from_pretrained`. Qwen3 loads first (pass 1), is freed, then the Coder base loads for passes 2–4. Within the Coder group, the three adapters (sweep 13 loaded twice for embed + hybrid passes, sweep 15 loaded once) are wrapped as named PEFT adapters and swapped via `model.set_adapter(name)` between passes. Each adapter is ~70 MB.

After each adapter completes, partial-union predictions are atomically written to `--output`. A mid-third-adapter timeout will still return a 2-way ensemble result, which is strictly better than a single-adapter result.

---

## Repository Layout

```
main.py                     Inference entry point (4-way maj2 ensemble default; --single for one adapter)
eval.py                     Course-provided evaluation script (table-/column-level P/R/F1)
prompt.py                   Shared prompt construction and output parsing utilities
schema_utils.py             Schema loading, serialization (compact/types/keys),
                            BM25 / embedding / hybrid table retrievers
training/                   Training-side scripts (NOT used at inference)
  ├── train_rapidfire.py    Training entry point (RapidFire AI RFGridSearch)
  ├── train_single.py       Plain TRL SFTTrainer fallback (used during early development)
  └── reproduce_champion.py Rebuilds the champion ensemble predictions from per-(adapter, retriever) files
data_prep/                  One-off data conversion / augmentation scripts (NOT used at inference)
  ├── format_training_data.py    Converts train.json / validation.json to JSONL chat-message format
  ├── format_two_stage_data.py   Splits each example into stage-A / stage-B SFT pairs
  ├── augment_data.py            Generates SBO training examples via the Claude API
  ├── generate_cot_traces.py     Generates chain-of-thought rationales via the Claude API (sweep 21)
  ├── expand_columns.py          Keyword-based column-expansion post-processor (exploratory)
  └── sql_to_schema_links.py     Extracts schema-link gold labels from SQL via sqlglot
scripts/                    Analysis / probe scripts (NOT used at inference)
  └── probe_ensembles.py    Combinatorial search over (adapter, retriever) ensemble configs
adapter/                    Submitted single LoRA adapter (sweep 13; rubric-required path)
adapter_ensemble/           Submitted 3-directory ensemble: sweep13, sweep15, sweep22
                            (sweep 13 is loaded twice at inference under embed + hybrid retrievers)
adapter_v2/                 Early artifact from the train_single.py path (kept for reference)
data/                       train.json, validation.json, preprocessed JSONL files,
                            and train_augmented.json (Claude-generated SBO augmentations)
predictions/                Per-(adapter, retriever) and ensemble validation outputs
logs/                       Per-sweep rapidfire.log, training.log, and metrics.json
                            (extracted from the RapidFire AI MLflow database)
docs/                       Course-provided project statement and sample_main.py
schemas/                    17 Spider-format database schemas (rubric-required path)
tests/                      Smoke tests (test_checklist.py exercises the rubric quick-start invariants)
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
