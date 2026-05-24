"""
Project 2 inference entry point.

CLI:
    python main.py --input <input.json> --output <output.json>

Optional flags:
    --schemas_dir <path>         (default ./schemas)
    --base_model  <hf_repo_id>   (default Qwen/Qwen2.5-1.5B-Instruct)
    --adapter_dir <path>         (default ./adapter; loaded if directory exists)
    --max_new_tokens <int>       (default 512)
    --batch_size <int>           (default 4)
    --max_tables <int>           (default 20; BM25 keep-cap; 40 covers 99.4% of val gold)
    --mock                       (skip model load; emit empty predictions -- for wiring tests)
"""
import argparse
import json
import os
import sys
from typing import Dict, List

from prompt import build_messages, parse_model_output
from schema_utils import canonicalize_prediction, load_schema


def predict_mock(items: List[Dict], schemas_dir: str) -> List[Dict]:
    """No-model baseline: emits {} for every question. Use to verify CLI wiring."""
    return [{'question_id': it['question_id'], 'schema_links': {}} for it in items]


# ---------- two-stage system prompts (mirror format_two_stage_data.py) ----------
_STAGE_A_SYSTEM = (
    "You are a schema-linking assistant. Given a database's TABLE NAMES and a "
    "natural-language question, identify the tables the underlying SQL would "
    "reference. Reply with a single JSON object on one line and nothing else, "
    "of the form: {\"tables\": [\"Table1\", \"Table2\"]}. Use only table names "
    "that appear in the provided list -- do not invent names."
)
_STAGE_B_SYSTEM = (
    "You are a schema-linking assistant. Given a database table's COLUMN names "
    "and a natural-language question, identify the columns of that table the "
    "underlying SQL would reference. Reply with a single JSON object on one "
    "line and nothing else, of the form: {\"columns\": [\"Col1\", \"Col2\"]}. "
    "A table referenced with no specific columns (e.g. `SELECT COUNT(*) FROM t`) "
    "should produce {\"columns\": []}. Use only column names that appear in the "
    "provided list -- do not invent names."
)


def _apply_template(tokenizer, msgs):
    if 'enable_thinking' in tokenizer.apply_chat_template.__code__.co_varnames:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _generate(model, tokenizer, prompt_text: str, max_new_tokens: int) -> str:
    import torch
    enc = tokenizer([prompt_text], return_tensors='pt', padding=True, truncation=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    gen_only = out[:, enc['input_ids'].shape[1]:]
    return tokenizer.batch_decode(gen_only, skip_special_tokens=True)[0]


def predict_two_stage(items: List[Dict],
                      schemas_dir: str,
                      base_model: str,
                      stage_a_adapter: str,
                      stage_b_adapter: str,
                      max_new_tokens: int,
                      stage_a_max_tables: int = 40) -> List[Dict]:
    """Two-stage inference:
      Pass 1 (stage A adapter): for each question, predict the table set.
      Pass 2 (stage B adapter): for each (question, predicted_table), predict columns.
      Merge into the final {table: [cols]} contract.

    Loads adapters sequentially (one at a time) so only one PEFT graph is
    active per pass -- avoids any adapter-switching gotchas at inference time.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    # Local imports for the stage-A schema serializer (table names only).
    from format_two_stage_data import build_stageA_user, build_stageB_user

    print(f"[main] [2stage] Loading base model: {base_model}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map='auto' if torch.cuda.is_available() else None,
    )
    base.eval()

    schema_cache: Dict[str, Dict] = {}
    def get_schema(db_id: str) -> Dict:
        if db_id not in schema_cache:
            schema_cache[db_id] = load_schema(schemas_dir, db_id)
        return schema_cache[db_id]

    # ---------- Pass 1: stage A (table prediction) ----------
    print(f"[main] [2stage] Loading stage-A adapter from: {stage_a_adapter}", file=sys.stderr)
    model = PeftModel.from_pretrained(base, stage_a_adapter)
    model.eval()

    table_predictions: Dict[int, List[str]] = {}
    n = len(items)
    for i, it in enumerate(items):
        sch = get_schema(it['db_id'])
        user = build_stageA_user(it['db_id'], it['question'], sch, max_tables=stage_a_max_tables)
        prompt = _apply_template(tokenizer, [
            {'role': 'system', 'content': _STAGE_A_SYSTEM},
            {'role': 'user',   'content': user},
        ])
        text = _generate(model, tokenizer, prompt, max_new_tokens)
        raw = parse_model_output(text)
        tbls = raw.get('tables', []) if isinstance(raw, dict) else []
        if not isinstance(tbls, list):
            tbls = []
        # Restrict to real schema tables (case-insensitive); preserve schema casing
        lc_tables = {t.lower(): t for t in sch['tables']}
        tbls_valid = [lc_tables[str(t).lower()] for t in tbls if str(t).lower() in lc_tables]
        # De-dup while preserving order
        seen = set()
        tbls_valid = [t for t in tbls_valid if not (t in seen or seen.add(t))]
        table_predictions[it['question_id']] = tbls_valid
        if (i + 1) % 10 == 0 or i + 1 == n:
            print(f"[main] [2stage] stage-A {i+1}/{n} done", file=sys.stderr)

    # Unload stage-A adapter before loading stage-B (avoid lingering active adapter).
    # The simplest robust approach: drop the PeftModel reference and rebuild from base.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---------- Pass 2: stage B (column prediction per predicted table) ----------
    print(f"[main] [2stage] Loading stage-B adapter from: {stage_b_adapter}", file=sys.stderr)
    model = PeftModel.from_pretrained(base, stage_b_adapter)
    model.eval()

    preds: List[Dict] = []
    total_b = sum(len(v) for v in table_predictions.values())
    done_b = 0
    for it in items:
        sch = get_schema(it['db_id'])
        schema_links: Dict[str, List[str]] = {}
        for canon_t in table_predictions[it['question_id']]:
            cols_all = sch['columns'][canon_t]
            user = build_stageB_user(it['db_id'], it['question'], canon_t, cols_all)
            prompt = _apply_template(tokenizer, [
                {'role': 'system', 'content': _STAGE_B_SYSTEM},
                {'role': 'user',   'content': user},
            ])
            text = _generate(model, tokenizer, prompt, max_new_tokens)
            raw = parse_model_output(text)
            cols_pred = raw.get('columns', []) if isinstance(raw, dict) else []
            if not isinstance(cols_pred, list):
                cols_pred = []
            lc_cols = {c.lower(): c for c in cols_all}
            cols_valid = [lc_cols[str(c).lower()] for c in cols_pred if str(c).lower() in lc_cols]
            # de-dup
            seen = set()
            schema_links[canon_t] = [c for c in cols_valid if not (c in seen or seen.add(c))]
            done_b += 1
        preds.append({'question_id': it['question_id'], 'schema_links': schema_links})

    print(f"[main] [2stage] stage-B {done_b}/{total_b} table-column queries done", file=sys.stderr)
    return preds


def predict_with_model(items: List[Dict],
                       schemas_dir: str,
                       base_model: str,
                       adapter_dir: str,
                       max_new_tokens: int,
                       batch_size: int,
                       max_tables: int = 20,
                       prompt_style: str = 'compact') -> List[Dict]:
    """Real-model inference path. Lazy-imports torch/transformers so that
    --mock works in environments without these installed."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[main] Loading base model: {base_model}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # required for batched causal-LM generation

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map='auto' if torch.cuda.is_available() else None,
    )

    if adapter_dir and os.path.isdir(adapter_dir):
        from peft import PeftModel
        print(f"[main] Loading adapter from: {adapter_dir}", file=sys.stderr)
        model = PeftModel.from_pretrained(model, adapter_dir)
    else:
        print(f"[main] No adapter at {adapter_dir}; running base model zero-shot", file=sys.stderr)

    model.eval()

    # Cache schemas so we don't re-parse per question.
    schema_cache: Dict[str, Dict] = {}
    def get_schema(db_id: str) -> Dict:
        if db_id not in schema_cache:
            schema_cache[db_id] = load_schema(schemas_dir, db_id)
        return schema_cache[db_id]

    preds: List[Dict] = []
    n = len(items)
    for start in range(0, n, batch_size):
        batch = items[start:start + batch_size]
        prompts = []
        schemas = []
        for it in batch:
            sch = get_schema(it['db_id'])
            schemas.append(sch)
            msgs = build_messages(it['db_id'], it['question'], sch, max_tables=max_tables, style=prompt_style)
            prompt_text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,  # Qwen3: skip CoT preamble; ignored by other tokenizers
            ) if 'enable_thinking' in tokenizer.apply_chat_template.__code__.co_varnames else \
                tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt_text)

        enc = tokenizer(prompts, return_tensors='pt', padding=True, truncation=False).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_only = out[:, enc['input_ids'].shape[1]:]
        decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)

        for it, sch, text in zip(batch, schemas, decoded):
            raw = parse_model_output(text)
            links = canonicalize_prediction(raw, sch)
            preds.append({'question_id': it['question_id'], 'schema_links': links})
        print(f"[main] {min(start + batch_size, n)}/{n} done", file=sys.stderr)

    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input',  required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--schemas_dir', default='./schemas')
    ap.add_argument('--base_model', default='Qwen/Qwen2.5-1.5B-Instruct')
    ap.add_argument('--adapter_dir', default='./adapter')
    ap.add_argument('--max_new_tokens', type=int, default=512)
    ap.add_argument('--batch_size', type=int, default=1,
                    help='Default 1: bs>1 with long left-padded bf16 prompts SIGFPEs in '
                         'the attention kernel on this stack (transformers 4.57 + torch 2.10 + H100).')
    ap.add_argument('--max_tables', type=int, default=20,
                    help='BM25 keep-cap when filtering large schemas. 40 covers 99.4%% '
                         'of val-set gold tables vs 91.8%% at the default 20.')
    ap.add_argument('--prompt_style', default='compact',
                    choices=['compact', 'types', 'keys', 'types_keys'],
                    help='Schema serialization style. MUST match the style the loaded '
                         'adapter was trained with -- adapter/ and adapter_v2/ are both compact.')
    ap.add_argument('--mock', action='store_true',
                    help='Skip model load; emit empty predictions (wiring smoke test).')
    ap.add_argument('--two_stage', action='store_true',
                    help='Use the two-stage table->columns inference pipeline. '
                         'Requires --stage_a_adapter and --stage_b_adapter.')
    ap.add_argument('--stage_a_adapter', default='./adapter_stage_a',
                    help='PEFT adapter dir for stage A (table-set prediction).')
    ap.add_argument('--stage_b_adapter', default='./adapter_stage_b',
                    help='PEFT adapter dir for stage B (per-table column prediction).')
    args = ap.parse_args()

    with open(args.input) as f:
        items = json.load(f)
    print(f"[main] Loaded {len(items)} questions from {args.input}", file=sys.stderr)

    if args.mock:
        preds = predict_mock(items, args.schemas_dir)
    elif args.two_stage:
        preds = predict_two_stage(
            items,
            schemas_dir=args.schemas_dir,
            base_model=args.base_model,
            stage_a_adapter=args.stage_a_adapter,
            stage_b_adapter=args.stage_b_adapter,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        preds = predict_with_model(
            items,
            schemas_dir=args.schemas_dir,
            base_model=args.base_model,
            adapter_dir=args.adapter_dir,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            max_tables=args.max_tables,
            prompt_style=args.prompt_style,
        )

    with open(args.output, 'w') as f:
        json.dump(preds, f, indent=2)
    print(f"[main] Wrote {len(preds)} predictions to {args.output}", file=sys.stderr)


if __name__ == '__main__':
    main()
