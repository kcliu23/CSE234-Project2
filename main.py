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


def predict_with_model(items: List[Dict],
                       schemas_dir: str,
                       base_model: str,
                       adapter_dir: str,
                       max_new_tokens: int,
                       batch_size: int,
                       max_tables: int = 20) -> List[Dict]:
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
            msgs = build_messages(it['db_id'], it['question'], sch, max_tables=max_tables)
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
    ap.add_argument('--mock', action='store_true',
                    help='Skip model load; emit empty predictions (wiring smoke test).')
    args = ap.parse_args()

    with open(args.input) as f:
        items = json.load(f)
    print(f"[main] Loaded {len(items)} questions from {args.input}", file=sys.stderr)

    if args.mock:
        preds = predict_mock(items, args.schemas_dir)
    else:
        preds = predict_with_model(
            items,
            schemas_dir=args.schemas_dir,
            base_model=args.base_model,
            adapter_dir=args.adapter_dir,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            max_tables=args.max_tables,
        )

    with open(args.output, 'w') as f:
        json.dump(preds, f, indent=2)
    print(f"[main] Wrote {len(preds)} predictions to {args.output}", file=sys.stderr)


if __name__ == '__main__':
    main()
