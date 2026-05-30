"""
Single-config HF Trainer fallback for the schema-linking SFT task.

Bypasses RapidFire AI — runs ONE training config end-to-end with TRL's
SFTTrainer, writing PEFT adapter checkpoints to DISK every `--save_steps`
steps. This is the safe alternative to train_rapidfire.py for environments
where the process may be SIGKILL'd mid-run (DSMLP cleanup at the ~4-6h mark
ate sweeps 5/6/7/8). Even a late kill leaves usable checkpoints behind.

Defaults are the winning config from sweep8's training metrics:
    Qwen2.5-1.5B-Instruct + LoRA r=16 q/k/v/o + lr=2e-4 + linear warmup.

Launch (under tmux so an SSH disconnect doesn't kill it):
    tmux new -s sft
    conda activate cse234
    python train_single.py \
        --train data/train.jsonl --val data/validation.jsonl \
        --output_dir ./adapter_v2
    # detach: Ctrl-b d ; reattach: tmux attach -t sft

After it finishes (or gets killed late), pick the best disk checkpoint and
point main.py at it via --adapter_dir.
"""
import argparse
import json
from typing import Any, Dict, List


def load_jsonl_as_hf_dataset(path: str):
    from datasets import Dataset
    rows: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return Dataset.from_list(rows)


def to_prompt_completion(row: Dict[str, Any]) -> Dict[str, Any]:
    """Split [system, user, assistant] into prompt + completion so that
    SFTTrainer applies loss only over the assistant turn."""
    msgs = row['messages']
    return {
        'prompt':     msgs[:-1],
        'completion': [msgs[-1]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True)
    ap.add_argument('--val',   required=True)
    ap.add_argument('--output_dir', default='./adapter_v2')
    ap.add_argument('--base_model', default='Qwen/Qwen2.5-1.5B-Instruct')
    ap.add_argument('--learning_rate', type=float, default=2e-4)
    ap.add_argument('--max_steps', type=int, default=300)
    ap.add_argument('--max_length', type=int, default=4096)
    ap.add_argument('--save_steps', type=int, default=50)
    ap.add_argument('--eval_steps', type=int, default=50)
    ap.add_argument('--lora_r', type=int, default=16)
    ap.add_argument('--lora_targets', default='q_proj,k_proj,v_proj,o_proj',
                    help='Comma-separated module name suffixes for LoRA.')
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    train_ds = load_jsonl_as_hf_dataset(args.train).map(to_prompt_completion, remove_columns=['messages'])
    val_ds   = load_jsonl_as_hf_dataset(args.val).map(to_prompt_completion,   remove_columns=['messages'])
    print(f"[train] train={len(train_ds)}  val={len(val_ds)}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype='auto',
        device_map='auto',
        use_cache=False,  # incompatible with gradient checkpointing
    )

    peft_config = LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
        target_modules=[t.strip() for t in args.lora_targets.split(',') if t.strip()],
        bias='none', task_type='CAUSAL_LM',
    )

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        lr_scheduler_type='linear',
        warmup_ratio=0.05,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        max_length=args.max_length,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy='steps',
        eval_steps=args.eval_steps,
        save_strategy='steps',
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        report_to='none',
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"[done] best adapter saved to {args.output_dir}")


if __name__ == '__main__':
    main()
