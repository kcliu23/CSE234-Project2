"""
RapidFire AI SFT training script for the schema-linking task.

Trains a small (<=2B) instruction-tuned model with LoRA on the prompts produced
by `prompt.build_messages`, supervised on the JSON target produced by
`prompt.target_string`. Uses RapidFire AI's RFGridSearch to compare 8 configs
concurrently with interactive-control hot-swapping across configs.

To run:
    python train_rapidfire.py --train data/train.jsonl --val data/validation.jsonl

You must run this inside a RapidFire AI environment (the package handles the
multi-config orchestration, chunked execution, and metrics logging).

Config sweep (8 = the spec's required minimum; clone-modify the winners via
the RapidFire IC UI to add more without re-launching):

    base_model: Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct
    lora:       (r=8, q+v) and (r=16, q+k+v+o)
    lr:         1e-4 and 2e-4

Knobs intentionally fixed for the first sweep:
    - epochs = 3 (~900 training steps at bs=1 grad_acc=4 over 301 examples)
    - lr_scheduler = linear with warmup_ratio=0.05
    - max_seq_length = 6144 (covers >99% of examples; p50 ~1350 tokens)
    - bf16 = True
    - per_device_train_batch_size = 1, gradient_accumulation_steps = 4
      (the long-schema SBO examples push memory hard at ~6k tokens)

If a smaller seq cap is needed for memory headroom, set --max_seq_length 4096
on the CLI -- truncates ~8% of examples (mostly SBODemoUS-Finance/Inventory).
"""
import argparse
import json
from typing import Any, Dict, List

# RapidFire AI imports are deferred to main() so this file can be imported and
# linted in environments where the package isn't installed.


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


def formatting_function(row: Dict[str, Any]) -> Dict[str, Any]:
    """Per-row transform consumed by RapidFire/TRL SFTTrainer.

    Splits the precomputed [system, user, assistant] message list into
    `prompt` (system + user, loss-masked) and `completion` (assistant, the
    only thing the model is supervised on).
    """
    msgs = row['messages']
    return {
        'prompt':     msgs[:-1],   # system + user
        'completion': [msgs[-1]],  # assistant turn
    }


def create_model(model_config):
    """RapidFire calls this once per config to materialize (model, tokenizer)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = model_config['model_name']
    model_kwargs = model_config['model_kwargs']

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Padding side is generation-only; SFTTrainer handles its own packing.
    return model, tokenizer


def build_config_group(max_seq_length: int, num_train_epochs: int):
    """Sweep 9 (tight): designed to fit under the DSMLP kill window.

    Background: sweeps 5/6/7/8 all got SIGKILL'd by DSMLP process cleanup
    around the 4-6h mark, before experiment.end() could flush rapidfire's
    in-SHM checkpoints to disk -- so every weight was lost. Sweep 8 reached
    264/300 steps on all 8 configs (~5h48m wall) before dying.

    Sweep 9 dials wall-time WAY down so end-to-end completes in ~1h, AND
    sweep 10 fixes the rapidfireai "no checkpoint on disk" issue by setting
    num_chunks=1 at launch:
        - 1 base model (1.5B, the proven leader on train metrics from sweep 8
          and the only one we have a real val number for: 0.6078 via the
          single-config fallback)
        - 2 LRs (2e-4 known-good on val; 3e-4 strong on sweep 8 train metrics
          but untested on val)
        - max_steps=200 (the 0.6078 adapter was already past the loss-elbow
          at step 200; the extra 100 steps gave diminishing returns)
        - num_chunks=1 at launch (see Sweep 9 post-mortem below)

    Sweep 9 post-mortem (rapidfireai disk-flush gotcha): rapidfireai only
    writes `final_checkpoint/<adapter>.safetensors` when
    `is_run_finished == (chunk_id == num_chunks-1) AND (steps >= total_steps)`
    -- see fit/backend/worker.py around L440. If max_steps is reached BEFORE
    the run gets to the last chunk (which it almost always is with
    num_chunks>1, since the scheduler interleaves chunks across configs),
    the disk save never fires and the SHM-only adapter is lost on process
    exit. Workaround: set num_chunks=1 so every run ends on "the last chunk"
    by construction.

    2 configs = 1 base model * 1 LoRA * 2 LRs.
    """
    from rapidfireai.automl import (
        List as RFList, RFLoraConfig, RFModelConfig, RFSFTConfig,
    )

    # Single LoRA config -- the proven sweep 4/8 winner.
    peft_config = RFLoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        bias='none', task_type='CAUSAL_LM',
    )

    # Hard cap at 200 steps regardless of --num_train_epochs.  max_steps
    # ALWAYS wins over num_train_epochs in TRL, but we still set epochs to a
    # number large enough that the cap kicks in (the chunked iterator needs a
    # finite epoch budget to drive its scheduler).
    max_steps = 200

    sft = RFSFTConfig(
        learning_rate=RFList([2e-4, 3e-4]),
        lr_scheduler_type='linear',
        warmup_ratio=0.05,
        max_steps=max_steps,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        max_length=max_seq_length,
        logging_steps=10,
        eval_strategy='steps',
        eval_steps=50,
        save_strategy='steps',
        save_steps=50,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        report_to='none',
    )

    common_model_kwargs = {
        'torch_dtype': 'auto',
        'device_map': 'auto',
        'use_cache': False,  # incompatible with gradient checkpointing
    }

    configs = RFList([
        RFModelConfig(
            model_name='Qwen/Qwen2.5-1.5B-Instruct',
            peft_config=peft_config,
            training_args=sft,
            model_type='causal_lm',
            model_kwargs=common_model_kwargs,
            formatting_func=formatting_function,
        ),
    ])

    from rapidfireai.automl import RFGridSearch
    return RFGridSearch(configs=configs, trainer_type='SFT')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True, help='train.jsonl from format_training_data.py')
    ap.add_argument('--val',   required=True, help='validation.jsonl from format_training_data.py')
    ap.add_argument('--experiment_name', default='schema-linking-sft-sweep1')
    ap.add_argument('--max_seq_length', type=int, default=4096,
                    help='Sequence cap; 4096 covers >99% of filtered-schema examples '
                         '(p99 ~3700 tokens after BM25 table-level filter).')
    ap.add_argument('--num_train_epochs', type=int, default=4)
    ap.add_argument('--num_chunks', type=int, default=8,
                    help='RapidFire interactive-swap granularity. More chunks = '
                         'finer config hot-swapping but more overhead.')
    args = ap.parse_args()

    from rapidfireai import Experiment

    train_ds = load_jsonl_as_hf_dataset(args.train)
    val_ds   = load_jsonl_as_hf_dataset(args.val)
    print(f"[train] train={len(train_ds)}  val={len(val_ds)}")

    experiment = Experiment(experiment_name=args.experiment_name, mode='fit')
    config_group = build_config_group(
        max_seq_length=args.max_seq_length,
        num_train_epochs=args.num_train_epochs,
    )

    experiment.run_fit(
        config_group, create_model,
        train_ds, val_ds,
        num_chunks=args.num_chunks, seed=42,
    )
    experiment.end()


if __name__ == '__main__':
    main()
