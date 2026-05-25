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
    """Sweep 20: cross-family base model for ensemble diversity.

    The 3-way ensemble (13+15+18, all Qwen-Coder-1.5B + r=32+MLP) is at
    0.6986. Sweep 19 (same family + cosine LR) was too correlated and
    hurt the ensemble. To break the correlation, sweep 20 swaps the
    base model to a different family entirely:
        - base: HuggingFaceTB/SmolLM2-1.7B-Instruct
          (Meta-like Llama-architecture, but pretrained on a different
          corpus -- so tokenizer, embedding space, and pretraining biases
          all differ from Qwen-Coder. Should produce different error modes.)
        - rubric suggested it explicitly; <=2B cap satisfied (1.7B).

    Identical to sweep 13 otherwise (compact prompt, r=32 + q/k/v/o + MLP,
    lr=3e-4, 200 steps, num_chunks=1). Llama-arch SmolLM2 has the same
    target_module names so the LoRA config carries over cleanly.

    If sweep 20 lands ~0.62+ AND its errors differ from sweep 13/15/18,
    ensembling it in should push past 0.70.

    Sweep 19: cosine-LR-schedule diversity-adapter for the ensemble.

    Champion is now a 3-way ensemble (sweep 13 + 15 + 18) at 0.6986 -- but the
    remaining val failures are CORRELATED (all 3 ensemble members miss the
    same hard examples).  Sweep 19 trains a 4th adapter with a different
    optimization trajectory (cosine LR schedule instead of linear) so its
    errors should be orthogonal to the existing 3.

    Identical to sweep 13 (Coder + r=32+MLP, 301 train, 200 steps) EXCEPT:
        - lr_scheduler_type='cosine'  (was 'linear')
        - lr=3e-4 (same as sweep 13, NOT sweep 18)

    Sweep 18: learning-rate ablation on the sweep 13 recipe.

    All knob-direction conclusions so far (Coder-1.5B + r=32 + MLP, compact,
    301 train, num_chunks=1):
        - sweep 13 lr=3e-4, steps=200 -> 0.6648 (champion)
        - sweep 16 lr=3e-4, steps=400 -> 0.6268 (overfit -- more steps hurt)
        - sweep 14/15 augmentation -> neutral
        - two-stage architecture (17a/b) -> -0.052 (worse)

    Sweep 18 tries lr=2e-4 to test whether 3e-4 was too hot for r=32 + MLP.
    Larger LoRA modules can require a smaller LR to converge cleanly --
    r=32+MLP has ~3.5x more trainable params than the r=16 q/k/v/o config
    that the lr=3e-4 sweep10 winner came from.  Identical to sweep 13
    except lr is dropped from 3e-4 -> 2e-4.

    Sweep 16: step-count ablation on the sweep 13 recipe.

    Series so far (all Coder-1.5B unless noted, lr=3e-4, num_chunks=1, compact):
        sweep 10  vanilla 1.5B  r=16 q/k/v/o   train=301  steps=200  -> 0.6229
        sweep 11  vanilla 1.5B  r=16 q/k/v/o   types_keys steps=200  -> 0.5946 (worse)
        sweep 12  Coder         r=16 q/k/v/o   train=301  steps=200  -> 0.6258
        sweep 13  Coder         r=32 +MLP      train=301  steps=200  -> 0.6648 (champion)
        sweep 14  Coder         r=32 +MLP      train=517  steps=200  -> 0.6379 (-0.027)
        sweep 15  Coder         r=32 +MLP      train=517  steps=400  -> 0.6600 (-0.005 vs 13)

    Sweep 15 recovered most of sweep 14's regression with 2x step budget,
    but didn't beat sweep 13.  Open question: was the issue (a) augmented
    data quality, or (b) just the step count?  Sweep 16 isolates by going
    BACK to the original 301-example train data and bumping steps to 400:
        - train data: data/train.jsonl  (original 301, pre-augmentation)
        - max_steps=400 (was 200)
        - Everything else identical to sweep 13

    If sweep 16 > sweep 13 -> 200 steps was undertrained even on 301 examples;
                              augmentation's job was just to bring more data
                              that 400 steps could exploit, and sweep 15's
                              ~tie with sweep 13 means augmentation was
                              roughly neutral overall.
    If sweep 16 == sweep 13 -> sweep 13's 200 steps was already converged;
                               the augmented-data distribution shift IS
                               what hurt sweep 14/15.
    If sweep 16 < sweep 13 -> 200 steps was OPTIMAL; we're now overfitting
                              on 301 examples at 400 steps.

    Wall time: ~2h (400 steps on 301 examples = ~5.3 epochs).

    Disk-flush gotcha (sweep 9 post-mortem): always launch with --num_chunks 1.

    1 config = 1 base model * 1 LoRA * 1 LR.
    """
    from rapidfireai.automl import (
        List as RFList, RFLoraConfig, RFModelConfig, RFSFTConfig,
    )

    # Sweep 13: r=32 + MLP modules on top of the sweep 12 winner config.
    peft_config = RFLoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj'],
        bias='none', task_type='CAUSAL_LM',
    )

    # Hard cap at 200 steps regardless of --num_train_epochs.  max_steps
    # ALWAYS wins over num_train_epochs in TRL, but we still set epochs to a
    # number large enough that the cap kicks in (the chunked iterator needs a
    # finite epoch budget to drive its scheduler).
    max_steps = 200

    sft = RFSFTConfig(
        learning_rate=3e-4,
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
            model_name='HuggingFaceTB/SmolLM2-1.7B-Instruct',
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
