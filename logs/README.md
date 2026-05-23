# RapidFire AI experiment logs

This folder contains the actual RapidFire AI training logs and metrics for the
CSE/DSC 234 Project 2 schema-linking SFT pipeline. Required by the rubric
(Section 5.5 of the project statement).

## Layout

```
logs/
├── README.md                  this file
├── sweep<N>/
│   ├── rapidfire.log          controller + worker debug log from rapidfireai
│   ├── training.log           TRL/SFTTrainer stdout (per-step loss, eval, etc.)
│   └── metrics.json           per-run config + final mlflow metrics, extracted
│                              from ~/rapidfireai/db/rapidfire_mlflow.db
```

The `metrics.json` files are the canonical source for "which config produced
which numbers." They are extracted from the same mlflow store that the
rapidfireai dashboard reads, so they match what was logged live.

## Sweep timeline and outcomes

Sweeps are numbered chronologically. Sweeps 1 and 3 are not present here:
they were early exploratory runs that predated proper mlflow registration.
Sweeps 2 onward are fully instrumented.

| sweep | # configs | target/reached steps | status | best run train-acc | key knobs varied |
|-------|-----------|----------------------|--------|--------------------|------------------|
| 2     | 8         | n/a / 228            | completed | run 8: 0.916  | base model (0.5B/1.5B) × lr × LoRA r × target_modules |
| 4     | 8         | 225 / 225            | completed | run 8: 0.899  | same axes as sweep 2 |
| 5     | 16        | 300 / 9              | died very early (DSMLP) | run 6: 0.792 | added r=32 + FFN LoRA (q/k/v/o + gate/up/down) |
| 6     | 8         | 300 / ~100           | killed mid-run (DSMLP ~4-6h) | run 6: 0.902 | r=32 + FFN, lr={3e-4, 5e-4} |
| 7     | 8         | 300 / ~70            | killed mid-run (DSMLP)     | run 7: 0.847 | same as sweep 6 |
| 8     | 8         | 300 / ~260           | killed mid-run (DSMLP)     | run 6: 0.933 | reverted to r=16 q/k/v/o; swept lr ∈ {1e-4, 2e-4, 3e-4, 5e-4} × {0.5B, 1.5B} |
| 9     | 2         | 200 / 200            | completed but **weights lost** (rapidfireai SHM/chunk bug, see below) | run 2: 0.970 | 1.5B × lr {2e-4, 3e-4} |
| 10    | 2         | 200 / in progress    | running, `num_chunks=1` fix to force disk-flush | TBD | same axes as sweep 9 |

Per-run granularity is in each sweep's `metrics.json`.

## RapidFire AI "lost weights" finding (sweep 9 post-mortem)

Sweeps 5/6/7/8 lost weights to DSMLP SIGKILL at the 4-6h mark before the
final disk-save. Sweep 9 ran the same configs in ~1h, finished cleanly, and
**still lost weights**. Inspection of `rapidfireai/fit/backend/worker.py`
around L440 showed `save_checkpoint_to_disk(..., last=True)` is gated on
`is_run_finished == (chunk_id == num_chunks-1) AND (steps >= total_steps)`.
With multi-chunk scheduling, runs hit their step budget on chunk N where
N < num_chunks-1, so the disk-save branch is never entered — only the
shared-memory checkpoint is updated, and SHM dies with the worker process.

**Workaround:** sweep 10 launches with `num_chunks=1`, so every run
necessarily ends on "the last chunk" by construction and the disk-save
branch fires. This is the configuration that should produce a usable
final_checkpoint/adapter_model.safetensors.

## Non-rapidfireai fallback (train_single.py)

When sweep 8 lost weights to DSMLP, a single-config fallback was built at
`../train_single.py` using plain `trl.SFTTrainer` (writes to disk natively).
It produced `../adapter_v2/` in ~14 minutes, scoring **leaderboard 0.6078**
on val with Qwen2.5-1.5B + LoRA r=16 q/k/v/o + lr=2e-4 + 300 steps. This is
not the primary submission path — rapidfireai sweep 10 is — but it is the
current best on-disk artifact.

## Provenance

Logs were copied verbatim from `~/rapidfireai/logs/schema-linking-sft-sweep*/`.
`metrics.json` files were extracted from `~/rapidfireai/db/rapidfire_mlflow.db`
(experiments table → runs table → params/metrics tables). The extractor lives
inline in shell history (not a committed script); the data here is the
canonical record.
