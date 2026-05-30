"""
reproduce_champion.py -- Recreate the champion predictions file (val leaderboard
0.6986) by union-ensembling 3 LoRA adapters' predictions.

The champion is a 3-way union ensemble:
    - sweep13: Qwen2.5-Coder-1.5B-Instruct + LoRA r=32 q/k/v/o + MLP, lr=3e-4, 200 steps
    - sweep15: same as sweep13 + augmented training data + max_steps=400
    - sweep18: same as sweep13 but lr=2e-4
All three use the compact prompt style and were trained via train_rapidfire.py
with --num_chunks 1.

The ensemble logic: for each question, union all tables predicted by any of the
3 models; for each table, union all columns predicted across the models.

By default this script reads pre-computed per-model predictions from
predictions/preds_sweep{13,15,18}.json and writes the ensemble output. To
regenerate the per-model predictions from scratch (e.g. on a fresh machine),
run main.py separately against each adapter under adapter_ensemble/, then
re-run this script.

Run:
    # Just rebuild the ensemble from existing per-model preds:
    python reproduce_champion.py

    # Regenerate per-model preds first (slow, ~5 min each):
    python reproduce_champion.py --regenerate \
        --input data/validation_input.json
"""
import argparse
import json
import os
import subprocess
from typing import Dict, List

ADAPTER_DIRS = {
    'sweep13': 'adapter_ensemble/sweep13',  # Qwen-Coder-1.5B
    'sweep15': 'adapter_ensemble/sweep15',  # Qwen-Coder-1.5B
    'sweep22': 'adapter_ensemble/sweep22',  # Qwen3-1.7B (cross-family diversity)
}
# Per-adapter base model -- main.py auto-detects from adapter_config.json
# but reproduce_champion.py uses this when --regenerate calls main.py directly.
ADAPTER_BASE = {
    'sweep13': 'Qwen/Qwen2.5-Coder-1.5B-Instruct',
    'sweep15': 'Qwen/Qwen2.5-Coder-1.5B-Instruct',
    'sweep22': 'Qwen/Qwen3-1.7B',
}
BASE_MODEL = 'Qwen/Qwen2.5-Coder-1.5B-Instruct'  # fallback only
DEFAULT_PREDS_DIR = 'predictions'


def regenerate_per_model_preds(input_path: str, preds_dir: str) -> None:
    """Run main.py once per adapter to (re)create predictions/preds_sweep{N}.json."""
    os.makedirs(preds_dir, exist_ok=True)
    for name, adir in ADAPTER_DIRS.items():
        out = os.path.join(preds_dir, f'preds_{name}.json')
        base = ADAPTER_BASE.get(name, BASE_MODEL)
        print(f"[reproduce] regenerating {out} via main.py + {adir} (base={base})")
        cmd = [
            'python', 'main.py',
            '--single',
            '--input', input_path,
            '--output', out,
            '--adapter_dir', adir,
            '--base_model', base,
            '--batch_size', '1',
        ]
        subprocess.run(cmd, check=True)


def load_preds(path: str) -> Dict[int, Dict[str, List[str]]]:
    return {p['question_id']: p['schema_links'] for p in json.load(open(path))}


def union_ensemble(*preds_list: Dict[int, Dict[str, List[str]]]) -> Dict[int, Dict[str, List[str]]]:
    """Per question: union of all predicted tables across models; for each
    table, union of all predicted columns. Casing is preserved from the first
    model that predicted that identifier (case-insensitive dedup)."""
    all_qids = set()
    for p in preds_list:
        all_qids |= set(p)
    out = {}
    for qid in all_qids:
        canon_table: Dict[str, str] = {}        # tlc -> canonical T
        cols: Dict[str, Dict[str, str]] = {}    # tlc -> {clc -> canonical c}
        for p in preds_list:
            for t, c_list in p.get(qid, {}).items():
                tlc = t.lower()
                canon_table.setdefault(tlc, t)
                cols.setdefault(tlc, {})
                for c in (c_list or []):
                    cols[tlc].setdefault(c.lower(), c)
        out[qid] = {canon_table[tlc]: sorted(cols[tlc].values()) for tlc in canon_table}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--regenerate', action='store_true',
                    help='Run main.py against each adapter first to refresh per-model preds.')
    ap.add_argument('--input', default='data/validation_input.json',
                    help='Used only with --regenerate.')
    ap.add_argument('--preds_dir', default=DEFAULT_PREDS_DIR)
    ap.add_argument('--output', default='predictions/preds_CHAMPION.json')
    args = ap.parse_args()

    if args.regenerate:
        regenerate_per_model_preds(args.input, args.preds_dir)

    per_model = []
    for name in ADAPTER_DIRS:
        path = os.path.join(args.preds_dir, f'preds_{name}.json')
        if not os.path.exists(path):
            raise SystemExit(f"Missing {path}. Run with --regenerate to recreate it.")
        per_model.append(load_preds(path))

    ensemble = union_ensemble(*per_model)
    out_list = [{'question_id': qid, 'schema_links': sl} for qid, sl in sorted(ensemble.items())]
    with open(args.output, 'w') as f:
        json.dump(out_list, f, indent=2)
    print(f"[reproduce] wrote {len(out_list)} ensemble predictions to {args.output}")


if __name__ == '__main__':
    main()
