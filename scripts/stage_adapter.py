"""
stage_adapter.py -- Take a fp32 LoRA adapter from RapidFire's experiment store,
convert to fp16 (so it fits under GitHub's 100 MB limit), and stage it into
adapter_ensemble/sweepN/. Run from the project root.

Usage:
    python scripts/stage_adapter.py 27
    python scripts/stage_adapter.py 27 --src /custom/path  # override default RF path
"""
import argparse
import json
import os
import shutil
import sys

import safetensors.torch as st
import torch

DEFAULT_SRC_TEMPLATE = ('/home/kal115/rapidfireai/rapidfire_experiments/'
                        'schema-linking-sft-sweep{n}/runs/1/checkpoints/final_checkpoint')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sweep', help='Sweep number (e.g. 27)')
    ap.add_argument('--src', default=None,
                    help='Source checkpoint directory (default: RapidFire experiment store path)')
    ap.add_argument('--dst', default=None,
                    help='Destination directory (default: adapter_ensemble/sweep<N>/)')
    ap.add_argument('--keep-fp32', action='store_true',
                    help='Skip fp16 conversion (use only if size is already under 100 MB)')
    args = ap.parse_args()

    src = args.src or DEFAULT_SRC_TEMPLATE.format(n=args.sweep)
    dst = args.dst or f'adapter_ensemble/sweep{args.sweep}'

    if not os.path.isdir(src):
        sys.exit(f"Source not found: {src}")
    os.makedirs(dst, exist_ok=True)

    safe_src = os.path.join(src, 'adapter_model.safetensors')
    safe_dst = os.path.join(dst, 'adapter_model.safetensors')

    sd = st.load_file(safe_src)
    print(f'[stage] loaded {len(sd)} tensors from {safe_src}')
    pre_mb = sum(v.numel() * v.element_size() for v in sd.values()) / 1e6
    print(f'[stage] source dtype={list(sd.values())[0].dtype}, size {pre_mb:.1f} MB')

    if not args.keep_fp32:
        sd = {k: v.to(torch.float16) for k, v in sd.items()}
        post_mb = sum(v.numel() * v.element_size() for v in sd.values()) / 1e6
        print(f'[stage] converted to fp16, size {post_mb:.1f} MB')

    st.save_file(sd, safe_dst)

    cfg_src = os.path.join(src, 'adapter_config.json')
    cfg_dst = os.path.join(dst, 'adapter_config.json')
    shutil.copy(cfg_src, cfg_dst)
    print(f'[stage] copied {cfg_src} -> {cfg_dst}')

    with open(cfg_dst) as f:
        cfg = json.load(f)
    print(f'[stage] base_model={cfg.get("base_model_name_or_path")}, '
          f'r={cfg.get("r")}, alpha={cfg.get("lora_alpha")}, '
          f'target_modules={cfg.get("target_modules")}')
    print(f'[stage] ready: {dst}')


if __name__ == '__main__':
    main()
