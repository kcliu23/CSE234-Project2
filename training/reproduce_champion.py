"""
reproduce_champion.py -- Recreate the champion predictions file from the
per-(adapter, retriever) predictions under predictions/.

The current champion is a 4-way ensemble with majority-2 aggregation:
    - sweep13 (Qwen2.5-Coder-1.5B-Instruct) + embed retrieval
    - sweep15 (Qwen2.5-Coder-1.5B-Instruct) + bm25  retrieval
    - sweep22 (Qwen3-1.7B)                 + embed retrieval
    - sweep13 (Qwen2.5-Coder-1.5B-Instruct) + hybrid retrieval

The 4th entry reuses sweep13's adapter weights with a different retriever at
inference time -- pure post-hoc diversity, no extra training.

Aggregation: an identifier (table or table.column) is kept iff it appears in
>=2 of the 4 per-(adapter, retriever) predictions. This matches main.py's
default `--aggregation maj2`. Validation leaderboard: 0.7230.

By default the script reads pre-computed per-(adapter, retriever) files from
predictions/ and writes the ensemble output. Run from the project root:
    python training/reproduce_champion.py
"""
import argparse
import json
import os
from typing import Dict, List

INGREDIENTS = [
    # (label, per-model predictions file). Each file is produced by running
    # main.py --single against the corresponding adapter with the named retriever.
    ('sweep13_embed',  'predictions/preds_sweep13_embed.json'),
    ('sweep15_bm25',   'predictions/preds_sweep15.json'),
    ('sweep22_embed',  'predictions/preds_sweep22.json'),
    ('sweep13_hybrid', 'predictions/preds_sweep13_hybrid.json'),
]
DEFAULT_PREDS_DIR = 'predictions'


def load_preds(path: str) -> Dict[int, Dict[str, List[str]]]:
    return {p['question_id']: p['schema_links'] for p in json.load(open(path))}


def maj2_ensemble(preds_list: List[Dict[int, Dict[str, List[str]]]]) -> Dict[int, Dict[str, List[str]]]:
    """Per question: keep tables and (table, column) pairs that appear in >=2
    of the input predictions. Casing is taken from the first model that
    predicted that identifier (case-insensitive dedup). Matches main.py's
    --aggregation maj2.
    """
    all_qids = set()
    for p in preds_list:
        all_qids |= set(p)
    out = {}
    for qid in all_qids:
        canon_table: Dict[str, str] = {}
        cols: Dict[str, Dict[str, str]] = {}
        tvotes: Dict[str, int] = {}
        cvotes: Dict[tuple, int] = {}
        for p in preds_list:
            seen_t, seen_c = set(), set()
            for t, c_list in p.get(qid, {}).items():
                tlc = t.lower()
                canon_table.setdefault(tlc, t)
                cols.setdefault(tlc, {})
                if tlc not in seen_t:
                    tvotes[tlc] = tvotes.get(tlc, 0) + 1
                    seen_t.add(tlc)
                for c in (c_list or []):
                    clc = c.lower()
                    cols[tlc].setdefault(clc, c)
                    tc = (tlc, clc)
                    if tc not in seen_c:
                        cvotes[tc] = cvotes.get(tc, 0) + 1
                        seen_c.add(tc)
        keep_t = {t for t, v in tvotes.items() if v >= 2}
        keep_c = {tc for tc, v in cvotes.items() if v >= 2}
        keep_t |= {t for (t, _) in keep_c}
        sl = {canon_table[tlc]: [] for tlc in keep_t if tlc in canon_table}
        for (tlc, clc) in keep_c:
            if tlc in keep_t and tlc in cols and clc in cols[tlc]:
                sl[canon_table[tlc]].append(cols[tlc][clc])
        for t in sl:
            sl[t] = sorted(sl[t])
        out[qid] = sl
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default='predictions/preds_CHAMPION.json')
    args = ap.parse_args()

    per_model = []
    for label, path in INGREDIENTS:
        if not os.path.exists(path):
            raise SystemExit(f"Missing {path}. Generate it with: python main.py --single "
                              f"--adapter_dir <dir> --retrieval <bm25|embed|hybrid> --output {path}")
        per_model.append(load_preds(path))
        print(f"[reproduce] loaded {label} -> {path}")

    ensemble = maj2_ensemble(per_model)
    out_list = [{'question_id': qid, 'schema_links': sl} for qid, sl in sorted(ensemble.items())]
    with open(args.output, 'w') as f:
        json.dump(out_list, f, indent=2)
    print(f"[reproduce] wrote {len(out_list)} ensemble predictions to {args.output}")


if __name__ == '__main__':
    main()
