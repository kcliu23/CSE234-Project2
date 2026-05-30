"""
probe_ensembles.py -- Combinatorial search over (per-sweep, per-retriever) prediction
ingredients + aggregation strategy, scored against validation gold.

Outputs a ranked table of (combo, aggregation, leaderboard) and writes the
top-scoring predictions to predictions/preds_PROBE_BEST.json. If it beats
0.7107 (current champion), the user can promote it to preds_FINAL_SUBMISSION.json.

No new training or model loads — pure post-hoc rescoring.
"""
import json
import os
import re
import subprocess
import tempfile
from itertools import combinations

PREDS_DIR = 'predictions'
GOLD = 'data/validation_gold_schema_links.json'
QUESTIONS = 'data/validation_input.json'
SCHEMAS = 'schemas'

INGREDIENTS = {
    'sweep13_embed':  f'{PREDS_DIR}/preds_sweep13_embed.json',
    'sweep13_bm25':   f'{PREDS_DIR}/preds_sweep13.json',
    'sweep13_hybrid': f'{PREDS_DIR}/preds_sweep13_hybrid.json',
    'sweep15_embed':  f'{PREDS_DIR}/preds_sweep15_embed.json',
    'sweep15_bm25':   f'{PREDS_DIR}/preds_sweep15.json',
    'sweep15_hybrid': f'{PREDS_DIR}/preds_sweep15_hybrid.json',
    'sweep22_embed':  f'{PREDS_DIR}/preds_sweep22.json',
    'sweep22_bm25':   f'{PREDS_DIR}/preds_sweep22_bm25.json',
    'sweep22_hybrid': f'{PREDS_DIR}/preds_sweep22_hybrid.json',
    'sweep26_embed':  f'{PREDS_DIR}/preds_sweep26_embed.json',
    'sweep26_bm25':   f'{PREDS_DIR}/preds_sweep26_bm25.json',
    'sweep26_hybrid': f'{PREDS_DIR}/preds_sweep26_hybrid.json',
}
INGREDIENTS = {k: v for k, v in INGREDIENTS.items() if os.path.exists(v)}

CHAMPION_KEYS = ('sweep13_embed', 'sweep15_bm25', 'sweep22_embed')

COMBOS = []
COMBOS.append(('champion_3way', list(CHAMPION_KEYS)))
for extra in INGREDIENTS:
    if extra in CHAMPION_KEYS:
        continue
    COMBOS.append((f'4way_+{extra}', list(CHAMPION_KEYS) + [extra]))
core = list(CHAMPION_KEYS)
for k1, k2 in combinations([k for k in INGREDIENTS if k not in CHAMPION_KEYS], 2):
    COMBOS.append((f'5way_+{k1}_+{k2}', core + [k1, k2]))


def load_pred_dict(path):
    return {p['question_id']: p['schema_links'] for p in json.load(open(path))}


def aggregate(pred_dicts, mode):
    """Return {qid: {table: [cols]}} given a list of pred dicts and an aggregation mode.

    mode='union'   : keep any (table, col) appearing in any model
    mode='maj2'    : keep (table, col) appearing in >=2 models; tables get the union
                     of cols passing the threshold; tables themselves need >=2 votes
    mode='maj2_or_lead': same as maj2 but always include anything from the first
                     model in the list (treated as the strongest/lead)
    """
    all_qids = set()
    for d in pred_dicts:
        all_qids |= set(d)
    out = {}
    for qid in all_qids:
        per_model = [d.get(qid, {}) for d in pred_dicts]
        table_votes = {}
        tablecol_votes = {}
        casing_t = {}
        casing_c = {}
        for sl in per_model:
            seen_t = set()
            for tname, cols in sl.items():
                lct = tname.lower()
                if lct not in seen_t:
                    table_votes[lct] = table_votes.get(lct, 0) + 1
                    casing_t.setdefault(lct, tname)
                    seen_t.add(lct)
                if isinstance(cols, list):
                    seen_c = set()
                    for c in cols:
                        lcc = (lct, c.lower())
                        if lcc not in seen_c:
                            tablecol_votes[lcc] = tablecol_votes.get(lcc, 0) + 1
                            casing_c.setdefault(lcc, c)
                            seen_c.add(lcc)
        lead = per_model[0] if per_model else {}
        lead_tables = {t.lower() for t in lead}
        lead_pairs = {(t.lower(), c.lower())
                      for t, cs in lead.items() if isinstance(cs, list) for c in cs}

        if mode == 'union':
            keep_t = set(table_votes)
            keep_c = set(tablecol_votes)
        elif mode == 'maj2':
            keep_t = {t for t, v in table_votes.items() if v >= 2}
            keep_c = {tc for tc, v in tablecol_votes.items() if v >= 2}
            keep_t |= {t for (t, _) in keep_c}  # ensure table appears if its col survived
        elif mode == 'maj2_or_lead':
            keep_t = {t for t, v in table_votes.items() if v >= 2} | lead_tables
            keep_c = {tc for tc, v in tablecol_votes.items() if v >= 2} | lead_pairs
            keep_t |= {t for (t, _) in keep_c}
        else:
            raise ValueError(mode)

        sl_out = {}
        for lct in keep_t:
            sl_out[casing_t[lct]] = []
        for (lct, lcc) in keep_c:
            if lct in keep_t:
                sl_out[casing_t[lct]].append(casing_c[(lct, lcc)])
        out[qid] = sl_out
    return out


def run_eval(pred_path):
    res = subprocess.run(
        ['python', 'eval.py',
         '--predictions', pred_path,
         '--gold', GOLD,
         '--questions_input', QUESTIONS,
         '--schemas_dir', SCHEMAS],
        capture_output=True, text=True)
    m = re.search(r'Leaderboard Score\s*:\s*([\d.]+)', res.stdout)
    if not m:
        return None, res.stdout + res.stderr
    return float(m.group(1)), res.stdout


def main():
    print(f"[probe] {len(INGREDIENTS)} ingredients available, {len(COMBOS)} combos x 3 aggregations = {len(COMBOS)*3} runs")
    rows = []
    best = (None, None, -1.0, None)
    for combo_name, keys in COMBOS:
        dicts = [load_pred_dict(INGREDIENTS[k]) for k in keys]
        for mode in ('union', 'maj2', 'maj2_or_lead'):
            agg = aggregate(dicts, mode)
            tmp = f'{PREDS_DIR}/_probe_{combo_name}_{mode}.json'
            with open(tmp, 'w') as f:
                json.dump([{'question_id': qid, 'schema_links': sl} for qid, sl in sorted(agg.items())], f)
            score, _stdout = run_eval(tmp)
            os.remove(tmp)
            rows.append((combo_name, mode, score))
            improved = score is not None and score > best[2]
            tag = ' <- new best' if improved else ''
            if improved:
                best = (combo_name, mode, score, agg)
            score_str = f'{score:.4f}' if score is not None else '  ERR'
            print(f"  {combo_name:32s} {mode:14s} -> {score_str}{tag}")

    print('\n[probe] top 10:')
    for r in sorted([r for r in rows if r[2] is not None], key=lambda r: -r[2])[:10]:
        print(f"  {r[2]:.4f}  {r[0]:32s} {r[1]}")

    if best[3] is not None:
        out = f'{PREDS_DIR}/preds_PROBE_BEST.json'
        with open(out, 'w') as f:
            json.dump([{'question_id': qid, 'schema_links': sl} for qid, sl in sorted(best[3].items())], f)
        print(f"\n[probe] best: {best[0]} + {best[1]} = {best[2]:.4f}")
        print(f"[probe] written to {out}")
        print(f"[probe] champion to beat: 0.7107")
        if best[2] > 0.7107:
            print(f"[probe] *** BEATS CHAMPION by {best[2]-0.7107:+.4f} ***")
        else:
            print(f"[probe] does not beat champion ({best[2]-0.7107:+.4f})")


if __name__ == '__main__':
    main()
