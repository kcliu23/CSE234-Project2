"""
Schema loading, serialization, and prediction-validation helpers.

Shared by the training-data formatter and main.py so that the prompt the model
sees at training time is identical to the prompt it sees at inference time.
"""
import json
import math
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple, Tuple


def db_id_to_filename(db_id: str) -> str:
    return db_id.replace(' ', '_').replace('/', '_') + '.json'


def load_schema(schemas_dir: str, db_id: str) -> Dict:
    """Return a dict with:
        tables       : [original_table_name, ...]
        columns      : {original_table_name: [original_col_name, ...]}
        col_types    : {original_table_name: {col_name: type_str}}
        primary_keys : list of (table, col) tuples
        foreign_keys : list of ((from_table, from_col), (to_table, to_col)) tuples
    Identifier casing is preserved from the source schema.
    """
    path = os.path.join(schemas_dir, db_id_to_filename(db_id))
    with open(path) as f:
        s = json.load(f)
    tables = list(s['table_names_original'])
    columns: Dict[str, List[str]] = {t: [] for t in tables}
    col_types: Dict[str, Dict[str, str]] = {t: {} for t in tables}
    types_raw = s.get('column_types', ['TEXT'] * len(s['column_names_original']))
    for (tidx, cname), ctype in zip(s['column_names_original'], types_raw):
        if tidx == -1:
            continue
        t = tables[tidx]
        columns[t].append(cname)
        col_types[t][cname] = (ctype or 'TEXT').upper()

    pks = []
    for pk in s.get('primary_keys', []):
        # Spider PKs are column indexes into column_names_original (or lists of them for composite keys)
        pk_list = pk if isinstance(pk, list) else [pk]
        for cidx in pk_list:
            if 0 <= cidx < len(s['column_names_original']):
                tidx, cname = s['column_names_original'][cidx]
                if tidx != -1:
                    pks.append((tables[tidx], cname))

    fks = []
    for fk in s.get('foreign_keys', []):
        if not (isinstance(fk, list) and len(fk) == 2):
            continue
        a, b = fk
        if not (0 <= a < len(s['column_names_original']) and 0 <= b < len(s['column_names_original'])):
            continue
        ta, ca = s['column_names_original'][a]
        tb, cb = s['column_names_original'][b]
        if ta == -1 or tb == -1:
            continue
        fks.append(((tables[ta], ca), (tables[tb], cb)))

    return {
        'db_id': db_id,
        'tables': tables,
        'columns': columns,
        'col_types': col_types,
        'primary_keys': pks,
        'foreign_keys': fks,
    }


def serialize_schema_compact(schema: Dict) -> str:
    """One line per table: `TableName: col1, col2, ...`.

    Compact, lossless on identifier names, no types. This is the baseline
    serialization; richer variants (types, PK/FK markers) are an experiment
    knob we'll add later.
    """
    lines = []
    for t in schema['tables']:
        cols = schema['columns'][t]
        lines.append(f"{t}: {', '.join(cols)}")
    return '\n'.join(lines)


def serialize_schema_with_types(schema: Dict) -> str:
    """`TableName: col1 (type), col2 (type), ...`. ~30% longer than compact."""
    lines = []
    for t in schema['tables']:
        cols = schema['columns'][t]
        types = schema['col_types'][t]
        parts = [f"{c} ({types.get(c, 'TEXT').lower()})" for c in cols]
        lines.append(f"{t}: {', '.join(parts)}")
    return '\n'.join(lines)


def serialize_schema_with_keys(schema: Dict) -> str:
    """Compact + PK marker (*) + inline FK arrows. ~10-15% longer than compact."""
    pk_set = set(schema['primary_keys'])
    fk_index = {(t, c): (t2, c2) for ((t, c), (t2, c2)) in schema['foreign_keys']}
    lines = []
    for t in schema['tables']:
        parts = []
        for c in schema['columns'][t]:
            marker = '*' if (t, c) in pk_set else ''
            if (t, c) in fk_index:
                t2, c2 = fk_index[(t, c)]
                parts.append(f"{c}{marker}->{t2}.{c2}")
            else:
                parts.append(f"{c}{marker}")
        lines.append(f"{t}: {', '.join(parts)}")
    return '\n'.join(lines)


def _format_columns(t: str, schema: Dict, style: str) -> List[str]:
    """Per-column string formatter for the requested style. Shared by the
    full and the BM25-filtered serializers so the two stay in lockstep."""
    cols = schema['columns'][t]
    types = schema['col_types'][t] if style in ('types', 'types_keys') else {}
    if style in ('keys', 'types_keys'):
        pk_set = set(schema['primary_keys'])
        fk_index = {(ft, fc): (tt, tc) for ((ft, fc), (tt, tc)) in schema['foreign_keys']}
    else:
        pk_set, fk_index = set(), {}
    parts = []
    for c in cols:
        s = c
        if style in ('types', 'types_keys'):
            s += f" ({types.get(c, 'TEXT').lower()})"
        if style in ('keys', 'types_keys') and (t, c) in pk_set:
            s += "*"
        if style in ('keys', 'types_keys') and (t, c) in fk_index:
            t2, c2 = fk_index[(t, c)]
            s += f"->{t2}.{c2}"
        parts.append(s)
    return parts


_TOKEN_SPLIT_RE = re.compile(r'([a-z])([A-Z])')
_TOKEN_SPLIT_RE2 = re.compile(r'([A-Z]+)([A-Z][a-z])')
_TOKEN_WORD_RE = re.compile(r'\w+')


def tokenize_identifier(s: str) -> List[str]:
    """Split a SQL identifier or NL phrase into a lowercase token list.
    Handles camelCase, snake_case, ABBR_Word, and bare words.
    """
    s = _TOKEN_SPLIT_RE.sub(r'\1 \2', s)
    s = _TOKEN_SPLIT_RE2.sub(r'\1 \2', s)
    s = re.sub(r'[_\-]+', ' ', s)
    return [t.lower() for t in _TOKEN_WORD_RE.findall(s)]


def _bm25_score_pairs(
    pairs: List[Tuple[str, str]],
    question: str,
    k1: float = 1.5,
    b: float = 0.75,
) -> List[float]:
    """BM25 ranking of (table, col) pairs by similarity to question.

    Each pair's document is the tokenized form of "Table Col". Self-contained:
    no external library needed. Returns one float score per input pair.
    """
    docs = [tokenize_identifier(f'{t} {c}') for t, c in pairs]
    q_terms = tokenize_identifier(question)
    if not docs or not q_terms:
        return [0.0] * len(pairs)

    N = len(docs)
    df: Dict[str, int] = {}
    for doc in docs:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1
    idf = {t: math.log(1.0 + (N - df_t + 0.5) / (df_t + 0.5)) for t, df_t in df.items()}

    avgdl = sum(len(d) for d in docs) / N
    scores = []
    for doc in docs:
        dl = max(1, len(doc))
        tf: Dict[str, int] = {}
        for term in doc:
            tf[term] = tf.get(term, 0) + 1
        s = 0.0
        for term in q_terms:
            if term not in idf:
                continue
            f = tf.get(term, 0)
            if f == 0:
                continue
            s += idf[term] * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        scores.append(s)
    return scores


def _bm25_score_tables(schema: Dict, question: str, k1: float = 1.5, b: float = 0.75) -> Dict[str, float]:
    """BM25 score for each table, where each table's doc is (table name +
    space-joined column names) tokenized into words.

    Empirically beats column-level BM25 for our task: recovers ~86% of gold
    (table, col) pairs at top-20 tables vs ~64% at top-400 columns. Reason:
    once a relevant table is found, ALL its columns are recovered, which is
    a much better match for the structure of schema linking.
    """
    table_docs: Dict[str, List[str]] = {}
    for t in schema['tables']:
        text = t + ' ' + ' '.join(schema['columns'][t])
        table_docs[t] = tokenize_identifier(text)
    q_terms = tokenize_identifier(question)
    if not q_terms or not table_docs:
        return {t: 0.0 for t in table_docs}

    N = len(table_docs)
    df: Dict[str, int] = {}
    for doc in table_docs.values():
        for term in set(doc):
            df[term] = df.get(term, 0) + 1
    idf = {t: math.log(1.0 + (N - df_t + 0.5) / (df_t + 0.5)) for t, df_t in df.items()}
    avgdl = sum(len(d) for d in table_docs.values()) / max(1, N)

    scores: Dict[str, float] = {}
    for tname, doc in table_docs.items():
        dl = max(1, len(doc))
        tf: Dict[str, int] = {}
        for term in doc:
            tf[term] = tf.get(term, 0) + 1
        s = 0.0
        for term in q_terms:
            if term not in idf:
                continue
            f = tf.get(term, 0)
            if f == 0:
                continue
            s += idf[term] * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        scores[tname] = s
    return scores


# Embedding-based table retriever -- alternative to BM25 keyword matching.
# Lazy-loaded so that anything importing schema_utils for serialization only
# doesn't pay the sentence-transformer import cost.
_EMBED_MODEL = None
_EMBED_TABLE_CACHE: Dict[int, Tuple[List[str], "any"]] = {}  # id(schema) -> (table_names, table_embs_tensor)


def _get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        # BGE-small-en-v1.5: ~30MB, ~384-dim, strong on retrieval benchmarks.
        # No auth required, downloads on first call.
        _EMBED_MODEL = SentenceTransformer('BAAI/bge-small-en-v1.5')
    return _EMBED_MODEL


def _embed_score_tables(schema: Dict, question: str) -> Dict[str, float]:
    """Embedding-based score per table: cosine similarity between the question
    embedding and each table's "document" embedding (table name + column names).
    Cached per schema id so the same schema isn't re-embedded per question."""
    model = _get_embed_model()
    cache_key = id(schema)
    if cache_key not in _EMBED_TABLE_CACHE:
        table_names = list(schema['tables'])
        # Build a natural-language-ish doc per table for richer embeddings
        # than raw identifiers (helps for camelCase / snake_case tokens).
        docs = []
        for t in table_names:
            tokens = ' '.join(tokenize_identifier(t)) or t
            col_tokens = ' '.join(' '.join(tokenize_identifier(c)) for c in schema['columns'][t])
            docs.append(f"table {tokens} columns {col_tokens}")
        # BGE recommends prefixing passages with a special instruction; not
        # strictly required for v1.5 but slightly improves recall.
        table_embs = model.encode(docs, normalize_embeddings=True, convert_to_tensor=True,
                                  show_progress_bar=False)
        _EMBED_TABLE_CACHE[cache_key] = (table_names, table_embs)
    table_names, table_embs = _EMBED_TABLE_CACHE[cache_key]
    # BGE recommends "Represent this sentence for searching relevant passages: ..."
    # as the query prefix; omitted for v1.5 with negligible impact.
    q_emb = model.encode([question], normalize_embeddings=True, convert_to_tensor=True,
                         show_progress_bar=False)[0]
    # Cosine sim = dot product (since both normalized)
    sims = (table_embs @ q_emb).cpu().tolist()
    return dict(zip(table_names, sims))


def serialize_schema_filtered(
    schema: Dict,
    question: str,
    gold_links: Optional[Dict[str, List[str]]] = None,
    max_tables: int = 20,
    threshold_cols: int = 500,
    style: str = 'compact',
    retrieval: str = 'bm25',
) -> str:
    """Schema serialization with table-level BM25 filtering for big schemas.

    Strategy:
      - If the schema has <= threshold_cols columns total, return the full
        serialization in the requested style (no point filtering).
      - Otherwise: score every table by BM25 against the question, keep the
        top `max_tables` tables, emit ALL their columns in the requested style.
      - If `gold_links` is provided (training time), force-include every
        gold-referenced table in the kept set.
      - Tables NOT in the kept set are dropped from the prompt entirely.

    `style` ∈ {'compact', 'types', 'keys', 'types_keys'}.  Mirrors the
    standalone serializer helpers above.  IMPORTANT: must match the style
    that the trained adapter was trained with -- different style at training
    vs. inference is the #1 cause of fine-tuned-model regressions.

    NOTE: `gold_links` must NOT be passed at inference time. Pass None there.
    """
    if style not in ('compact', 'types', 'keys', 'types_keys'):
        raise ValueError(f"Unknown style: {style!r}")

    n_total = sum(len(c) for c in schema['columns'].values())
    if n_total <= threshold_cols:
        # Full serialization in the requested style.
        if style == 'compact':
            return serialize_schema_compact(schema)
        if style == 'types':
            return serialize_schema_with_types(schema)
        if style == 'keys':
            return serialize_schema_with_keys(schema)
        # types_keys: combine
        lines = []
        for t in schema['tables']:
            parts = _format_columns(t, schema, 'types_keys')
            lines.append(f"{t}: {', '.join(parts)}")
        return '\n'.join(lines)

    if retrieval == 'embed':
        table_scores = _embed_score_tables(schema, question)
        top_tables = [t for t, _ in sorted(table_scores.items(), key=lambda kv: -kv[1])[:max_tables]]
    elif retrieval == 'bm25':
        table_scores = _bm25_score_tables(schema, question)
        top_tables = [t for t, _ in sorted(table_scores.items(), key=lambda kv: -kv[1])[:max_tables]]
    elif retrieval == 'hybrid':
        # Reciprocal Rank Fusion (RRF): combine BM25 and embed rankings via
        # 1/(60+rank) summed across the two methods. Scale-free, no
        # normalization needed. k=60 is the standard RRF constant from the
        # Cormack et al. 2009 paper.
        bm25_scores = _bm25_score_tables(schema, question)
        embed_scores = _embed_score_tables(schema, question)
        bm25_rank = {t: r for r, t in enumerate(sorted(bm25_scores, key=lambda x: -bm25_scores[x]))}
        embed_rank = {t: r for r, t in enumerate(sorted(embed_scores, key=lambda x: -embed_scores[x]))}
        K_RRF = 60
        fused = {t: 1.0 / (K_RRF + bm25_rank.get(t, len(bm25_rank)))
                  + 1.0 / (K_RRF + embed_rank.get(t, len(embed_rank)))
                 for t in schema['tables']}
        top_tables = [t for t, _ in sorted(fused.items(), key=lambda kv: -kv[1])[:max_tables]]
    else:
        raise ValueError(f"Unknown retrieval: {retrieval!r}")
    kept_tables = set(top_tables)

    # Oracle augmentation at training time: include every gold-referenced table.
    if gold_links:
        lc_tables = {t.lower(): t for t in schema['tables']}
        for gtbl in gold_links:
            tlc = str(gtbl).lower()
            if tlc in lc_tables:
                kept_tables.add(lc_tables[tlc])

    # Emit kept tables in original schema order (stable, helps debugging).
    lines = []
    for t in schema['tables']:
        if t not in kept_tables:
            continue
        parts = _format_columns(t, schema, style)
        lines.append(f"{t}: {', '.join(parts)}")
    return '\n'.join(lines)


def canonicalize_prediction(pred: Dict, schema: Dict) -> Dict[str, List[str]]:
    """Drop identifiers not in the schema; rewrite remaining ones to the
    schema's original casing.

    Tables not in the schema are dropped entirely (and their column lists with
    them). Columns not in their (valid) table's column set are dropped.

    This is the prediction-time hallucination filter. The eval script also
    drops hallucinations from the *score*, but it counts them as false
    positives -- so filtering here before emitting predictions strictly
    improves precision.
    """
    lc_tables = {t.lower(): t for t in schema['tables']}
    lc_cols = {t: {c.lower(): c for c in schema['columns'][t]} for t in schema['tables']}

    # Merge across duplicate-with-different-casing keys (e.g. "injury" and "INJURY").
    merged: Dict[str, List[str]] = {}
    seen_per_table: Dict[str, set] = {}
    if not isinstance(pred, dict):
        return merged
    for tbl, cols in pred.items():
        tlc = str(tbl).lower()
        if tlc not in lc_tables:
            continue
        canon_t = lc_tables[tlc]
        if canon_t not in merged:
            merged[canon_t] = []
            seen_per_table[canon_t] = set()
        if isinstance(cols, list):
            for c in cols:
                clc = str(c).lower()
                if clc in lc_cols[canon_t] and clc not in seen_per_table[canon_t]:
                    merged[canon_t].append(lc_cols[canon_t][clc])
                    seen_per_table[canon_t].add(clc)
        # Tables with empty col lists are KEPT: wildcard semantics
        # ({"t": []} for `select count(*) from t`).
    return merged
