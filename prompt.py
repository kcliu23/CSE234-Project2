"""
Prompt template and output-parsing utilities for schema linking.

The same `build_messages` function MUST be used at training time (when we
construct supervised examples from train.json) and at inference time
(main.py) -- otherwise the model sees a different prompt distribution and
schema-linking quality collapses.
"""
import json
import re
from typing import Dict, List, Optional

from schema_utils import serialize_schema_compact, serialize_schema_filtered


SYSTEM_PROMPT = (
    "You are a schema-linking assistant. Given a database schema and a natural-language "
    "question, identify the tables and columns that the underlying SQL query would reference. "
    "Reply with a single JSON object on one line and nothing else. "
    "The JSON keys are table names from the schema; the value for each key is a list of "
    "column names from that table that the SQL would reference. "
    "A table referenced with no specific columns (e.g. `SELECT COUNT(*) FROM t`) must appear "
    "with an empty list: {\"t\": []}. "
    "Use only identifiers that appear in the provided schema -- do not invent table or column names."
)


def build_user_message(db_id: str, question: str, schema_text: str) -> str:
    return (
        f"Database: {db_id}\n"
        f"Schema:\n{schema_text}\n\n"
        f"Question: {question}\n\n"
        f"Output the schema_links JSON now."
    )


def build_messages(
    db_id: str,
    question: str,
    schema: Dict,
    gold_links: Optional[Dict[str, List[str]]] = None,
    max_tables: int = 20,
    threshold_cols: int = 500,
) -> List[Dict[str, str]]:
    """Returns a chat-format message list. The model's `apply_chat_template`
    will turn this into a single prompt string at train and inference time.

    Uses table-level BM25 filtering for big schemas (>threshold_cols cols);
    full compact serialization for small schemas. Identical behavior at
    train and inference -- the only difference is `gold_links`, which is
    passed at training time (oracle-includes gold-referenced tables) and
    None at inference time.
    """
    schema_text = serialize_schema_filtered(
        schema, question, gold_links=gold_links,
        max_tables=max_tables, threshold_cols=threshold_cols,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": build_user_message(db_id, question, schema_text)},
    ]


def target_string(schema_links: Dict[str, List[str]]) -> str:
    """Canonical target string the model is supervised to emit.

    Sorted keys + sorted column lists for stable decoding. Empty col lists
    survive (required for wildcard examples). One line, no extra whitespace.
    """
    canon = {t: sorted(cols) for t, cols in sorted(schema_links.items())}
    return json.dumps(canon, separators=(',', ': '))


_JSON_OBJ_RE = re.compile(r'\{[\s\S]*\}')


def parse_model_output(text: str) -> Dict:
    """Best-effort recovery of a {table: [cols]} dict from a model's raw output.

    Strategy:
      1. Strip common chat-template/end-of-turn tokens.
      2. Try json.loads on the whole stripped text.
      3. Otherwise, find the FIRST balanced {...} block and json.loads that.
      4. Otherwise, return {} (the caller will downstream-filter as empty).
    """
    if not isinstance(text, str):
        return {}
    s = text.strip()
    # Strip code fences if the model wrapped its output.
    s = re.sub(r'^```(?:json)?\s*', '', s)
    s = re.sub(r'\s*```$', '', s)
    # Strip common end-of-turn markers from a few chat templates.
    for tok in ('<|im_end|>', '<|endoftext|>', '<|eot_id|>', '</s>'):
        s = s.replace(tok, '')
    s = s.strip()

    # Direct parse first.
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Find first balanced JSON object by scanning braces. This handles cases
    # where the model emitted a preamble like "Sure! Here you go:\n{...}".
    depth = 0
    start = -1
    in_str = False
    escape = False
    for i, ch in enumerate(s):
        if in_str:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start >= 0:
                candidate = s[start:i + 1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    start = -1  # try next opening brace
                    continue
    return {}
