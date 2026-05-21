"""
sql_to_schema_links.py -- Helper utility to extract schema-link ground truth
from an SQL query given the target database schema.

This is the SAME extractor that was used to build the train/validation gold
schema_links from the SNAILS gold SQL queries. We release it so you can
generate ground-truth schema_links for any additional SQL queries you write
or generate (e.g., for data augmentation).

Usage (single query, via CLI):
    python sql_to_schema_links.py --schemas_dir schemas/ \
        --db_id NTSB \
        --sql "select count(*) from CASE_VEHICLE where VEHICLEMAKE = 'FORD'"

Usage (batch, via CLI):
    python sql_to_schema_links.py --schemas_dir schemas/ \
        --batch_in queries.json --batch_out queries_with_links.json
    # queries.json: [{"question_id": 1, "db_id": "NTSB", "question": "...",
    #                 "gold_sql": "select ..."}, ...]
    # queries_with_links.json: same but with "schema_links" added.

The extractor parses the SQL via sqlglot (T-SQL dialect, since the SNAILS gold
queries are MS SQL Server style), qualifies bare column references against the
schema, restores original-case identifiers, and outputs:

    {"<Table1>": ["<Col1>", "<Col2>", ...], "<Table2>": [...]}

Notes
-----
- Tables that appear in FROM/JOIN with no columns referenced still get a
  (possibly empty) entry, since their identity is itself a schema link.
- COUNT(*) and other wildcard column references DO NOT produce column entries.
- Identifiers in the SQL that do not match the provided schema are skipped
  (this happens when, e.g., the SQL references a column that has been
  renamed). Inspect the SQL or schema if you see surprising empties.
- The default SQL dialect is ``tsql`` (T-SQL / MS SQL Server), matching the
  SNAILS source corpus. If you generate augmentation queries in another
  dialect (Postgres, MySQL, etc.) and hit parse errors or wrong outputs,
  override with ``--dialect <name>``. sqlglot is fairly tolerant across
  mainstream dialects, so most generic SQL parses fine under tsql.
- The extractor uses sqlglot's ``allow_partial_qualification=True``: if a
  column reference cannot be resolved to a known table (e.g., a typo in a
  hand-written augmentation query), it is silently dropped rather than
  raising. For augmented training data, sanity-check the output dict
  against your SQL to catch unintended drops.
- Requires: pip install sqlglot
"""
import argparse
import json
import os

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.qualify_columns import qualify_columns
from sqlglot.schema import MappingSchema


def load_schema(schemas_dir, db_id):
    """Load Spider-format schema and return {table: {col: type}}."""
    fname = db_id.replace(' ', '_').replace('/', '_') + '.json'
    path = os.path.join(schemas_dir, fname)
    with open(path) as f:
        s = json.load(f)
    table_names = s['table_names_original']
    column_names = s['column_names_original']
    column_types = s.get('column_types', ['TEXT'] * len(column_names))
    schema = {t: {} for t in table_names}
    for (tidx, cname), ctype in zip(column_names, column_types):
        if tidx == -1:
            continue
        schema[table_names[tidx]][cname] = (ctype or 'TEXT').upper()
    return schema


def extract_schema_links(sql, schema, dialect='tsql'):
    """Parse SQL and extract a {table: [cols]} dict, with identifiers in
    original schema casing.

    Critically, this does NOT expand ``SELECT *`` into all columns of the FROM
    table. A ``select *`` references the TABLE (so the table appears in the
    output) but does not name any specific columns. The same applies to
    ``count(*)`` and other star wildcards.
    """
    lc_tables = {t.lower(): t for t in schema}
    lc_cols   = {t: {c.lower(): c for c in cols} for t, cols in schema.items()}
    lc_schema = {t.lower(): {c.lower(): ctype for c, ctype in cols.items()}
                 for t, cols in schema.items()}
    try:
        parsed = sqlglot.parse_one(sql, dialect=dialect)
        # Step 1: qualify table refs and resolve aliases, but do NOT yet
        # qualify columns (because the default would expand *).
        qualified = qualify(parsed, schema=lc_schema, dialect=dialect,
                            validate_qualify_columns=False, identify=False,
                            infer_schema=False, qualify_columns=False,
                            expand_alias_refs=True)
        # Step 2: qualify columns with expand_stars=False so SELECT * does
        # not pollute the schema-link ground truth.
        qualified = qualify_columns(qualified,
                                    schema=MappingSchema(lc_schema, dialect=dialect),
                                    expand_alias_refs=True,
                                    expand_stars=False,
                                    infer_schema=False,
                                    allow_partial_qualification=True,
                                    dialect=dialect)
    except Exception as e:
        return None, str(e)

    used_tables_lc = set()
    alias_to_table = {}
    for tbl in qualified.find_all(exp.Table):
        tname_lc = tbl.name.lower()
        if tname_lc in lc_tables:
            used_tables_lc.add(tname_lc)
        if tbl.alias:
            alias_to_table[tbl.alias.lower()] = tname_lc
        alias_to_table[tname_lc] = tname_lc

    table_cols_lc = {}
    for c in qualified.find_all(exp.Column):
        col_table_lc = (c.table or '').lower()
        col_name_lc  = c.name.lower()
        actual_table_lc = alias_to_table.get(col_table_lc, col_table_lc)
        if actual_table_lc in lc_tables:
            orig_table = lc_tables[actual_table_lc]
            if col_name_lc in lc_cols[orig_table]:
                table_cols_lc.setdefault(actual_table_lc, set()).add(col_name_lc)

    for tname_lc in used_tables_lc:
        table_cols_lc.setdefault(tname_lc, set())

    out = {}
    for tname_lc, cols_lc in table_cols_lc.items():
        orig_table = lc_tables[tname_lc]
        orig_cols = sorted({lc_cols[orig_table][c] for c in cols_lc})
        out[orig_table] = orig_cols
    return out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--schemas_dir', required=True)
    ap.add_argument('--db_id', help='Used with --sql for a single-query run.')
    ap.add_argument('--sql', help='Single SQL query to parse.')
    ap.add_argument('--batch_in',  help='Input JSON list with db_id+gold_sql fields.')
    ap.add_argument('--batch_out', help='Output JSON path for batch mode.')
    ap.add_argument('--dialect', default='tsql')
    args = ap.parse_args()

    if args.sql:
        schema = load_schema(args.schemas_dir, args.db_id)
        links, err = extract_schema_links(args.sql, schema, args.dialect)
        if err:
            print(f"PARSE ERROR: {err}")
            return
        print(json.dumps(links, indent=2))
        return

    if args.batch_in:
        with open(args.batch_in) as f:
            items = json.load(f)
        out = []
        n_err = 0
        schema_cache = {}
        for it in items:
            db_id = it['db_id']
            if db_id not in schema_cache:
                schema_cache[db_id] = load_schema(args.schemas_dir, db_id)
            links, err = extract_schema_links(it['gold_sql'], schema_cache[db_id], args.dialect)
            if err:
                n_err += 1
                links = {}
            new_it = dict(it)
            new_it['schema_links'] = links
            out.append(new_it)
        with open(args.batch_out, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"Wrote {len(out)} records to {args.batch_out}  (parse errors: {n_err})")
        return

    ap.print_help()


if __name__ == '__main__':
    main()
