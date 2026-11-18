import re

EXTRACTION_PROMPT = """
You are an expert SQL parser. Your task is to extract specific components from the provided SQL query into a JSON format.
You must extract ONLY objects that exist in the database schema. Do NOT extract intermediate tables (CTEs), subquery aliases, or derived tables.
Copy table and column names exactly as they are written in the SQL; do not add or remove underscores or otherwise correct their spelling.

1. "tables": A list of all DATABASE table names referenced in the query (FROM, JOIN). Ignore CTE names and subquery aliases.
2. "columns": A list of all fully qualified DATABASE columns used (SELECT, WHERE, GROUP BY, HAVING, ORDER BY, JOIN conditions). Format: "TABLE_NAME.COLUMN_NAME". Ignore aliases, map back to real table names. If table name is unknown, use the alias or inferred name, but ensure it maps to a real table. Do not expand "*" into columns.
3. "join_keys": A list of equality join conditions between two *different* database tables, in the form "TABLE1.COL1 = TABLE2.COL2". When a join goes through a CTE, subquery or derived table, trace each side back to the database table column it comes from and report that. Do not report a table joined with itself.
4. "predicates": Every predicate in WHERE, HAVING or ON clauses (including inside CTEs and subqueries) that compares a database column with a literal value, a list of literals, or a subquery (e.g. "COL NOT IN (SELECT ...)"), with any operator (=, <>, !=, <, >, <=, >=, IN, NOT IN, LIKE, NOT LIKE, REGEXP, IS NULL, IS NOT NULL). Trace the column back to its database table. Do not judge whether the predicate is domain-specific; list all of them. If the column is wrapped in a function (UPPER(...), LOWER(...)), report the bare column and put the function in "function". If the right-hand side is a subquery, set "values" to null and "subquery" to true.

SQL Query:
<<<
{sql}
>>>

Output JSON format:
{{
  "tables": ["TABLE1", ...],
  "columns": ["TABLE1.COL1", ...],
  "join_keys": ["TABLE1.ID = TABLE2.REF_ID", ...],
  "predicates": [
    {{"table": "SIS_DEPARTMENT", "column": "DEPARTMENT_NAME", "op": "=", "values": ["Mathematics"]}},
    {{"table": "SIS_DEPARTMENT", "column": "DEPARTMENT_CODE", "op": "NOT IN", "values": ["5", "7"]}}
  ]
}}
"""

# Every metric returns None when the record's gold set is empty, so averages skip it instead of counting 0.
# Known limitation: X = 'a' OR X = 'b' is not merged into X IN ('a', 'b').

NUM = re.compile(r"^-?\d+(?:\.\d+)?$")


def _ident(x):
    """Table or column name: uppercase, no backticks/quotes, database prefix dropped (dw.TABLE -> TABLE)."""
    return str(x).replace("`", "").replace('"', "").split(".")[-1].strip().upper()


def _col(x):
    """TABLE.COLUMN with the same normalization; a database prefix (dw.TABLE.COLUMN) is dropped."""
    parts = str(x).replace("`", "").replace('"', "").strip().upper().split(".")
    return ".".join(p.strip() for p in parts[-2:])


def _join(x):
    """Order-independent join key from "A = B" or [A, B]."""
    sides = x if isinstance(x, (list, tuple)) else str(x).split("=")
    return frozenset(_col(s) for s in sides)


def _different_tables(join):
    """Gold join keys never join a table with itself (self-joins, CTEs traced back to the same table)."""
    return len({c.split(".")[0] for c in join}) == len(join) == 2


def _f1(precision, recall):
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def set_f1(generated, gold):
    if not gold:
        return None
    tp = len(generated & gold)
    precision = tp / len(generated) if generated else 0.0
    return _f1(precision, tp / len(gold))


def _components(joins):
    """Column -> equivalence-class id under the transitive closure of the join equalities."""
    parent = {}

    def find(c):
        parent.setdefault(c, c)
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    for j in joins:
        x, y = sorted(j)
        parent[find(x)] = find(y)
    return {c: find(c) for c in parent}


def _implied(join, components):
    x, y = sorted(join)
    return x in components and y in components and components[x] == components[y]


def join_f1(generated, gold):
    """F1 over join keys up to transitivity: a generated pair is correct if the gold equalities imply it, and a gold
    pair is recovered if the generated equalities imply it (A = B, B = C and A = B, A = C join the same columns)."""
    if not gold:
        return None
    gold_c, gen_c = _components(gold), _components(generated)
    precision = sum(_implied(j, gold_c) for j in generated) / len(generated) if generated else 0.0
    recall = sum(_implied(j, gen_c) for j in gold) / len(gold)
    return _f1(precision, recall)


def set_recall(generated, gold):
    if not gold:
        return None
    return len(generated & gold) / len(gold)


def _value(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        s = s[1:-1]
    if NUM.match(s):
        return float(s)
    return s.lower()  # MySQL's default collation is case-insensitive


def canon(p, with_table=True):
    """Comparable predicate key: != -> <>, single-value IN/NOT IN -> =/<>, values as a set; 'function' ignored."""
    op = re.sub(r"\s+", " ", str(p["op"]).strip().upper())
    if op == "!=":
        op = "<>"
    if op in ("IS NULL", "IS NOT NULL"):
        vals = ()
    elif p.get("subquery") or p.get("values") is None:
        vals = ("<SUBQUERY>",)
    else:
        vals = tuple(sorted({_value(v) for v in p["values"]}, key=repr))
        if op == "IN" and len(vals) == 1:
            op = "="
        elif op == "NOT IN" and len(vals) == 1:
            op = "<>"
    key = (_ident(p["column"]), op, vals)
    return ((_ident(p["table"]),) + key) if with_table else key


def clean_predicates(predicates):
    """Drop malformed LLM predicates (missing table/column/op) and wrap a string 'values' in a list."""
    out = []
    for p in predicates if isinstance(predicates, list) else []:
        if not isinstance(p, dict) or not all(p.get(k) for k in ("table", "column", "op")):
            continue
        p = dict(p)
        if isinstance(p.get("values"), (str, int, float)):
            p["values"] = [p["values"]]
        elif p.get("values") is not None and not isinstance(p["values"], list):
            continue
        out.append(p)
    return out


def dk_scores(generated, gold, with_table=True):
    """Recall over gold DK predicates; precision only over generated predicates on a gold DK column, so filters on
    other columns (term codes, DELETED = 0, ...) are not penalized while a wrong value/operator on the right column is.
    with_table=False ignores the table, crediting a valid alternative table for the same column."""
    if not gold:
        return None
    gold_keys = {canon(p, with_table) for p in gold}
    if with_table:
        gold_cols = {(_ident(p["table"]), _ident(p["column"])) for p in gold}
        on_cols = [g for g in generated if (_ident(g["table"]), _ident(g["column"])) in gold_cols]
    else:
        gold_cols = {_ident(p["column"]) for p in gold}
        on_cols = [g for g in generated if _ident(g["column"]) in gold_cols]
    gen_keys = {canon(g, with_table) for g in on_cols}
    tp = len(gold_keys & gen_keys)
    recall = tp / len(gold_keys)
    precision = tp / len(gen_keys) if gen_keys else 0.0
    return dict(precision=precision, recall=recall, f1=_f1(precision, recall))


def _strings(xs):
    return [x for x in xs if isinstance(x, (str, list, tuple))] if isinstance(xs, list) else []


def normalized(extraction_output, gold_entry):
    """(generated, gold) pairs per subtask after normalization: sets for tables/columns/join keys, predicate
    lists for domain knowledge."""
    extraction_output = extraction_output if isinstance(extraction_output, dict) else {}
    gold_dk = gold_entry.get("domain_knowledge", [])
    if not all(isinstance(d, dict) for d in gold_dk):
        raise ValueError(f"{gold_entry.get('id')}: gold domain_knowledge is in the old string format; "
                         "score against the v2 annotations")
    return {
        "tables": ({_ident(t) for t in _strings(extraction_output.get("tables")) if isinstance(t, str)},
                   {_ident(t) for t in gold_entry.get("tables", [])}),
        "columns": ({_col(c) for c in _strings(extraction_output.get("columns")) if isinstance(c, str)},
                    {_col(c) for cols in gold_entry.get("column_mapping", {}).values() for c in cols}),
        "join_keys": ({j for j in map(_join, _strings(extraction_output.get("join_keys"))) if _different_tables(j)},
                      {_join(jk) for jk in gold_entry.get("join_keys", [])}),
        "domain_knowledge": (clean_predicates(extraction_output.get("predicates")), gold_dk),
    }


def eval_extraction_output(extraction_output, gold_entry):
    sets = normalized(extraction_output, gold_entry)
    gen_dk, gold_dk = sets["domain_knowledge"]
    dk = dk_scores(gen_dk, gold_dk) or {}
    dk_lenient = dk_scores(gen_dk, gold_dk, with_table=False) or {}
    return {
        "table_retrieval_f1": set_f1(*sets["tables"]),
        # Gold lists only the columns the question refers to, a subset of the columns the SQL uses, so recall only
        "column_mapping_recall": set_recall(*sets["columns"]),
        "join_key_f1": join_f1(*sets["join_keys"]),
        "domain_knowledge_f1": dk.get("f1"),
        "domain_knowledge_precision": dk.get("precision"),
        "domain_knowledge_recall": dk.get("recall"),
        "domain_knowledge_f1_lenient": dk_lenient.get("f1"),
    }
