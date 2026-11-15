import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

from database_utils.execution import execute_sql

# Port of CHESS database_utils/db_values/{preprocess,search}.py: the MinHash LSH over distinct text values
# is built offline from MySQL; text columns come from dev_tables.json types instead of SQLite PRAGMA.
# dev_tables.json has no primary keys, so CHESS's primary-key exclusion never applies.

SKIP_COLUMN_KEYWORDS = ["_id", " id", "url", "email", "web", "time", "phone", "date", "address"]


def _is_text_type(column_type: str) -> bool:
    # CHESS keeps SQLite TEXT columns; the MySQL/Oracle equivalents are (VAR)CHAR(2), *TEXT, ENUM and SET
    t = column_type.split("(")[0].upper()
    return "CHAR" in t or "TEXT" in t or t in ("ENUM", "SET")


def get_unique_values(mysql_creds: Dict[str, Any], tables: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, List[str]]]:
    unique_values: Dict[str, Dict[str, List[str]]] = {}
    for table_name, table in tqdm(tables.items(), desc="Fetching distinct values"):
        columns = [c for c, t in zip(table["column_names"], table["column_types"]) if _is_text_type(t)]
        table_values: Dict[str, List[str]] = {}
        for column in columns:
            if any(keyword in column.lower() for keyword in SKIP_COLUMN_KEYWORDS) or column.endswith("Id"):
                continue
            try:
                result = execute_sql(mysql_creds, f"""
                    SELECT SUM(CHAR_LENGTH(unique_values)), COUNT(unique_values)
                    FROM (
                        SELECT DISTINCT `{column}` AS unique_values
                        FROM `{table_name}`
                        WHERE `{column}` IS NOT NULL
                    ) AS subquery
                """, fetch="one", timeout=480)
            except Exception as e:
                logging.warning(f"Skipping {table_name}.{column}: {e}")
                result = 0, 0
            sum_of_lengths, count_distinct = result
            if sum_of_lengths is None or count_distinct == 0:
                continue
            sum_of_lengths = int(sum_of_lengths)
            average_length = sum_of_lengths / count_distinct
            if ("name" in column.lower() and sum_of_lengths < 5000000) or (sum_of_lengths < 2000000 and average_length < 25) or count_distinct < 100:
                try:
                    values = [str(value[0]) for value in execute_sql(mysql_creds, f"SELECT DISTINCT `{column}` FROM `{table_name}` WHERE `{column}` IS NOT NULL", fetch="all", timeout=480)]
                except Exception as e:
                    logging.warning(f"Skipping {table_name}.{column}: {e}")
                    values = []
                table_values[column] = values
        unique_values[table_name] = table_values
    return unique_values


def _ngrams(string: str, n_gram: int) -> List[bytes]:
    return [string[i:i + n_gram].encode('utf8') for i in range(len(string) - n_gram + 1)]


def _create_minhash(signature_size: int, string: str, n_gram: int) -> MinHash:
    m = MinHash(num_perm=signature_size)
    m.update_batch(_ngrams(string, n_gram))
    return m


def _band_keys(hashvalues: np.ndarray, b: int, r: int) -> np.ndarray:
    """One key per band and row, shape (b, n); rows sharing a key in any band are LSH candidates."""
    if r == 1:
        return np.ascontiguousarray(hashvalues[:, :b].T)
    bands = np.ascontiguousarray(hashvalues[:, :b * r].reshape(len(hashvalues), b, r))
    return bands.view(np.dtype((np.void, bands.dtype.itemsize * r)))[..., 0].T.copy()


def make_lsh(unique_values: Dict[str, Dict[str, List[str]]], signature_size: int, n_gram: int, threshold: float) -> Dict[str, Any]:
    """
    Equivalent of inserting every value into datasketch's MinHashLSH: the same bands (b, r) and the same
    candidates, but each band is a sorted array searched by bisection instead of a dict of buckets,
    which takes a fraction of the memory (the dw index needed 11GB as a MinHashLSH).
    """
    entries = [(table_name, column_name, value)
               for table_name, table_values in unique_values.items()
               for column_name, column_values in table_values.items()
               for value in column_values]
    params = MinHashLSH(threshold=threshold, num_perm=signature_size)
    b, r = params.b, params.r
    hashvalues = None
    batch_size = 10000
    for start in tqdm(range(0, len(entries), batch_size), desc="Creating LSH"):
        batch = entries[start:start + batch_size]
        # bulk() shares one set of permutations, so it matches _create_minhash value for value
        minhashes = MinHash.bulk([_ngrams(value, n_gram) for _, _, value in batch], num_perm=signature_size)
        if hashvalues is None:
            hashvalues = np.empty((len(entries), signature_size), dtype=minhashes[0].hashvalues.dtype)
        for offset, minhash in enumerate(minhashes):
            hashvalues[start + offset] = minhash.hashvalues
    if hashvalues is None:
        hashvalues = np.empty((0, signature_size), dtype=np.uint32)

    keys = _band_keys(hashvalues, b, r)
    order = np.argsort(keys, axis=1, kind="stable").astype(np.int32)
    sorted_keys = np.take_along_axis(keys, order, axis=1)

    table_names = sorted({table_name for table_name, _, _ in entries})
    table_ids = {name: i for i, name in enumerate(table_names)}
    entry_tables = np.array([table_ids[table_name] for table_name, _, _ in entries], dtype=np.int32)
    return {"b": b, "r": r, "sorted_keys": sorted_keys, "order": order, "hashvalues": hashvalues,
            "entries": entries, "table_names": table_names, "entry_tables": entry_tables,
            "signature_size": signature_size, "n_gram": n_gram, "threshold": threshold}


def make_db_lsh(mysql_creds: Dict[str, Any], tables: Dict[str, Dict[str, Any]], output_dir: Path,
                signature_size: int = 100, n_gram: int = 3, threshold: float = 0.01, refetch: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    values_path = output_dir / "unique_values.pkl"
    if values_path.exists() and not refetch:
        # Fetching is the slow step; reuse the values of an earlier (possibly interrupted) build
        print(f"Reusing distinct values from {values_path}")
        with open(values_path, "rb") as file:
            unique_values = pickle.load(file)
    else:
        unique_values = get_unique_values(mysql_creds, tables)
        with open(values_path, "wb") as file:
            pickle.dump(unique_values, file)
    index = make_lsh(unique_values, signature_size, n_gram, threshold)
    # Written last: its presence marks the preprocessing as complete
    with open(output_dir / "lsh.pkl", "wb") as file:
        pickle.dump(index, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_db_lsh(output_dir: Path) -> Dict[str, Any]:
    with open(output_dir / "lsh.pkl", "rb") as file:
        return pickle.load(file)


def _lsh_candidates(index: Dict[str, Any], query_minhash: MinHash) -> np.ndarray:
    query_keys = _band_keys(query_minhash.hashvalues[None, :], index["b"], index["r"])[:, 0]
    candidates = []
    for band, key in enumerate(query_keys):
        sorted_keys = index["sorted_keys"][band]
        lo = np.searchsorted(sorted_keys, key, side="left")
        hi = np.searchsorted(sorted_keys, key, side="right")
        if hi > lo:
            candidates.append(index["order"][band][lo:hi])
    if not candidates:
        return np.empty(0, dtype=np.int32)
    return np.unique(np.concatenate(candidates))


def query_lsh(index: Dict[str, Any], keyword: str, signature_size: int = 100, n_gram: int = 3, top_n: int = 10,
              allowed_tables: Optional[set] = None) -> Dict[str, Dict[str, List[str]]]:
    """
    Returns the top_n values most similar to the keyword, grouped by table and column.
    allowed_tables (lowercased names) drops other tables' values before the top_n cut, so they cannot crowd out allowed ones.
    """
    query_minhash = _create_minhash(signature_size, keyword, n_gram)
    entries = index["entries"]
    results = _lsh_candidates(index, query_minhash)
    if allowed_tables is not None:
        allowed_ids = [i for i, name in enumerate(index["table_names"]) if name.lower() in allowed_tables]
        results = results[np.isin(index["entry_tables"][results], allowed_ids)]
    if len(results) == 0:
        return {}
    similarities = (index["hashvalues"][results] == query_minhash.hashvalues).mean(axis=1)
    top = results[np.argsort(-similarities, kind="stable")[:top_n]]

    similar_values_trimmed: Dict[str, Dict[str, List[str]]] = {}
    for result in top:
        table_name, column_name, value = entries[result]
        similar_values_trimmed.setdefault(table_name, {}).setdefault(column_name, []).append(value)
    return similar_values_trimmed
