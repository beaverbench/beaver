import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

CHESS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CHESS_DIR / "src"))
sys.path.insert(0, str(CHESS_DIR.parent))

from database_utils.db_values import make_db_lsh
from utils.ex_acc import get_mysql_credentials

# Builds CHESS's value index (MinHash LSH over distinct text values) for one BEAVER database, using the
# settings in CHESS's run/run_preprocess.sh. The column catalog (vector DB) is not built: BEAVER has no
# column descriptions. dw and dw_real share the dw database, so they share one index.

PREPROCESSED_DIR = CHESS_DIR / "preprocessed_data"


def index_dir(db_id: str) -> Path:
    return PREPROCESSED_DIR / db_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=str(CHESS_DIR.parent.parent / "data"))
    parser.add_argument("--signature_size", type=int, default=100)
    parser.add_argument("--n_gram", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--force", action="store_true", help="Rebuild the index even if it exists")
    parser.add_argument("--refetch", action="store_true", help="With --force, also refetch the distinct values from MySQL")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
    load_dotenv(CHESS_DIR.parent.parent / ".env")

    with open(os.path.join(args.data_dir, args.dataset, "dev_tables.json")) as f:
        tables = json.load(f)
    db_ids = {t["db"] for t in tables.values()}
    assert len(db_ids) == 1, f"Expected one database in dev_tables.json, found {db_ids}"
    db_id = db_ids.pop()

    output_dir = index_dir(db_id)
    if (output_dir / "lsh.pkl").exists() and not args.force:
        print(f"Value index for {db_id} already exists at {output_dir}")
        return

    mysql_creds = get_mysql_credentials(args.dataset)
    if not mysql_creds:
        raise SystemExit("MYSQL_HOST / MYSQL_USER / MYSQL_PASSWORD are not set in .env")

    print(f"Building value index for {db_id} ({len(tables)} tables) at {output_dir}")
    make_db_lsh(mysql_creds, tables, output_dir, args.signature_size, args.n_gram, args.threshold, refetch=args.refetch)


if __name__ == "__main__":
    main()
