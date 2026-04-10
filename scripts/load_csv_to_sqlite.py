#!/usr/bin/env python3
"""Load DrugBank table CSVs into a local SQLite database."""

from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from table_metadata_index import normalize_table_name


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_TABLES_DIR = REPO_ROOT / "data" / "Pharma" / "drugbank-tables"
DEFAULT_DB_PATH = REPO_ROOT / "data" / "drugbank.db"


def load_csvs(tables_dir: Path, db_path: Path) -> int:
    csv_files = sorted(tables_dir.glob("*.csv"))
    if not csv_files:
        return 0

    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path))
    try:
        for path in tqdm(csv_files, desc="Loading CSVs to SQLite", unit="table"):
            table_name = normalize_table_name(path.stem)
            df = pd.read_csv(path)
            # Keep table/column names SQLite-safe and deterministic.
            df.columns = [normalize_table_name(str(col)) for col in df.columns]
            df.to_sql(table_name, connection, if_exists="replace", index=False)
    finally:
        connection.close()

    return len(csv_files)


def main() -> int:
    parser = argparse.ArgumentParser(description="Load DrugBank CSV tables into SQLite.")
    parser.add_argument("--tables-dir", default=str(DEFAULT_TABLES_DIR))
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    args = parser.parse_args()

    tables_dir = Path(args.tables_dir).expanduser().resolve()
    db_path = Path(args.db_path).expanduser().resolve()

    if not tables_dir.exists():
        print(f"Tables directory not found: {tables_dir}")
        return 1

    loaded = load_csvs(tables_dir=tables_dir, db_path=db_path)
    if loaded == 0:
        print(f"No CSV files found under {tables_dir}")
        return 1

    print(f"Loaded {loaded} tables into {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
