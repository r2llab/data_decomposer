#!/usr/bin/env python3
"""
Script to group PubMed passage files by drug mentions. Reads passage files from a directory,
extracts drug names and synonyms from DrugBank SQLite database, matches mentions in text,
and outputs a JSON mapping of drug names to lists of passage filenames.
"""
import argparse
import sqlite3
import re
import os
import json
from collections import defaultdict
import logging

def load_drug_names(db_path):
    """
    Load primary drug names and English synonyms from DrugBank SQLite database.
    Returns a mapping from lower-case name/synonym to primary drug name.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Map DrugBank ID to primary name
    cursor.execute("SELECT primary_key, name FROM drugbank_drug")
    id_to_name = {row[0]: row[1] for row in cursor.fetchall()}
    # Load English synonyms
    cursor.execute(
        "SELECT synonym, drugbank_id FROM drugbank_drug_syn "
        "WHERE language LIKE '%english%'"
    )
    synonyms_map = {}
    for syn, drug_id in cursor.fetchall():
        primary = id_to_name.get(drug_id)
        if primary:
            synonyms_map[syn.lower()] = primary
    # Include primary names as their own synonym
    for primary in id_to_name.values():
        synonyms_map[primary.lower()] = primary
    conn.close()
    return synonyms_map

def group_passages_by_drug(input_dir, synonyms_map):
    """
    Iterate over text files in input_dir, match drug synonyms, and group passage filenames.
    Returns a dict mapping primary drug names to sets of filenames.
    """
    result = defaultdict(set)
    # Prepare file list and log progress
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    total = len(files)
    logging.info(f"Found {total} passage files in '{input_dir}' to process")
    for idx, fname in enumerate(files, start=1):
        if idx == 1 or idx % 100 == 0:
            logging.info(f"Processing file {idx}/{total}: {fname}")
        fpath = os.path.join(input_dir, fname)
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                text = f.read().lower()
        except Exception:
            continue
        # Match each synonym in the text
        for syn_lower, primary in synonyms_map.items():
            # Match whole-word synonyms in lower-case text
            pattern = r'\b' + re.escape(syn_lower) + r'\b'
            if re.search(pattern, text):
                result[primary].add(fname)
    # Convert sets to sorted lists
    return {drug: sorted(files) for drug, files in result.items()}

def main():
    script_dir = os.path.dirname(__file__)
    default_input = os.path.abspath(
        os.path.join(script_dir, '..', 'data', 'Pharma', 'pubmed-targets')
    )
    default_db = os.path.abspath(
        os.path.join(script_dir, '..', 'data', 'drugbank.db')
    )
    parser = argparse.ArgumentParser(
        description='Group PubMed target passages by drug mentions.'
    )
    parser.add_argument(
        '--input-dir', type=str, default=default_input,
        help='Directory containing PubMed passage files.'
    )
    parser.add_argument(
        '--db', type=str, default=default_db,
        help='Path to DrugBank SQLite database.'
    )
    parser.add_argument(
        '--output-file', type=str, default='grouped_passages_by_drug.json',
        help='Output JSON file for grouping results.'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Enable debug logging'
    )
    args = parser.parse_args()
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s %(levelname)s: %(message)s'
    )

    synonyms_map = load_drug_names(args.db)
    logging.info(f"Loaded {len(synonyms_map)} synonyms mapping to {len(set(synonyms_map.values()))} primary drugs")
    grouped = group_passages_by_drug(args.input_dir, synonyms_map)

    with open(args.output_file, 'w', encoding='utf-8') as out_file:
        json.dump(grouped, out_file, ensure_ascii=False, indent=2)
    logging.info(f"Saved grouped passages by drug to {args.output_file}")

if __name__ == '__main__':
    main()