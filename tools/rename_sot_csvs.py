"""
tools/rename_sot_csvs.py
========================
One-time utility: rename existing customer SOT CSVs to match the atom
canonical_id that their formulas reference.

Run from the project root:
    python tools/rename_sot_csvs.py [--dry-run] [--customer CUSTOMER_ID]

How matching works
------------------
For each CSV file under customer_runtime/customers/<id>/sot_csv/:
  1. Load the customer's canonical_index.json to find every atom canonical_id
     actually referenced in compiled formulas.
  2. Score each candidate atom by:
       - String similarity between CSV stem and atom canonical_id (weight 0.70)
       - Column overlap between CSV headers and atom fields in atoms.json (0.30)
  3. Rename to the best match if it differs from the current filename.

Safe to re-run — skips files already named correctly and skips if the
target name already exists.
"""

import argparse
import csv
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_atoms() -> dict:
    path = PROJECT_ROOT / "data" / "atoms.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    atoms = raw.get("_default", raw) if isinstance(raw, dict) else {}
    return {
        a["canonical_id"]: {f["name"].lower() for f in a.get("fields", [])}
        for a in atoms.values()
        if "canonical_id" in a
    }


def referenced_atoms(idx: dict) -> set:
    refs = set()
    for entry in idx.values():
        formula = entry.get("formula_line", "")
        for m in re.finditer(r"MEASURE[A-Z_]*\(['\"](\w+)['\"]", formula):
            refs.add(m.group(1))
        for m in re.finditer(r"FROM\s+(\w+)", formula, re.IGNORECASE):
            refs.add(m.group(1))
    return refs


def best_match(csv_stem: str, csv_cols: set, candidates: set, atom_fields: dict) -> tuple:
    best_cid, best_score = None, -1.0
    for atom_cid in candidates:
        sim = SequenceMatcher(None, csv_stem, atom_cid).ratio()
        afields = atom_fields.get(atom_cid, set())
        col_overlap = len(csv_cols & afields) / max(len(afields), 1) if afields else 0.0
        score = sim * 0.70 + col_overlap * 0.30
        if score > best_score:
            best_score, best_cid = score, atom_cid
    return best_cid, best_score


def process_customer(cust_dir: Path, atom_fields: dict, dry_run: bool) -> None:
    idx_path = cust_dir / "knowledge_store" / "canonical_index.json"
    if not idx_path.exists():
        print(f"  [skip] No canonical_index.json: {cust_dir.name}")
        return

    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    refs = referenced_atoms(idx)
    if not refs:
        print(f"  [skip] No atom references found: {cust_dir.name}")
        return

    sot_root = cust_dir / "sot_csv"
    if not sot_root.exists():
        print(f"  [skip] No sot_csv dir: {cust_dir.name}")
        return

    for v_dir in sorted(sot_root.iterdir()):
        if not v_dir.is_dir():
            continue
        for csv_file in sorted(v_dir.glob("*.csv")):
            stem = csv_file.stem
            if stem in refs:
                print(f"  OK     {v_dir.name}/{csv_file.name}")
                continue

            with open(csv_file, newline="", encoding="utf-8-sig") as f:
                cols = {c.lower().strip() for c in (csv.DictReader(f).fieldnames or [])}

            cid, score = best_match(stem, cols, refs, atom_fields)
            if not cid:
                print(f"  NOMATCH {v_dir.name}/{csv_file.name}")
                continue

            target = v_dir / f"{cid}.csv"
            if target.exists():
                print(f"  SKIP   {v_dir.name}/{csv_file.name} → {cid}.csv (target exists)")
                continue

            action = "DRY-RUN" if dry_run else "RENAME"
            print(f"  {action} {v_dir.name}/{csv_file.name} → {cid}.csv  (score={score:.2f})")
            if not dry_run:
                csv_file.rename(target)


def main():
    parser = argparse.ArgumentParser(description="Rename SOT CSVs to atom canonical_id names")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing")
    parser.add_argument("--customer", help="Process only this customer_id")
    args = parser.parse_args()

    atom_fields = load_atoms()
    customers_dir = PROJECT_ROOT / "customer_runtime" / "customers"

    if not customers_dir.exists():
        print(f"No customers dir found at {customers_dir}")
        sys.exit(1)

    for cust_dir in sorted(customers_dir.iterdir()):
        if not cust_dir.is_dir():
            continue
        if args.customer and cust_dir.name != args.customer:
            continue
        print(f"\n{cust_dir.name}:")
        process_customer(cust_dir, atom_fields, dry_run=args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
