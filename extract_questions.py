#!/usr/bin/env python3
"""Extract text-like cells from an Excel file and write them to a plain text file.

Usage:
  python extract_questions.py input.xlsx output.txt

Behavior:
  - Reads all sheets from the Excel workbook.
  - Collects non-empty string-like cell values in row-major order.
  - Deduplicates while preserving first-seen order.
  - Writes one question per line to the output UTF-8 text file.

This script is intentionally permissive about which cells count as "questions" to
work reasonably for a variety of spreadsheet layouts.
"""
import sys
from pathlib import Path

def extract_texts_from_excel(path):
    try:
        import pandas as pd
    except Exception as e:
        raise RuntimeError("pandas is required to run this script. Install with: pip install pandas openpyxl") from e

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")

    # Read all sheets without inferring a header so we see all cells
    sheets = pd.read_excel(p, sheet_name=None, header=None, dtype=object, engine="openpyxl")

    seen = set()
    items = []

    def normalize(s):
        if s is None:
            return None
        if isinstance(s, float) and pd.isna(s):
            return None
        # Convert numbers to string as well
        s = str(s)
        s = s.strip()
        if not s:
            return None
        return s

    for sheet_name, df in sheets.items():
        # iterate row-major
        for row in df.itertuples(index=False):
            for cell in row:
                txt = normalize(cell)
                if not txt:
                    continue
                # Heuristic: skip purely numeric short values (e.g., IDs)
                if txt.isdigit() and len(txt) < 4:
                    continue
                if txt in seen:
                    continue
                seen.add(txt)
                items.append(txt)

    return items


def write_txt(items, outpath):
    p = Path(outpath)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Write with UTF-8
    with p.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(it + "\n")


def main(argv):
    if len(argv) < 3:
        print("Usage: python extract_questions.py input.xlsx output.txt")
        return 2

    input_path = argv[1]
    output_path = argv[2]

    items = extract_texts_from_excel(input_path)
    write_txt(items, output_path)
    print(f"Wrote {len(items)} entries to {output_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
