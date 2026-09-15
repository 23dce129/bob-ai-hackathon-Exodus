"""
generate_dossier_fixtures.py — One-time script to create test fixture files
from the synthetic dossier outline JSON.

Run from src/backend/:
    .venv/Scripts/python.exe generate_dossier_fixtures.py

Creates:
    src/tests/fixtures/dossier_outline_clean.csv    — full 75-row outline
    src/tests/fixtures/dossier_outline_clean.xlsx   — same data as XLSX
    src/tests/fixtures/dossier_outline_missing_cols.csv  — no section_number col
    src/tests/fixtures/dossier_outline_blank_numbers.csv — some blank section_numbers
    src/tests/fixtures/dossier_outline_duplicates.csv    — duplicate section_numbers
    src/tests/fixtures/dossier_outline_semicolon.csv     — semicolon-delimited
    src/tests/fixtures/dossier_outline_tsv.csv           — tab-delimited .csv
    src/tests/fixtures/dossier_outline_latin1.csv        — Latin-1 encoded
    src/tests/fixtures/dossier_outline_alt_headers.csv   — alternate column aliases
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent.parent
DOSSIER_JSON = REPO_ROOT / "demo" / "data" / "synthetic_dossier_outline.json"
FIXTURES_DIR = REPO_ROOT / "src" / "tests" / "fixtures"
FIXTURES_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Load source data
# ---------------------------------------------------------------------------
raw = json.loads(DOSSIER_JSON.read_text(encoding="utf-8"))
sections = raw["sections"]

# Build the canonical DataFrame (canonical column names)
rows = []
for s in sections:
    rows.append({
        "section_number": s["section_id"],
        "section_title":  s["title"],
        "description":    s.get("notes", ""),
        "source":         s.get("document_ref", ""),
        "status":         s["status"],
    })

df_clean = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# 1. Clean CSV (comma, UTF-8)
# ---------------------------------------------------------------------------
df_clean.to_csv(FIXTURES_DIR / "dossier_outline_clean.csv", index=False, encoding="utf-8")
print(f"Written: dossier_outline_clean.csv  ({len(df_clean)} rows)")

# ---------------------------------------------------------------------------
# 2. Clean XLSX
# ---------------------------------------------------------------------------
df_clean.to_excel(FIXTURES_DIR / "dossier_outline_clean.xlsx", index=False, engine="openpyxl")
print(f"Written: dossier_outline_clean.xlsx ({len(df_clean)} rows)")

# ---------------------------------------------------------------------------
# 3. Missing both required columns (has only description + status)
# ---------------------------------------------------------------------------
df_missing = df_clean[["description", "status"]].copy()
df_missing.to_csv(FIXTURES_DIR / "dossier_outline_missing_cols.csv", index=False)
print("Written: dossier_outline_missing_cols.csv")

# ---------------------------------------------------------------------------
# 4. Some rows have blank section_number
# ---------------------------------------------------------------------------
df_blanks = df_clean.copy()
df_blanks.loc[[2, 5, 10], "section_number"] = ""  # rows 3, 6, 11 (1-based)
df_blanks.to_csv(FIXTURES_DIR / "dossier_outline_blank_numbers.csv", index=False)
print("Written: dossier_outline_blank_numbers.csv (3 blank section_numbers)")

# ---------------------------------------------------------------------------
# 5. Duplicate section_number (rows 0 and 1 both claim "1.1")
# ---------------------------------------------------------------------------
df_dupes = df_clean.copy()
df_dupes.loc[1, "section_number"] = df_dupes.loc[0, "section_number"]  # duplicate 1.1
df_dupes.to_csv(FIXTURES_DIR / "dossier_outline_duplicates.csv", index=False)
print("Written: dossier_outline_duplicates.csv (1 duplicate section_number)")

# ---------------------------------------------------------------------------
# 6. Semicolon-delimited CSV
# ---------------------------------------------------------------------------
df_clean.to_csv(FIXTURES_DIR / "dossier_outline_semicolon.csv", index=False, sep=";")
print("Written: dossier_outline_semicolon.csv (semicolon delimiter)")

# ---------------------------------------------------------------------------
# 7. Tab-delimited (saved as .csv so the parser must detect it)
# ---------------------------------------------------------------------------
df_clean.to_csv(FIXTURES_DIR / "dossier_outline_tsv.csv", index=False, sep="\t")
print("Written: dossier_outline_tsv.csv (tab delimiter)")

# ---------------------------------------------------------------------------
# 8. Latin-1 encoded CSV (with a non-ASCII character in one description)
#    Strip any characters outside Latin-1 range from all string cells first,
#    then inject a Latin-1 accented string to prove the encoding is honoured.
# ---------------------------------------------------------------------------
def _to_latin1_safe(val: object) -> object:
    """Replace any character outside the Latin-1 range (U+0100+) with '?'."""
    if not isinstance(val, str):
        return val
    return val.encode("latin-1", errors="replace").decode("latin-1")

df_latin1 = df_clean.map(_to_latin1_safe)
df_latin1.loc[0, "description"] = "Toc v\xe9rifi\xe9 avec accents"  # é = \xe9
df_latin1.to_csv(
    FIXTURES_DIR / "dossier_outline_latin1.csv",
    index=False,
    encoding="latin-1",
)
print("Written: dossier_outline_latin1.csv (Latin-1 encoding)")

# ---------------------------------------------------------------------------
# 9. Alternate header aliases
#    Use: "id", "name", "notes", "file", "state" instead of canonical names
# ---------------------------------------------------------------------------
df_alt = df_clean.rename(columns={
    "section_number": "id",
    "section_title":  "name",
    "description":    "notes",
    "source":         "file",
    "status":         "state",
})
df_alt.to_csv(FIXTURES_DIR / "dossier_outline_alt_headers.csv", index=False)
print("Written: dossier_outline_alt_headers.csv (alternate column aliases)")

# ---------------------------------------------------------------------------
# 10. Only section_number column (no section_title) — valid: title will be ""
# ---------------------------------------------------------------------------
df_number_only = df_clean[["section_number", "status"]].copy()
df_number_only.to_csv(FIXTURES_DIR / "dossier_outline_number_only.csv", index=False)
print("Written: dossier_outline_number_only.csv (no section_title column)")

# ---------------------------------------------------------------------------
# 11. Only section_title column (no section_number) — valid: number will be ""
# ---------------------------------------------------------------------------
df_title_only = df_clean[["section_title", "description"]].copy()
df_title_only.to_csv(FIXTURES_DIR / "dossier_outline_title_only.csv", index=False)
print("Written: dossier_outline_title_only.csv (no section_number column)")

print(f"\nAll fixtures written to: {FIXTURES_DIR}")
