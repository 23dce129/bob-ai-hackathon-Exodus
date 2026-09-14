"""
faers_ingestor.py — Load, validate, and normalise FDA FAERS adverse-event data.

Public API
----------
    load_faers(path)          → pd.DataFrame  (main entry point)
    to_parquet(df, path)      → None
    load_parquet(path)        → pd.DataFrame
    to_duckdb(df, db_path)    → None

Pipeline (called internally by load_faers)
------------------------------------------
    load_faers_csv(path)      → raw DataFrame
    validate_schema(df)       → None  (raises ValueError on bad input)
    normalize_dataframe(df)   → clean DataFrame

Column mapping
--------------
The FDA FAERS quarterly ASCII files use these column names (vary slightly
across quarters).  All are mapped to the project's internal schema defined
in schemas.REQUIRED_INTERNAL_COLUMNS.

    Raw FAERS name(s)          Internal name   Notes
    ─────────────────────────────────────────────────────────────────────
    primaryid                  report_id
    drugname | drug_name       drug_name        normalised (see below)
    pt                         adverse_event    MedDRA preferred term
    outc_cod                   outcome_code     DE/HO/LT/DS/CA/RI/OT
    event_dt | fda_dt          report_date      datetime, NaT if invalid
    age                        age_years        float, NaN if missing
    sex                        sex              M/F/UNK

Drug-name normalisation
-----------------------
    1. Lowercase and strip surrounding whitespace.
    2. Remove common salt/formulation suffixes:
       hydrochloride, hcl, sodium, potassium, calcium, acetate,
       sulfate, sulphate, tartrate, maleate, fumarate, mesylate,
       phosphate, bromide, citrate, lactate, gluconate
    3. Collapse internal whitespace.

Adverse-event normalisation
----------------------------
    Lowercase and strip only.  MedDRA preferred terms are controlled
    vocabulary; aggressive normalisation would corrupt them.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Union

import duckdb
import pandas as pd

from core.schemas import REQUIRED_INTERNAL_COLUMNS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column aliases — maps every known FAERS variant to our internal name
# ---------------------------------------------------------------------------
_COLUMN_ALIASES: dict[str, str] = {
    "primaryid":   "report_id",
    "caseid":      "report_id",      # fallback if primaryid absent
    "drugname":    "drug_name",
    "drug_name":   "drug_name",
    "pt":          "adverse_event",
    "outc_cod":    "outcome_code",
    "event_dt":    "report_date",
    "fda_dt":      "report_date",
    "rept_dt":     "report_date",
    "age":         "age_years",
    "age_in_years":"age_years",
    "sex":         "sex",
    "gndr_cod":    "sex",
}

# Salt/formulation suffixes to strip from drug names.
# The pattern strips one suffix per call; apply repeatedly for stacked suffixes
# (e.g. "metformin hydrochloride sodium" would need two passes — rare in practice).
_SALT_PATTERN = re.compile(
    r"\b(hydrochloride|hydro chloride|hcl|sodium|potassium|calcium|"
    r"acetate|sulfate|sulphate|tartrate|maleate|fumarate|mesylate|"
    r"phosphate|bromide|citrate|lactate|gluconate|monohydrate|dihydrate|"
    r"anhydrous|trihydrate)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def normalize_drug_name(name: str) -> str:
    """
    Return a normalised drug name suitable for grouping and comparison.

    Steps:
        1. Coerce to str (handles float NaN passed from pandas).
        2. Lowercase + strip.
        3. Remove salt/formulation suffixes.
        4. Collapse runs of whitespace.

    Examples
    --------
    >>> normalize_drug_name("WARFARIN SODIUM ")
    'warfarin'
    >>> normalize_drug_name("Ibuprofen Hydrochloride")
    'ibuprofen'
    >>> normalize_drug_name("metformin hcl")
    'metformin'
    """
    if pd.isna(name):
        return ""
    clean = str(name).lower().strip()
    clean = _SALT_PATTERN.sub("", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def normalize_adverse_event(term: str) -> str:
    """
    Return a normalised adverse-event term.

    MedDRA preferred terms are controlled vocabulary, so only lowercase
    and strip — do not remove any words.

    Examples
    --------
    >>> normalize_adverse_event("  Nausea  ")
    'nausea'
    >>> normalize_adverse_event("DRUG-INDUCED LIVER INJURY")
    'drug-induced liver injury'
    """
    if pd.isna(term):
        return ""
    return str(term).lower().strip()


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_faers_csv(path: Union[str, Path]) -> pd.DataFrame:
    """
    Read a raw FAERS quarterly CSV into a DataFrame.

    Handles:
    - UTF-8 and Latin-1 encoding (FAERS uses Latin-1 in older quarters).
    - Comma and tab delimiters (FAERS ASCII files are dollar-separated in
      some quarters; we fall back to comma).
    - Columns are lowercased immediately so alias mapping is case-insensitive.

    Raises
    ------
    FileNotFoundError
        If `path` does not exist.
    ValueError
        If the file cannot be parsed as CSV with any supported delimiter.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FAERS data file not found: {path}")

    for encoding in ("utf-8", "latin-1"):
        for sep in (",", "\t", "$"):
            try:
                df = pd.read_csv(path, sep=sep, encoding=encoding, low_memory=False)
                if df.shape[1] > 1:
                    df.columns = [c.lower().strip() for c in df.columns]
                    logger.debug(
                        "Loaded %d rows from %s (encoding=%s, sep=%r)",
                        len(df), path, encoding, sep,
                    )
                    return df
            except Exception:  # noqa: BLE001
                continue

    raise ValueError(f"Could not parse {path} as CSV with any supported encoding/delimiter.")


def validate_schema(df: pd.DataFrame) -> None:
    """
    Check that the DataFrame contains at least one column that maps to each
    required internal column.  Raises ValueError listing every missing column.

    Does not modify the DataFrame.
    """
    # Which internal columns can be satisfied by columns present in df?
    present = set(df.columns)
    # Build reverse map: internal_name → list of aliases that satisfy it
    satisfiable: set[str] = set()
    for raw_col, internal_col in _COLUMN_ALIASES.items():
        if raw_col in present:
            satisfiable.add(internal_col)

    missing = [c for c in REQUIRED_INTERNAL_COLUMNS if c not in satisfiable]
    if missing:
        raise ValueError(
            f"FAERS data is missing required columns: {missing}. "
            f"Columns found: {sorted(present)}"
        )


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map raw FAERS column names to internal names, normalise values,
    and return a clean DataFrame with exactly REQUIRED_INTERNAL_COLUMNS.

    Rows where both drug_name and adverse_event are empty after normalisation
    are dropped (they carry no information for signal detection).
    Rows where only one of the two is empty are also dropped.

    Warns (does not raise) if more than 20 % of rows are dropped.
    """
    out = pd.DataFrame()

    # ── report_id ────────────────────────────────────────────────────────────
    if "primaryid" in df.columns:
        out["report_id"] = df["primaryid"].astype(str)
    elif "caseid" in df.columns:
        out["report_id"] = df["caseid"].astype(str)
    else:
        out["report_id"] = pd.RangeIndex(len(df)).astype(str)

    # ── drug_name ─────────────────────────────────────────────────────────────
    drug_col = next((c for c in ("drug_name", "drugname") if c in df.columns), None)
    if drug_col:
        out["drug_name"] = df[drug_col].apply(normalize_drug_name)

    # ── adverse_event ─────────────────────────────────────────────────────────
    if "pt" in df.columns:
        out["adverse_event"] = df["pt"].apply(normalize_adverse_event)

    # ── outcome_code ──────────────────────────────────────────────────────────
    if "outc_cod" in df.columns:
        out["outcome_code"] = df["outc_cod"].astype(str).str.upper().str.strip()
    else:
        out["outcome_code"] = "OT"

    # ── report_date ───────────────────────────────────────────────────────────
    date_col = next(
        (c for c in ("event_dt", "fda_dt", "rept_dt") if c in df.columns), None
    )
    if date_col:
        out["report_date"] = pd.to_datetime(
            df[date_col], format="%Y%m%d", errors="coerce"
        )
    else:
        out["report_date"] = pd.NaT

    # ── age_years ─────────────────────────────────────────────────────────────
    age_col = next((c for c in ("age", "age_in_years", "age_years") if c in df.columns), None)
    if age_col:
        out["age_years"] = pd.to_numeric(df[age_col], errors="coerce")
    else:
        out["age_years"] = float("nan")

    # ── sex ───────────────────────────────────────────────────────────────────
    sex_col = next((c for c in ("sex", "gndr_cod") if c in df.columns), None)
    if sex_col:
        out["sex"] = (
            df[sex_col]
            .astype(str)
            .str.upper()
            .str.strip()
            .replace({"M": "M", "F": "F", "MALE": "M", "FEMALE": "F", "UNK": "UNK",
                       "NS": "UNK", "NAN": "UNK", "NONE": "UNK", "": "UNK"})
        )
        out["sex"] = out["sex"].where(out["sex"].isin(["M", "F", "UNK"]), "UNK")
    else:
        out["sex"] = "UNK"

    # ── Drop unusable rows ────────────────────────────────────────────────────
    original_len = len(out)
    out = out[
        (out["drug_name"].str.len() > 0) &
        (out["adverse_event"].str.len() > 0)
    ].copy()

    dropped = original_len - len(out)
    if original_len > 0 and dropped / original_len > 0.20:
        logger.warning(
            "%.0f%% of rows were dropped during normalisation (%d of %d). "
            "Check that the input file matches the expected FAERS format.",
            100 * dropped / original_len, dropped, original_len,
        )

    out = out.reset_index(drop=True)
    return out[REQUIRED_INTERNAL_COLUMNS]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def to_parquet(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """
    Save a normalised FAERS DataFrame as a Parquet file.

    Parquet uses columnar compression and is significantly faster to reload
    than CSV for large datasets.  This is the recommended format for
    storing processed FAERS data between runs.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")
    logger.info("Saved %d rows to %s", len(df), path)


def load_parquet(path: Union[str, Path]) -> pd.DataFrame:
    """Load a previously saved Parquet file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    return pd.read_parquet(path, engine="pyarrow")


def to_duckdb(
    df: pd.DataFrame,
    db_path: Union[str, Path],
    table_name: str = "faers_reports",
) -> None:
    """
    Write (or replace) a table in a DuckDB database file.

    Creates the database file if it does not exist.  If the table already
    exists it is replaced, so this function is safe to call multiple times
    with updated data.

    DuckDB stores the data as a persistent columnar database, enabling fast
    SQL aggregations in later phases (PRR batch computation, trend queries).
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    try:
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.execute(
            f"CREATE TABLE {table_name} AS SELECT * FROM df"
        )
        count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        logger.info("Wrote %d rows to DuckDB table '%s' at %s", count, table_name, db_path)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_faers(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a FAERS CSV file and return a clean, normalised DataFrame.

    This is the single public entry point for the ingestion pipeline.
    It calls load_faers_csv → validate_schema → normalize_dataframe in sequence.

    Parameters
    ----------
    path : str or Path
        Path to the raw FAERS quarterly CSV file.

    Returns
    -------
    pd.DataFrame
        Columns: report_id, drug_name, adverse_event, outcome_code,
                 report_date, age_years, sex.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If required columns are missing after schema validation.
    """
    raw = load_faers_csv(path)
    validate_schema(raw)
    clean = normalize_dataframe(raw)
    logger.info(
        "load_faers: %d rows loaded from %s → %d rows after normalisation",
        len(raw), path, len(clean),
    )
    return clean
