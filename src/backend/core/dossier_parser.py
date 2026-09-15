"""
dossier_parser.py — Ingest regulatory dossier outlines from CSV or XLSX files.

This module accepts a user-supplied dossier outline (a table listing which CTD
sections are present in a submission) and normalises each row into a
``DossierRow`` dataclass.

No regulatory decisions are made here.
The parser does NOT consult the CTD knowledge base, does NOT classify sections
as required/conditional, and does NOT produce gap reports.  Those concerns
belong to readiness_scorer.py (next phase).

Supported formats
-----------------
* CSV  — any encoding (UTF-8 / UTF-8-BOM / Latin-1); comma, semicolon, or tab
         delimiter auto-detected.
* XLSX — single-sheet Excel workbooks (.xlsx only; .xls not supported).

Required columns (case-insensitive, multiple aliases accepted)
--------------------------------------------------------------
+------------------+-----------------------------------------------------------+
| Internal name    | Accepted header aliases                                   |
+------------------+-----------------------------------------------------------+
| section_number   | section_number, section_id, section, number, id, ctd_id   |
| section_title    | section_title, title, name, section_name                  |
+------------------+-----------------------------------------------------------+

Optional columns (normalised if present, empty string if absent)
----------------------------------------------------------------
+------------------+-----------------------------------------------------------+
| Internal name    | Accepted header aliases                                   |
+------------------+-----------------------------------------------------------+
| description      | description, notes, note, details, summary                |
| source           | source, document_ref, doc_ref, file, filename, reference  |
| status           | status, state, submission_status                          |
+------------------+-----------------------------------------------------------+

Normalisation rules
-------------------
* section_number : str — stripped, lowercased, leading zeroes preserved.
* section_title  : str — stripped; internal whitespace collapsed to single space.
* description    : str — stripped.
* source         : str — stripped.
* status         : str — stripped, lowercased.  Empty string when not supplied.

Validation errors (collected, not raised immediately)
-----------------------------------------------------
Errors are accumulated in ``DossierParseResult.errors`` so the caller can see
all problems at once.  A ``DossierParseError`` is raised only if:
  - the file does not exist,
  - the file format is unsupported,
  - the file cannot be parsed at all, or
  - the required columns (section_number OR section_title) are both absent.

Row-level issues (blank section_number, duplicate section_number) produce
``DossierValidationError`` entries in the ``errors`` list but do not stop
parsing.

Public API
----------
    DossierRow             — normalised dataclass for a single section entry
    DossierParseResult     — parsing output: rows + errors + source metadata
    DossierParseError      — raised for file-level failures
    DossierValidationError — collected for row-level issues (not raised)
    parse_dossier(path)    — main entry point
    to_dataframe(result)   — convert DossierParseResult rows to pd.DataFrame
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column alias maps
# ---------------------------------------------------------------------------

# Each internal field → ordered list of header aliases to try (first match wins)
_COLUMN_ALIASES: dict[str, list[str]] = {
    "section_number": [
        "section_number", "section_id", "section_no", "section",
        "number", "id", "ctd_id", "ctd_section",
    ],
    "section_title": [
        "section_title", "title", "name", "section_name", "heading",
    ],
    "description": [
        "description", "notes", "note", "details", "summary", "text",
    ],
    "source": [
        "source", "document_ref", "doc_ref", "file", "filename",
        "reference", "document", "path",
    ],
    "status": [
        "status", "state", "submission_status", "section_status",
    ],
}

# Required internal fields — parsing fails if BOTH are absent
_REQUIRED_FIELDS = {"section_number", "section_title"}

# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DossierRow:
    """A single normalised dossier section entry.

    Attributes
    ----------
    section_number : str
        CTD section identifier, e.g. ``"2.7.4"``.  Empty string when the
        source row had no value (a validation error is also recorded).
    section_title : str
        Human-readable section title.  Empty string when absent.
    description : str
        Free-text notes or details supplied by the applicant.  Empty string
        when not provided.
    source : str
        Document reference or file path.  Empty string when not provided.
    status : str
        Normalised status string as supplied by the applicant
        (e.g. ``"present"``, ``"missing"``, ``"needs_review"``).
        Empty string when not provided — the parser does NOT impute a value.
    row_index : int
        1-based row number in the original file (header = row 0).
        Used for error messages.
    """

    section_number: str
    section_title: str
    description: str
    source: str
    status: str
    row_index: int

    def to_dict(self) -> dict:
        return {
            "section_number": self.section_number,
            "section_title": self.section_title,
            "description": self.description,
            "source": self.source,
            "status": self.status,
            "row_index": self.row_index,
        }


@dataclass
class DossierValidationError:
    """A row-level validation issue collected during parsing.

    These are non-fatal: they are accumulated in ``DossierParseResult.errors``
    but do not stop the parse.  The offending row is still included in
    ``DossierParseResult.rows`` (with the best available normalised values).

    Attributes
    ----------
    row_index : int
        1-based row number in the original file.
    field_name : str
        Internal field name that triggered the error.
    message : str
        Human-readable description of the problem.
    raw_value : str
        The raw string value (or empty string) that caused the issue.
    """

    row_index: int
    field_name: str
    message: str
    raw_value: str = ""

    def __str__(self) -> str:
        return f"Row {self.row_index} [{self.field_name}]: {self.message}"


@dataclass
class DossierParseResult:
    """Output of ``parse_dossier()``.

    Attributes
    ----------
    rows : list[DossierRow]
        All normalised section entries (including rows that produced
        validation errors).
    errors : list[DossierValidationError]
        Accumulated row-level validation issues.  Empty when the file is clean.
    source_path : str
        Absolute path of the parsed file.
    source_format : str
        ``"csv"`` or ``"xlsx"``.
    row_count : int
        Number of data rows read (excluding the header).
    column_map : dict[str, str]
        Maps internal field name → actual header found in the file.
        Only contains entries for fields that were found.
    """

    rows: list[DossierRow] = field(default_factory=list)
    errors: list[DossierValidationError] = field(default_factory=list)
    source_path: str = ""
    source_format: str = ""
    row_count: int = 0
    column_map: dict[str, str] = field(default_factory=dict)

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def error_count(self) -> int:
        return len(self.errors)

    def to_dict(self) -> dict:
        return {
            "source_path": self.source_path,
            "source_format": self.source_format,
            "row_count": self.row_count,
            "column_map": self.column_map,
            "parsed_rows": len(self.rows),
            "error_count": self.error_count,
            "errors": [str(e) for e in self.errors],
        }


# ---------------------------------------------------------------------------
# Exception for file-level failures
# ---------------------------------------------------------------------------

class DossierParseError(Exception):
    """Raised when the dossier file cannot be parsed at all.

    This is distinct from ``DossierValidationError`` (row-level issues):
    ``DossierParseError`` means nothing was returned.
    """


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_column(headers: list[str], field_name: str) -> str | None:
    """Return the first header that matches any alias for *field_name*, or None.

    Comparison is case-insensitive with surrounding whitespace stripped.
    """
    normalised_headers = {h.lower().strip(): h for h in headers}
    for alias in _COLUMN_ALIASES[field_name]:
        if alias in normalised_headers:
            return normalised_headers[alias]
    return None


def _build_column_map(headers: list[str]) -> dict[str, str]:
    """Map every known internal field to its actual header in the file."""
    col_map: dict[str, str] = {}
    for field_name in _COLUMN_ALIASES:
        found = _find_column(headers, field_name)
        if found is not None:
            col_map[field_name] = found
    return col_map


def _normalise_section_number(raw: object) -> str:
    """Strip and lowercase a section-number value.

    Returns empty string for null/NaN inputs.
    """
    if pd.isna(raw):
        return ""
    return str(raw).strip().lower()


def _normalise_title(raw: object) -> str:
    """Strip and collapse internal whitespace in a title string."""
    if pd.isna(raw):
        return ""
    return re.sub(r"\s+", " ", str(raw).strip())


def _normalise_text(raw: object) -> str:
    """Strip a free-text field; return empty string for null."""
    if pd.isna(raw):
        return ""
    return str(raw).strip()


def _normalise_status(raw: object) -> str:
    """Lowercase and strip a status field; return empty string for null."""
    if pd.isna(raw):
        return ""
    return str(raw).strip().lower()


# ---------------------------------------------------------------------------
# File readers
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> pd.DataFrame:
    """Try multiple encodings and delimiters; return first successful parse."""
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        for sep in (",", ";", "\t"):
            try:
                df = pd.read_csv(
                    path,
                    sep=sep,
                    encoding=encoding,
                    dtype=str,       # keep everything as strings; no type coercion
                    keep_default_na=False,  # do not convert "" to NaN here
                    na_values=["", "NA", "N/A", "NULL", "null", "nan", "NaN", "None"],
                )
                if df.shape[1] > 1:
                    logger.debug(
                        "CSV loaded: %d rows, encoding=%s, sep=%r",
                        len(df), encoding, sep,
                    )
                    return df
            except Exception:  # noqa: BLE001
                continue

    raise DossierParseError(
        f"Could not parse '{path.name}' as CSV with any supported "
        "encoding (UTF-8, UTF-8-BOM, Latin-1) or delimiter (comma, semicolon, tab)."
    )


def _read_xlsx(path: Path) -> pd.DataFrame:
    """Read the first sheet of an XLSX workbook."""
    try:
        df = pd.read_excel(
            path,
            sheet_name=0,
            dtype=str,
            keep_default_na=False,
            na_values=["", "NA", "N/A", "NULL", "null", "nan", "NaN", "None"],
            engine="openpyxl",
        )
        logger.debug("XLSX loaded: %d rows from '%s'", len(df), path.name)
        return df
    except Exception as exc:
        raise DossierParseError(
            f"Could not read XLSX file '{path.name}': {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Core parsing logic
# ---------------------------------------------------------------------------

def _parse_dataframe(
    df: pd.DataFrame,
    source_path: str,
    source_format: str,
) -> DossierParseResult:
    """Convert a raw DataFrame (from CSV or XLSX) into a DossierParseResult."""
    result = DossierParseResult(
        source_path=source_path,
        source_format=source_format,
        row_count=len(df),
    )

    # Normalise column headers once
    headers = list(df.columns)
    col_map = _build_column_map(headers)
    result.column_map = col_map

    # Validate that at least one required column is present
    missing_required = _REQUIRED_FIELDS - set(col_map.keys())
    if missing_required == _REQUIRED_FIELDS:
        # Both required fields are absent — cannot produce any useful rows
        raise DossierParseError(
            f"File '{Path(source_path).name}' is missing both required columns. "
            "Need at least one of: "
            f"{_COLUMN_ALIASES['section_number']} (for section_number) "
            f"or {_COLUMN_ALIASES['section_title']} (for section_title). "
            f"Headers found: {headers}"
        )

    # Track seen section numbers for duplicate detection
    seen_numbers: dict[str, int] = {}  # section_number → first row_index that set it

    for df_idx, row in df.iterrows():
        row_index = int(df_idx) + 1  # 1-based

        # ── section_number ────────────────────────────────────────────────────
        raw_number = row.get(col_map.get("section_number", ""), pd.NA)
        section_number = _normalise_section_number(raw_number)

        # Only flag blank section_number when the column was present in the file.
        # If the column is absent entirely, every row is blank by design — that is
        # not an error (the user simply did not supply section numbers).
        if not section_number and "section_number" in col_map:
            result.errors.append(DossierValidationError(
                row_index=row_index,
                field_name="section_number",
                message="section_number is blank or missing",
                raw_value=str(raw_number) if not pd.isna(raw_number) else "",
            ))

        # ── section_title ─────────────────────────────────────────────────────
        raw_title = row.get(col_map.get("section_title", ""), pd.NA)
        section_title = _normalise_title(raw_title)

        # ── duplicate section_number check ────────────────────────────────────
        if section_number and section_number in seen_numbers:
            result.errors.append(DossierValidationError(
                row_index=row_index,
                field_name="section_number",
                message=(
                    f"Duplicate section_number '{section_number}' — "
                    f"first seen at row {seen_numbers[section_number]}"
                ),
                raw_value=section_number,
            ))
        elif section_number:
            seen_numbers[section_number] = row_index

        # ── optional fields ───────────────────────────────────────────────────
        raw_desc = row.get(col_map.get("description", ""), pd.NA)
        raw_source = row.get(col_map.get("source", ""), pd.NA)
        raw_status = row.get(col_map.get("status", ""), pd.NA)

        result.rows.append(DossierRow(
            section_number=section_number,
            section_title=section_title,
            description=_normalise_text(raw_desc),
            source=_normalise_text(raw_source),
            status=_normalise_status(raw_status),
            row_index=row_index,
        ))

    logger.info(
        "parse_dossier: %d rows parsed from '%s' (%s), %d validation error(s)",
        len(result.rows), Path(source_path).name, source_format, result.error_count,
    )
    return result


# ---------------------------------------------------------------------------
# Helpers — export
# ---------------------------------------------------------------------------

def to_dataframe(result: DossierParseResult) -> pd.DataFrame:
    """Convert ``DossierParseResult.rows`` to a ``pd.DataFrame``.

    Returns an empty DataFrame with the correct columns when there are no rows.
    """
    if not result.rows:
        return pd.DataFrame(columns=[
            "section_number", "section_title", "description",
            "source", "status", "row_index",
        ])
    return pd.DataFrame([r.to_dict() for r in result.rows])


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def parse_dossier(path: Union[str, Path]) -> DossierParseResult:
    """Parse a dossier outline CSV or XLSX file into normalised ``DossierRow`` objects.

    Parameters
    ----------
    path : str or Path
        Path to a ``.csv`` or ``.xlsx`` file.

    Returns
    -------
    DossierParseResult
        ``rows``   — list of normalised ``DossierRow`` objects (one per data row).
        ``errors`` — list of ``DossierValidationError`` for row-level issues.
        ``column_map`` — maps internal field → actual header found.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    DossierParseError
        If the file format is unsupported, unreadable, or both required columns
        (section_number AND section_title) are absent.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dossier file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        raw_df = _read_csv(path)
        fmt = "csv"
    elif suffix in (".xlsx", ".xlsm"):
        raw_df = _read_xlsx(path)
        fmt = "xlsx"
    else:
        raise DossierParseError(
            f"Unsupported file format '{suffix}'. "
            "parse_dossier accepts .csv and .xlsx files only."
        )

    return _parse_dataframe(raw_df, str(path.resolve()), fmt)
