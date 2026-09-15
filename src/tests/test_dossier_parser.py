"""
test_dossier_parser.py — Unit tests for the dossier outline ingestion pipeline.

Test strategy
-------------
Tests are grouped into six sections:

  1. Happy-path CSV    — clean file, all columns present; assert shape, types,
                         normalisation, and exact values from the synthetic dossier.
  2. Happy-path XLSX   — same assertions on the Excel version of the same data.
  3. Format variants   — semicolon-, tab-, and Latin-1-encoded CSVs all parse to
                         identical row count and content.
  4. Column aliases    — alternate header names (id, name, notes, file, state)
                         resolve to the correct internal fields.
  5. Partial columns   — number-only and title-only files parse without error;
                         the absent field is an empty string on every row.
  6. Validation errors — blank section_numbers, duplicate section_numbers, and
                         missing-both-required-columns all produce the correct
                         errors or raise the correct exception.
  7. Error handling    — non-existent file, unsupported extension, empty file,
                         header-only file.
  8. to_dataframe()    — output DataFrame has correct dtypes and shape.
  9. Integration       — synthetic dossier content spot-checks (specific IDs,
                         statuses, and notes from the demo dataset).

Key invariants asserted throughout
------------------------------------
  - Every DossierRow has exactly the five string fields + row_index (int).
  - section_number is always lowercase and stripped.
  - status is always lowercase and stripped.
  - description and source are stripped (never None).
  - row_index is 1-based.
  - DossierParseError is raised for file-level failures only.
  - DossierValidationError entries are collected, not raised.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from core.dossier_parser import (
    DossierParseError,
    DossierParseResult,
    DossierRow,
    DossierValidationError,
    parse_dossier,
    to_dataframe,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures"
DOSSIER_JSON = Path(__file__).parent.parent.parent / "demo" / "data" / "synthetic_dossier_outline.json"

# Load expected values from the canonical JSON once
_RAW_DOSSIER = json.loads(DOSSIER_JSON.read_text(encoding="utf-8"))
_EXPECTED_SECTIONS = _RAW_DOSSIER["sections"]
_EXPECTED_COUNT = len(_EXPECTED_SECTIONS)  # 75


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section_by_id(result: DossierParseResult, section_id: str) -> DossierRow | None:
    """Return the first DossierRow whose section_number matches section_id."""
    for row in result.rows:
        if row.section_number == section_id.lower():
            return row
    return None


# ===========================================================================
# Section 1 — Happy-path CSV
# ===========================================================================

class TestCleanCSV:
    """Parse the clean comma-delimited CSV and verify structure and content."""

    @pytest.fixture(scope="class")
    @classmethod
    def result(cls) -> DossierParseResult:
        return parse_dossier(FIXTURES / "dossier_outline_clean.csv")

    def test_returns_parse_result(self, result):
        assert isinstance(result, DossierParseResult)

    def test_source_format_is_csv(self, result):
        assert result.source_format == "csv"

    def test_row_count(self, result):
        assert result.row_count == _EXPECTED_COUNT

    def test_rows_length(self, result):
        assert len(result.rows) == _EXPECTED_COUNT

    def test_no_validation_errors(self, result):
        assert result.has_errors is False
        assert result.error_count == 0
        assert result.errors == []

    def test_every_row_is_dossier_row(self, result):
        for row in result.rows:
            assert isinstance(row, DossierRow)

    def test_every_row_index_is_positive_int(self, result):
        for row in result.rows:
            assert isinstance(row.row_index, int)
            assert row.row_index >= 1

    def test_row_indices_are_sequential(self, result):
        indices = [r.row_index for r in result.rows]
        assert indices == list(range(1, _EXPECTED_COUNT + 1))

    def test_every_field_is_string(self, result):
        for row in result.rows:
            assert isinstance(row.section_number, str)
            assert isinstance(row.section_title, str)
            assert isinstance(row.description, str)
            assert isinstance(row.source, str)
            assert isinstance(row.status, str)

    def test_section_numbers_are_lowercase(self, result):
        for row in result.rows:
            assert row.section_number == row.section_number.lower()

    def test_status_values_are_lowercase(self, result):
        for row in result.rows:
            assert row.status == row.status.lower()

    def test_no_leading_trailing_whitespace_in_any_field(self, result):
        for row in result.rows:
            for val in (row.section_number, row.section_title, row.description,
                        row.source, row.status):
                assert val == val.strip()

    def test_column_map_contains_all_five_fields(self, result):
        assert set(result.column_map.keys()) == {
            "section_number", "section_title", "description", "source", "status"
        }

    # --- Spot-check specific section data against the JSON source ---

    def test_section_1_1_present(self, result):
        row = _section_by_id(result, "1.1")
        assert row is not None
        assert "contents" in row.section_title.lower()
        assert row.status == "present"

    def test_section_1_8_missing(self, result):
        row = _section_by_id(result, "1.8")
        assert row is not None
        assert row.status == "missing"
        assert "pharmacovigilance" in row.section_title.lower()

    def test_section_2_7_4_missing(self, result):
        row = _section_by_id(result, "2.7.4")
        assert row is not None
        assert row.status == "missing"

    def test_section_2_5_needs_review(self, result):
        row = _section_by_id(result, "2.5")
        assert row is not None
        assert row.status == "needs_review"

    def test_section_5_3_5_needs_review(self, result):
        row = _section_by_id(result, "5.3.5")
        assert row is not None
        assert row.status == "needs_review"

    def test_section_4_2_3_5_missing(self, result):
        row = _section_by_id(result, "4.2.3.5")
        assert row is not None
        assert row.status == "missing"

    def test_section_4_2_3_6_not_applicable(self, result):
        row = _section_by_id(result, "4.2.3.6")
        assert row is not None
        assert row.status == "not_applicable"

    def test_section_3_2_s_2_present_with_source(self, result):
        row = _section_by_id(result, "3.2.s.2")
        assert row is not None
        assert row.status == "present"
        assert row.source != ""

    def test_missing_section_has_notes_in_description(self, result):
        row = _section_by_id(result, "2.7.4")
        assert row is not None
        # The JSON notes field for 2.7.4 starts with "MISSING —"
        # .upper() is used so the check works regardless of case
        assert "MISSING" in row.description.upper()

    def test_all_expected_section_numbers_present(self, result):
        """Every section_id from the JSON appears in parsed rows."""
        found = {r.section_number for r in result.rows}
        for s in _EXPECTED_SECTIONS:
            assert s["section_id"].lower() in found

    def test_status_values_are_known_set(self, result):
        known = {"present", "missing", "needs_review", "waived", "not_applicable", ""}
        for row in result.rows:
            assert row.status in known, f"Unexpected status {row.status!r} at row {row.row_index}"

    def test_to_dict_returns_correct_keys(self, result):
        d = result.rows[0].to_dict()
        assert set(d.keys()) == {
            "section_number", "section_title", "description",
            "source", "status", "row_index",
        }

    def test_parse_result_to_dict(self, result):
        d = result.to_dict()
        assert d["source_format"] == "csv"
        assert d["row_count"] == _EXPECTED_COUNT
        assert d["parsed_rows"] == _EXPECTED_COUNT
        assert d["error_count"] == 0


# ===========================================================================
# Section 2 — Happy-path XLSX
# ===========================================================================

class TestCleanXLSX:
    """Same data as the clean CSV; verify XLSX parsing produces identical output."""

    @pytest.fixture(scope="class")
    @classmethod
    def result(cls) -> DossierParseResult:
        return parse_dossier(FIXTURES / "dossier_outline_clean.xlsx")

    def test_source_format_is_xlsx(self, result):
        assert result.source_format == "xlsx"

    def test_row_count(self, result):
        assert result.row_count == _EXPECTED_COUNT

    def test_rows_length(self, result):
        assert len(result.rows) == _EXPECTED_COUNT

    def test_no_validation_errors(self, result):
        assert result.has_errors is False

    def test_section_1_8_missing(self, result):
        row = _section_by_id(result, "1.8")
        assert row is not None
        assert row.status == "missing"

    def test_section_2_3_present(self, result):
        row = _section_by_id(result, "2.3")
        assert row is not None
        assert row.status == "present"

    def test_column_map_has_five_fields(self, result):
        assert len(result.column_map) == 5

    def test_section_numbers_lowercase(self, result):
        for row in result.rows:
            assert row.section_number == row.section_number.lower()

    def test_xlsx_and_csv_produce_same_section_numbers(self):
        csv_result = parse_dossier(FIXTURES / "dossier_outline_clean.csv")
        xlsx_result = parse_dossier(FIXTURES / "dossier_outline_clean.xlsx")
        csv_nums = [r.section_number for r in csv_result.rows]
        xlsx_nums = [r.section_number for r in xlsx_result.rows]
        assert csv_nums == xlsx_nums

    def test_xlsx_and_csv_produce_same_statuses(self):
        csv_result = parse_dossier(FIXTURES / "dossier_outline_clean.csv")
        xlsx_result = parse_dossier(FIXTURES / "dossier_outline_clean.xlsx")
        csv_statuses = [r.status for r in csv_result.rows]
        xlsx_statuses = [r.status for r in xlsx_result.rows]
        assert csv_statuses == xlsx_statuses


# ===========================================================================
# Section 3 — Format variants
# ===========================================================================

class TestFormatVariants:
    """Semicolon, tab, and Latin-1 delimited/encoded CSVs all parse correctly."""

    def test_semicolon_csv_row_count(self):
        result = parse_dossier(FIXTURES / "dossier_outline_semicolon.csv")
        assert result.row_count == _EXPECTED_COUNT
        assert result.has_errors is False

    def test_semicolon_csv_section_1_8_missing(self):
        result = parse_dossier(FIXTURES / "dossier_outline_semicolon.csv")
        row = _section_by_id(result, "1.8")
        assert row is not None and row.status == "missing"

    def test_tsv_csv_row_count(self):
        result = parse_dossier(FIXTURES / "dossier_outline_tsv.csv")
        assert result.row_count == _EXPECTED_COUNT
        assert result.has_errors is False

    def test_tsv_csv_section_numbers_match_clean(self):
        clean = parse_dossier(FIXTURES / "dossier_outline_clean.csv")
        tsv = parse_dossier(FIXTURES / "dossier_outline_tsv.csv")
        assert [r.section_number for r in clean.rows] == [r.section_number for r in tsv.rows]

    def test_latin1_csv_row_count(self):
        result = parse_dossier(FIXTURES / "dossier_outline_latin1.csv")
        assert result.row_count == _EXPECTED_COUNT
        assert result.has_errors is False

    def test_latin1_csv_first_row_description_contains_accent(self):
        result = parse_dossier(FIXTURES / "dossier_outline_latin1.csv")
        # First row description was set to "Toc vérifié avec accents"
        assert "\xe9" in result.rows[0].description


# ===========================================================================
# Section 4 — Column aliases
# ===========================================================================

class TestColumnAliases:
    """Alternate header names (id, name, notes, file, state) resolve correctly."""

    @pytest.fixture(scope="class")
    @classmethod
    def result(cls) -> DossierParseResult:
        return parse_dossier(FIXTURES / "dossier_outline_alt_headers.csv")

    def test_row_count(self, result):
        assert result.row_count == _EXPECTED_COUNT

    def test_no_errors(self, result):
        assert result.has_errors is False

    def test_column_map_has_five_fields(self, result):
        assert set(result.column_map.keys()) == {
            "section_number", "section_title", "description", "source", "status"
        }

    def test_column_map_actual_headers(self, result):
        # The alt-headers file uses: id, name, notes, file, state
        assert result.column_map["section_number"] == "id"
        assert result.column_map["section_title"] == "name"
        assert result.column_map["description"] == "notes"
        assert result.column_map["source"] == "file"
        assert result.column_map["status"] == "state"

    def test_section_1_8_missing(self, result):
        row = _section_by_id(result, "1.8")
        assert row is not None and row.status == "missing"

    def test_alt_produces_same_content_as_clean(self, result):
        clean = parse_dossier(FIXTURES / "dossier_outline_clean.csv")
        alt_nums = [r.section_number for r in result.rows]
        clean_nums = [r.section_number for r in clean.rows]
        assert alt_nums == clean_nums


# ===========================================================================
# Section 5 — Partial columns
# ===========================================================================

class TestPartialColumns:
    """Files that supply only one of the two required columns are still valid."""

    def test_number_only_row_count(self):
        result = parse_dossier(FIXTURES / "dossier_outline_number_only.csv")
        assert result.row_count == _EXPECTED_COUNT

    def test_number_only_no_errors(self):
        result = parse_dossier(FIXTURES / "dossier_outline_number_only.csv")
        assert result.has_errors is False

    def test_number_only_section_title_is_empty(self):
        result = parse_dossier(FIXTURES / "dossier_outline_number_only.csv")
        for row in result.rows:
            assert row.section_title == ""

    def test_number_only_section_number_populated(self):
        result = parse_dossier(FIXTURES / "dossier_outline_number_only.csv")
        # At least some rows should have a section_number
        numbers = [r.section_number for r in result.rows if r.section_number]
        assert len(numbers) > 0

    def test_number_only_column_map_has_section_number(self):
        result = parse_dossier(FIXTURES / "dossier_outline_number_only.csv")
        assert "section_number" in result.column_map
        assert "section_title" not in result.column_map

    def test_title_only_row_count(self):
        result = parse_dossier(FIXTURES / "dossier_outline_title_only.csv")
        assert result.row_count == _EXPECTED_COUNT

    def test_title_only_no_errors(self):
        result = parse_dossier(FIXTURES / "dossier_outline_title_only.csv")
        # When section_number column is absent, blank-number errors are suppressed.
        # The only errors possible would be duplicates — there are none here.
        blank_errors = [
            e for e in result.errors
            if e.field_name == "section_number" and "blank" in e.message
        ]
        assert blank_errors == [], (
            "Should not report blank section_number errors when the column is absent"
        )

    def test_title_only_section_number_is_empty(self):
        result = parse_dossier(FIXTURES / "dossier_outline_title_only.csv")
        for row in result.rows:
            assert row.section_number == ""

    def test_title_only_section_title_populated(self):
        result = parse_dossier(FIXTURES / "dossier_outline_title_only.csv")
        titles = [r.section_title for r in result.rows if r.section_title]
        assert len(titles) > 0

    def test_title_only_column_map_has_section_title(self):
        result = parse_dossier(FIXTURES / "dossier_outline_title_only.csv")
        assert "section_title" in result.column_map
        assert "section_number" not in result.column_map


# ===========================================================================
# Section 6 — Validation errors (row-level)
# ===========================================================================

class TestValidationErrors:
    """Row-level issues are collected; parsing still succeeds."""

    def test_blank_section_numbers_collected(self):
        result = parse_dossier(FIXTURES / "dossier_outline_blank_numbers.csv")
        # 3 rows have blank section_number
        blank_errors = [
            e for e in result.errors if e.field_name == "section_number"
            and "blank" in e.message
        ]
        assert len(blank_errors) == 3

    def test_blank_rows_still_in_results(self):
        result = parse_dossier(FIXTURES / "dossier_outline_blank_numbers.csv")
        # Total rows parsed = all 75
        assert len(result.rows) == _EXPECTED_COUNT

    def test_blank_error_row_indices_are_correct(self):
        result = parse_dossier(FIXTURES / "dossier_outline_blank_numbers.csv")
        # Blank rows were set at df indices 2, 5, 10 → 1-based row_index 3, 6, 11
        blank_indices = sorted(
            e.row_index for e in result.errors
            if e.field_name == "section_number" and "blank" in e.message
        )
        assert blank_indices == [3, 6, 11]

    def test_duplicate_section_number_detected(self):
        result = parse_dossier(FIXTURES / "dossier_outline_duplicates.csv")
        dup_errors = [
            e for e in result.errors
            if e.field_name == "section_number" and "duplicate" in e.message.lower()
        ]
        assert len(dup_errors) == 1

    def test_duplicate_error_message_contains_section_id(self):
        result = parse_dossier(FIXTURES / "dossier_outline_duplicates.csv")
        dup_errors = [e for e in result.errors if "duplicate" in e.message.lower()]
        assert len(dup_errors) == 1
        # The duplicated value is "1.1"
        assert "1.1" in dup_errors[0].raw_value

    def test_duplicate_rows_still_parsed(self):
        result = parse_dossier(FIXTURES / "dossier_outline_duplicates.csv")
        assert len(result.rows) == _EXPECTED_COUNT

    def test_validation_error_str_representation(self):
        result = parse_dossier(FIXTURES / "dossier_outline_blank_numbers.csv")
        err = result.errors[0]
        s = str(err)
        assert "Row" in s
        assert "section_number" in s

    def test_has_errors_is_true_for_blank_file(self):
        result = parse_dossier(FIXTURES / "dossier_outline_blank_numbers.csv")
        assert result.has_errors is True

    def test_error_count_matches_errors_list(self):
        result = parse_dossier(FIXTURES / "dossier_outline_blank_numbers.csv")
        assert result.error_count == len(result.errors)

    def test_missing_both_required_raises_parse_error(self):
        with pytest.raises(DossierParseError) as exc_info:
            parse_dossier(FIXTURES / "dossier_outline_missing_cols.csv")
        assert "missing both required" in str(exc_info.value).lower()


# ===========================================================================
# Section 7 — File-level error handling
# ===========================================================================

class TestFileErrors:
    """File-level failures raise DossierParseError or FileNotFoundError."""

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_dossier(tmp_path / "does_not_exist.csv")

    def test_unsupported_extension_raises_parse_error(self, tmp_path):
        p = tmp_path / "file.json"
        p.write_text('{"x": 1}')
        with pytest.raises(DossierParseError) as exc_info:
            parse_dossier(p)
        assert "unsupported" in str(exc_info.value).lower()

    def test_unsupported_xls_extension_raises_parse_error(self, tmp_path):
        p = tmp_path / "file.xls"
        p.write_bytes(b"garbage")
        with pytest.raises(DossierParseError):
            parse_dossier(p)

    def test_header_only_csv_parses_with_zero_rows(self, tmp_path):
        p = tmp_path / "header_only.csv"
        p.write_text("section_number,section_title,status\n", encoding="utf-8")
        result = parse_dossier(p)
        assert result.row_count == 0
        assert result.rows == []
        assert result.has_errors is False

    def test_empty_csv_raises_parse_error(self, tmp_path):
        p = tmp_path / "empty.csv"
        p.write_bytes(b"")
        with pytest.raises(DossierParseError):
            parse_dossier(p)

    def test_garbage_csv_raises_parse_error(self, tmp_path):
        p = tmp_path / "garbage.csv"
        p.write_bytes(b"\x00\x01\x02\x03")
        with pytest.raises(DossierParseError):
            parse_dossier(p)

    def test_single_column_csv_raises_parse_error(self, tmp_path):
        """A file with only one column can't satisfy section_number AND section_title;
        it will also be rejected as having only 1 column (shape check in _read_csv)."""
        p = tmp_path / "one_col.csv"
        p.write_text("description\nnotes here\n", encoding="utf-8")
        # This has 1 column — _read_csv requires df.shape[1] > 1, so it will fail
        with pytest.raises(DossierParseError):
            parse_dossier(p)


# ===========================================================================
# Section 8 — to_dataframe()
# ===========================================================================

class TestToDataframe:
    """to_dataframe() converts DossierParseResult into a well-formed DataFrame."""

    @pytest.fixture(scope="class")
    @classmethod
    def df(cls) -> pd.DataFrame:
        result = parse_dossier(FIXTURES / "dossier_outline_clean.csv")
        return to_dataframe(result)

    def test_returns_dataframe(self, df):
        assert isinstance(df, pd.DataFrame)

    def test_row_count(self, df):
        assert len(df) == _EXPECTED_COUNT

    def test_columns(self, df):
        assert set(df.columns) == {
            "section_number", "section_title", "description",
            "source", "status", "row_index",
        }

    def test_all_string_columns_are_object_dtype(self, df):
        for col in ("section_number", "section_title", "description", "source", "status"):
            assert df[col].dtype == object

    def test_row_index_column_is_numeric(self, df):
        assert pd.api.types.is_integer_dtype(df["row_index"])

    def test_no_null_values(self, df):
        assert df.isnull().sum().sum() == 0

    def test_section_numbers_in_expected_set(self, df):
        expected = {s["section_id"].lower() for s in _EXPECTED_SECTIONS}
        assert set(df["section_number"]) == expected

    def test_empty_result_returns_empty_dataframe(self):
        empty_result = DossierParseResult()
        df = to_dataframe(empty_result)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert set(df.columns) == {
            "section_number", "section_title", "description",
            "source", "status", "row_index",
        }


# ===========================================================================
# Section 9 — Integration: synthetic dossier content spot-checks
# ===========================================================================

class TestSyntheticDossierIntegration:
    """Deep spot-checks of actual content from the demo dossier."""

    @pytest.fixture(scope="class")
    @classmethod
    def result(cls) -> DossierParseResult:
        return parse_dossier(FIXTURES / "dossier_outline_clean.csv")

    def test_four_missing_sections(self, result):
        missing = [r for r in result.rows if r.status == "missing"]
        assert len(missing) == 4

    def test_critical_missing_section_ids(self, result):
        missing_ids = {r.section_number for r in result.rows if r.status == "missing"}
        assert missing_ids == {"1.8", "2.4", "2.7.4", "4.2.3.5"}

    def test_seven_needs_review_sections(self, result):
        review = [r for r in result.rows if r.status == "needs_review"]
        assert len(review) == 7

    def test_needs_review_section_ids(self, result):
        review_ids = {r.section_number for r in result.rows if r.status == "needs_review"}
        assert review_ids == {"1.3", "2.5", "2.6.6", "4.2.3.2", "4.2.3.4", "5.3.5", "5.3.6"}

    def test_two_not_applicable_sections(self, result):
        na = [r for r in result.rows if r.status == "not_applicable"]
        assert len(na) == 2

    def test_not_applicable_section_ids(self, result):
        na_ids = {r.section_number for r in result.rows if r.status == "not_applicable"}
        assert na_ids == {"1.7", "4.2.3.6"}

    def test_present_section_count(self, result):
        present = [r for r in result.rows if r.status == "present"]
        assert len(present) == 62

    def test_section_2_7_4_description_mentions_missing(self, result):
        row = _section_by_id(result, "2.7.4")
        assert "MISSING" in row.description.upper()

    def test_section_1_8_description_mentions_pharmacovigilance(self, result):
        row = _section_by_id(result, "1.8")
        assert "pharmacovigilance" in row.description.lower() or "rems" in row.description.lower()

    def test_section_3_2_s_has_source(self, result):
        row = _section_by_id(result, "3.2.s")
        assert row is not None
        assert row.source != ""

    def test_section_5_3_5_description_mentions_hepatic(self, result):
        row = _section_by_id(result, "5.3.5")
        assert "hepatic" in row.description.lower() or "hepatotoxicity" in row.description.lower()

    def test_all_section_ids_cover_all_five_modules(self, result):
        prefixes = {r.section_number.split(".")[0] for r in result.rows if r.section_number}
        # M1 sections start with "1", M2 with "2", ... M5 with "5"
        # Also includes IDs like "3.2.s.1" where split gives "3"
        for expected_prefix in ("1", "2", "3", "4", "5"):
            assert expected_prefix in prefixes, f"No sections with prefix {expected_prefix!r}"

    def test_m3_sections_all_present_or_conditional(self, result):
        """Module 3 quality package is complete in the synthetic dossier — no missing."""
        m3_rows = [r for r in result.rows if r.section_number.startswith("3.")]
        missing_m3 = [r for r in m3_rows if r.status == "missing"]
        assert missing_m3 == [], f"Unexpected missing M3 sections: {[r.section_number for r in missing_m3]}"

    def test_m5_has_no_waived_sections(self, result):
        m5_waived = [r for r in result.rows
                     if r.section_number.startswith("5.") and r.status == "waived"]
        assert m5_waived == []

    def test_no_status_is_none(self, result):
        for row in result.rows:
            assert row.status is not None

    def test_section_1_9_source_is_non_empty(self, result):
        row = _section_by_id(result, "1.9")
        assert row is not None and row.source != ""
