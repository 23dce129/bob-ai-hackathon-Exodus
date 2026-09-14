"""
test_faers_ingestor.py — Unit tests for the FAERS ingestion pipeline.

All tests use either:
  - the bundled faers_sample.csv (via the `sample_df` or `clean_df` fixture), or
  - small in-memory DataFrames constructed inside the test.

No network calls are made.  No external FAERS data is required.
"""
from __future__ import annotations

import io
from pathlib import Path

import duckdb
import pandas as pd
import pytest

# Add backend to sys.path so imports work when pytest runs from src/tests/
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from core.faers_ingestor import (
    load_faers,
    load_faers_csv,
    load_parquet,
    normalize_adverse_event,
    normalize_dataframe,
    normalize_drug_name,
    to_duckdb,
    to_parquet,
    validate_schema,
)
from core.schemas import REQUIRED_INTERNAL_COLUMNS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_CSV = Path(__file__).parent.parent / "backend" / "data" / "faers_sample.csv"

MINIMAL_CSV_CONTENT = (
    "primaryid,drug_name,pt,outc_cod,event_dt,age,sex\n"
    "1,Warfarin Sodium,bleeding,DE,20220101,65.0,M\n"
    "2,ibuprofen hcl,Nausea,HO,20220215,45.0,F\n"
    "3,METFORMIN,nausea,OT,20220301,55.0,UNK\n"
)


def _make_df_from_string(csv_text: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(csv_text))


# ---------------------------------------------------------------------------
# normalize_drug_name
# ---------------------------------------------------------------------------

class TestNormalizeDrugName:
    def test_uppercase_stripped_and_lowercased(self):
        assert normalize_drug_name("WARFARIN") == "warfarin"

    def test_salt_suffix_sodium_removed(self):
        assert normalize_drug_name("warfarin sodium") == "warfarin"

    def test_salt_suffix_hcl_removed(self):
        assert normalize_drug_name("metformin hcl") == "warfarin".replace("warfarin", "metformin")
        assert normalize_drug_name("metformin hcl") == "metformin"

    def test_salt_suffix_hydrochloride_removed(self):
        assert normalize_drug_name("Ibuprofen Hydrochloride") == "ibuprofen"

    def test_leading_trailing_whitespace_stripped(self):
        assert normalize_drug_name("  aspirin  ") == "aspirin"

    def test_nan_returns_empty_string(self):
        import math
        assert normalize_drug_name(float("nan")) == ""

    def test_name_without_suffix_unchanged(self):
        assert normalize_drug_name("aspirin") == "aspirin"

    def test_uppercase_full_name_normalised(self):
        assert normalize_drug_name("WARFARIN SODIUM ") == "warfarin"


# ---------------------------------------------------------------------------
# normalize_adverse_event
# ---------------------------------------------------------------------------

class TestNormalizeAdverseEvent:
    def test_lowercased(self):
        assert normalize_adverse_event("Nausea") == "nausea"

    def test_uppercase_lowercased(self):
        assert normalize_adverse_event("DRUG-INDUCED LIVER INJURY") == "drug-induced liver injury"

    def test_whitespace_stripped(self):
        assert normalize_adverse_event("  bleeding  ") == "bleeding"

    def test_nan_returns_empty_string(self):
        import math
        assert normalize_adverse_event(float("nan")) == ""

    def test_medra_term_preserved(self):
        # MedDRA terms contain hyphens and spaces — must not be altered
        term = "drug-induced liver injury"
        assert normalize_adverse_event(term) == term


# ---------------------------------------------------------------------------
# validate_schema
# ---------------------------------------------------------------------------

class TestValidateSchema:
    def test_valid_schema_does_not_raise(self):
        df = _make_df_from_string(MINIMAL_CSV_CONTENT)
        validate_schema(df)  # must not raise

    def test_missing_drug_name_raises(self):
        df = _make_df_from_string(MINIMAL_CSV_CONTENT).drop(columns=["drug_name"])
        with pytest.raises(ValueError, match="drug_name"):
            validate_schema(df)

    def test_missing_pt_raises(self):
        df = _make_df_from_string(MINIMAL_CSV_CONTENT).drop(columns=["pt"])
        with pytest.raises(ValueError, match="adverse_event"):
            validate_schema(df)

    def test_missing_primaryid_raises(self):
        df = _make_df_from_string(MINIMAL_CSV_CONTENT).drop(columns=["primaryid"])
        with pytest.raises(ValueError, match="report_id"):
            validate_schema(df)

    def test_error_message_lists_all_missing(self):
        df = pd.DataFrame({"unrelated_col": [1, 2]})
        with pytest.raises(ValueError) as exc_info:
            validate_schema(df)
        msg = str(exc_info.value)
        assert "report_id" in msg
        assert "drug_name" in msg
        assert "adverse_event" in msg


# ---------------------------------------------------------------------------
# normalize_dataframe
# ---------------------------------------------------------------------------

class TestNormalizeDataframe:
    def test_all_internal_columns_present(self):
        raw = _make_df_from_string(MINIMAL_CSV_CONTENT)
        clean = normalize_dataframe(raw)
        for col in REQUIRED_INTERNAL_COLUMNS:
            assert col in clean.columns, f"Missing column: {col}"

    def test_drug_names_are_lowercased(self):
        raw = _make_df_from_string(MINIMAL_CSV_CONTENT)
        clean = normalize_dataframe(raw)
        assert all(name == name.lower() for name in clean["drug_name"])

    def test_salt_suffix_removed_in_dataframe(self):
        raw = _make_df_from_string(MINIMAL_CSV_CONTENT)
        clean = normalize_dataframe(raw)
        assert "warfarin" in clean["drug_name"].values
        assert "warfarin sodium" not in clean["drug_name"].values

    def test_adverse_events_are_lowercased(self):
        raw = _make_df_from_string(MINIMAL_CSV_CONTENT)
        clean = normalize_dataframe(raw)
        assert all(e == e.lower() for e in clean["adverse_event"])

    def test_report_date_is_datetime_dtype(self):
        raw = _make_df_from_string(MINIMAL_CSV_CONTENT)
        clean = normalize_dataframe(raw)
        assert pd.api.types.is_datetime64_any_dtype(clean["report_date"])

    def test_unparseable_date_becomes_nat(self):
        csv = (
            "primaryid,drug_name,pt,outc_cod,event_dt,age,sex\n"
            "1,aspirin,nausea,OT,BADDATE,50.0,M\n"
        )
        raw = _make_df_from_string(csv)
        clean = normalize_dataframe(raw)
        assert pd.isna(clean["report_date"].iloc[0])

    def test_null_drug_name_rows_dropped(self):
        csv = (
            "primaryid,drug_name,pt,outc_cod,event_dt,age,sex\n"
            "1,,nausea,OT,20220101,50.0,M\n"
            "2,aspirin,nausea,OT,20220101,50.0,M\n"
        )
        raw = _make_df_from_string(csv)
        clean = normalize_dataframe(raw)
        assert len(clean) == 1
        assert clean["drug_name"].iloc[0] == "aspirin"

    def test_null_adverse_event_rows_dropped(self):
        csv = (
            "primaryid,drug_name,pt,outc_cod,event_dt,age,sex\n"
            "1,aspirin,,OT,20220101,50.0,M\n"
            "2,aspirin,nausea,OT,20220101,50.0,M\n"
        )
        raw = _make_df_from_string(csv)
        clean = normalize_dataframe(raw)
        assert len(clean) == 1

    def test_sex_normalised_to_allowed_values(self):
        csv = (
            "primaryid,drug_name,pt,outc_cod,event_dt,age,sex\n"
            "1,aspirin,nausea,OT,20220101,50.0,MALE\n"
            "2,aspirin,nausea,OT,20220101,50.0,FEMALE\n"
            "3,aspirin,nausea,OT,20220101,50.0,NS\n"
        )
        raw = _make_df_from_string(csv)
        clean = normalize_dataframe(raw)
        assert set(clean["sex"].unique()).issubset({"M", "F", "UNK"})


# ---------------------------------------------------------------------------
# load_faers_csv
# ---------------------------------------------------------------------------

class TestLoadFaersCsv:
    def test_loads_sample_csv(self):
        df = load_faers_csv(SAMPLE_CSV)
        assert len(df) > 0

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_faers_csv("/nonexistent/path/faers.csv")

    def test_columns_lowercased_on_load(self):
        df = load_faers_csv(SAMPLE_CSV)
        assert all(c == c.lower() for c in df.columns)


# ---------------------------------------------------------------------------
# load_faers (public entry point — end-to-end)
# ---------------------------------------------------------------------------

class TestLoadFaers:
    def test_returns_dataframe(self):
        df = load_faers(SAMPLE_CSV)
        assert isinstance(df, pd.DataFrame)

    def test_has_all_required_columns(self):
        df = load_faers(SAMPLE_CSV)
        for col in REQUIRED_INTERNAL_COLUMNS:
            assert col in df.columns

    def test_row_count_reasonable(self):
        df = load_faers(SAMPLE_CSV)
        # Sample has 291 rows; after dropping nulls should still have most
        assert len(df) >= 250

    def test_no_empty_drug_names(self):
        df = load_faers(SAMPLE_CSV)
        assert (df["drug_name"].str.len() > 0).all()

    def test_no_empty_adverse_events(self):
        df = load_faers(SAMPLE_CSV)
        assert (df["adverse_event"].str.len() > 0).all()

    def test_all_drug_names_lowercase(self):
        df = load_faers(SAMPLE_CSV)
        assert all(name == name.lower() for name in df["drug_name"])


# ---------------------------------------------------------------------------
# Parquet round-trip
# ---------------------------------------------------------------------------

class TestParquetRoundtrip:
    def test_save_and_reload_identical(self, tmp_parquet):
        df = load_faers(SAMPLE_CSV)
        to_parquet(df, tmp_parquet)
        reloaded = load_parquet(tmp_parquet)
        pd.testing.assert_frame_equal(df.reset_index(drop=True),
                                      reloaded.reset_index(drop=True))

    def test_parquet_file_created(self, tmp_parquet):
        df = load_faers(SAMPLE_CSV)
        to_parquet(df, tmp_parquet)
        assert tmp_parquet.exists()

    def test_load_parquet_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_parquet(tmp_path / "ghost.parquet")


# ---------------------------------------------------------------------------
# DuckDB persistence
# ---------------------------------------------------------------------------

class TestDuckDB:
    def test_table_created(self, tmp_duckdb):
        df = load_faers(SAMPLE_CSV)
        to_duckdb(df, tmp_duckdb)
        conn = duckdb.connect(str(tmp_duckdb))
        tables = conn.execute("SHOW TABLES").fetchall()
        conn.close()
        table_names = [t[0] for t in tables]
        assert "faers_reports" in table_names

    def test_row_count_matches(self, tmp_duckdb):
        df = load_faers(SAMPLE_CSV)
        to_duckdb(df, tmp_duckdb)
        conn = duckdb.connect(str(tmp_duckdb))
        count = conn.execute("SELECT COUNT(*) FROM faers_reports").fetchone()[0]
        conn.close()
        assert count == len(df)

    def test_idempotent_write(self, tmp_duckdb):
        """Calling to_duckdb twice should not duplicate rows."""
        df = load_faers(SAMPLE_CSV)
        to_duckdb(df, tmp_duckdb)
        to_duckdb(df, tmp_duckdb)
        conn = duckdb.connect(str(tmp_duckdb))
        count = conn.execute("SELECT COUNT(*) FROM faers_reports").fetchone()[0]
        conn.close()
        assert count == len(df)
