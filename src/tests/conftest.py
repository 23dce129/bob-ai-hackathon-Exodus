"""
conftest.py — Shared pytest fixtures for PharmaGuard AI tests.

Fixtures defined here are automatically available to all test files
without importing.  pytest discovers this file by convention.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

# Resolve the backend root regardless of where pytest is invoked from.
BACKEND_DIR = Path(__file__).parent.parent / "backend"
SAMPLE_CSV = BACKEND_DIR / "data" / "faers_sample.csv"


# ---------------------------------------------------------------------------
# sample_df  — real sample CSV loaded once per session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    """
    Load and return the bundled faers_sample.csv as a *raw* DataFrame.
    (Not normalised — used by ingestor tests to verify normalisation.)
    """
    return pd.read_csv(SAMPLE_CSV)


# ---------------------------------------------------------------------------
# clean_df  — normalised version of the sample, loaded once per session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def clean_df() -> pd.DataFrame:
    """
    Normalised FAERS DataFrame produced by load_faers() on the sample CSV.
    Used by PRR tests that need a fully processed dataset.
    """
    from core.faers_ingestor import load_faers  # noqa: PLC0415
    return load_faers(SAMPLE_CSV)


# ---------------------------------------------------------------------------
# tiny_df  — hardcoded minimal DataFrame with known exact counts
#
# Drug-event layout (all counts are exact and hand-verified):
#
#   drugA / eventX :  a=10, drug_total=12, event_total=14, n=50
#       → PRR = (10/12) / (4/38) = 0.8333 / 0.1053 ≈ 7.917
#
#   drugA / eventY :  a=2  → below min_cases threshold (not flagged)
#
#   drugB / eventX :  a=4, PRR just above 2  → may or may not be flagged
#
#   drugC / eventZ :  a=1, c=0 → PRR undefined (only drug with this event)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tiny_df() -> pd.DataFrame:
    """
    Minimal synthetic DataFrame with hand-calculated expected values.
    Use this fixture when a test must assert exact numeric results.
    """
    rows = []

    # drugA/eventX — 10 reports  (strong signal)
    rows += [{"drug_name": "drugA", "adverse_event": "eventX"}] * 10
    # drugA/eventY — 2 reports   (below case threshold)
    rows += [{"drug_name": "drugA", "adverse_event": "eventY"}] * 2
    # drugA/eventZ — 0 (not present)

    # drugB/eventX — 4 reports
    rows += [{"drug_name": "drugB", "adverse_event": "eventX"}] * 4
    # drugB/eventY — 16 reports  (drugB is mostly associated with eventY)
    rows += [{"drug_name": "drugB", "adverse_event": "eventY"}] * 16
    # drugB/eventZ — 4 reports
    rows += [{"drug_name": "drugB", "adverse_event": "eventZ"}] * 4

    # drugC/eventZ — 1 report (only drug ever reporting eventZ → c=0)
    rows += [{"drug_name": "drugC", "adverse_event": "eventZ"}] * 1

    # padding: drugB fills remaining entries
    rows += [{"drug_name": "drugB", "adverse_event": "eventW"}] * 13

    df = pd.DataFrame(rows)
    # Add dummy columns required by ingestor schema (not used by PRR tests)
    df["report_id"]    = [str(i) for i in range(len(df))]
    df["outcome_code"] = "OT"
    df["report_date"]  = pd.NaT
    df["age_years"]    = float("nan")
    df["sex"]          = "UNK"
    return df


# ---------------------------------------------------------------------------
# tmp_parquet  — temporary file path (cleaned up after test)
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_parquet(tmp_path: Path) -> Path:
    """Return a temporary .parquet file path inside pytest's tmp_path."""
    return tmp_path / "test_faers.parquet"


# ---------------------------------------------------------------------------
# tmp_duckdb  — temporary DuckDB file path (cleaned up after test)
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_duckdb(tmp_path: Path) -> Path:
    """Return a temporary .duckdb file path inside pytest's tmp_path."""
    return tmp_path / "test_faers.duckdb"
