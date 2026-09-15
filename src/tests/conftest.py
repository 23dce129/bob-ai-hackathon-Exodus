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


# ---------------------------------------------------------------------------
# Traceability fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def kb():
    """Loaded CTD knowledge base singleton."""
    from core.ctd_knowledge_base import load_knowledge_base  # noqa: PLC0415
    return load_knowledge_base()


@pytest.fixture(scope="session")
def empty_match_summary():
    """A MatchSummary with zero rows — simulates an empty dossier."""
    from core.section_matcher import MatchSummary  # noqa: PLC0415
    return MatchSummary(total_rows=0)


@pytest.fixture(scope="session")
def bleeding_signal() -> dict:
    """A synthetic signal dict for warfarin / haemorrhage (bleeding category)."""
    return {
        "drug_name": "warfarin",
        "adverse_event": "haemorrhage",
        "prr": 7.2,
        "p_value": 0.001,
        "signal_flag": True,
    }


@pytest.fixture(scope="session")
def hepato_signal() -> dict:
    """A synthetic signal dict for atorvastatin / hepatotoxicity."""
    return {
        "drug_name": "atorvastatin",
        "adverse_event": "hepatotoxicity",
        "prr": 3.5,
        "p_value": 0.02,
        "signal_flag": True,
    }


@pytest.fixture(scope="session")
def full_match_summary(kb):
    """
    A MatchSummary that marks every KB section as PRESENT (confidence 1.0).
    Used to test the fully-documented path (score == 1.0).
    """
    from core.dossier_parser import DossierRow  # noqa: PLC0415
    from core.section_matcher import (  # noqa: PLC0415
        MatchSummary, SectionMatch, MatchType, MatchStatus,
    )

    matches = []
    for i, section in enumerate(kb.all_sections()):
        row = DossierRow(
            row_index=i,
            section_number=section.id,
            section_title=section.title,
            description="",
            source="",
            status="present",
        )
        matches.append(SectionMatch(
            dossier_row=row,
            expected_section=section,
            match_type=MatchType.EXACT_NUMBER,
            confidence=1.0,
            status=MatchStatus.PRESENT,
            match_reason="fixture: all sections marked PRESENT",
            status_source="dossier_supplied",
        ))

    summary = MatchSummary(
        matches=matches,
        total_rows=len(matches),
        exact_number_count=len(matches),
    )
    summary.status_counts = {"PRESENT": len(matches)}
    return summary
