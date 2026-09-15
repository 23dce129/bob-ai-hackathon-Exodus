"""
signals.py — FastAPI routes for signal detection and temporal trend analysis.

Endpoints
---------
GET  /api/signals/sample
    Run signal detection on the bundled sample dataset.
    Returns flagged signals plus trend data for each flagged pair.

POST /api/signals/analyze
    Same as /sample but accepts optional filters (drug_name, thresholds).

GET  /api/signals/trend/{drug_name}/{adverse_event}
    Return the temporal trend for a single (drug, event) pair.

GET  /api/signals/trends/{drug_name}
    Return temporal trends for all events reported for a given drug.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from core.faers_ingestor import load_faers
from core.prr_calculator import calculate_all_signals
from core.priority_scorer import ScoringConfig as _ScoringConfig, score_all_signals
from core.signal_clusterer import cluster_reports
from core.temporal_analyser import analyse_all_trends, analyse_trend

# Resolve the bundled sample dataset path relative to this file
_SAMPLE_CSV = Path(__file__).parent.parent.parent / "data" / "faers_sample.csv"

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    drug_name: Optional[str] = Field(
        default=None,
        description="Filter results to this drug only. Omit for all drugs.",
    )
    min_prr: float = Field(default=2.0, ge=1.0, description="Minimum PRR threshold.")
    min_cases: int = Field(default=3, ge=1, description="Minimum case count.")
    max_pval: float = Field(default=0.05, gt=0.0, le=1.0, description="Maximum p-value.")
    include_trends: bool = Field(
        default=True,
        description="Attach temporal trend data to each flagged signal.",
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_sample() -> object:
    """Load and cache the sample dataset. Raises 500 if file is missing."""
    if not _SAMPLE_CSV.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Sample dataset not found at {_SAMPLE_CSV}. "
                   "Ensure faers_sample.csv is present in src/backend/data/.",
        )
    return load_faers(_SAMPLE_CSV)


def _build_response(df, drug_filter, min_prr, min_cases, max_pval, include_trends):
    """Run PRR + optional trend analysis and return a serialisable dict."""
    signals = calculate_all_signals(
        df,
        min_cases=min_cases,
        min_prr=min_prr,
        max_pval=max_pval,
        drug_filter=drug_filter,
    )
    flagged = [s for s in signals if s.signal_flag]
    all_signals_dicts = [s.to_dict() for s in signals]
    flagged_dicts     = [s.to_dict() for s in flagged]

    trends_by_pair: dict = {}
    if include_trends:
        drugs_to_analyse = {drug_filter} if drug_filter else {s.drug_name for s in flagged}
        for drug in drugs_to_analyse:
            trend_results = analyse_all_trends(df, drug_filter=drug)
            for tr in trend_results:
                key = f"{tr.drug_name}|{tr.adverse_event}"
                trends_by_pair[key] = tr.to_dict()

    # Attach trend inline to each flagged signal dict
    for sig in flagged_dicts:
        key = f"{sig['drug_name']}|{sig['adverse_event']}"
        sig["trend"] = trends_by_pair.get(key)

    return {
        "total_evaluated": len(signals),
        "total_flagged": len(flagged),
        "drug_filter": drug_filter,
        "thresholds": {
            "min_prr": min_prr,
            "min_cases": min_cases,
            "max_pval": max_pval,
        },
        "flagged_signals": flagged_dicts,
        "all_signals": all_signals_dicts,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/sample", summary="Detect signals on bundled sample dataset")
def get_sample_signals(
    drug_name: Optional[str] = Query(default=None, description="Filter to one drug."),
    min_prr: float = Query(default=2.0, ge=1.0),
    min_cases: int = Query(default=3, ge=1),
    max_pval: float = Query(default=0.05, gt=0.0, le=1.0),
    include_trends: bool = Query(default=True),
):
    """
    Run PRR signal detection on the bundled FAERS sample dataset.
    No file upload needed — uses the committed faers_sample.csv.
    Each flagged signal includes temporal trend data when include_trends=true.
    """
    df = _load_sample()
    return _build_response(df, drug_name, min_prr, min_cases, max_pval, include_trends)


@router.post("/analyze", summary="Detect signals with custom parameters")
def analyze_signals(body: AnalyzeRequest):
    """
    Run PRR signal detection with custom thresholds on the sample dataset.
    Post a JSON body to override defaults.
    """
    df = _load_sample()
    return _build_response(
        df,
        body.drug_name,
        body.min_prr,
        body.min_cases,
        body.max_pval,
        body.include_trends,
    )


@router.get(
    "/trend/{drug_name}/{adverse_event}",
    summary="Temporal trend for one drug–event pair",
)
def get_trend(drug_name: str, adverse_event: str):
    """
    Return the full temporal trend analysis for a single (drug, adverse_event) pair.

    Path parameters are case-insensitive and will be lowercased before lookup.
    Returns 404 if no reports exist for this pair.
    """
    df = _load_sample()
    drug  = drug_name.lower().strip()
    event = adverse_event.lower().strip()

    pair_exists = (
        (df["drug_name"] == drug) & (df["adverse_event"] == event)
    ).any()
    if not pair_exists:
        raise HTTPException(
            status_code=404,
            detail=f"No reports found for drug='{drug}' / event='{event}'.",
        )

    result = analyse_trend(df, drug, event)
    return result.to_dict()


@router.get("/trends/{drug_name}", summary="All temporal trends for a drug")
def get_drug_trends(drug_name: str):
    """
    Return temporal trends for every adverse event reported for the given drug.
    Results are sorted by |trend_score| descending; insufficient-data pairs last.
    Returns 404 if the drug does not appear in the dataset.
    """
    df = _load_sample()
    drug = drug_name.lower().strip()

    if not (df["drug_name"] == drug).any():
        raise HTTPException(
            status_code=404,
            detail=f"Drug '{drug}' not found in the dataset.",
        )

    trends = analyse_all_trends(df, drug_filter=drug)
    return {
        "drug_name": drug,
        "total_pairs": len(trends),
        "trends": [t.to_dict() for t in trends],
    }


@router.get("/priority", summary="Priority scores for all flagged signals")
def get_priority_scores(
    drug_name: Optional[str] = Query(default=None, description="Filter to one drug."),
    min_prr: float = Query(default=2.0, ge=1.0),
    min_cases: int = Query(default=3, ge=1),
    max_pval: float = Query(default=0.05, gt=0.0, le=1.0),
):
    """
    Run PRR signal detection, then compute a priority score for every flagged signal.

    Each score combines six evidence dimensions:
        PRR strength (30 pts) + Report volume (20 pts) +
        Statistical strength (20 pts) + Serious outcome rate (15 pts) +
        Reporting trend (10 pts) + Cluster strength (5 pts) = 100 pts max.

    Results are sorted by final_score descending.

    ⚠️  HACKATHON SCREENING SCORE ONLY — not an FDA-approved risk score.
    """
    df = _load_sample()

    # Run detection
    signals = calculate_all_signals(
        df, min_cases=min_cases, min_prr=min_prr, max_pval=max_pval,
        drug_filter=drug_name,
    )
    flagged = [s for s in signals if s.signal_flag]

    if not flagged:
        return {
            "total_flagged": 0,
            "priority_scores": [],
            "disclaimer": (
                "⚠️  HACKATHON SCREENING SCORE ONLY.  "
                "Not an FDA-approved or regulatory risk score."
            ),
        }

    # Build trend lookup
    trend_list = analyse_all_trends(df, drug_filter=drug_name)
    trends = {f"{t.drug_name}|{t.adverse_event}": t for t in trend_list}

    # Build clusters
    clusters = cluster_reports(df)

    # Score
    priority_scores = score_all_signals(
        flagged, df, trends=trends, clusters=clusters,
    )

    return {
        "total_flagged": len(priority_scores),
        "priority_scores": [ps.to_dict() for ps in priority_scores],
        "disclaimer": priority_scores[0].disclaimer if priority_scores else "",
    }
