"""
temporal_analyser.py — Deterministic temporal trend analysis for drug–event pairs.

Purpose
-------
Compute how the reporting rate for a specific (drug, adverse_event) pair has
changed over time.  This is a purely statistical/arithmetical module.
No language model is involved at any point.

Method
------
1.  Extract all reports for the (drug, event) pair that have a valid date.
2.  Assign each report to a calendar quarter (YYYY-Q1/Q2/Q3/Q4).
3.  Count reports per quarter → time series.
4.  Split the time series into two equal halves: "previous period" (older
    half) and "recent period" (newer half).
5.  Compute:
    - recent_count  = sum of counts in recent half
    - previous_count = sum of counts in previous half
    - pct_change    = (recent - previous) / previous × 100  (None if previous = 0)
    - trend_score   = (recent_rate - previous_rate) / max_rate
                      where rate = count / n_quarters_in_period
                      Range [-1.0, +1.0]; positive = increasing.
6.  Classify direction:
    - trend_score >  STABLE_BAND → "increasing"
    - trend_score < -STABLE_BAND → "decreasing"
    - else                       → "stable"
    - fewer than TREND_MIN_QUARTERS quarters with data → "insufficient_data"

Why quarters, not months or years?
-----------------------------------
FDA FAERS data is published quarterly.  Using quarters aligns naturally with
the publication cadence, gives enough granularity to detect emerging signals,
and avoids the month-level noise that distorts short time series.

Why split in half rather than use a fixed lookback window?
-----------------------------------------------------------
A fixed window (e.g. "last 6 months vs. prior 6 months") breaks down when
the dataset only covers 2–3 quarters.  Splitting in half is always well-defined
for any number of quarters ≥ 2 and scales correctly for multi-year datasets.

STABLE_BAND
-----------
Set to 0.1 (10 % of the max rate).  This prevents single-report fluctuations
from being classified as trends on small datasets.  The band is exposed as a
module constant so callers can override it for testing.

Public API
----------
    analyse_trend(df, drug, event)                     → TrendResult
    analyse_all_trends(df, drug_filter=None)           → list[TrendResult]
    quarterly_counts(df, drug, event)                  → list[QuarterCount]
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from core.schemas import (
    TREND_DECREASING,
    TREND_INCREASING,
    TREND_INSUFFICIENT_DATA,
    TREND_MIN_QUARTERS,
    TREND_STABLE,
    QuarterCount,
    TrendResult,
)

logger = logging.getLogger(__name__)

# Fraction of max_rate that defines the "stable" band around zero.
# trend_score in (-STABLE_BAND, +STABLE_BAND) → "stable".
STABLE_BAND: float = 0.10


# ---------------------------------------------------------------------------
# Quarter helpers
# ---------------------------------------------------------------------------

def _quarter_label(ts: pd.Timestamp) -> str:
    """Return an ISO-style quarter label, e.g. '2023-Q2'."""
    return f"{ts.year}-Q{ts.quarter}"


def quarterly_counts(
    df: pd.DataFrame,
    drug: str,
    event: str,
) -> list[QuarterCount]:
    """
    Count reports per calendar quarter for a specific (drug, event) pair.

    Only rows with a valid (non-NaT) report_date are counted.  Rows with
    missing dates are excluded from the time series but ARE counted in
    TrendResult.total_reports via a separate path in analyse_trend().

    Parameters
    ----------
    df : pd.DataFrame
        Normalised FAERS DataFrame with columns: drug_name, adverse_event,
        report_date.
    drug : str
        Normalised drug name.
    event : str
        Normalised adverse event term.

    Returns
    -------
    list[QuarterCount]
        Sorted oldest → newest.  Only quarters that fall within the observed
        [min_date, max_date] range are included; interior zero-count quarters
        ARE included so the time series has no gaps.
    """
    # Filter to this pair with valid dates
    mask = (
        (df["drug_name"] == drug)
        & (df["adverse_event"] == event)
        & df["report_date"].notna()
    )
    pair_df = df.loc[mask, "report_date"].copy()

    if pair_df.empty:
        return []

    # Assign quarter labels
    pair_ts = pd.to_datetime(pair_df)
    quarter_series = pair_ts.apply(_quarter_label)
    counts = quarter_series.value_counts().sort_index()

    # Build the full quarter spine (no gaps inside the range)
    min_ts = pair_ts.min()
    max_ts = pair_ts.max()
    # Generate all quarter-period starts between min and max
    all_quarters = pd.period_range(
        start=pd.Period(min_ts, freq="Q"),
        end=pd.Period(max_ts, freq="Q"),
        freq="Q",
    )

    result: list[QuarterCount] = []
    for period in all_quarters:
        label = f"{period.year}-Q{period.quarter}"
        count = int(counts.get(label, 0))
        result.append(
            QuarterCount(
                quarter=label,
                year=period.year,
                quarter_num=period.quarter,
                count=count,
            )
        )

    return result


# ---------------------------------------------------------------------------
# Trend score computation
# ---------------------------------------------------------------------------

def _compute_trend_score(
    recent_count: int,
    previous_count: int,
    n_recent: int,
    n_previous: int,
) -> float:
    """
    Compute the normalised trend score ∈ [-1.0, +1.0].

    Formula
    -------
        recent_rate   = recent_count   / n_recent    (reports per quarter)
        previous_rate = previous_count / n_previous  (reports per quarter)
        max_rate      = max(recent_rate, previous_rate)

        If max_rate == 0: score = 0.0 (no reports in either period)
        Otherwise:        score = (recent_rate - previous_rate) / max_rate

    This normalisation keeps the score in [-1, +1] regardless of absolute
    reporting volume.  A score of +1.0 means all reports are in the recent
    period and none in the previous period.  A score of -1.0 means the
    opposite.

    Parameters
    ----------
    recent_count : int
        Total reports in the recent half.
    previous_count : int
        Total reports in the previous half.
    n_recent : int
        Number of quarters in the recent half.
    n_previous : int
        Number of quarters in the previous half.

    Returns
    -------
    float
        Trend score in [-1.0, +1.0].
    """
    if n_recent == 0 or n_previous == 0:
        return 0.0

    recent_rate   = recent_count   / n_recent
    previous_rate = previous_count / n_previous
    max_rate = max(recent_rate, previous_rate)

    if max_rate == 0.0:
        return 0.0

    return (recent_rate - previous_rate) / max_rate


def _classify_direction(
    trend_score: float,
    n_quarters_with_data: int,
    stable_band: float = STABLE_BAND,
) -> str:
    """
    Map a numeric trend_score to a direction label.

    Parameters
    ----------
    trend_score : float
    n_quarters_with_data : int
        Quarters that had at least one report.  If < TREND_MIN_QUARTERS,
        returns TREND_INSUFFICIENT_DATA regardless of score.
    stable_band : float
        Half-width of the stable band around 0.

    Returns
    -------
    str — one of TREND_INCREASING | TREND_DECREASING | TREND_STABLE |
                 TREND_INSUFFICIENT_DATA
    """
    if n_quarters_with_data < TREND_MIN_QUARTERS:
        return TREND_INSUFFICIENT_DATA
    if trend_score > stable_band:
        return TREND_INCREASING
    if trend_score < -stable_band:
        return TREND_DECREASING
    return TREND_STABLE


# ---------------------------------------------------------------------------
# Public entry point — single pair
# ---------------------------------------------------------------------------

def analyse_trend(
    df: pd.DataFrame,
    drug: str,
    event: str,
    stable_band: float = STABLE_BAND,
) -> TrendResult:
    """
    Compute the temporal trend for one (drug, adverse_event) pair.

    Parameters
    ----------
    df : pd.DataFrame
        Normalised FAERS DataFrame.
    drug : str
        Normalised drug name.
    event : str
        Normalised adverse event term.
    stable_band : float
        Override for the stable-band threshold (useful in tests).

    Returns
    -------
    TrendResult
    """
    # Total reports (including those with missing dates, for completeness)
    total_mask = (df["drug_name"] == drug) & (df["adverse_event"] == event)
    total_reports = int(total_mask.sum())

    # Date range (from rows with valid dates only)
    dated = df.loc[total_mask & df["report_date"].notna(), "report_date"]
    date_range_start: Optional[str] = None
    date_range_end: Optional[str] = None
    if not dated.empty:
        date_range_start = pd.to_datetime(dated.min()).date().isoformat()
        date_range_end   = pd.to_datetime(dated.max()).date().isoformat()

    # Per-quarter counts
    quarters = quarterly_counts(df, drug, event)
    n_total_quarters = len(quarters)

    # Number of quarters that actually have ≥1 report
    n_quarters_with_data = sum(1 for q in quarters if q.count > 0)

    if n_total_quarters == 0:
        # No dated reports at all
        return TrendResult(
            drug_name=drug,
            adverse_event=event,
            quarters=[],
            recent_count=0,
            previous_count=0,
            pct_change=None,
            trend_score=0.0,
            direction=TREND_INSUFFICIENT_DATA,
            total_reports=total_reports,
            date_range_start=None,
            date_range_end=None,
        )

    # Split into previous (older) half and recent (newer) half.
    # For odd lengths, the extra quarter goes to the recent half so we
    # never miss the most recent data.
    #   n=1 → previous=[], recent=[q0]
    #   n=2 → previous=[q0], recent=[q1]
    #   n=3 → previous=[q0], recent=[q1,q2]
    #   n=4 → previous=[q0,q1], recent=[q2,q3]
    split = n_total_quarters // 2
    previous_quarters = quarters[:split]
    recent_quarters   = quarters[split:]

    previous_count = sum(q.count for q in previous_quarters)
    recent_count   = sum(q.count for q in recent_quarters)
    n_previous = len(previous_quarters)
    n_recent   = len(recent_quarters)

    # Percentage change
    pct_change: Optional[float] = None
    if previous_count > 0:
        pct_change = ((recent_count - previous_count) / previous_count) * 100.0

    # Trend score
    trend_score = _compute_trend_score(
        recent_count, previous_count, n_recent, n_previous
    )

    # Direction
    direction = _classify_direction(n_quarters_with_data, n_quarters_with_data, stable_band)
    # Re-classify using trend_score now that we have it
    direction = _classify_direction(trend_score, n_quarters_with_data, stable_band)

    logger.debug(
        "analyse_trend: %s / %s → direction=%s score=%.3f "
        "recent=%d previous=%d quarters=%d",
        drug, event, direction, trend_score,
        recent_count, previous_count, n_total_quarters,
    )

    return TrendResult(
        drug_name=drug,
        adverse_event=event,
        quarters=quarters,
        recent_count=recent_count,
        previous_count=previous_count,
        pct_change=pct_change,
        trend_score=round(trend_score, 6),
        direction=direction,
        total_reports=total_reports,
        date_range_start=date_range_start,
        date_range_end=date_range_end,
    )


# ---------------------------------------------------------------------------
# Public entry point — all pairs (or filtered to one drug)
# ---------------------------------------------------------------------------

def analyse_all_trends(
    df: pd.DataFrame,
    drug_filter: Optional[str] = None,
) -> list[TrendResult]:
    """
    Compute temporal trends for every (drug, adverse_event) pair in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Normalised FAERS DataFrame.
    drug_filter : str, optional
        If supplied, only analyse pairs for this drug.

    Returns
    -------
    list[TrendResult]
        Sorted by |trend_score| descending (most extreme trends first),
        with "insufficient_data" results at the end.
    """
    if df.empty:
        return []

    drugs = [drug_filter] if drug_filter else sorted(df["drug_name"].unique())
    events = sorted(df["adverse_event"].unique())

    results: list[TrendResult] = []
    for drug in drugs:
        for event in events:
            # Skip pairs where this drug has never reported this event
            pair_exists = (
                (df["drug_name"] == drug) & (df["adverse_event"] == event)
            ).any()
            if not pair_exists:
                continue
            results.append(analyse_trend(df, drug, event))

    # Sort: insufficient_data last; within each group, sort by |trend_score| descending
    results.sort(
        key=lambda r: (
            r.direction == TREND_INSUFFICIENT_DATA,
            -abs(r.trend_score),
        )
    )

    return results
