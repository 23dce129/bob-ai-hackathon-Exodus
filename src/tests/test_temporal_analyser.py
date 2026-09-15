"""
test_temporal_analyser.py — Unit tests for deterministic temporal trend analysis.

Test data design
----------------
All synthetic DataFrames are built using _make_dated_df(), which accepts a
list of (date_string, drug, event) tuples.  Dates are in YYYYMMDD format and
are parsed by the same path used in faers_ingestor.normalize_dataframe().

For each trend direction we construct a dataset with a known hand-calculated
outcome and assert the implementation agrees exactly.

Hand-calculations for each scenario are shown inline in the test docstrings.

Constants used by the analyser (checked once here so any accidental change
to the module breaks a test):
    STABLE_BAND      = 0.10
    TREND_MIN_QUARTERS = 2
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from core.temporal_analyser import (
    STABLE_BAND,
    _classify_direction,
    _compute_trend_score,
    analyse_all_trends,
    analyse_trend,
    quarterly_counts,
)
from core.schemas import (
    TREND_DECREASING,
    TREND_INCREASING,
    TREND_INSUFFICIENT_DATA,
    TREND_MIN_QUARTERS,
    TREND_STABLE,
    QuarterCount,
    TrendResult,
)

# ---------------------------------------------------------------------------
# Synthetic DataFrame factory
# ---------------------------------------------------------------------------

def _make_dated_df(
    triples: list[tuple[str, str, str]],   # (YYYYMMDD, drug, event)
    outcome: str = "OT",
) -> pd.DataFrame:
    """
    Build a minimal normalised FAERS DataFrame from (date, drug, event) triples.
    Dates are stored as pd.Timestamp via to_datetime with format="%Y%m%d".
    """
    rows = []
    for i, (date_str, drug, event) in enumerate(triples):
        rows.append({
            "report_id":     str(i),
            "drug_name":     drug,
            "adverse_event": event,
            "outcome_code":  outcome,
            "report_date":   pd.to_datetime(date_str, format="%Y%m%d"),
            "age_years":     float("nan"),
            "sex":           "UNK",
        })
    return pd.DataFrame(rows)


def _make_nodates_df(drug: str, event: str, n: int = 5) -> pd.DataFrame:
    """All reports have NaT dates — no temporal information."""
    rows = [
        {
            "report_id":     str(i),
            "drug_name":     drug,
            "adverse_event": event,
            "outcome_code":  "OT",
            "report_date":   pd.NaT,
            "age_years":     float("nan"),
            "sex":           "UNK",
        }
        for i in range(n)
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pre-built scenario DataFrames (used across multiple tests)
# ---------------------------------------------------------------------------

# INCREASING: 1 report in Q1 2022, 8 reports in Q1-Q2 2023
# Quarters: 2022-Q1(1), 2022-Q2(0), 2022-Q3(0), 2022-Q4(0), 2023-Q1(4), 2023-Q2(4)
# → 6 quarters total; split: previous=[Q1,Q2,Q3] (sum=1), recent=[Q4,Q1,Q2] (sum=8)
# previous_rate = 1/3, recent_rate = 8/3, max_rate = 8/3
# trend_score = (8/3 - 1/3) / (8/3) = (7/3)/(8/3) = 7/8 = 0.875  >> STABLE_BAND
_INCREASING_TRIPLES = (
    [("20220101", "drugX", "eventA")]           # Q1 2022
    + [("20230101", "drugX", "eventA")] * 4     # Q1 2023
    + [("20230401", "drugX", "eventA")] * 4     # Q2 2023
)

# DECREASING: heavy early, sparse recent
# Q1 2022(8), Q2 2022(4), Q1 2023(1)
# Quarters span: 2022-Q1..2023-Q1 = 5 quarters
# split: previous=[Q1,Q2] (sum=12), recent=[Q3,Q4,Q1_2023] (sum=1)
# previous_rate = 12/2 = 6, recent_rate = 1/3, max_rate = 6
# trend_score = (1/3 - 6) / 6 = (-17/3)/6 = -17/18 ≈ -0.944  << -STABLE_BAND
_DECREASING_TRIPLES = (
    [("20220101", "drugX", "eventA")] * 8       # Q1 2022
    + [("20220401", "drugX", "eventA")] * 4     # Q2 2022
    + [("20230101", "drugX", "eventA")]          # Q1 2023
)

# STABLE: uniform rate across all quarters
# 4 reports per quarter, 4 quarters
# previous=[Q1,Q2] sum=8, recent=[Q3,Q4] sum=8
# previous_rate = 4, recent_rate = 4, trend_score = 0.0
_STABLE_TRIPLES = (
    [("20220101", "drugX", "eventA")] * 4       # Q1 2022
    + [("20220401", "drugX", "eventA")] * 4     # Q2 2022
    + [("20220701", "drugX", "eventA")] * 4     # Q3 2022
    + [("20221001", "drugX", "eventA")] * 4     # Q4 2022
)

# SINGLE QUARTER: only one quarter with data → insufficient_data
_SINGLE_QUARTER_TRIPLES = [("20220101", "drugX", "eventA")] * 5


# ---------------------------------------------------------------------------
# 1. quarterly_counts
# ---------------------------------------------------------------------------

class TestQuarterlyCounts:

    def test_returns_list_of_quarter_counts(self):
        df = _make_dated_df([("20220101", "aspirin", "nausea")])
        result = quarterly_counts(df, "aspirin", "nausea")
        assert isinstance(result, list)
        assert all(isinstance(q, QuarterCount) for q in result)

    def test_correct_quarter_label_q1(self):
        df = _make_dated_df([("20220101", "aspirin", "nausea")])
        result = quarterly_counts(df, "aspirin", "nausea")
        assert result[0].quarter == "2022-Q1"

    def test_correct_quarter_label_q3(self):
        df = _make_dated_df([("20220801", "aspirin", "nausea")])
        result = quarterly_counts(df, "aspirin", "nausea")
        assert result[0].quarter == "2022-Q3"

    def test_counts_per_quarter_correct(self):
        triples = (
            [("20220101", "aspirin", "nausea")] * 3   # Q1 2022
            + [("20220401", "aspirin", "nausea")] * 5  # Q2 2022
        )
        df = _make_dated_df(triples)
        result = quarterly_counts(df, "aspirin", "nausea")
        counts = {q.quarter: q.count for q in result}
        assert counts["2022-Q1"] == 3
        assert counts["2022-Q2"] == 5

    def test_interior_gap_quarters_included_with_zero(self):
        """Q1 2022 and Q3 2022 have reports; Q2 2022 (the gap) must appear with count=0."""
        triples = (
            [("20220101", "aspirin", "nausea")]        # Q1 2022
            + [("20220701", "aspirin", "nausea")]       # Q3 2022
        )
        df = _make_dated_df(triples)
        result = quarterly_counts(df, "aspirin", "nausea")
        quarters = {q.quarter for q in result}
        assert "2022-Q2" in quarters
        gap = next(q for q in result if q.quarter == "2022-Q2")
        assert gap.count == 0

    def test_no_leading_zero_quarters(self):
        """Only the range min→max is filled; no padding before the first report."""
        df = _make_dated_df([("20230101", "aspirin", "nausea")])
        result = quarterly_counts(df, "aspirin", "nausea")
        assert result[0].quarter == "2023-Q1"
        assert len(result) == 1

    def test_sorted_oldest_to_newest(self):
        triples = (
            [("20230101", "aspirin", "nausea")]
            + [("20220101", "aspirin", "nausea")]
        )
        df = _make_dated_df(triples)
        result = quarterly_counts(df, "aspirin", "nausea")
        labels = [q.quarter for q in result]
        assert labels == sorted(labels)

    def test_empty_for_missing_drug(self):
        df = _make_dated_df([("20220101", "aspirin", "nausea")])
        result = quarterly_counts(df, "warfarin", "nausea")
        assert result == []

    def test_empty_for_all_nat_dates(self):
        df = _make_nodates_df("aspirin", "nausea", n=5)
        result = quarterly_counts(df, "aspirin", "nausea")
        assert result == []

    def test_other_drug_reports_excluded(self):
        """Reports for drugB must not appear in drugA's quarter counts."""
        triples = (
            [("20220101", "drugA", "nausea")] * 3
            + [("20220101", "drugB", "nausea")] * 10
        )
        df = _make_dated_df(triples)
        result = quarterly_counts(df, "drugA", "nausea")
        total = sum(q.count for q in result)
        assert total == 3


# ---------------------------------------------------------------------------
# 2. _compute_trend_score
# ---------------------------------------------------------------------------

class TestComputeTrendScore:

    def test_equal_rates_score_zero(self):
        # recent_rate = 4/2 = 2, previous_rate = 4/2 = 2 → score = 0
        assert _compute_trend_score(4, 4, 2, 2) == pytest.approx(0.0)

    def test_all_recent_score_plus_one(self):
        # previous_count=0 → previous_rate=0, recent_rate=5/1=5, max=5
        # score = (5-0)/5 = 1.0
        assert _compute_trend_score(5, 0, 1, 1) == pytest.approx(1.0)

    def test_all_previous_score_minus_one(self):
        # recent_count=0 → recent_rate=0, previous_rate=5, max=5
        # score = (0-5)/5 = -1.0
        assert _compute_trend_score(0, 5, 1, 1) == pytest.approx(-1.0)

    def test_score_in_minus_one_to_plus_one(self):
        for rc, pc, nr, np_ in [(3,1,2,2), (1,3,2,2), (10,2,4,4), (0,0,2,2)]:
            s = _compute_trend_score(rc, pc, nr, np_)
            assert -1.0 <= s <= 1.0, f"score={s} out of range for ({rc},{pc},{nr},{np_})"

    def test_zero_n_recent_returns_zero(self):
        assert _compute_trend_score(5, 3, 0, 2) == pytest.approx(0.0)

    def test_zero_n_previous_returns_zero(self):
        assert _compute_trend_score(5, 3, 2, 0) == pytest.approx(0.0)

    def test_all_zeros_returns_zero(self):
        assert _compute_trend_score(0, 0, 2, 2) == pytest.approx(0.0)

    def test_known_increasing_value(self):
        """
        recent=8, previous=1, n_recent=3, n_previous=3
        recent_rate=8/3, previous_rate=1/3, max_rate=8/3
        score = (8/3 - 1/3)/(8/3) = (7/3)/(8/3) = 7/8 = 0.875
        """
        score = _compute_trend_score(8, 1, 3, 3)
        assert score == pytest.approx(7 / 8, rel=1e-6)

    def test_known_decreasing_value(self):
        """
        recent=1, previous=12, n_recent=3, n_previous=2
        recent_rate=1/3, previous_rate=6, max_rate=6
        score = (1/3 - 6)/6 = (-17/3)/6 = -17/18 ≈ -0.9444
        """
        score = _compute_trend_score(1, 12, 3, 2)
        assert score == pytest.approx(-17 / 18, rel=1e-6)


# ---------------------------------------------------------------------------
# 3. _classify_direction
# ---------------------------------------------------------------------------

class TestClassifyDirection:

    def test_above_stable_band_is_increasing(self):
        assert _classify_direction(STABLE_BAND + 0.01, 4) == TREND_INCREASING

    def test_below_negative_stable_band_is_decreasing(self):
        assert _classify_direction(-(STABLE_BAND + 0.01), 4) == TREND_DECREASING

    def test_within_band_is_stable(self):
        assert _classify_direction(0.0, 4) == TREND_STABLE
        assert _classify_direction(STABLE_BAND - 0.001, 4) == TREND_STABLE
        assert _classify_direction(-(STABLE_BAND - 0.001), 4) == TREND_STABLE

    def test_exact_stable_band_boundary_is_stable(self):
        # score == STABLE_BAND is NOT > STABLE_BAND → stable
        assert _classify_direction(STABLE_BAND, 4) == TREND_STABLE

    def test_insufficient_quarters_overrides_score(self):
        # Even a score of +1.0 must return insufficient_data when n_quarters < 2
        assert _classify_direction(1.0,  0) == TREND_INSUFFICIENT_DATA
        assert _classify_direction(1.0,  1) == TREND_INSUFFICIENT_DATA
        assert _classify_direction(1.0,  TREND_MIN_QUARTERS) == TREND_INCREASING

    def test_exactly_min_quarters_is_sufficient(self):
        result = _classify_direction(0.5, TREND_MIN_QUARTERS)
        assert result != TREND_INSUFFICIENT_DATA


# ---------------------------------------------------------------------------
# 4. analyse_trend — increasing
# ---------------------------------------------------------------------------

class TestIncreasingTrend:
    """
    Dataset: drugX / eventA
    Quarters (derived from _INCREASING_TRIPLES):
        2022-Q1: 1
        2022-Q2: 0   (interior gap — filled with 0)
        2022-Q3: 0
        2022-Q4: 0
        2023-Q1: 4
        2023-Q2: 4

    Total quarters: 6 → split = 3
    previous = [2022-Q1, 2022-Q2, 2022-Q3]  sum = 1
    recent   = [2022-Q4, 2023-Q1, 2023-Q2]  sum = 8

    trend_score = _compute_trend_score(8, 1, 3, 3) = 7/8 = 0.875
    pct_change  = (8-1)/1 × 100 = 700.0
    direction   = "increasing"
    """

    @pytest.fixture(scope="class")
    def result(self) -> TrendResult:
        df = _make_dated_df(_INCREASING_TRIPLES)
        return analyse_trend(df, "drugX", "eventA")

    def test_direction_is_increasing(self, result):
        assert result.direction == TREND_INCREASING

    def test_trend_score_positive(self, result):
        assert result.trend_score > 0

    def test_trend_score_above_stable_band(self, result):
        assert result.trend_score > STABLE_BAND

    def test_recent_count_greater_than_previous(self, result):
        assert result.recent_count > result.previous_count

    def test_pct_change_positive(self, result):
        assert result.pct_change is not None
        assert result.pct_change > 0

    def test_pct_change_known_value(self, result):
        # (8-1)/1 × 100 = 700.0
        assert result.pct_change == pytest.approx(700.0, rel=1e-4)

    def test_trend_score_known_value(self, result):
        assert result.trend_score == pytest.approx(7 / 8, rel=1e-4)

    def test_quarters_sorted_oldest_to_newest(self, result):
        labels = [q.quarter for q in result.quarters]
        assert labels == sorted(labels)

    def test_total_reports_correct(self, result):
        assert result.total_reports == len(_INCREASING_TRIPLES)

    def test_date_range_set(self, result):
        assert result.date_range_start is not None
        assert result.date_range_end is not None

    def test_to_dict_direction(self, result):
        assert result.to_dict()["direction"] == TREND_INCREASING


# ---------------------------------------------------------------------------
# 5. analyse_trend — decreasing
# ---------------------------------------------------------------------------

class TestDecreasingTrend:
    """
    Dataset: drugX / eventA
    Quarters:
        2022-Q1: 8
        2022-Q2: 4
        2022-Q3: 0
        2022-Q4: 0
        2023-Q1: 1

    Total quarters: 5 → split = 2
    previous = [2022-Q1, 2022-Q2]            sum = 12
    recent   = [2022-Q3, 2022-Q4, 2023-Q1]  sum = 1

    trend_score = _compute_trend_score(1, 12, 3, 2) = -17/18 ≈ -0.944
    pct_change  = (1-12)/12 × 100 = -91.67
    direction   = "decreasing"
    """

    @pytest.fixture(scope="class")
    def result(self) -> TrendResult:
        df = _make_dated_df(_DECREASING_TRIPLES)
        return analyse_trend(df, "drugX", "eventA")

    def test_direction_is_decreasing(self, result):
        assert result.direction == TREND_DECREASING

    def test_trend_score_negative(self, result):
        assert result.trend_score < 0

    def test_trend_score_below_negative_stable_band(self, result):
        assert result.trend_score < -STABLE_BAND

    def test_recent_count_less_than_previous(self, result):
        assert result.recent_count < result.previous_count

    def test_pct_change_negative(self, result):
        assert result.pct_change is not None
        assert result.pct_change < 0

    def test_pct_change_known_value(self, result):
        # (1-12)/12 × 100 = -91.6667
        assert result.pct_change == pytest.approx(-91.6667, rel=1e-3)

    def test_trend_score_known_value(self, result):
        assert result.trend_score == pytest.approx(-17 / 18, rel=1e-4)

    def test_to_dict_direction(self, result):
        assert result.to_dict()["direction"] == TREND_DECREASING


# ---------------------------------------------------------------------------
# 6. analyse_trend — stable
# ---------------------------------------------------------------------------

class TestStableTrend:
    """
    Dataset: 4 reports in each of Q1–Q4 2022 (uniform rate).

    Quarters: 4 → split = 2
    previous = [Q1, Q2]  sum = 8, n = 2
    recent   = [Q3, Q4]  sum = 8, n = 2

    trend_score = (8/2 - 8/2) / max(4,4) = 0/4 = 0.0
    pct_change  = (8-8)/8 × 100 = 0.0
    direction   = "stable"
    """

    @pytest.fixture(scope="class")
    def result(self) -> TrendResult:
        df = _make_dated_df(_STABLE_TRIPLES)
        return analyse_trend(df, "drugX", "eventA")

    def test_direction_is_stable(self, result):
        assert result.direction == TREND_STABLE

    def test_trend_score_zero(self, result):
        assert result.trend_score == pytest.approx(0.0, abs=1e-9)

    def test_pct_change_zero(self, result):
        assert result.pct_change == pytest.approx(0.0, abs=1e-9)

    def test_recent_equals_previous(self, result):
        assert result.recent_count == result.previous_count

    def test_to_dict_direction(self, result):
        assert result.to_dict()["direction"] == TREND_STABLE


# ---------------------------------------------------------------------------
# 7. analyse_trend — insufficient historical data
# ---------------------------------------------------------------------------

class TestInsufficientData:

    def test_single_quarter_is_insufficient(self):
        df = _make_dated_df(_SINGLE_QUARTER_TRIPLES)
        result = analyse_trend(df, "drugX", "eventA")
        assert result.direction == TREND_INSUFFICIENT_DATA

    def test_all_nat_dates_is_insufficient(self):
        df = _make_nodates_df("drugX", "eventA", n=10)
        result = analyse_trend(df, "drugX", "eventA")
        assert result.direction == TREND_INSUFFICIENT_DATA

    def test_all_nat_dates_quarters_empty(self):
        df = _make_nodates_df("drugX", "eventA", n=10)
        result = analyse_trend(df, "drugX", "eventA")
        assert result.quarters == []

    def test_no_reports_at_all(self):
        df = _make_dated_df([("20220101", "otherDrug", "otherEvent")])
        result = analyse_trend(df, "drugX", "eventA")
        assert result.direction == TREND_INSUFFICIENT_DATA
        assert result.total_reports == 0

    def test_trend_score_zero_when_insufficient(self):
        df = _make_dated_df(_SINGLE_QUARTER_TRIPLES)
        result = analyse_trend(df, "drugX", "eventA")
        assert result.trend_score == pytest.approx(0.0, abs=1e-9)

    def test_pct_change_none_when_previous_zero(self):
        """Single quarter: previous_count=0 → pct_change must be None."""
        df = _make_dated_df(_SINGLE_QUARTER_TRIPLES)
        result = analyse_trend(df, "drugX", "eventA")
        assert result.pct_change is None

    def test_total_reports_correct_despite_no_dates(self):
        df = _make_nodates_df("drugX", "eventA", n=7)
        result = analyse_trend(df, "drugX", "eventA")
        assert result.total_reports == 7


# ---------------------------------------------------------------------------
# 8. analyse_trend — structure and serialisation
# ---------------------------------------------------------------------------

class TestTrendResultStructure:

    @pytest.fixture(scope="class")
    def result(self) -> TrendResult:
        df = _make_dated_df(_INCREASING_TRIPLES)
        return analyse_trend(df, "drugX", "eventA")

    def test_to_dict_has_all_keys(self, result):
        d = result.to_dict()
        required = {
            "drug_name", "adverse_event", "quarters",
            "recent_count", "previous_count", "pct_change",
            "trend_score", "direction", "total_reports",
            "date_range_start", "date_range_end",
        }
        assert required.issubset(d.keys())

    def test_quarters_in_to_dict_are_dicts(self, result):
        d = result.to_dict()
        assert all(isinstance(q, dict) for q in d["quarters"])

    def test_quarter_dict_has_all_keys(self, result):
        q = result.to_dict()["quarters"][0]
        assert {"quarter", "year", "quarter_num", "count"}.issubset(q.keys())

    def test_pct_change_rounded_to_2dp(self, result):
        d = result.to_dict()
        if d["pct_change"] is not None:
            # String representation should have at most 2 decimal places
            parts = str(d["pct_change"]).split(".")
            assert len(parts[-1]) <= 2

    def test_trend_score_rounded_to_4dp(self, result):
        d = result.to_dict()
        parts = str(abs(d["trend_score"])).split(".")
        assert len(parts[-1]) <= 4


# ---------------------------------------------------------------------------
# 9. analyse_all_trends
# ---------------------------------------------------------------------------

class TestAnalyseAllTrends:

    def test_returns_list(self, clean_df):
        result = analyse_all_trends(clean_df)
        assert isinstance(result, list)

    def test_all_elements_are_trend_results(self, clean_df):
        result = analyse_all_trends(clean_df)
        assert all(isinstance(r, TrendResult) for r in result)

    def test_drug_filter_limits_to_one_drug(self, clean_df):
        result = analyse_all_trends(clean_df, drug_filter="warfarin")
        assert all(r.drug_name == "warfarin" for r in result)

    def test_insufficient_data_results_are_last(self, clean_df):
        result = analyse_all_trends(clean_df)
        if len(result) < 2:
            return
        first_insufficient = next(
            (i for i, r in enumerate(result) if r.direction == TREND_INSUFFICIENT_DATA),
            None,
        )
        if first_insufficient is not None:
            # Everything after the first insufficient must also be insufficient
            after = result[first_insufficient:]
            assert all(r.direction == TREND_INSUFFICIENT_DATA for r in after)

    def test_empty_dataframe_returns_empty_list(self):
        from core.schemas import REQUIRED_INTERNAL_COLUMNS
        df = pd.DataFrame(columns=REQUIRED_INTERNAL_COLUMNS)
        assert analyse_all_trends(df) == []

    def test_each_pair_appears_at_most_once(self, clean_df):
        result = analyse_all_trends(clean_df)
        pairs = [(r.drug_name, r.adverse_event) for r in result]
        assert len(pairs) == len(set(pairs)), "Duplicate (drug, event) pairs in results"


# ---------------------------------------------------------------------------
# 10. API route smoke test (uses FastAPI TestClient)
# ---------------------------------------------------------------------------

class TestSignalsRoutes:

    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        from main import app
        return TestClient(app)

    def test_health_still_works(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_sample_signals_returns_200(self, client):
        r = client.get("/api/signals/sample")
        assert r.status_code == 200

    def test_sample_signals_has_flagged_signals_key(self, client):
        r = client.get("/api/signals/sample")
        assert "flagged_signals" in r.json()

    def test_sample_signals_total_evaluated_positive(self, client):
        r = client.get("/api/signals/sample")
        assert r.json()["total_evaluated"] > 0

    def test_flagged_signals_have_trend_key(self, client):
        r = client.get("/api/signals/sample?include_trends=true")
        data = r.json()
        for sig in data["flagged_signals"]:
            assert "trend" in sig, f"Signal {sig['drug_name']}/{sig['adverse_event']} missing 'trend'"

    def test_trend_endpoint_known_drug_event(self, client):
        r = client.get("/api/signals/trend/warfarin/bleeding")
        assert r.status_code == 200
        body = r.json()
        assert body["drug_name"] == "warfarin"
        assert body["adverse_event"] == "bleeding"
        assert "direction" in body
        assert "quarters" in body

    def test_trend_endpoint_case_insensitive(self, client):
        r = client.get("/api/signals/trend/WARFARIN/BLEEDING")
        assert r.status_code == 200

    def test_trend_endpoint_unknown_pair_returns_404(self, client):
        r = client.get("/api/signals/trend/unknowndrug123/unknownevent456")
        assert r.status_code == 404

    def test_drug_trends_endpoint_returns_200(self, client):
        r = client.get("/api/signals/trends/warfarin")
        assert r.status_code == 200

    def test_drug_trends_endpoint_has_trends_key(self, client):
        r = client.get("/api/signals/trends/warfarin")
        assert "trends" in r.json()

    def test_drug_trends_endpoint_unknown_drug_returns_404(self, client):
        r = client.get("/api/signals/trends/completelyunknowndrug999")
        assert r.status_code == 404

    def test_analyze_post_returns_200(self, client):
        r = client.post("/api/signals/analyze", json={"drug_name": "warfarin"})
        assert r.status_code == 200

    def test_analyze_post_filters_to_drug(self, client):
        r = client.post("/api/signals/analyze", json={"drug_name": "warfarin"})
        data = r.json()
        for sig in data["all_signals"]:
            assert sig["drug_name"] == "warfarin"
