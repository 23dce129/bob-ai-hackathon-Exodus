"""
test_priority_scorer.py — Unit tests for the Safety Signal Priority Engine.

Test strategy
-------------
Each scoring component is tested in isolation with hand-calculated expected
values, then the full score_signal() pipeline is tested end-to-end.

All expected values in this file were computed by hand before being asserted
against the implementation.  The calculation is shown in the test docstring.

Key invariants asserted throughout:
  - final_score ∈ [0, config.max_score]
  - sum(c.earned for c in components) ≈ final_score  (within rounding)
  - disclaimer is always present and mentions "HACKATHON"
  - formula is always a non-empty string
  - component names are deterministic
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from core.priority_scorer import (
    _score_cluster,
    _score_prr,
    _score_serious_outcomes,
    _score_statistical,
    _score_trend,
    _score_volume,
    score_all_signals,
    score_signal,
)
from core.schemas import (
    DEFAULT_PVAL_FULL,
    DEFAULT_PVAL_NONE,
    DEFAULT_PRR_SATURATION,
    DEFAULT_SERIOUS_SATURATION,
    DEFAULT_VOLUME_SATURATION,
    DEFAULT_WEIGHT_CLUSTER,
    DEFAULT_WEIGHT_PRR,
    DEFAULT_WEIGHT_SERIOUS,
    DEFAULT_WEIGHT_STATISTICAL,
    DEFAULT_WEIGHT_TREND,
    DEFAULT_WEIGHT_VOLUME,
    ComponentScore,
    PriorityScore,
    ScoringConfig,
    SignalResult,
    TrendResult,
    TREND_INCREASING,
    TREND_DECREASING,
    TREND_STABLE,
    TREND_INSUFFICIENT_DATA,
)
from core.signal_clusterer import ClusterResult


# ---------------------------------------------------------------------------
# Helpers — build minimal test objects
# ---------------------------------------------------------------------------

def _make_signal(
    drug: str = "warfarin",
    event: str = "bleeding",
    prr: float = 8.3,
    p_value: float = 0.001,
    case_count: int = 40,
    flagged: bool = True,
) -> SignalResult:
    return SignalResult(
        drug_name=drug,
        adverse_event=event,
        a=case_count, b=60, c=10, d=890,
        prr=prr,
        ci_lower_95=prr * 0.7,
        ci_upper_95=prr * 1.4,
        chi2_statistic=22.0,
        p_value=p_value,
        test_method="chi2",
        signal_flag=flagged,
        flag_reason="Flagged." if flagged else "Not flagged.",
    )


def _make_df(
    drug: str = "warfarin",
    event: str = "bleeding",
    n_serious: int = 20,
    n_non_serious: int = 20,
    outcome_serious: str = "HO",
    outcome_non_serious: str = "OT",
) -> pd.DataFrame:
    """Build a minimal FAERS DataFrame with controllable serious-outcome counts."""
    rows = (
        [{"drug_name": drug, "adverse_event": event,
          "outcome_code": outcome_serious, "report_id": str(i),
          "report_date": pd.NaT, "age_years": float("nan"), "sex": "UNK"}
         for i in range(n_serious)]
        + [{"drug_name": drug, "adverse_event": event,
            "outcome_code": outcome_non_serious, "report_id": str(n_serious + i),
            "report_date": pd.NaT, "age_years": float("nan"), "sex": "UNK"}
           for i in range(n_non_serious)]
    )
    return pd.DataFrame(rows)


def _make_trend(
    direction: str = TREND_INCREASING,
    trend_score: float = 0.875,
) -> TrendResult:
    return TrendResult(
        drug_name="warfarin",
        adverse_event="bleeding",
        quarters=[],
        recent_count=8,
        previous_count=1,
        pct_change=700.0,
        trend_score=trend_score,
        direction=direction,
        total_reports=9,
        date_range_start="2022-01-01",
        date_range_end="2023-06-30",
    )


def _make_cluster(
    drug: str = "warfarin",
    purity: float = 0.9,
    is_noise: bool = False,
) -> ClusterResult:
    return ClusterResult(
        cluster_id=0 if not is_noise else -1,
        is_noise=is_noise,
        size=10,
        dominant_drug=drug,
        dominant_events=["bleeding"],
        dominant_outcomes=["HO (Hospitalisation)"],
        drug_purity=purity,
        event_purity=0.8,
        label=f"{drug} / bleeding (n=10, purity={purity:.2f})",
        report_ids=[str(i) for i in range(10)],
    )


# ---------------------------------------------------------------------------
# 1. ScoringConfig
# ---------------------------------------------------------------------------

class TestScoringConfig:

    def test_default_max_score_is_100(self):
        config = ScoringConfig()
        assert config.max_score == 100

    def test_weights_sum_to_100_by_default(self):
        c = ScoringConfig()
        total = (c.weight_prr + c.weight_volume + c.weight_statistical
                 + c.weight_serious + c.weight_trend + c.weight_cluster)
        assert total == 100

    def test_custom_weights_change_max_score(self):
        c = ScoringConfig(weight_prr=50, weight_volume=50, weight_statistical=0,
                          weight_serious=0, weight_trend=0, weight_cluster=0)
        assert c.max_score == 100

    def test_trend_direction_bonus_defaults_populated(self):
        c = ScoringConfig()
        assert "increasing" in c.trend_direction_bonus
        assert "decreasing" in c.trend_direction_bonus
        assert c.trend_direction_bonus["increasing"] == 1.0
        assert c.trend_direction_bonus["decreasing"] == 0.0


# ---------------------------------------------------------------------------
# 2. _score_prr
# ---------------------------------------------------------------------------

class TestScorePRR:
    """
    Default saturation = 20.0, weight = 30.

    PRR=8.3:  norm = min(8.3/20, 1.0) = 0.415  → earned = 0.415 × 30 = 12.45
    PRR=20.0: norm = min(20/20, 1.0)  = 1.0    → earned = 1.0 × 30  = 30.0
    PRR=25.0: norm = min(25/20, 1.0)  = 1.0    → earned = 1.0 × 30  = 30.0  (saturated)
    PRR=None: earned = 0
    """

    def test_known_prr_8_3(self):
        result = _score_prr(8.3, 30, 20.0)
        assert result.earned == pytest.approx(0.415 * 30, rel=1e-4)

    def test_prr_at_saturation_earns_full(self):
        result = _score_prr(20.0, 30, 20.0)
        assert result.earned == pytest.approx(30.0)

    def test_prr_above_saturation_capped(self):
        result = _score_prr(40.0, 30, 20.0)
        assert result.earned == pytest.approx(30.0)

    def test_prr_none_earns_zero(self):
        result = _score_prr(None, 30, 20.0)
        assert result.earned == 0.0

    def test_prr_zero_earns_zero(self):
        result = _score_prr(0.0, 30, 20.0)
        assert result.earned == 0.0

    def test_name_is_prr(self):
        result = _score_prr(5.0, 30, 20.0)
        assert result.name == "PRR"

    def test_maximum_is_weight(self):
        result = _score_prr(5.0, 30, 20.0)
        assert result.maximum == 30

    def test_formula_is_non_empty(self):
        result = _score_prr(5.0, 30, 20.0)
        assert len(result.formula) > 0

    def test_raw_value_is_prr(self):
        result = _score_prr(8.3, 30, 20.0)
        assert result.raw_value == pytest.approx(8.3)

    def test_summary_line_format(self):
        result = _score_prr(20.0, 30, 20.0)
        assert result.summary_line() == "PRR: 30.0/30"


# ---------------------------------------------------------------------------
# 3. _score_volume
# ---------------------------------------------------------------------------

class TestScoreVolume:
    """
    Default saturation = 100, weight = 20.

    count=40:   norm = 40/100 = 0.4    → earned = 0.4 × 20 = 8.0
    count=100:  norm = 1.0             → earned = 20.0
    count=150:  norm = 1.0 (capped)    → earned = 20.0
    """

    def test_known_count_40(self):
        result = _score_volume(40, 20, 100)
        assert result.earned == pytest.approx(8.0)

    def test_at_saturation_earns_full(self):
        result = _score_volume(100, 20, 100)
        assert result.earned == pytest.approx(20.0)

    def test_above_saturation_capped(self):
        result = _score_volume(200, 20, 100)
        assert result.earned == pytest.approx(20.0)

    def test_zero_count_earns_zero(self):
        result = _score_volume(0, 20, 100)
        assert result.earned == 0.0

    def test_name_is_volume(self):
        assert _score_volume(10, 20, 100).name == "Volume"

    def test_raw_value_is_count(self):
        assert _score_volume(40, 20, 100).raw_value == 40.0


# ---------------------------------------------------------------------------
# 4. _score_statistical
# ---------------------------------------------------------------------------

class TestScoreStatistical:
    """
    Default pval_full = 0.001, pval_none = 0.05, weight = 20.

    p=0.0001:  ≤ pval_full → norm=1.0  → earned=20.0
    p=0.001:   = pval_full → norm=1.0  → earned=20.0
    p=0.05:    = pval_none → norm=0.0  → earned=0.0
    p=0.06:    > pval_none → norm=0.0  → earned=0.0
    p=0.025:   midpoint    → norm=(0.05-0.025)/(0.05-0.001)=0.025/0.049≈0.5102
                             → earned≈10.2

    General formula: norm = (pval_none - p) / (pval_none - pval_full)
    """

    def test_p_below_full_earns_max(self):
        result = _score_statistical(0.0001, 20, 0.001, 0.05)
        assert result.earned == pytest.approx(20.0)

    def test_p_equal_full_earns_max(self):
        result = _score_statistical(0.001, 20, 0.001, 0.05)
        assert result.earned == pytest.approx(20.0)

    def test_p_equal_none_earns_zero(self):
        result = _score_statistical(0.05, 20, 0.001, 0.05)
        assert result.earned == pytest.approx(0.0)

    def test_p_above_none_earns_zero(self):
        result = _score_statistical(0.10, 20, 0.001, 0.05)
        assert result.earned == pytest.approx(0.0)

    def test_p_midpoint_interpolated(self):
        p = 0.025
        expected_norm = (0.05 - p) / (0.05 - 0.001)
        result = _score_statistical(p, 20, 0.001, 0.05)
        assert result.earned == pytest.approx(expected_norm * 20, rel=1e-4)

    def test_p_none_earns_zero(self):
        result = _score_statistical(None, 20, 0.001, 0.05)
        assert result.earned == 0.0

    def test_name_is_statistical_strength(self):
        assert _score_statistical(0.01, 20, 0.001, 0.05).name == "Statistical strength"

    def test_raw_value_is_pvalue(self):
        result = _score_statistical(0.025, 20, 0.001, 0.05)
        assert result.raw_value == pytest.approx(0.025)


# ---------------------------------------------------------------------------
# 5. _score_serious_outcomes
# ---------------------------------------------------------------------------

class TestScoreSeriousOutcomes:
    """
    Default saturation = 0.50, weight = 15.

    20 serious / 40 total = rate 0.5 → norm = min(0.5/0.5, 1.0) = 1.0 → earned = 15
    10 serious / 40 total = rate 0.25 → norm = 0.25/0.5 = 0.5 → earned = 7.5
    0 serious  / 40 total = rate 0.0  → earned = 0
    """

    def test_50_pct_serious_earns_full(self):
        df = _make_df(n_serious=20, n_non_serious=20)
        result = _score_serious_outcomes(df, "warfarin", "bleeding", 15, 0.50)
        assert result.earned == pytest.approx(15.0)

    def test_25_pct_serious_earns_half(self):
        df = _make_df(n_serious=10, n_non_serious=30)
        result = _score_serious_outcomes(df, "warfarin", "bleeding", 15, 0.50)
        assert result.earned == pytest.approx(7.5)

    def test_zero_serious_earns_zero(self):
        df = _make_df(n_serious=0, n_non_serious=40, outcome_serious="OT")
        result = _score_serious_outcomes(df, "warfarin", "bleeding", 15, 0.50)
        assert result.earned == pytest.approx(0.0)

    def test_above_saturation_capped_at_full(self):
        # 100% serious rate → norm = min(1.0/0.5, 1.0) = 1.0
        df = _make_df(n_serious=40, n_non_serious=0)
        result = _score_serious_outcomes(df, "warfarin", "bleeding", 15, 0.50)
        assert result.earned == pytest.approx(15.0)

    def test_death_code_is_serious(self):
        df = _make_df(n_serious=40, n_non_serious=0, outcome_serious="DE")
        result = _score_serious_outcomes(df, "warfarin", "bleeding", 15, 0.50)
        assert result.earned == pytest.approx(15.0)

    def test_ot_code_is_not_serious(self):
        df = _make_df(n_serious=0, n_non_serious=40, outcome_non_serious="OT")
        result = _score_serious_outcomes(df, "warfarin", "bleeding", 15, 0.50)
        assert result.earned == pytest.approx(0.0)

    def test_missing_pair_earns_zero(self):
        df = _make_df()
        result = _score_serious_outcomes(df, "unknowndrug", "unknownevent", 15, 0.50)
        assert result.earned == 0.0

    def test_name_is_serious_outcomes(self):
        df = _make_df()
        result = _score_serious_outcomes(df, "warfarin", "bleeding", 15, 0.50)
        assert result.name == "Serious outcomes"

    def test_raw_value_is_rate(self):
        df = _make_df(n_serious=20, n_non_serious=20)
        result = _score_serious_outcomes(df, "warfarin", "bleeding", 15, 0.50)
        assert result.raw_value == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 6. _score_trend
# ---------------------------------------------------------------------------

class TestScoreTrend:
    """
    Default bonuses: increasing=1.0, stable=0.5, decreasing=0.0, insufficient=0.5.
    Weight = 10.

    increasing → 1.0 × 10 = 10.0
    stable     → 0.5 × 10 = 5.0
    decreasing → 0.0 × 10 = 0.0
    insufficient → 0.5 × 10 = 5.0
    """
    BONUSES = {
        "increasing": 1.0, "stable": 0.5,
        "decreasing": 0.0, "insufficient_data": 0.5,
    }

    def test_increasing_earns_full(self):
        t = _make_trend(TREND_INCREASING)
        result = _score_trend(t, 10, self.BONUSES)
        assert result.earned == pytest.approx(10.0)

    def test_stable_earns_half(self):
        t = _make_trend(TREND_STABLE)
        result = _score_trend(t, 10, self.BONUSES)
        assert result.earned == pytest.approx(5.0)

    def test_decreasing_earns_zero(self):
        t = _make_trend(TREND_DECREASING)
        result = _score_trend(t, 10, self.BONUSES)
        assert result.earned == pytest.approx(0.0)

    def test_insufficient_earns_half(self):
        t = _make_trend(TREND_INSUFFICIENT_DATA)
        result = _score_trend(t, 10, self.BONUSES)
        assert result.earned == pytest.approx(5.0)

    def test_none_trend_earns_half(self):
        result = _score_trend(None, 10, self.BONUSES)
        assert result.earned == pytest.approx(5.0)

    def test_name_is_trend(self):
        result = _score_trend(None, 10, self.BONUSES)
        assert result.name == "Trend"

    def test_raw_value_is_trend_score(self):
        t = _make_trend(TREND_INCREASING, trend_score=0.875)
        result = _score_trend(t, 10, self.BONUSES)
        assert result.raw_value == pytest.approx(0.875)

    def test_raw_value_none_when_no_trend(self):
        result = _score_trend(None, 10, self.BONUSES)
        assert result.raw_value is None


# ---------------------------------------------------------------------------
# 7. _score_cluster
# ---------------------------------------------------------------------------

class TestScoreCluster:
    """
    Weight = 5.

    purity=0.9 → earned = 0.9 × 5 = 4.5
    purity=1.0 → earned = 1.0 × 5 = 5.0
    no match   → earned = 0.0
    noise only → earned = 0.0
    """

    def test_known_purity_0_9(self):
        clusters = [_make_cluster("warfarin", purity=0.9)]
        result = _score_cluster(clusters, "warfarin", 5)
        assert result.earned == pytest.approx(4.5)

    def test_full_purity(self):
        clusters = [_make_cluster("warfarin", purity=1.0)]
        result = _score_cluster(clusters, "warfarin", 5)
        assert result.earned == pytest.approx(5.0)

    def test_no_matching_drug_earns_zero(self):
        clusters = [_make_cluster("ibuprofen", purity=0.9)]
        result = _score_cluster(clusters, "warfarin", 5)
        assert result.earned == pytest.approx(0.0)

    def test_noise_cluster_ignored(self):
        clusters = [_make_cluster("warfarin", purity=1.0, is_noise=True)]
        result = _score_cluster(clusters, "warfarin", 5)
        assert result.earned == pytest.approx(0.0)

    def test_best_purity_used_when_multiple_clusters(self):
        clusters = [
            _make_cluster("warfarin", purity=0.6),
            _make_cluster("warfarin", purity=0.9),
        ]
        result = _score_cluster(clusters, "warfarin", 5)
        assert result.earned == pytest.approx(0.9 * 5)

    def test_empty_clusters_earns_zero(self):
        result = _score_cluster([], "warfarin", 5)
        assert result.earned == 0.0

    def test_none_clusters_earns_zero(self):
        result = _score_cluster(None, "warfarin", 5)
        assert result.earned == 0.0

    def test_name_is_cluster_strength(self):
        result = _score_cluster([], "warfarin", 5)
        assert result.name == "Cluster strength"


# ---------------------------------------------------------------------------
# 8. score_signal — integration (full pipeline)
# ---------------------------------------------------------------------------

class TestScoreSignal:

    @pytest.fixture(scope="class")
    @classmethod
    def scored(cls) -> PriorityScore:
        """
        Full score for warfarin/bleeding.

        Setup:
            PRR = 8.3,  saturation=20  → norm=0.415 → earned=12.45  (weight=30)
            Volume = 40, saturation=100 → norm=0.4   → earned=8.0   (weight=20)
            p_value = 0.001 = pval_full              → earned=20.0   (weight=20)
            Serious: 20/40 = 0.5, saturation=0.5     → earned=15.0   (weight=15)
            Trend: increasing                        → earned=10.0   (weight=10)
            Cluster: purity=0.9                      → earned=4.5    (weight=5)

        Total = 12.45 + 8.0 + 20.0 + 15.0 + 10.0 + 4.5 = 69.95 → rounded = 70
        """
        sig = _make_signal(prr=8.3, p_value=0.001, case_count=40)
        df  = _make_df(n_serious=20, n_non_serious=20)
        trend = _make_trend(TREND_INCREASING, trend_score=0.875)
        clusters = [_make_cluster("warfarin", purity=0.9)]
        return score_signal(sig, df, trend=trend, clusters=clusters)

    def test_returns_priority_score_instance(self, scored):
        assert isinstance(scored, PriorityScore)

    def test_final_score_is_integer(self, scored):
        assert isinstance(scored.final_score, int)

    def test_final_score_in_valid_range(self, scored):
        assert 0 <= scored.final_score <= 100

    def test_final_score_known_value(self, scored):
        # 12.45 + 8.0 + 20.0 + 15.0 + 10.0 + 4.5 = 69.95 → 70
        assert scored.final_score == 70

    def test_has_six_components(self, scored):
        assert len(scored.components) == 6

    def test_component_names_in_order(self, scored):
        names = [c.name for c in scored.components]
        assert names == [
            "PRR", "Volume", "Statistical strength",
            "Serious outcomes", "Trend", "Cluster strength",
        ]

    def test_disclaimer_always_present(self, scored):
        assert scored.disclaimer is not None
        assert len(scored.disclaimer) > 0

    def test_disclaimer_mentions_hackathon(self, scored):
        assert "HACKATHON" in scored.disclaimer.upper()

    def test_disclaimer_not_fda_approved(self, scored):
        assert "NOT" in scored.disclaimer

    def test_to_dict_has_required_keys(self, scored):
        d = scored.to_dict()
        required = {"drug_name", "adverse_event", "final_score",
                    "max_possible_score", "components", "disclaimer"}
        assert required.issubset(d.keys())

    def test_to_dict_max_possible_score_is_100(self, scored):
        assert scored.to_dict()["max_possible_score"] == 100

    def test_each_component_has_formula(self, scored):
        for c in scored.components:
            assert isinstance(c.formula, str)
            assert len(c.formula) > 0

    def test_display_contains_signal_score_line(self, scored):
        display = scored.display()
        assert "Signal score:" in display

    def test_display_contains_contributors_section(self, scored):
        display = scored.display()
        assert "Contributors:" in display

    def test_display_contains_all_component_names(self, scored):
        display = scored.display()
        for c in scored.components:
            assert c.name in display

    def test_display_contains_disclaimer(self, scored):
        display = scored.display()
        assert "HACKATHON" in display.upper()

    def test_component_earned_sum_matches_final_score(self, scored):
        total = sum(c.earned for c in scored.components)
        assert round(total) == scored.final_score

    def test_no_claims_of_causality_in_disclaimer(self, scored):
        assert "causality" in scored.disclaimer.lower()

    def test_score_without_trend_is_lower(self):
        sig = _make_signal(prr=8.3, p_value=0.001, case_count=40)
        df  = _make_df(n_serious=20, n_non_serious=20)
        clusters = [_make_cluster("warfarin", purity=0.9)]
        scored_with    = score_signal(sig, df, trend=_make_trend(TREND_INCREASING), clusters=clusters)
        scored_without = score_signal(sig, df, trend=None, clusters=clusters)
        # No trend → neutral (0.5 × weight), increasing → full (1.0 × weight)
        assert scored_with.final_score > scored_without.final_score

    def test_decreasing_trend_lower_score_than_increasing(self):
        sig = _make_signal(prr=8.3, p_value=0.001, case_count=40)
        df  = _make_df()
        increasing = score_signal(sig, df, trend=_make_trend(TREND_INCREASING))
        decreasing = score_signal(sig, df, trend=_make_trend(TREND_DECREASING))
        assert increasing.final_score > decreasing.final_score

    def test_custom_config_changes_score(self):
        """Doubling weight_prr and halving weight_volume changes the final score."""
        sig = _make_signal(prr=5.0, p_value=0.01, case_count=10)
        df  = _make_df(n_serious=5, n_non_serious=35)
        default_config = ScoringConfig()
        custom_config  = ScoringConfig(weight_prr=40, weight_volume=10,
                                       weight_statistical=20, weight_serious=15,
                                       weight_trend=10, weight_cluster=5)
        s_default = score_signal(sig, df, config=default_config)
        s_custom  = score_signal(sig, df, config=custom_config)
        # High PRR should score higher under config that weights PRR more
        assert s_custom.final_score != s_default.final_score


# ---------------------------------------------------------------------------
# 9. score_all_signals
# ---------------------------------------------------------------------------

class TestScoreAllSignals:

    def test_returns_list(self):
        sig = _make_signal()
        df  = _make_df()
        result = score_all_signals([sig], df)
        assert isinstance(result, list)

    def test_sorted_by_final_score_descending(self):
        sig_high = _make_signal(prr=15.0, case_count=80)
        sig_low  = _make_signal(drug="ibuprofen", event="rash", prr=2.5, case_count=5,
                                p_value=0.04)
        df = pd.concat([_make_df("warfarin", "bleeding"),
                        _make_df("ibuprofen", "rash")], ignore_index=True)
        results = score_all_signals([sig_high, sig_low], df)
        scores = [r.final_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_signals_returns_empty_list(self):
        df = _make_df()
        assert score_all_signals([], df) == []

    def test_all_disclaimers_present(self):
        sig = _make_signal()
        df  = _make_df()
        results = score_all_signals([sig], df)
        for r in results:
            assert "HACKATHON" in r.disclaimer.upper()

    def test_score_on_sample_csv(self, clean_df):
        """Smoke test: run full pipeline on real data."""
        from core.prr_calculator import calculate_all_signals
        from core.temporal_analyser import analyse_all_trends
        from core.signal_clusterer import cluster_reports

        signals  = calculate_all_signals(clean_df)
        flagged  = [s for s in signals if s.signal_flag]
        if not flagged:
            pytest.skip("No flagged signals in sample data")

        trend_list = analyse_all_trends(clean_df)
        trends   = {f"{t.drug_name}|{t.adverse_event}": t for t in trend_list}
        clusters = cluster_reports(clean_df)

        results = score_all_signals(flagged, clean_df, trends=trends, clusters=clusters)

        assert len(results) == len(flagged)
        for r in results:
            assert 0 <= r.final_score <= 100
            assert len(r.components) == 6


# ---------------------------------------------------------------------------
# 10. API endpoint smoke test
# ---------------------------------------------------------------------------

class TestPriorityEndpoint:

    @pytest.fixture(scope="class")
    @classmethod
    def client(cls):
        from fastapi.testclient import TestClient
        from main import app
        return TestClient(app)

    def test_priority_endpoint_returns_200(self, client):
        r = client.get("/api/signals/priority")
        assert r.status_code == 200

    def test_response_has_priority_scores_key(self, client):
        r = client.get("/api/signals/priority")
        assert "priority_scores" in r.json()

    def test_response_has_disclaimer(self, client):
        r = client.get("/api/signals/priority")
        body = r.json()
        assert "disclaimer" in body
        assert "HACKATHON" in body["disclaimer"].upper()

    def test_each_score_has_components(self, client):
        r = client.get("/api/signals/priority")
        for ps in r.json()["priority_scores"]:
            assert "components" in ps
            assert len(ps["components"]) == 6

    def test_each_component_has_formula(self, client):
        r = client.get("/api/signals/priority")
        for ps in r.json()["priority_scores"]:
            for c in ps["components"]:
                assert "formula" in c
                assert len(c["formula"]) > 0

    def test_scores_sorted_descending(self, client):
        r = client.get("/api/signals/priority")
        scores = [ps["final_score"] for ps in r.json()["priority_scores"]]
        assert scores == sorted(scores, reverse=True)

    def test_priority_filtered_to_drug(self, client):
        r = client.get("/api/signals/priority?drug_name=warfarin")
        assert r.status_code == 200
        for ps in r.json()["priority_scores"]:
            assert ps["drug_name"] == "warfarin"
