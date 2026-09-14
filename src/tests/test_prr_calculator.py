"""
test_prr_calculator.py — Unit tests for the PRR statistical engine.

Tests are split into:
  1. ContingencyTable construction
  2. PRR calculation (point estimate and CI)
  3. Chi-square / Fisher test selection and results
  4. Signal flagging logic
  5. SignalResult structure (caution field, serialisation)
  6. Batch calculation on the bundled sample dataset

All numeric assertions use pytest.approx() with a tolerance of 0.01
to avoid floating-point equality failures.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from core.prr_calculator import (
    DEFAULT_MAX_PVAL,
    DEFAULT_MIN_CASES,
    DEFAULT_MIN_PRR,
    build_contingency_table,
    calculate_all_signals,
    calculate_chi2,
    calculate_prr,
    flag_signal,
)
from core.schemas import (
    Chi2Result,
    ContingencyTable,
    PRRResult,
    SignalResult,
    _PRR_CAUTION,
)


# ---------------------------------------------------------------------------
# Helpers — build ContingencyTable directly from raw counts
# ---------------------------------------------------------------------------

def ct(a: int, b: int, c: int, d: int) -> ContingencyTable:
    return ContingencyTable(drug="testdrug", event="testevent", a=a, b=b, c=c, d=d)


def prr_r(ct_: ContingencyTable) -> PRRResult:
    return calculate_prr(ct_)


def chi2_r(ct_: ContingencyTable) -> Chi2Result:
    return calculate_chi2(ct_)


# ---------------------------------------------------------------------------
# 1. ContingencyTable
# ---------------------------------------------------------------------------

class TestContingencyTable:
    def test_cells_stored_correctly(self):
        table = ct(10, 5, 4, 31)
        assert table.a == 10
        assert table.b == 5
        assert table.c == 4
        assert table.d == 31

    def test_n_is_sum_of_all_cells(self):
        table = ct(10, 5, 4, 31)
        assert table.n == 50

    def test_drug_total(self):
        table = ct(10, 5, 4, 31)
        assert table.drug_total == 15

    def test_event_total(self):
        table = ct(10, 5, 4, 31)
        assert table.event_total == 14

    def test_negative_cell_raises(self):
        with pytest.raises(ValueError):
            ContingencyTable(drug="d", event="e", a=-1, b=5, c=4, d=31)

    def test_zero_cells_allowed(self):
        # c=0 is valid — PRR will be marked uncalculable, not error
        table = ct(5, 5, 0, 40)
        assert table.c == 0


class TestBuildContingencyTable:
    def test_correct_counts_from_tiny_df(self, tiny_df):
        table = build_contingency_table(tiny_df, "drugA", "eventX")
        assert table.a == 10
        assert table.b == 2   # drugA/eventY only
        assert table.c == 4   # drugB/eventX
        assert table.d == tiny_df.shape[0] - table.a - table.b - table.c

    def test_drug_not_in_df_returns_zero_a(self, tiny_df):
        table = build_contingency_table(tiny_df, "nonexistent_drug", "eventX")
        assert table.a == 0

    def test_event_not_in_df_returns_zero_a(self, tiny_df):
        table = build_contingency_table(tiny_df, "drugA", "nonexistent_event")
        assert table.a == 0


# ---------------------------------------------------------------------------
# 2. PRR calculation
# ---------------------------------------------------------------------------

class TestCalculatePRR:
    def test_known_prr_value(self):
        # a=10, b=2, c=4, d=34  → PRR = (10/12)/(4/38) ≈ 7.917
        table = ct(10, 2, 4, 34)
        result = calculate_prr(table)
        assert result.calculable is True
        assert result.prr == pytest.approx(7.917, rel=0.01)

    def test_prr_equal_one_when_proportions_equal(self):
        # Drug has same proportion of event as background
        # a=5, b=5, c=10, d=10 → PRR = (5/10)/(10/20) = 1.0
        table = ct(5, 5, 10, 10)
        result = calculate_prr(table)
        assert result.prr == pytest.approx(1.0, rel=0.01)

    def test_ci_bounds_bracket_point_estimate(self):
        table = ct(10, 2, 4, 34)
        result = calculate_prr(table)
        assert result.ci_lower_95 < result.prr < result.ci_upper_95

    def test_ci_lower_positive(self):
        table = ct(10, 2, 4, 34)
        result = calculate_prr(table)
        assert result.ci_lower_95 > 0

    def test_zero_a_returns_uncalculable(self):
        # a=0 means drug never reported this event
        table = ct(0, 10, 5, 35)
        result = calculate_prr(table)
        assert result.calculable is False
        assert result.prr is None
        assert result.ci_lower_95 is None
        assert result.ci_upper_95 is None

    def test_zero_c_returns_uncalculable(self):
        # c=0 means no other drug ever reported this event
        table = ct(5, 5, 0, 40)
        result = calculate_prr(table)
        assert result.calculable is False
        assert result.prr is None

    def test_high_prr_has_wide_ci(self):
        # Small counts → wide CI
        table = ct(3, 0, 1, 46)
        result = calculate_prr(table)
        if result.calculable:
            ci_width = result.ci_upper_95 - result.ci_lower_95
            assert ci_width > 1.0  # CI must be wide for n=4

    def test_large_counts_have_narrow_ci(self):
        # Large counts → narrow CI
        table = ct(100, 50, 200, 650)
        result = calculate_prr(table)
        ci_width = result.ci_upper_95 - result.ci_lower_95
        assert ci_width < 1.0  # CI should be narrow for n=1000


# ---------------------------------------------------------------------------
# 3. Chi-square / Fisher test
# ---------------------------------------------------------------------------

class TestCalculateChi2:
    def test_high_association_low_pvalue(self):
        # Strong association: drug D heavily over-reports event E
        table = ct(40, 5, 10, 245)
        result = calculate_chi2(table)
        assert result.p_value < 0.05

    def test_no_association_high_pvalue(self):
        # Equal proportions → no association
        table = ct(5, 5, 50, 50)
        result = calculate_chi2(table)
        assert result.p_value > 0.05

    def test_method_is_chi2_for_large_counts(self):
        table = ct(40, 10, 80, 370)
        result = calculate_chi2(table)
        assert result.method == "chi2"

    def test_method_is_fisher_for_small_expected_counts(self):
        # Expected cell counts < 5 → Fisher
        table = ct(2, 1, 1, 46)
        result = calculate_chi2(table)
        assert result.method == "fisher"

    def test_fisher_statistic_is_none(self):
        table = ct(2, 1, 1, 46)
        result = calculate_chi2(table)
        if result.method == "fisher":
            assert result.statistic is None

    def test_pvalue_between_zero_and_one(self):
        table = ct(10, 5, 20, 65)
        result = calculate_chi2(table)
        assert 0.0 <= result.p_value <= 1.0

    def test_zero_n_returns_pvalue_one(self):
        table = ct(0, 0, 0, 0)
        result = calculate_chi2(table)
        assert result.p_value == 1.0


# ---------------------------------------------------------------------------
# 4. Signal flagging
# ---------------------------------------------------------------------------

class TestFlagSignal:
    def _all_met(
        self,
        a: int = 5,
        prr: float = 3.0,
        pval: float = 0.01,
    ) -> tuple[bool, str]:
        ct_ = ct(a, 10, 5, 80)
        prr_res = PRRResult(prr=prr, ci_lower_95=1.5, ci_upper_95=6.0, calculable=True)
        chi_res = Chi2Result(statistic=8.0, p_value=pval, method="chi2")
        return flag_signal(ct_, prr_res, chi_res)

    def test_all_criteria_met_returns_true(self):
        flagged, _ = self._all_met()
        assert flagged is True

    def test_reason_says_flagged_when_true(self):
        _, reason = self._all_met()
        assert reason.lower().startswith("flagged")

    def test_below_min_cases_not_flagged(self):
        # a=2 < default min_cases=3
        ct_ = ct(2, 10, 5, 80)
        prr_res = PRRResult(prr=4.0, ci_lower_95=1.5, ci_upper_95=10.0, calculable=True)
        chi_res = Chi2Result(statistic=5.0, p_value=0.02, method="chi2")
        flagged, reason = flag_signal(ct_, prr_res, chi_res)
        assert flagged is False
        assert "case" in reason.lower()

    def test_low_prr_not_flagged(self):
        ct_ = ct(5, 10, 20, 65)
        prr_res = PRRResult(prr=1.5, ci_lower_95=0.8, ci_upper_95=2.8, calculable=True)
        chi_res = Chi2Result(statistic=2.0, p_value=0.04, method="chi2")
        flagged, reason = flag_signal(ct_, prr_res, chi_res)
        assert flagged is False
        assert "prr" in reason.lower() or "threshold" in reason.lower()

    def test_high_pvalue_not_flagged(self):
        ct_ = ct(5, 10, 5, 80)
        prr_res = PRRResult(prr=3.0, ci_lower_95=1.5, ci_upper_95=6.0, calculable=True)
        chi_res = Chi2Result(statistic=1.5, p_value=0.20, method="chi2")
        flagged, reason = flag_signal(ct_, prr_res, chi_res)
        assert flagged is False
        assert "p-value" in reason.lower() or "significance" in reason.lower()

    def test_uncalculable_prr_not_flagged(self):
        ct_ = ct(5, 10, 0, 85)
        prr_res = PRRResult(prr=None, ci_lower_95=None, ci_upper_95=None, calculable=False)
        chi_res = Chi2Result(statistic=None, p_value=0.01, method="fisher")
        flagged, reason = flag_signal(ct_, prr_res, chi_res)
        assert flagged is False

    def test_reason_is_non_empty_string(self):
        _, reason = self._all_met()
        assert isinstance(reason, str)
        assert len(reason) > 0

    def test_reason_is_human_readable_no_variable_names(self):
        _, reason = self._all_met()
        # Should not contain raw Python variable names
        assert "prr_result" not in reason
        assert "chi2_result" not in reason
        assert "ct_" not in reason

    def test_custom_thresholds_respected(self):
        # Raise min_prr to 5.0 → PRR=3.0 should now fail
        ct_ = ct(5, 10, 5, 80)
        prr_res = PRRResult(prr=3.0, ci_lower_95=1.5, ci_upper_95=6.0, calculable=True)
        chi_res = Chi2Result(statistic=8.0, p_value=0.01, method="chi2")
        flagged, _ = flag_signal(ct_, prr_res, chi_res, min_prr=5.0)
        assert flagged is False


# ---------------------------------------------------------------------------
# 5. SignalResult structure
# ---------------------------------------------------------------------------

class TestSignalResult:
    def _make_result(self, flagged: bool = True) -> SignalResult:
        return SignalResult(
            drug_name="warfarin",
            adverse_event="bleeding",
            a=10, b=5, c=4, d=81,
            prr=7.9, ci_lower_95=3.1, ci_upper_95=20.1,
            chi2_statistic=22.1, p_value=0.0001,
            test_method="chi2",
            signal_flag=flagged,
            flag_reason="Flagged: PRR=7.9." if flagged else "Not flagged.",
        )

    def test_caution_always_present(self):
        result = self._make_result()
        assert result.caution == _PRR_CAUTION

    def test_caution_is_always_set_regardless_of_flag(self):
        flagged = self._make_result(flagged=True)
        not_flagged = self._make_result(flagged=False)
        assert flagged.caution == not_flagged.caution == _PRR_CAUTION

    def test_caution_mentions_causality(self):
        result = self._make_result()
        assert "causality" in result.caution.lower() or "causal" in result.caution.lower()

    def test_to_dict_includes_caution(self):
        result = self._make_result()
        d = result.to_dict()
        assert "caution" in d
        assert d["caution"] == _PRR_CAUTION

    def test_to_dict_includes_contingency(self):
        result = self._make_result()
        d = result.to_dict()
        assert "contingency" in d
        assert d["contingency"]["a"] == 10

    def test_to_dict_has_all_expected_keys(self):
        result = self._make_result()
        d = result.to_dict()
        required_keys = {
            "drug_name", "adverse_event", "contingency",
            "prr", "ci_lower_95", "ci_upper_95",
            "chi2_statistic", "p_value", "test_method",
            "signal_flag", "flag_reason", "caution",
        }
        assert required_keys.issubset(d.keys())


# ---------------------------------------------------------------------------
# 6. Batch calculation on sample dataset
# ---------------------------------------------------------------------------

class TestCalculateAllSignals:
    def test_returns_list(self, clean_df):
        results = calculate_all_signals(clean_df)
        assert isinstance(results, list)

    def test_results_are_signal_result_instances(self, clean_df):
        results = calculate_all_signals(clean_df)
        assert all(isinstance(r, SignalResult) for r in results)

    def test_all_results_have_caution(self, clean_df):
        results = calculate_all_signals(clean_df)
        assert all(r.caution == _PRR_CAUTION for r in results)

    def test_warfarin_bleeding_is_flagged(self, clean_df):
        results = calculate_all_signals(clean_df)
        wb = next(
            (r for r in results if r.drug_name == "warfarin" and r.adverse_event == "bleeding"),
            None,
        )
        assert wb is not None, "warfarin/bleeding pair not found in results"
        assert wb.signal_flag is True, (
            f"warfarin/bleeding should be flagged but was not. reason={wb.flag_reason}"
        )

    def test_amoxicillin_anaphylaxis_not_flagged_too_few_cases(self, clean_df):
        # Sample has only 2 anaphylaxis reports for amoxicillin → below min_cases=3
        results = calculate_all_signals(clean_df)
        aa = next(
            (r for r in results
             if r.drug_name == "amoxicillin" and r.adverse_event == "anaphylaxis"),
            None,
        )
        if aa is not None:
            assert aa.signal_flag is False
            assert "case" in aa.flag_reason.lower()

    def test_sorted_by_prr_descending(self, clean_df):
        results = calculate_all_signals(clean_df)
        prrs = [r.prr for r in results if r.prr is not None]
        assert prrs == sorted(prrs, reverse=True)

    def test_drug_filter_limits_results(self, clean_df):
        all_results = calculate_all_signals(clean_df)
        filtered = calculate_all_signals(clean_df, drug_filter="warfarin")
        assert all(r.drug_name == "warfarin" for r in filtered)
        assert len(filtered) < len(all_results)

    def test_empty_dataframe_returns_empty_list(self):
        import pandas as pd
        from core.schemas import REQUIRED_INTERNAL_COLUMNS
        empty_df = pd.DataFrame(columns=REQUIRED_INTERNAL_COLUMNS)
        assert calculate_all_signals(empty_df) == []

    def test_no_prr_none_for_flagged_signals(self, clean_df):
        results = calculate_all_signals(clean_df)
        for r in results:
            if r.signal_flag:
                assert r.prr is not None, (
                    f"Flagged signal {r.drug_name}/{r.adverse_event} has prr=None"
                )

    def test_flagged_signals_meet_all_thresholds(self, clean_df):
        results = calculate_all_signals(clean_df)
        for r in results:
            if r.signal_flag:
                assert r.a >= DEFAULT_MIN_CASES
                assert r.prr is not None and r.prr >= DEFAULT_MIN_PRR
                assert r.p_value is not None and r.p_value <= DEFAULT_MAX_PVAL
