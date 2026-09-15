"""
test_readiness_scorer.py — Unit tests for the CTD readiness scorer.

Test strategy
-------------
Tests are grouped into seven sections:

  1. Infrastructure  — constants, config validation, disclaimer invariant.
  2. Section weights — _section_weight() for all requirement/safety combinations.
  3. Module scoring  — controlled synthetic inputs with hand-calculated values.
  4. Overall scoring — weighted average formula, exclusion of empty modules.
  5. Config overrides — needs_review_partial, critical_penalty, module_weights.
  6. Edge cases      — all-present, all-missing, all-NA, empty, single-section.
  7. Integration     — full synthetic dossier; exact values verified by running
                       the scorer and checking the arithmetic above.

Key invariants asserted throughout
------------------------------------
  - disclaimer is always equal to READINESS_DISCLAIMER (cannot be omitted).
  - overall_score and all module scores are in [0.0, 1.0].
  - total_present + total_needs_review + total_missing + total_not_applicable
    == total_rows in the summary (no double-counting, no drops).
  - NOT_APPLICABLE sections are NEVER counted in present/missing/review totals
    and are NEVER included in possible_weight.
  - NEEDS_REVIEW sections earn exactly needs_review_partial × full_weight.
  - Each critical-missing section deducts exactly critical_penalty from the
    module's raw_ratio, but score cannot go below 0.0.
  - Scores are never described as "FDA approval readiness".

All hand-calculated values were derived from the _verify_scoring.py output
and manually verified against the formula in the module docstring.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from core.ctd_knowledge_base import CTDSection, load_knowledge_base
from core.dossier_parser import DossierRow, parse_dossier
from core.readiness_scorer import (
    CRITICAL_PENALTY,
    NEEDS_REVIEW_PARTIAL,
    READINESS_DISCLAIMER,
    ModuleScore,
    ReadinessConfig,
    ReadinessResult,
    _earned_weight,
    _section_weight,
    score_readiness,
)
from core.section_matcher import (
    MatchStatus,
    MatchSummary,
    MatchType,
    SectionMatch,
    build_matcher,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers — build minimal SectionMatch / MatchSummary objects
# ---------------------------------------------------------------------------

def _make_kb_section(
    section_id: str,
    requirement: str = "required",
    safety_relevant: bool = False,
) -> CTDSection:
    """Return a CTDSection-like object from the real KB, or a synthetic one."""
    kb = load_knowledge_base()
    sec = kb.get_section(section_id)
    if sec is not None:
        return sec
    # Fallback: build a minimal dataclass instance for unit tests
    return CTDSection(
        id=section_id,
        number=section_id,
        title=f"Synthetic section {section_id}",
        parent_id="M1",
        level=2,
        requirement=requirement,
        applicability=["NDA"],
        submission_types_note="",
        ich_source=True,
        safety_relevant=safety_relevant,
        traceability_categories=[],
    )


def _make_row(section_number: str = "1.1", row_index: int = 1) -> DossierRow:
    return DossierRow(
        section_number=section_number,
        section_title="",
        description="",
        source="",
        status="",
        row_index=row_index,
    )


def _make_match(
    section_id: str,
    status: MatchStatus,
    requirement: str = "required",
    safety_relevant: bool = False,
    row_index: int = 1,
) -> SectionMatch:
    sec = _make_kb_section(section_id, requirement, safety_relevant)
    return SectionMatch(
        dossier_row=_make_row(section_id, row_index),
        expected_section=sec,
        match_type=MatchType.EXACT_NUMBER,
        confidence=1.0,
        status=status,
        match_reason=f"Test match for {section_id}",
        status_source="dossier_supplied",
    )


def _make_unmatched(row_index: int = 1) -> SectionMatch:
    return SectionMatch(
        dossier_row=_make_row("9.9.9", row_index),
        expected_section=None,
        match_type=MatchType.UNMATCHED,
        confidence=0.0,
        status=MatchStatus.MISSING,
        match_reason="Test unmatched row",
        status_source="inferred",
    )


def _make_summary(matches: list[SectionMatch]) -> MatchSummary:
    s = MatchSummary(total_rows=len(matches))
    s.matches = matches
    return s


# ===========================================================================
# Section 1 — Infrastructure
# ===========================================================================

class TestInfrastructure:
    """Constants, config validation, disclaimer invariant."""

    def test_disclaimer_contains_heuristic(self):
        assert "HEURISTIC" in READINESS_DISCLAIMER

    def test_disclaimer_says_not_fda(self):
        assert "NOT" in READINESS_DISCLAIMER
        assert "FDA" in READINESS_DISCLAIMER

    def test_disclaimer_says_not_approval(self):
        assert "approval" in READINESS_DISCLAIMER.lower()

    def test_needs_review_partial_default(self):
        assert NEEDS_REVIEW_PARTIAL == 0.5

    def test_critical_penalty_default(self):
        assert CRITICAL_PENALTY == 0.10

    def test_readiness_config_defaults(self):
        cfg = ReadinessConfig()
        assert cfg.needs_review_partial == 0.5
        assert cfg.critical_penalty == 0.10
        assert "M2" in cfg.module_weights
        assert cfg.module_weights["M2"] > cfg.module_weights["M1"]

    def test_readiness_config_rejects_bad_partial(self):
        with pytest.raises(ValueError, match="needs_review_partial"):
            ReadinessConfig(needs_review_partial=1.5)

    def test_readiness_config_rejects_negative_penalty(self):
        with pytest.raises(ValueError, match="critical_penalty"):
            ReadinessConfig(critical_penalty=-0.01)

    def test_result_disclaimer_equals_constant(self):
        summary = _make_summary([_make_match("1.1", MatchStatus.PRESENT)])
        rr = score_readiness(summary)
        assert rr.disclaimer == READINESS_DISCLAIMER

    def test_result_disclaimer_cannot_be_overridden(self):
        """disclaimer is a field(init=False) — passing it as kwarg is a TypeError."""
        with pytest.raises(TypeError):
            ReadinessResult(
                overall_score=1.0,
                overall_score_pct=100.0,
                module_scores={},
                total_present=0,
                total_needs_review=0,
                total_missing=0,
                total_not_applicable=0,
                total_critical_missing=0,
                config=ReadinessConfig(),
                disclaimer="CUSTOM DISCLAIMER",  # must raise
            )

    def test_result_scores_in_range(self):
        summary = _make_summary([
            _make_match("1.1", MatchStatus.PRESENT),
            _make_match("1.8", MatchStatus.MISSING, safety_relevant=True),
        ])
        rr = score_readiness(summary)
        assert 0.0 <= rr.overall_score <= 1.0
        for ms in rr.module_scores.values():
            assert 0.0 <= ms.score <= 1.0


# ===========================================================================
# Section 2 — Section weights
# ===========================================================================

class TestSectionWeights:
    """_section_weight() for all requirement / safety combinations."""

    def test_required_not_safety(self):
        # base=1.0, multiplier=1.0 → 1.0
        sec = _make_kb_section("X", "required", False)
        assert _section_weight(sec) == pytest.approx(1.0)

    def test_required_safety(self):
        # base=1.0, multiplier=1.5 → 1.5
        sec = _make_kb_section("X", "required", True)
        assert _section_weight(sec) == pytest.approx(1.5)

    def test_conditional_not_safety(self):
        # base=0.5, multiplier=1.0 → 0.5
        sec = _make_kb_section("X", "conditional", False)
        assert _section_weight(sec) == pytest.approx(0.5)

    def test_conditional_safety(self):
        # base=0.5, multiplier=1.5 → 0.75
        sec = _make_kb_section("X", "conditional", True)
        assert _section_weight(sec) == pytest.approx(0.75)

    def test_real_section_1_8_weight(self):
        """1.8 Pharmacovigilance: required + safety_relevant → 1.5."""
        sec = _make_kb_section("1.8")
        assert _section_weight(sec) == pytest.approx(1.5)

    def test_real_section_2_7_4_weight(self):
        """2.7.4 Summary of Clinical Safety: required + safety_relevant → 1.5."""
        sec = _make_kb_section("2.7.4")
        assert _section_weight(sec) == pytest.approx(1.5)

    def test_real_section_3_2_s_1_weight(self):
        """3.2.S.1 General Information: required + not safety → 1.0."""
        sec = _make_kb_section("3.2.S.1")
        assert _section_weight(sec) == pytest.approx(1.0)

    def test_real_section_4_2_3_4_weight(self):
        """4.2.3.4 Carcinogenicity: conditional + safety → 0.75."""
        sec = _make_kb_section("4.2.3.4")
        assert _section_weight(sec) == pytest.approx(0.75)

    def test_earned_present_full(self):
        assert _earned_weight(1.5, MatchStatus.PRESENT, 0.5) == pytest.approx(1.5)

    def test_earned_needs_review_half(self):
        assert _earned_weight(1.5, MatchStatus.NEEDS_REVIEW, 0.5) == pytest.approx(0.75)

    def test_earned_missing_zero(self):
        assert _earned_weight(1.5, MatchStatus.MISSING, 0.5) == pytest.approx(0.0)

    def test_earned_not_applicable_zero(self):
        assert _earned_weight(1.5, MatchStatus.NOT_APPLICABLE, 0.5) == pytest.approx(0.0)

    def test_earned_needs_review_with_custom_partial(self):
        assert _earned_weight(2.0, MatchStatus.NEEDS_REVIEW, 0.25) == pytest.approx(0.5)


# ===========================================================================
# Section 3 — Module scoring (hand-calculated)
# ===========================================================================

class TestModuleScoring:
    """Controlled single-module inputs with hand-verified arithmetic."""

    def test_all_present_module_score_is_1(self):
        """Two required sections, both PRESENT → score = 1.0."""
        matches = [
            _make_match("1.1", MatchStatus.PRESENT, "required", False),
            _make_match("1.2", MatchStatus.PRESENT, "required", False),
        ]
        rr = score_readiness(_make_summary(matches))
        ms = rr.module_scores["M1"]
        assert ms.score == pytest.approx(1.0)
        assert ms.raw_ratio == pytest.approx(1.0)
        assert ms.penalty_applied == pytest.approx(0.0)

    def test_all_missing_non_critical_score(self):
        """Two required non-safety sections, both MISSING → score = 0.0."""
        matches = [
            _make_match("1.1", MatchStatus.MISSING, "required", False),
            _make_match("1.2", MatchStatus.MISSING, "required", False),
        ]
        rr = score_readiness(_make_summary(matches))
        ms = rr.module_scores["M1"]
        assert ms.raw_ratio == pytest.approx(0.0)
        assert ms.penalty_applied == pytest.approx(0.0)
        assert ms.score == pytest.approx(0.0)

    def test_one_missing_one_present_ratio(self):
        """
        1.1 (required, not-safety, w=1.0) PRESENT → earned=1.0
        1.2 (required, not-safety, w=1.0) MISSING → earned=0.0
        possible=2.0, earned=1.0, raw_ratio=0.5, no penalty → score=0.5
        """
        matches = [
            _make_match("1.1", MatchStatus.PRESENT, "required", False),
            _make_match("1.2", MatchStatus.MISSING, "required", False),
        ]
        rr = score_readiness(_make_summary(matches))
        ms = rr.module_scores["M1"]
        assert ms.possible_weight == pytest.approx(2.0)
        assert ms.earned_weight == pytest.approx(1.0)
        assert ms.raw_ratio == pytest.approx(0.5)
        assert ms.penalty_applied == pytest.approx(0.0)
        assert ms.score == pytest.approx(0.5)

    def test_needs_review_earns_partial(self):
        """
        1.1 (required, w=1.0) NEEDS_REVIEW → earned = 1.0 × 0.5 = 0.5
        possible=1.0, earned=0.5, ratio=0.5
        """
        matches = [_make_match("1.1", MatchStatus.NEEDS_REVIEW, "required", False)]
        rr = score_readiness(_make_summary(matches))
        ms = rr.module_scores["M1"]
        assert ms.earned_weight == pytest.approx(0.5)
        assert ms.raw_ratio == pytest.approx(0.5)
        assert ms.needs_review_count == 1
        assert ms.present_count == 0

    def test_not_applicable_excluded_from_denominator(self):
        """
        1.1 (w=1.0) PRESENT  → earned=1.0
        1.7 (conditional, safety, w=0.75) NOT_APPLICABLE → excluded
        possible = 1.0 only (1.7 excluded), earned=1.0 → ratio=1.0
        """
        matches = [
            _make_match("1.1", MatchStatus.PRESENT, "required", False),
            _make_match("1.7", MatchStatus.NOT_APPLICABLE, "conditional", True),
        ]
        rr = score_readiness(_make_summary(matches))
        ms = rr.module_scores["M1"]
        assert ms.not_applicable_count == 1
        assert ms.possible_weight == pytest.approx(1.0)
        assert ms.raw_ratio == pytest.approx(1.0)
        assert ms.score == pytest.approx(1.0)

    def test_critical_missing_applies_penalty(self):
        """
        2.7.4 (required, safety, w=1.5) MISSING → critical_missing=1
        penalty = 1 × 0.10 = 0.10
        raw_ratio = 0.0 / 1.5 = 0.0
        score = max(0, 0.0 - 0.10) = 0.0  (clamped)
        """
        matches = [_make_match("2.7.4", MatchStatus.MISSING, "required", True)]
        rr = score_readiness(_make_summary(matches))
        ms = rr.module_scores["M2"]
        assert ms.critical_missing_count == 1
        assert ms.penalty_applied == pytest.approx(0.10)
        assert ms.score == pytest.approx(0.0)

    def test_critical_missing_reduces_non_zero_score(self):
        """
        2.7.3 (required, not-safety, w=1.0) PRESENT → earned=1.0
        2.7.4 (required, safety, w=1.5) MISSING → critical_missing=1
        possible=2.5, earned=1.0
        raw_ratio = 1.0/2.5 = 0.4
        penalty = 0.10
        score = 0.4 - 0.10 = 0.30
        """
        matches = [
            _make_match("2.7.3", MatchStatus.PRESENT, "required", False),
            _make_match("2.7.4", MatchStatus.MISSING, "required", True),
        ]
        rr = score_readiness(_make_summary(matches))
        ms = rr.module_scores["M2"]
        assert ms.raw_ratio == pytest.approx(1.0 / 2.5)
        assert ms.penalty_applied == pytest.approx(0.10)
        assert ms.score == pytest.approx(0.30, abs=1e-6)

    def test_two_critical_missing_double_penalty(self):
        """
        Two required+safety sections both MISSING → penalty = 2 × 0.10 = 0.20
        """
        matches = [
            _make_match("2.7.4", MatchStatus.MISSING, "required", True),
            _make_match("1.8", MatchStatus.MISSING, "required", True),
        ]
        rr = score_readiness(_make_summary(matches))
        total_penalty = sum(
            ms.penalty_applied for ms in rr.module_scores.values()
        )
        assert total_penalty == pytest.approx(0.20)

    def test_score_never_below_zero(self):
        """Ten critical-missing sections: penalty=1.0 > raw_ratio → clamp to 0."""
        # Use multiple safe synthetic section ids not in KB → UNMATCHED → no penalty
        # Use real required+safety sections instead
        # Create 10 critical-missing matches across M2 sections
        critical_ids = ["2.7.4", "2.4", "2.5", "2.6", "2.6.2",
                        "2.6.6", "2.6.7", "2.7", "2.7.2", "2.7.6"]
        matches = [
            _make_match(sid, MatchStatus.MISSING, "required", True)
            for sid in critical_ids
        ]
        rr = score_readiness(_make_summary(matches))
        for ms in rr.module_scores.values():
            assert ms.score >= 0.0

    def test_safety_weight_higher_than_non_safety(self):
        """Safety sections have higher weight → missing one hurts more."""
        # One required+safety PRESENT, possible=1.5, earned=1.5 → ratio=1.0
        m_safe = [_make_match("1.8", MatchStatus.PRESENT, "required", True)]
        # One required PRESENT, possible=1.0, earned=1.0 → ratio=1.0
        m_plain = [_make_match("1.1", MatchStatus.PRESENT, "required", False)]
        # Both complete → both score 1.0
        rr_safe = score_readiness(_make_summary(m_safe))
        rr_plain = score_readiness(_make_summary(m_plain))
        assert rr_safe.module_scores["M1"].score == pytest.approx(1.0)
        assert rr_plain.module_scores["M1"].score == pytest.approx(1.0)

        # But missing: safety section missing hurts more due to higher weight
        m_safe_miss = [
            _make_match("1.8", MatchStatus.MISSING, "required", True),   # w=1.5 missing
            _make_match("1.1", MatchStatus.PRESENT, "required", False),   # w=1.0 present
        ]
        rr_miss = score_readiness(_make_summary(m_safe_miss))
        ms = rr_miss.module_scores["M1"]
        # raw = 1.0 / 2.5 = 0.4; penalty = 0.1 → score = 0.3
        assert ms.raw_ratio < 0.5   # safety missing pulls down harder than plain

    def test_conditional_weight_less_than_required(self):
        """Conditional sections contribute less to possible_weight."""
        matches_req = [_make_match("1.1", MatchStatus.PRESENT, "required", False)]
        matches_cond = [_make_match("1.5", MatchStatus.PRESENT, "conditional", False)]
        rr_req = score_readiness(_make_summary(matches_req))
        rr_cond = score_readiness(_make_summary(matches_cond))
        ms_req = rr_req.module_scores["M1"]
        ms_cond = rr_cond.module_scores["M1"]
        assert ms_req.possible_weight > ms_cond.possible_weight


# ===========================================================================
# Section 4 — Overall scoring
# ===========================================================================

class TestOverallScoring:
    """Weighted average formula."""

    def test_overall_score_in_range(self):
        summary = _make_summary([
            _make_match("1.1", MatchStatus.PRESENT),
            _make_match("2.7.4", MatchStatus.PRESENT),
        ])
        rr = score_readiness(summary)
        assert 0.0 <= rr.overall_score <= 1.0

    def test_overall_score_pct_equals_score_times_100(self):
        summary = _make_summary([_make_match("1.1", MatchStatus.PRESENT)])
        rr = score_readiness(summary)
        assert rr.overall_score_pct == pytest.approx(rr.overall_score * 100)

    def test_all_modules_all_present_overall_is_1(self):
        """When every section is PRESENT, every module = 1.0 → overall = 1.0."""
        kb = load_knowledge_base()
        matches = [
            _make_match(s.id, MatchStatus.PRESENT, s.requirement, s.safety_relevant)
            for s in kb.all_sections()
        ]
        rr = score_readiness(_make_summary(matches))
        assert rr.overall_score == pytest.approx(1.0)

    def test_m2_weight_higher_than_m1(self):
        """M2 (0.30) outweighs M1 (0.10) in default config."""
        cfg = ReadinessConfig()
        assert cfg.module_weights["M2"] > cfg.module_weights["M1"]

    def test_empty_module_excluded_from_average(self):
        """A module with no applicable sections (possible_weight=0) is skipped."""
        # Only M1 sections supplied
        summary = _make_summary([
            _make_match("1.1", MatchStatus.PRESENT),
            _make_match("1.2", MatchStatus.PRESENT),
        ])
        rr = score_readiness(summary)
        # M2–M5 have no matches → excluded
        assert "M2" not in rr.module_scores or rr.module_scores.get("M2") is None or \
               rr.module_scores["M2"].possible_weight == 0.0

    def test_custom_module_weights(self):
        """Overriding module weights changes the overall score."""
        matches = [
            _make_match("1.1", MatchStatus.PRESENT),   # M1
            _make_match("2.3", MatchStatus.MISSING),    # M2
        ]
        cfg_default = ReadinessConfig()
        cfg_m1_heavy = ReadinessConfig(module_weights={"M1": 0.9, "M2": 0.1,
                                                        "M3": 0.0, "M4": 0.0, "M5": 0.0})
        rr_default = score_readiness(_make_summary(matches), config=cfg_default)
        rr_m1 = score_readiness(_make_summary(matches), config=cfg_m1_heavy)
        # With M1 weighted heavily and M1 is complete, m1-heavy should score higher
        assert rr_m1.overall_score > rr_default.overall_score


# ===========================================================================
# Section 5 — Config overrides
# ===========================================================================

class TestConfigOverrides:
    """needs_review_partial, critical_penalty, and module_weights overrides."""

    def test_zero_partial_needs_review_earns_nothing(self):
        """With needs_review_partial=0, NEEDS_REVIEW sections count as missing."""
        matches = [_make_match("1.1", MatchStatus.NEEDS_REVIEW)]
        cfg = ReadinessConfig(needs_review_partial=0.0)
        rr = score_readiness(_make_summary(matches), config=cfg)
        ms = rr.module_scores["M1"]
        assert ms.earned_weight == pytest.approx(0.0)
        assert ms.raw_ratio == pytest.approx(0.0)

    def test_full_partial_needs_review_earns_full(self):
        """With needs_review_partial=1.0, NEEDS_REVIEW = PRESENT."""
        matches = [_make_match("1.1", MatchStatus.NEEDS_REVIEW)]
        cfg = ReadinessConfig(needs_review_partial=1.0)
        rr = score_readiness(_make_summary(matches), config=cfg)
        ms = rr.module_scores["M1"]
        assert ms.earned_weight == pytest.approx(1.0)
        assert ms.raw_ratio == pytest.approx(1.0)

    def test_zero_penalty_no_reduction(self):
        """With critical_penalty=0, a critical-missing section doesn't reduce score.
        2.7.4 is required+safety_relevant; 2.7.3 is required+not-safety.
        When 2.7.4 is MISSING it is critical → default config applies penalty,
        zero-penalty config does not.
        """
        matches = [
            _make_match("2.7.3", MatchStatus.PRESENT, "required", False),   # present
            _make_match("2.7.4", MatchStatus.MISSING, "required", True),    # critical miss
        ]
        cfg_no_penalty = ReadinessConfig(critical_penalty=0.0)
        cfg_default = ReadinessConfig()
        rr_none = score_readiness(_make_summary(matches), config=cfg_no_penalty)
        rr_def = score_readiness(_make_summary(matches), config=cfg_default)
        ms_none = rr_none.module_scores["M2"]
        ms_def = rr_def.module_scores["M2"]
        assert ms_none.penalty_applied == pytest.approx(0.0)
        # With zero penalty, critical_missing_count is still tracked but no deduction
        assert ms_none.critical_missing_count == 1
        assert ms_none.score > ms_def.score  # without penalty, score is higher

    def test_large_penalty_clamps_to_zero(self):
        """critical_penalty=1.0 means one critical miss → score = 0 minimum."""
        cfg = ReadinessConfig(critical_penalty=1.0)
        matches = [
            _make_match("2.7.4", MatchStatus.MISSING, "required", True),
            _make_match("2.7.3", MatchStatus.PRESENT, "required", False),
        ]
        rr = score_readiness(_make_summary(matches), config=cfg)
        ms = rr.module_scores["M2"]
        assert ms.score == pytest.approx(0.0)

    def test_custom_needs_review_partial_used(self):
        matches = [_make_match("1.1", MatchStatus.NEEDS_REVIEW)]
        cfg = ReadinessConfig(needs_review_partial=0.75)
        rr = score_readiness(_make_summary(matches), config=cfg)
        ms = rr.module_scores["M1"]
        assert ms.earned_weight == pytest.approx(1.0 * 0.75)


# ===========================================================================
# Section 6 — Edge cases
# ===========================================================================

class TestEdgeCases:
    """All-present, all-missing, all-NA, empty, single-section."""

    def test_empty_summary_overall_is_zero(self):
        rr = score_readiness(_make_summary([]))
        assert rr.overall_score == pytest.approx(0.0)
        assert rr.total_present == 0
        assert rr.total_missing == 0

    def test_single_present_section(self):
        rr = score_readiness(_make_summary([_make_match("1.1", MatchStatus.PRESENT)]))
        assert rr.total_present == 1
        assert rr.total_missing == 0
        assert rr.module_scores["M1"].score == pytest.approx(1.0)

    def test_single_missing_section(self):
        rr = score_readiness(_make_summary([_make_match("1.1", MatchStatus.MISSING)]))
        assert rr.total_present == 0
        assert rr.total_missing == 1
        assert rr.module_scores["M1"].score == pytest.approx(0.0)

    def test_single_not_applicable_section(self):
        rr = score_readiness(_make_summary([
            _make_match("1.7", MatchStatus.NOT_APPLICABLE, "conditional", True)
        ]))
        assert rr.total_not_applicable == 1
        assert rr.total_present == 0
        assert rr.total_missing == 0
        # Module has possible_weight = 0 → score vacuously = 1.0
        ms = rr.module_scores.get("M1")
        if ms:
            assert ms.possible_weight == pytest.approx(0.0)

    def test_all_not_applicable_does_not_count_as_missing(self):
        """A module where every section is NA should not count sections as missing."""
        rr = score_readiness(_make_summary([
            _make_match("1.7", MatchStatus.NOT_APPLICABLE),
            _make_match("1.5", MatchStatus.NOT_APPLICABLE),
        ]))
        assert rr.total_missing == 0
        assert rr.total_not_applicable == 2

    def test_count_consistency(self):
        """present + review + missing + na == total rows."""
        matches = [
            _make_match("1.1", MatchStatus.PRESENT),
            _make_match("1.3", MatchStatus.NEEDS_REVIEW),
            _make_match("1.8", MatchStatus.MISSING),
            _make_match("1.7", MatchStatus.NOT_APPLICABLE),
        ]
        rr = score_readiness(_make_summary(matches))
        total = (rr.total_present + rr.total_needs_review
                 + rr.total_missing + rr.total_not_applicable)
        assert total == 4

    def test_to_dict_has_all_required_keys(self):
        rr = score_readiness(_make_summary([_make_match("1.1", MatchStatus.PRESENT)]))
        d = rr.to_dict()
        assert "disclaimer" in d
        assert "overall_score" in d
        assert "overall_score_pct" in d
        assert "total_present" in d
        assert "total_missing" in d
        assert "total_needs_review" in d
        assert "total_not_applicable" in d
        assert "total_critical_missing" in d
        assert "module_scores" in d

    def test_module_score_to_dict_keys(self):
        rr = score_readiness(_make_summary([_make_match("1.1", MatchStatus.PRESENT)]))
        md = rr.module_scores["M1"].to_dict()
        assert "module_id" in md
        assert "score" in md
        assert "score_pct" in md
        assert "present_count" in md
        assert "missing_count" in md
        assert "needs_review_count" in md
        assert "not_applicable_count" in md
        assert "critical_missing_count" in md
        assert "penalty_applied" in md

    def test_disclaimer_never_empty(self):
        rr = score_readiness(_make_summary([]))
        assert len(rr.disclaimer) > 20


# ===========================================================================
# Section 7 — Integration: synthetic dossier (exact values)
# ===========================================================================

class TestSyntheticDossierIntegration:
    """Full dossier from demo data; values verified against _verify_scoring.py output."""

    @pytest.fixture(scope="class")
    @classmethod
    def result(cls) -> ReadinessResult:
        parse_result = parse_dossier(FIXTURES / "dossier_outline_clean.csv")
        matcher = build_matcher()
        summary = matcher.match_dossier(parse_result.rows)
        return score_readiness(summary)

    # --- Top-level structure ---

    def test_returns_readiness_result(self, result):
        assert isinstance(result, ReadinessResult)

    def test_disclaimer_present(self, result):
        assert result.disclaimer == READINESS_DISCLAIMER

    def test_disclaimer_not_fda_approval(self, result):
        assert "FDA" in result.disclaimer
        assert "NOT" in result.disclaimer

    def test_overall_score_in_range(self, result):
        assert 0.0 <= result.overall_score <= 1.0

    def test_overall_score_pct_consistent(self, result):
        assert result.overall_score_pct == pytest.approx(result.overall_score * 100)

    def test_five_modules_present(self, result):
        assert set(result.module_scores.keys()) == {"M1", "M2", "M3", "M4", "M5"}

    # --- Aggregate counts ---

    def test_total_present_equals_62(self, result):
        assert result.total_present == 62

    def test_total_needs_review_equals_7(self, result):
        assert result.total_needs_review == 7

    def test_total_missing_equals_4(self, result):
        assert result.total_missing == 4

    def test_total_not_applicable_equals_2(self, result):
        assert result.total_not_applicable == 2

    def test_count_sums_to_75(self, result):
        total = (result.total_present + result.total_needs_review
                 + result.total_missing + result.total_not_applicable)
        assert total == 75

    def test_total_critical_missing_equals_4(self, result):
        """All four missing sections (1.8, 2.4, 2.7.4, 4.2.3.5) are
        required+safety → 4 critical missing."""
        assert result.total_critical_missing == 4

    # --- Module scores (exact values from _verify_scoring.py) ---

    def test_m1_score(self, result):
        assert result.module_scores["M1"].score == pytest.approx(0.6097, abs=1e-3)

    def test_m1_raw_ratio(self, result):
        assert result.module_scores["M1"].raw_ratio == pytest.approx(0.7097, abs=1e-3)

    def test_m1_penalty(self, result):
        assert result.module_scores["M1"].penalty_applied == pytest.approx(0.10)

    def test_m1_critical_missing(self, result):
        assert result.module_scores["M1"].critical_missing_count == 1

    def test_m2_score(self, result):
        assert result.module_scores["M2"].score == pytest.approx(0.6200, abs=1e-3)

    def test_m2_raw_ratio(self, result):
        assert result.module_scores["M2"].raw_ratio == pytest.approx(0.8200, abs=1e-3)

    def test_m2_penalty(self, result):
        assert result.module_scores["M2"].penalty_applied == pytest.approx(0.20)

    def test_m2_critical_missing(self, result):
        assert result.module_scores["M2"].critical_missing_count == 2

    def test_m3_score_is_1(self, result):
        """Module 3 is fully complete in the synthetic dossier."""
        assert result.module_scores["M3"].score == pytest.approx(1.0)

    def test_m3_no_missing(self, result):
        assert result.module_scores["M3"].missing_count == 0
        assert result.module_scores["M3"].critical_missing_count == 0

    def test_m4_score(self, result):
        assert result.module_scores["M4"].score == pytest.approx(0.7190, abs=1e-3)

    def test_m4_critical_missing(self, result):
        assert result.module_scores["M4"].critical_missing_count == 1

    def test_m5_no_critical_missing(self, result):
        assert result.module_scores["M5"].critical_missing_count == 0

    def test_m5_score_high(self, result):
        """M5 has no missing sections → score driven only by NEEDS_REVIEW partial."""
        assert result.module_scores["M5"].score > 0.80

    # --- Overall score ---

    def test_overall_score_roughly_expected(self, result):
        # Verified: 0.7671
        assert result.overall_score == pytest.approx(0.7671, abs=1e-3)

    def test_overall_score_reflects_missing_sections(self, result):
        """Score < 1.0 because there are missing and needs_review sections."""
        assert result.overall_score < 1.0

    def test_m3_complete_does_not_inflate_overall_above_missing_modules(self, result):
        """A fully-complete M3 doesn't make overall = 1.0 when M2 has gaps."""
        assert result.module_scores["M3"].score == pytest.approx(1.0)
        assert result.overall_score < 1.0

    # --- section_details audit trail ---

    def test_section_details_present_on_each_module(self, result):
        for mid, ms in result.module_scores.items():
            assert len(ms.section_details) > 0, f"M{mid} has no section_details"

    def test_section_details_contain_match_reason(self, result):
        for mid, ms in result.module_scores.items():
            for detail in ms.section_details:
                assert "match_reason" in detail
                assert len(detail["match_reason"]) > 0

    def test_to_dict_roundtrip(self, result):
        d = result.to_dict()
        assert d["total_present"] == 62
        assert d["total_missing"] == 4
        assert d["total_critical_missing"] == 4
        assert "M3" in d["module_scores"]
        assert d["module_scores"]["M3"]["score"] == pytest.approx(1.0)
