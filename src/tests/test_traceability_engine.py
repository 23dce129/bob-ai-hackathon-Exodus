"""
test_traceability_engine.py — Unit tests for the traceability engine.

Test strategy
-------------
Tests are grouped into eight sections:

  1. Infrastructure       — module constants, disclaimer invariant.
  2. classify_signal_category — known terms, aliases, fallback, edge cases.
  3. _build_covered_section_ids  — covered dict built correctly from MatchSummary.
  4. _compute_traceability_score — score formula: all-present, all-missing,
                                    mixed, NOT_APPLICABLE exclusion, empty.
  5. _determine_priority  — critical / high / normal logic.
  6. trace_signal         — empty dossier, full dossier, partial dossier,
                             signal without flag, unknown adverse event.
  7. trace_all_signals    — empty list, single signal, multiple signals,
                             sort order, overall score, fully_documented count.
  8. Serialisation        — to_dict() on all result types has expected keys.

Key invariants asserted throughout
------------------------------------
  - disclaimer is always equal to TRACEABILITY_DISCLAIMER (cannot be omitted).
  - traceability_score ∈ [0.0, 1.0] for every result.
  - No result text contains "proves", "establishes causality", "FDA rejection",
    or "will be rejected" (regulatory safety language guard).
  - critical_gaps contains only section ids that are safety_relevant=True AND
    status MISSING in the corresponding entries list.
  - trace_all_signals([]) returns an empty TraceabilityReport cleanly.
  - results in TraceabilityReport are sorted ascending by traceability_score.
  - NOT_APPLICABLE sections are excluded from the score denominator.
  - A fully-present dossier → traceability_score == 1.0 for every signal.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest

# Ensure the backend package is importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from core.ctd_knowledge_base import CTDSection, load_knowledge_base
from core.dossier_parser import DossierRow
from core.section_matcher import (
    MatchStatus,
    MatchSummary,
    MatchType,
    SectionMatch,
)
from core.traceability_engine import (
    TRACEABILITY_DISCLAIMER,
    TRACEABILITY_PARTIAL,
    TraceabilityEntry,
    TraceabilityReport,
    TraceabilityResult,
    _build_covered_section_ids,
    _compute_traceability_score,
    _determine_priority,
    classify_signal_category,
    trace_all_signals,
    trace_signal,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_section(
    section_id: str,
    title: str = "Test Section",
    safety_relevant: bool = False,
    requirement: str = "required",
    traceability_categories: Optional[list[str]] = None,
) -> CTDSection:
    """Build a minimal CTDSection for unit tests that don't need the real KB."""
    return CTDSection(
        id=section_id,
        number=section_id,
        title=title,
        parent_id="M2",
        level=2,
        requirement=requirement,
        applicability=["NDA"],
        submission_types_note="",
        ich_source=True,
        safety_relevant=safety_relevant,
        traceability_categories=traceability_categories or [],
    )


def _make_dossier_row(section_id: str, row_index: int = 0, status: str = "present") -> DossierRow:
    """Build a minimal DossierRow."""
    return DossierRow(
        section_number=section_id,
        section_title="",
        description="",
        source="",
        status=status,
        row_index=row_index,
    )


def _make_section_match(
    section: CTDSection,
    status: MatchStatus = MatchStatus.PRESENT,
    confidence: float = 1.0,
    row_index: int = 0,
) -> SectionMatch:
    """Build a minimal SectionMatch for unit tests."""
    row = _make_dossier_row(section.id, row_index=row_index, status=status.value.lower())
    return SectionMatch(
        dossier_row=row,
        expected_section=section,
        match_type=MatchType.EXACT_NUMBER,
        confidence=confidence,
        status=status,
        match_reason="test fixture",
        status_source="dossier_supplied",
    )


def _make_entry(
    section_id: str,
    status: MatchStatus,
    safety_relevant: bool = False,
) -> TraceabilityEntry:
    """Build a minimal TraceabilityEntry for score-computation tests."""
    priority = _determine_priority(
        _make_section(section_id, safety_relevant=safety_relevant), status
    )
    return TraceabilityEntry(
        ctd_section_id=section_id,
        ctd_section_title="Test",
        relevance_reason="test",
        dossier_match=(status != MatchStatus.MISSING),
        match_confidence=1.0 if status != MatchStatus.MISSING else 0.0,
        status=status,
        priority=priority,
        safety_relevant=safety_relevant,
        requirement="required",
    )


# ===========================================================================
# 1. Infrastructure
# ===========================================================================

class TestInfrastructure:

    def test_disclaimer_constant_is_non_empty(self):
        assert TRACEABILITY_DISCLAIMER
        assert len(TRACEABILITY_DISCLAIMER) > 20

    def test_disclaimer_contains_heuristic_language(self):
        assert "heuristic" in TRACEABILITY_DISCLAIMER.lower()

    def test_disclaimer_does_not_assert_approval_or_rejection(self):
        # The disclaimer must NOT assert that a submission WILL be approved.
        # It MAY say "does not mean the submission will be rejected" (protective language).
        lower = TRACEABILITY_DISCLAIMER.lower()
        assert "will be approved" not in lower
        # Ensure it does not unconditionally state rejection — the phrase
        # "will be rejected" is only acceptable when preceded by a negation.
        assert "submission will be rejected" not in lower or "not mean" in lower

    def test_partial_constant_value(self):
        assert TRACEABILITY_PARTIAL == 0.5

    def test_traceability_result_embeds_disclaimer(self, bleeding_signal, empty_match_summary, kb):
        result = trace_signal(bleeding_signal, empty_match_summary, kb=kb)
        assert result.disclaimer == TRACEABILITY_DISCLAIMER

    def test_traceability_report_embeds_disclaimer(self, empty_match_summary, kb):
        report = trace_all_signals([], empty_match_summary, kb=kb)
        assert report.disclaimer == TRACEABILITY_DISCLAIMER


# ===========================================================================
# 2. classify_signal_category
# ===========================================================================

class TestClassifySignalCategory:

    # Bleeding / haemorrhage
    @pytest.mark.parametrize("term", [
        "haemorrhage", "hemorrhage", "bleeding", "GI bleed", "thrombocytopenia",
        "epistaxis", "ecchymosis", "coagulation disorder",
    ])
    def test_bleeding_terms(self, term):
        assert classify_signal_category(term) == "bleeding"

    # Hepatotoxicity / liver
    @pytest.mark.parametrize("term", [
        "hepatotoxicity", "elevated ALT", "elevated AST", "jaundice",
        "drug-induced liver injury", "DILI", "cholestasis", "hepatitis",
        "elevated bilirubin",
    ])
    def test_hepatotoxicity_terms(self, term):
        assert classify_signal_category(term) == "hepatotoxicity"

    # Cardiac
    @pytest.mark.parametrize("term", [
        "cardiac arrest", "myocardial infarction", "QT prolongation",
        "arrhythmia", "tachycardia", "atrial fibrillation", "bradycardia",
        "angina pectoris",
    ])
    def test_cardiac_terms(self, term):
        assert classify_signal_category(term) == "cardiac"

    # Renal
    @pytest.mark.parametrize("term", [
        "renal failure", "acute kidney injury", "nephrotoxicity",
        "proteinuria", "elevated creatinine", "glomerulonephritis",
    ])
    def test_renal_terms(self, term):
        assert classify_signal_category(term) == "renal"

    # Neurological
    @pytest.mark.parametrize("term", [
        "seizure", "convulsion", "peripheral neuropathy",
        "encephalopathy", "stroke", "cognitive impairment",
    ])
    def test_neurological_terms(self, term):
        assert classify_signal_category(term) == "neurological"

    # Carcinogenicity
    @pytest.mark.parametrize("term", [
        "carcinogenicity", "tumour formation", "lymphoma",
        "leukemia", "malignancy", "genotoxicity",
    ])
    def test_carcinogenicity_terms(self, term):
        assert classify_signal_category(term) == "carcinogenicity"

    # Reproductive
    @pytest.mark.parametrize("term", [
        "teratogenicity", "embryotoxicity", "fetal toxicity",
        "fertility impairment", "congenital anomaly",
    ])
    def test_reproductive_terms(self, term):
        assert classify_signal_category(term) == "reproductive"

    # Anaphylaxis
    @pytest.mark.parametrize("term", [
        "anaphylaxis", "anaphylactic reaction", "hypersensitivity",
        "angioedema", "urticaria",
    ])
    def test_anaphylaxis_terms(self, term):
        assert classify_signal_category(term) == "anaphylaxis"

    # Infection
    @pytest.mark.parametrize("term", [
        "sepsis", "opportunistic infection", "immunosuppression",
        "tuberculosis",
    ])
    def test_infection_terms(self, term):
        assert classify_signal_category(term) == "infection"

    # Fallback to "general"
    @pytest.mark.parametrize("term", [
        "nausea", "vomiting", "headache", "fatigue", "dizziness", "rash",
        "insomnia", "weight_gain_unexpected_xyz",
    ])
    def test_general_fallback(self, term):
        assert classify_signal_category(term) == "general"

    # Edge cases
    def test_empty_string_returns_general(self):
        assert classify_signal_category("") == "general"

    def test_whitespace_only_returns_general(self):
        assert classify_signal_category("   ") == "general"

    def test_case_insensitive(self):
        assert classify_signal_category("HEPATOTOXICITY") == "hepatotoxicity"
        assert classify_signal_category("Haemorrhage") == "bleeding"

    def test_deterministic_same_term_twice(self):
        assert classify_signal_category("hepatotoxicity") == classify_signal_category("hepatotoxicity")


# ===========================================================================
# 3. _build_covered_section_ids
# ===========================================================================

class TestBuildCoveredSectionIds:

    def test_empty_summary_returns_empty_dict(self, empty_match_summary):
        covered = _build_covered_section_ids(empty_match_summary)
        assert covered == {}

    def test_present_section_is_included(self):
        section = _make_section("2.7.4")
        match = _make_section_match(section, status=MatchStatus.PRESENT, confidence=1.0)
        summary = MatchSummary(matches=[match], total_rows=1)
        covered = _build_covered_section_ids(summary)
        assert "2.7.4" in covered
        assert covered["2.7.4"] == (MatchStatus.PRESENT, 1.0)

    def test_needs_review_section_is_included(self):
        section = _make_section("5.3.5")
        match = _make_section_match(section, status=MatchStatus.NEEDS_REVIEW, confidence=0.85)
        summary = MatchSummary(matches=[match], total_rows=1)
        covered = _build_covered_section_ids(summary)
        assert "5.3.5" in covered
        assert covered["5.3.5"][0] == MatchStatus.NEEDS_REVIEW

    def test_missing_section_is_included_in_covered(self):
        # MISSING rows that DID resolve to a KB section are still recorded
        section = _make_section("4.2.3")
        match = _make_section_match(section, status=MatchStatus.MISSING, confidence=0.0)
        summary = MatchSummary(matches=[match], total_rows=1)
        covered = _build_covered_section_ids(summary)
        assert "4.2.3" in covered
        assert covered["4.2.3"][0] == MatchStatus.MISSING

    def test_unmatched_row_excluded(self):
        row = _make_dossier_row("unknown_id", row_index=0)
        match = SectionMatch(
            dossier_row=row,
            expected_section=None,
            match_type=MatchType.UNMATCHED,
            confidence=0.0,
            status=MatchStatus.MISSING,
            match_reason="no match found",
            status_source="inferred",
        )
        summary = MatchSummary(matches=[match], total_rows=1)
        covered = _build_covered_section_ids(summary)
        assert "unknown_id" not in covered
        assert len(covered) == 0

    def test_duplicate_section_id_highest_confidence_wins(self):
        section = _make_section("2.4")
        low_conf = _make_section_match(section, status=MatchStatus.PRESENT, confidence=0.7)
        high_conf = _make_section_match(section, status=MatchStatus.PRESENT, confidence=1.0, row_index=1)
        summary = MatchSummary(matches=[low_conf, high_conf], total_rows=2)
        covered = _build_covered_section_ids(summary)
        assert covered["2.4"][1] == 1.0


# ===========================================================================
# 4. _compute_traceability_score
# ===========================================================================

class TestComputeTraceabilityScore:

    def test_all_present_returns_1_0(self):
        entries = [
            _make_entry("2.4", MatchStatus.PRESENT),
            _make_entry("2.7.4", MatchStatus.PRESENT),
            _make_entry("5.3.5", MatchStatus.PRESENT),
        ]
        assert _compute_traceability_score(entries) == pytest.approx(1.0)

    def test_all_missing_returns_0_0(self):
        entries = [
            _make_entry("2.4", MatchStatus.MISSING),
            _make_entry("2.7.4", MatchStatus.MISSING),
        ]
        assert _compute_traceability_score(entries) == pytest.approx(0.0)

    def test_all_needs_review_returns_partial(self):
        entries = [
            _make_entry("2.4", MatchStatus.NEEDS_REVIEW),
            _make_entry("2.7.4", MatchStatus.NEEDS_REVIEW),
        ]
        # Each NEEDS_REVIEW = 0.5 credit, N=2 → score = (0.5+0.5)/2 = 0.5
        assert _compute_traceability_score(entries) == pytest.approx(0.5)

    def test_mixed_present_missing(self):
        entries = [
            _make_entry("2.4", MatchStatus.PRESENT),
            _make_entry("2.7.4", MatchStatus.MISSING),
            _make_entry("4.2.3", MatchStatus.MISSING),
            _make_entry("5.3.5", MatchStatus.PRESENT),
        ]
        # 2 PRESENT → 2.0; 2 MISSING → 0.0; N=4 → score = 2.0/4 = 0.5
        assert _compute_traceability_score(entries) == pytest.approx(0.5)

    def test_not_applicable_excluded_from_denominator(self):
        entries = [
            _make_entry("2.4", MatchStatus.PRESENT),
            _make_entry("5.3.1", MatchStatus.NOT_APPLICABLE),  # excluded
        ]
        # N=1 (only PRESENT); score = 1.0
        assert _compute_traceability_score(entries) == pytest.approx(1.0)

    def test_all_not_applicable_returns_1_0(self):
        entries = [
            _make_entry("2.4", MatchStatus.NOT_APPLICABLE),
        ]
        # N=0 → trivially 1.0
        assert _compute_traceability_score(entries) == pytest.approx(1.0)

    def test_empty_entries_returns_1_0(self):
        assert _compute_traceability_score([]) == pytest.approx(1.0)

    def test_one_present_one_missing_one_review(self):
        entries = [
            _make_entry("2.4", MatchStatus.PRESENT),
            _make_entry("2.7.4", MatchStatus.MISSING),
            _make_entry("4.2.3", MatchStatus.NEEDS_REVIEW),
        ]
        # (1.0 + 0.0 + 0.5) / 3 = 0.5
        assert _compute_traceability_score(entries) == pytest.approx(0.5)

    def test_score_is_in_0_1_range(self):
        import random
        random.seed(42)
        statuses = [MatchStatus.PRESENT, MatchStatus.MISSING,
                    MatchStatus.NEEDS_REVIEW, MatchStatus.NOT_APPLICABLE]
        for _ in range(50):
            entries = [
                _make_entry(str(i), random.choice(statuses))
                for i in range(random.randint(0, 10))
            ]
            score = _compute_traceability_score(entries)
            assert 0.0 <= score <= 1.0


# ===========================================================================
# 5. _determine_priority
# ===========================================================================

class TestDeterminePriority:

    def test_safety_relevant_missing_is_critical(self):
        section = _make_section("2.7.4", safety_relevant=True)
        assert _determine_priority(section, MatchStatus.MISSING) == "critical"

    def test_safety_relevant_present_is_high(self):
        section = _make_section("2.7.4", safety_relevant=True)
        assert _determine_priority(section, MatchStatus.PRESENT) == "high"

    def test_safety_relevant_needs_review_is_high(self):
        section = _make_section("2.7.4", safety_relevant=True)
        assert _determine_priority(section, MatchStatus.NEEDS_REVIEW) == "high"

    def test_not_safety_relevant_missing_is_normal(self):
        section = _make_section("3.1", safety_relevant=False)
        assert _determine_priority(section, MatchStatus.MISSING) == "normal"

    def test_not_safety_relevant_present_is_normal(self):
        section = _make_section("3.1", safety_relevant=False)
        assert _determine_priority(section, MatchStatus.PRESENT) == "normal"

    def test_safety_relevant_not_applicable_is_high(self):
        section = _make_section("5.3.1", safety_relevant=True)
        assert _determine_priority(section, MatchStatus.NOT_APPLICABLE) == "high"


# ===========================================================================
# 6. trace_signal
# ===========================================================================

class TestTraceSignal:

    # ---- Basic shape -------------------------------------------------------

    def test_returns_traceability_result(self, bleeding_signal, empty_match_summary, kb):
        result = trace_signal(bleeding_signal, empty_match_summary, kb=kb)
        assert isinstance(result, TraceabilityResult)

    def test_drug_name_preserved(self, bleeding_signal, empty_match_summary, kb):
        result = trace_signal(bleeding_signal, empty_match_summary, kb=kb)
        assert result.drug_name == "warfarin"

    def test_adverse_event_preserved(self, bleeding_signal, empty_match_summary, kb):
        result = trace_signal(bleeding_signal, empty_match_summary, kb=kb)
        assert result.adverse_event == "haemorrhage"

    def test_prr_preserved(self, bleeding_signal, empty_match_summary, kb):
        result = trace_signal(bleeding_signal, empty_match_summary, kb=kb)
        assert result.prr == pytest.approx(7.2)

    def test_p_value_preserved(self, bleeding_signal, empty_match_summary, kb):
        result = trace_signal(bleeding_signal, empty_match_summary, kb=kb)
        assert result.p_value == pytest.approx(0.001)

    def test_signal_flag_preserved(self, bleeding_signal, empty_match_summary, kb):
        result = trace_signal(bleeding_signal, empty_match_summary, kb=kb)
        assert result.signal_flag is True

    def test_signal_category_classified_correctly(self, bleeding_signal, empty_match_summary, kb):
        result = trace_signal(bleeding_signal, empty_match_summary, kb=kb)
        assert result.signal_category == "bleeding"

    def test_disclaimer_always_present(self, bleeding_signal, empty_match_summary, kb):
        result = trace_signal(bleeding_signal, empty_match_summary, kb=kb)
        assert result.disclaimer == TRACEABILITY_DISCLAIMER

    # ---- Empty dossier → all sections MISSING ------------------------------

    def test_empty_dossier_score_is_0(self, bleeding_signal, empty_match_summary, kb):
        result = trace_signal(bleeding_signal, empty_match_summary, kb=kb)
        assert result.traceability_score == pytest.approx(0.0)

    def test_empty_dossier_all_entries_missing(self, bleeding_signal, empty_match_summary, kb):
        result = trace_signal(bleeding_signal, empty_match_summary, kb=kb)
        for entry in result.entries:
            assert entry.status == MatchStatus.MISSING

    def test_empty_dossier_all_missing_dossier_match_false(self, bleeding_signal, empty_match_summary, kb):
        result = trace_signal(bleeding_signal, empty_match_summary, kb=kb)
        for entry in result.entries:
            assert entry.dossier_match is False

    def test_empty_dossier_has_entries(self, bleeding_signal, empty_match_summary, kb):
        result = trace_signal(bleeding_signal, empty_match_summary, kb=kb)
        assert len(result.entries) > 0

    def test_empty_dossier_has_critical_gaps(self, bleeding_signal, empty_match_summary, kb):
        result = trace_signal(bleeding_signal, empty_match_summary, kb=kb)
        # With an empty dossier all safety-relevant sections are critical gaps
        assert len(result.critical_gaps) > 0

    # ---- Full dossier → all sections PRESENT -------------------------------

    def test_full_dossier_score_is_1(self, bleeding_signal, full_match_summary, kb):
        result = trace_signal(bleeding_signal, full_match_summary, kb=kb)
        assert result.traceability_score == pytest.approx(1.0)

    def test_full_dossier_no_critical_gaps(self, bleeding_signal, full_match_summary, kb):
        result = trace_signal(bleeding_signal, full_match_summary, kb=kb)
        assert result.critical_gaps == []

    def test_full_dossier_all_entries_present_or_na(self, bleeding_signal, full_match_summary, kb):
        result = trace_signal(bleeding_signal, full_match_summary, kb=kb)
        for entry in result.entries:
            assert entry.status in (MatchStatus.PRESENT, MatchStatus.NOT_APPLICABLE)

    # ---- Critical gaps logic -----------------------------------------------

    def test_critical_gaps_are_safety_relevant_and_missing(self, bleeding_signal, empty_match_summary, kb):
        result = trace_signal(bleeding_signal, empty_match_summary, kb=kb)
        for gap_id in result.critical_gaps:
            matching = [e for e in result.entries if e.ctd_section_id == gap_id]
            assert len(matching) == 1
            assert matching[0].safety_relevant is True
            assert matching[0].status == MatchStatus.MISSING

    def test_critical_gaps_ids_are_strings(self, bleeding_signal, empty_match_summary, kb):
        result = trace_signal(bleeding_signal, empty_match_summary, kb=kb)
        for gap_id in result.critical_gaps:
            assert isinstance(gap_id, str)

    # ---- Entry fields ------------------------------------------------------

    def test_entries_have_non_empty_relevance_reason(self, bleeding_signal, empty_match_summary, kb):
        result = trace_signal(bleeding_signal, empty_match_summary, kb=kb)
        for entry in result.entries:
            assert entry.relevance_reason
            assert len(entry.relevance_reason) > 5

    def test_relevance_reason_does_not_claim_causality(self, bleeding_signal, empty_match_summary, kb):
        result = trace_signal(bleeding_signal, empty_match_summary, kb=kb)
        forbidden = ["proves", "establishes causality", "will be rejected", "fda rejection"]
        for entry in result.entries:
            reason_lower = entry.relevance_reason.lower()
            for phrase in forbidden:
                assert phrase not in reason_lower, (
                    f"Forbidden phrase '{phrase}' found in relevance_reason: {entry.relevance_reason}"
                )

    def test_entries_confidence_in_0_1(self, bleeding_signal, empty_match_summary, kb):
        result = trace_signal(bleeding_signal, empty_match_summary, kb=kb)
        for entry in result.entries:
            assert 0.0 <= entry.match_confidence <= 1.0

    def test_entries_priority_values(self, bleeding_signal, empty_match_summary, kb):
        valid_priorities = {"critical", "high", "normal"}
        result = trace_signal(bleeding_signal, empty_match_summary, kb=kb)
        for entry in result.entries:
            assert entry.priority in valid_priorities

    def test_entries_sorted_by_section_id(self, bleeding_signal, empty_match_summary, kb):
        result = trace_signal(bleeding_signal, empty_match_summary, kb=kb)
        ids = [e.ctd_section_id for e in result.entries]
        assert ids == sorted(ids)

    # ---- Signal without optional fields ------------------------------------

    def test_minimal_signal_dict(self, empty_match_summary, kb):
        """Only drug_name and adverse_event are required; prr/p_value/signal_flag default."""
        signal = {"drug_name": "aspirin", "adverse_event": "bleeding"}
        result = trace_signal(signal, empty_match_summary, kb=kb)
        assert result.drug_name == "aspirin"
        assert result.prr is None
        assert result.p_value is None
        assert result.signal_flag is False

    # ---- Unknown adverse event falls back to "general" ---------------------

    def test_unknown_adverse_event_category_is_general(self, empty_match_summary, kb):
        signal = {"drug_name": "drugX", "adverse_event": "nausea"}
        result = trace_signal(signal, empty_match_summary, kb=kb)
        assert result.signal_category == "general"

    def test_general_category_has_entries(self, empty_match_summary, kb):
        signal = {"drug_name": "drugX", "adverse_event": "nausea"}
        result = trace_signal(signal, empty_match_summary, kb=kb)
        assert len(result.entries) > 0

    # ---- Hepatotoxicity signal ----------------------------------------------

    def test_hepato_signal_category(self, hepato_signal, empty_match_summary, kb):
        result = trace_signal(hepato_signal, empty_match_summary, kb=kb)
        assert result.signal_category == "hepatotoxicity"

    def test_hepato_empty_dossier_score_is_0(self, hepato_signal, empty_match_summary, kb):
        result = trace_signal(hepato_signal, empty_match_summary, kb=kb)
        assert result.traceability_score == pytest.approx(0.0)

    def test_hepato_full_dossier_score_is_1(self, hepato_signal, full_match_summary, kb):
        result = trace_signal(hepato_signal, full_match_summary, kb=kb)
        assert result.traceability_score == pytest.approx(1.0)

    # ---- Score is in [0, 1] ------------------------------------------------

    def test_traceability_score_in_range(self, bleeding_signal, empty_match_summary, kb):
        result = trace_signal(bleeding_signal, empty_match_summary, kb=kb)
        assert 0.0 <= result.traceability_score <= 1.0

    # ---- Partial dossier gives intermediate score --------------------------

    def test_partial_dossier_intermediate_score(self, bleeding_signal, kb):
        """A dossier with some relevant sections present gives 0 < score < 1."""
        # Fetch the bleeding-relevant sections from the KB
        relevant = kb.get_sections_by_traceability_category("bleeding")
        if len(relevant) < 2:
            pytest.skip("Insufficient bleeding sections in KB for this test")

        # Mark only the first relevant section as PRESENT
        first_section = relevant[0]
        match = _make_section_match(first_section, status=MatchStatus.PRESENT)
        partial_summary = MatchSummary(matches=[match], total_rows=1)
        partial_summary.status_counts = {"PRESENT": 1}

        result = trace_signal(bleeding_signal, partial_summary, kb=kb)
        # Score should be strictly between 0 and 1 (one of several present)
        assert 0.0 < result.traceability_score < 1.0


# ===========================================================================
# 7. trace_all_signals
# ===========================================================================

class TestTraceAllSignals:

    def test_empty_list_returns_report(self, empty_match_summary, kb):
        report = trace_all_signals([], empty_match_summary, kb=kb)
        assert isinstance(report, TraceabilityReport)

    def test_empty_list_results_empty(self, empty_match_summary, kb):
        report = trace_all_signals([], empty_match_summary, kb=kb)
        assert report.results == []
        assert report.total_signals == 0

    def test_empty_list_overall_score_is_1(self, empty_match_summary, kb):
        # Trivially fully documented when there are no signals
        report = trace_all_signals([], empty_match_summary, kb=kb)
        assert report.overall_score == pytest.approx(1.0)

    def test_empty_list_disclaimer(self, empty_match_summary, kb):
        report = trace_all_signals([], empty_match_summary, kb=kb)
        assert report.disclaimer == TRACEABILITY_DISCLAIMER

    def test_single_signal(self, bleeding_signal, empty_match_summary, kb):
        report = trace_all_signals([bleeding_signal], empty_match_summary, kb=kb)
        assert report.total_signals == 1
        assert len(report.results) == 1

    def test_multiple_signals(self, bleeding_signal, hepato_signal, empty_match_summary, kb):
        report = trace_all_signals([bleeding_signal, hepato_signal], empty_match_summary, kb=kb)
        assert report.total_signals == 2
        assert len(report.results) == 2

    def test_results_sorted_ascending_by_score(self, bleeding_signal, hepato_signal, full_match_summary, empty_match_summary, kb):
        """Mix fully-present and empty-dossier scenarios to get different scores."""
        # bleeding in empty dossier → score 0
        # hepato in full dossier → score 1
        # We need to produce different scores so mix dossiers per-signal manually.
        # Use a single partial dossier to get a predictable sort.
        relevant_bleeding = kb.get_sections_by_traceability_category("bleeding")
        relevant_hepato = kb.get_sections_by_traceability_category("hepatotoxicity")

        if not relevant_bleeding or not relevant_hepato:
            pytest.skip("KB lacks sections for this test")

        # Mark only ONE bleeding section as PRESENT (low score)
        match_bleeding = _make_section_match(relevant_bleeding[0], status=MatchStatus.PRESENT)
        # Mark ALL hepato sections as PRESENT (high score)
        matches_hepato = [
            _make_section_match(s, status=MatchStatus.PRESENT, row_index=i + 1)
            for i, s in enumerate(relevant_hepato)
        ]
        all_matches = [match_bleeding] + matches_hepato
        summary = MatchSummary(matches=all_matches, total_rows=len(all_matches))
        summary.status_counts = {"PRESENT": len(all_matches)}

        report = trace_all_signals([bleeding_signal, hepato_signal], summary, kb=kb)
        scores = [r.traceability_score for r in report.results]
        assert scores == sorted(scores), "Results must be sorted ascending by traceability_score"

    def test_overall_score_is_mean(self, bleeding_signal, hepato_signal, empty_match_summary, kb):
        report = trace_all_signals([bleeding_signal, hepato_signal], empty_match_summary, kb=kb)
        expected_mean = sum(r.traceability_score for r in report.results) / len(report.results)
        assert report.overall_score == pytest.approx(expected_mean)

    def test_fully_documented_count(self, bleeding_signal, hepato_signal, full_match_summary, kb):
        report = trace_all_signals([bleeding_signal, hepato_signal], full_match_summary, kb=kb)
        assert report.fully_documented == 2

    def test_fully_documented_zero_when_empty_dossier(self, bleeding_signal, hepato_signal, empty_match_summary, kb):
        report = trace_all_signals([bleeding_signal, hepato_signal], empty_match_summary, kb=kb)
        assert report.fully_documented == 0

    def test_critical_gap_count(self, bleeding_signal, hepato_signal, empty_match_summary, kb):
        report = trace_all_signals([bleeding_signal, hepato_signal], empty_match_summary, kb=kb)
        # With empty dossier each signal should have critical gaps
        assert report.critical_gap_count > 0

    def test_critical_gap_count_zero_when_full_dossier(self, bleeding_signal, hepato_signal, full_match_summary, kb):
        report = trace_all_signals([bleeding_signal, hepato_signal], full_match_summary, kb=kb)
        assert report.critical_gap_count == 0

    def test_disclaimer_always_set(self, empty_match_summary, kb):
        report = trace_all_signals([], empty_match_summary, kb=kb)
        assert report.disclaimer == TRACEABILITY_DISCLAIMER

    def test_overall_score_in_0_1(self, bleeding_signal, empty_match_summary, kb):
        report = trace_all_signals([bleeding_signal], empty_match_summary, kb=kb)
        assert 0.0 <= report.overall_score <= 1.0


# ===========================================================================
# 8. Serialisation — to_dict()
# ===========================================================================

class TestSerialisation:

    def test_traceability_entry_to_dict_keys(self, bleeding_signal, empty_match_summary, kb):
        result = trace_signal(bleeding_signal, empty_match_summary, kb=kb)
        assert len(result.entries) > 0
        d = result.entries[0].to_dict()
        expected_keys = {
            "ctd_section_id", "ctd_section_title", "relevance_reason",
            "dossier_match", "match_confidence", "status", "priority",
            "safety_relevant", "requirement",
        }
        assert expected_keys.issubset(d.keys())

    def test_traceability_entry_status_is_string(self, bleeding_signal, empty_match_summary, kb):
        result = trace_signal(bleeding_signal, empty_match_summary, kb=kb)
        for entry in result.entries:
            d = entry.to_dict()
            assert isinstance(d["status"], str)

    def test_traceability_result_to_dict_keys(self, bleeding_signal, empty_match_summary, kb):
        result = trace_signal(bleeding_signal, empty_match_summary, kb=kb)
        d = result.to_dict()
        expected_keys = {
            "drug_name", "adverse_event", "signal_category",
            "prr", "p_value", "signal_flag",
            "traceability_score", "critical_gaps", "entries", "disclaimer",
        }
        assert expected_keys.issubset(d.keys())

    def test_traceability_result_entries_are_list(self, bleeding_signal, empty_match_summary, kb):
        result = trace_signal(bleeding_signal, empty_match_summary, kb=kb)
        d = result.to_dict()
        assert isinstance(d["entries"], list)

    def test_traceability_result_disclaimer_is_string(self, bleeding_signal, empty_match_summary, kb):
        result = trace_signal(bleeding_signal, empty_match_summary, kb=kb)
        d = result.to_dict()
        assert isinstance(d["disclaimer"], str)
        assert d["disclaimer"] == TRACEABILITY_DISCLAIMER

    def test_traceability_report_to_dict_keys(self, bleeding_signal, empty_match_summary, kb):
        report = trace_all_signals([bleeding_signal], empty_match_summary, kb=kb)
        d = report.to_dict()
        expected_keys = {
            "overall_score", "total_signals", "fully_documented",
            "critical_gap_count", "results", "disclaimer",
        }
        assert expected_keys.issubset(d.keys())

    def test_traceability_report_results_is_list(self, bleeding_signal, empty_match_summary, kb):
        report = trace_all_signals([bleeding_signal], empty_match_summary, kb=kb)
        d = report.to_dict()
        assert isinstance(d["results"], list)

    def test_confidence_rounded_in_to_dict(self, bleeding_signal, empty_match_summary, kb):
        result = trace_signal(bleeding_signal, empty_match_summary, kb=kb)
        for entry in result.entries:
            d = entry.to_dict()
            # Should be rounded to 4 dp — check it's a float
            assert isinstance(d["match_confidence"], float)

    def test_score_rounded_in_to_dict(self, bleeding_signal, empty_match_summary, kb):
        result = trace_signal(bleeding_signal, empty_match_summary, kb=kb)
        d = result.to_dict()
        assert isinstance(d["traceability_score"], float)
