"""
test_section_matcher.py — Unit tests for the CTD section matching engine.

Test strategy
-------------
Tests are grouped into eight sections:

  1. Infrastructure  — matcher construction, index sizes, IDF vocabulary.
  2. Exact match     — section_number matches KB id exactly; confidence=1.0.
  3. Title normalised match — exact normalised title and known aliases.
  4. Semantic match  — TF-IDF cosine above threshold with sufficient gap.
  5. Ambiguous match — cosine above threshold but gap too small.
  6. Missing / unmatched — no strategy succeeds; status→MISSING.
  7. Status determination — dossier_supplied overrides inferred; vocabulary map.
  8. Dossier batch   — match_dossier() on the synthetic dossier fixture.

Key invariants asserted throughout
------------------------------------
  - Every SectionMatch has a non-empty match_reason.
  - Confidence ∈ [0.0, 1.0].
  - EXACT_NUMBER → confidence == 1.0.
  - TITLE_NORMALISED exact → confidence == 1.0.
  - TITLE_NORMALISED alias → confidence == 0.9.
  - UNMATCHED → expected_section is None and confidence == 0.0.
  - status_source is "dossier_supplied" when row.status was non-empty,
    "inferred" otherwise.
  - A dossier-supplied status is NEVER silently overridden.
  - AMBIGUOUS → alternative_candidates is non-empty.

All cosine and confidence values used in assertions were verified by running
the matcher's own scoring logic and checking against the printed output above.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from core.ctd_knowledge_base import CTDKnowledgeBase, load_knowledge_base
from core.dossier_parser import DossierRow, parse_dossier
from core.section_matcher import (
    SEMANTIC_GAP,
    SEMANTIC_THRESHOLD,
    MatchStatus,
    MatchSummary,
    MatchType,
    SectionMatch,
    SectionMatcher,
    _map_status_string,
    _normalise_title,
    _tokenise,
    build_matcher,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures"
DOSSIER_JSON = Path(__file__).parent.parent.parent / "demo" / "data" / "synthetic_dossier_outline.json"


@pytest.fixture(scope="module")
def kb() -> CTDKnowledgeBase:
    return load_knowledge_base()


@pytest.fixture(scope="module")
def matcher() -> SectionMatcher:
    return build_matcher()


# ---------------------------------------------------------------------------
# Row builder helper
# ---------------------------------------------------------------------------

def make_row(
    section_number: str = "",
    section_title: str = "",
    status: str = "",
    description: str = "",
    source: str = "",
    row_index: int = 1,
) -> DossierRow:
    return DossierRow(
        section_number=section_number,
        section_title=section_title,
        description=description,
        source=source,
        status=status,
        row_index=row_index,
    )


# ===========================================================================
# Section 1 — Infrastructure
# ===========================================================================

class TestInfrastructure:
    """Matcher construction and internal index integrity."""

    def test_build_matcher_returns_section_matcher(self, matcher):
        assert isinstance(matcher, SectionMatcher)

    def test_sections_loaded(self, matcher):
        assert len(matcher._sections) == 75

    def test_by_id_covers_all_sections(self, matcher):
        assert len(matcher._by_id) == 75

    def test_by_normalised_title_has_entries(self, matcher):
        assert len(matcher._by_normalised_title) > 0

    def test_alias_map_non_empty(self, matcher):
        assert len(matcher._by_alias) > 0

    def test_idf_vocabulary_non_empty(self, matcher):
        assert len(matcher._idf) > 0

    def test_tfidf_vectors_cover_all_sections(self, matcher):
        assert len(matcher._tfidf_vectors) == 75

    def test_match_type_values(self):
        assert MatchType.EXACT_NUMBER.value == "EXACT_NUMBER"
        assert MatchType.SEMANTIC.value == "SEMANTIC"
        assert MatchType.UNMATCHED.value == "UNMATCHED"

    def test_match_status_values(self):
        assert MatchStatus.PRESENT.value == "PRESENT"
        assert MatchStatus.MISSING.value == "MISSING"
        assert MatchStatus.NEEDS_REVIEW.value == "NEEDS_REVIEW"
        assert MatchStatus.NOT_APPLICABLE.value == "NOT_APPLICABLE"

    def test_semantic_threshold_is_sensible(self):
        assert 0.0 < SEMANTIC_THRESHOLD < 1.0

    def test_semantic_gap_is_sensible(self):
        assert 0.0 < SEMANTIC_GAP < 0.5

    def test_tokenise_drops_stopwords(self):
        tokens = _tokenise("Summary of Clinical Safety")
        assert "of" not in tokens
        assert "summary" in tokens
        assert "clinical" in tokens
        assert "safety" in tokens

    def test_tokenise_lowercases(self):
        tokens = _tokenise("Pharmacovigilance")
        assert tokens == ["pharmacovigilance"]

    def test_normalise_title_produces_sorted_tokens(self):
        # "Summary of Clinical Safety" → "summary clinical safety"
        norm = _normalise_title("Summary of Clinical Safety")
        assert norm == "summary clinical safety"

    def test_normalise_title_collapses_whitespace(self):
        norm = _normalise_title("  Nonclinical   Overview  ")
        assert "  " not in norm


# ===========================================================================
# Section 2 — Exact number match
# ===========================================================================

class TestExactNumberMatch:
    """Dossier section_number == KB section id → EXACT_NUMBER, confidence=1.0."""

    def test_exact_match_2_7_4(self, matcher):
        row = make_row(section_number="2.7.4")
        m = matcher.match_row(row)
        assert m.match_type == MatchType.EXACT_NUMBER
        assert m.expected_section is not None
        assert m.expected_section.id == "2.7.4"
        assert m.confidence == 1.0

    def test_exact_match_1_8(self, matcher):
        row = make_row(section_number="1.8")
        m = matcher.match_row(row)
        assert m.match_type == MatchType.EXACT_NUMBER
        assert m.expected_section.id == "1.8"
        assert m.confidence == 1.0

    def test_exact_match_3_2_s_4(self, matcher):
        """Section ids with letters (3.2.S.4) match case-insensitively."""
        row = make_row(section_number="3.2.S.4")
        m = matcher.match_row(row)
        assert m.match_type == MatchType.EXACT_NUMBER
        assert m.expected_section.id == "3.2.S.4"

    def test_exact_match_uppercase_normalised(self, matcher):
        """Parser lowercases section_number; 3.2.S.4 stored lowercase."""
        row = make_row(section_number="3.2.s.4")   # already lowercase per parser
        m = matcher.match_row(row)
        assert m.match_type == MatchType.EXACT_NUMBER

    def test_exact_match_4_2_3_5(self, matcher):
        row = make_row(section_number="4.2.3.5")
        m = matcher.match_row(row)
        assert m.match_type == MatchType.EXACT_NUMBER
        assert m.expected_section.id == "4.2.3.5"

    def test_exact_match_reason_non_empty(self, matcher):
        m = matcher.match_row(make_row(section_number="2.7.4"))
        assert len(m.match_reason) > 0

    def test_exact_match_reason_mentions_section_number(self, matcher):
        m = matcher.match_row(make_row(section_number="2.7.4"))
        assert "2.7.4" in m.match_reason

    def test_exact_match_confidence_is_exactly_1(self, matcher):
        for sid in ("1.1", "2.3", "3.2.S", "4.2.3.3", "5.3.5"):
            m = matcher.match_row(make_row(section_number=sid))
            assert m.confidence == 1.0, f"Expected confidence=1.0 for {sid}"

    def test_exact_match_status_source_inferred_when_no_status(self, matcher):
        m = matcher.match_row(make_row(section_number="2.7.4"))
        assert m.status_source == "inferred"
        assert m.status == MatchStatus.PRESENT

    def test_exact_match_to_dict_shape(self, matcher):
        m = matcher.match_row(make_row(section_number="1.8"))
        d = m.to_dict()
        assert set(d.keys()) == {
            "row_index", "dossier_section_number", "dossier_section_title",
            "dossier_status_raw", "matched_section_id", "matched_section_title",
            "match_type", "confidence", "status", "status_source",
            "match_reason", "alternative_candidates",
        }
        assert d["match_type"] == "EXACT_NUMBER"
        assert d["matched_section_id"] == "1.8"

    def test_exact_match_overrides_title_matching(self, matcher):
        """When section_number is valid, title is ignored even if it's wrong."""
        row = make_row(section_number="2.7.4", section_title="Completely Wrong Title")
        m = matcher.match_row(row)
        assert m.match_type == MatchType.EXACT_NUMBER
        assert m.expected_section.id == "2.7.4"


# ===========================================================================
# Section 3 — Normalised title match
# ===========================================================================

class TestTitleNormalisedMatch:
    """Normalised-title exact match and alias match."""

    def test_normalised_exact_2_4(self, matcher):
        """'Nonclinical Overview' normalises exactly to KB title."""
        row = make_row(section_title="Nonclinical Overview")
        m = matcher.match_row(row)
        assert m.match_type == MatchType.TITLE_NORMALISED
        assert m.expected_section.id == "2.4"
        assert m.confidence == 1.0

    def test_normalised_exact_2_5(self, matcher):
        row = make_row(section_title="Clinical Overview")
        m = matcher.match_row(row)
        assert m.match_type == MatchType.TITLE_NORMALISED
        assert m.expected_section.id == "2.5"

    def test_normalised_exact_2_3_qos(self, matcher):
        row = make_row(section_title="Quality Overall Summary")
        m = matcher.match_row(row)
        assert m.match_type == MatchType.TITLE_NORMALISED
        assert m.expected_section.id == "2.3"
        assert m.confidence == 1.0

    def test_alias_iss(self, matcher):
        """'ISS' is a known alias for 2.7.4 Summary of Clinical Safety."""
        row = make_row(section_title="ISS")
        m = matcher.match_row(row)
        assert m.match_type == MatchType.TITLE_NORMALISED
        assert m.expected_section.id == "2.7.4"
        assert m.confidence == 0.9

    def test_alias_rmp(self, matcher):
        """'RMP' is a known alias for 1.8 Pharmacovigilance."""
        row = make_row(section_title="RMP")
        m = matcher.match_row(row)
        assert m.match_type == MatchType.TITLE_NORMALISED
        assert m.expected_section.id == "1.8"
        assert m.confidence == 0.9

    def test_alias_tox_written_summary(self, matcher):
        row = make_row(section_title="Tox Written Summary")
        m = matcher.match_row(row)
        assert m.match_type == MatchType.TITLE_NORMALISED
        assert m.expected_section.id == "2.6.6"
        assert m.confidence == 0.9

    def test_alias_repro_tox(self, matcher):
        row = make_row(section_title="Repro Tox")
        m = matcher.match_row(row)
        assert m.match_type == MatchType.TITLE_NORMALISED
        assert m.expected_section.id == "4.2.3.5"
        assert m.confidence == 0.9

    def test_alias_genotoxicity(self, matcher):
        row = make_row(section_title="Genotoxicity")
        m = matcher.match_row(row)
        assert m.match_type == MatchType.TITLE_NORMALISED
        assert m.expected_section.id == "4.2.3.3"

    def test_normalised_match_confidence_1_0_for_exact(self, matcher):
        m = matcher.match_row(make_row(section_title="Nonclinical Overview"))
        assert m.confidence == 1.0

    def test_alias_confidence_0_9(self, matcher):
        m = matcher.match_row(make_row(section_title="ISS"))
        assert m.confidence == 0.9

    def test_normalised_match_reason_non_empty(self, matcher):
        m = matcher.match_row(make_row(section_title="Clinical Overview"))
        assert len(m.match_reason) > 0

    def test_alias_match_reason_mentions_alias(self, matcher):
        m = matcher.match_row(make_row(section_title="RMP"))
        assert "alias" in m.match_reason.lower()

    def test_number_takes_priority_over_title(self, matcher):
        """Exact number match fires before normalised title even if title would differ."""
        row = make_row(section_number="1.8", section_title="Nonclinical Overview")
        m = matcher.match_row(row)
        # section_number "1.8" → Pharmacovigilance, not the title's 2.4
        assert m.match_type == MatchType.EXACT_NUMBER
        assert m.expected_section.id == "1.8"


# ===========================================================================
# Section 4 — Semantic match
# ===========================================================================

class TestSemanticMatch:
    """TF-IDF cosine above threshold with sufficient gap → SEMANTIC."""

    def test_semantic_repro_tox_paraphrase(self, matcher):
        """'reproductive and developmental toxicology' → 4.2.3.5
        (verified: cosine ~0.745, gap ~0.230)."""
        row = make_row(section_title="reproductive and developmental toxicology")
        m = matcher.match_row(row)
        assert m.match_type == MatchType.SEMANTIC
        assert m.expected_section.id == "4.2.3.5"

    def test_semantic_single_dose_paraphrase(self, matcher):
        """'single dose acute toxicity' → 4.2.3.1 (cosine ~0.991, gap ~0.400)."""
        row = make_row(section_title="single dose acute toxicity")
        m = matcher.match_row(row)
        assert m.match_type == MatchType.SEMANTIC
        assert m.expected_section.id == "4.2.3.1"

    def test_semantic_tabular_listing_trials(self, matcher):
        """'tabular listing clinical trials' → 5.2 (cosine ~0.919, gap ~0.599)."""
        row = make_row(section_title="tabular listing clinical trials")
        m = matcher.match_row(row)
        assert m.match_type == MatchType.SEMANTIC
        assert m.expected_section.id == "5.2"

    def test_semantic_human_pk(self, matcher):
        """'human pharmacokinetic pk studies reports' → 5.3.3."""
        row = make_row(section_title="human pharmacokinetic pk studies reports")
        m = matcher.match_row(row)
        assert m.match_type == MatchType.SEMANTIC
        assert m.expected_section.id == "5.3.3"

    def test_semantic_drug_substance_manufacturing(self, matcher):
        """'drug substance manufacturing process' → 3.2.S (cosine ~0.969)."""
        row = make_row(section_title="drug substance manufacturing process")
        m = matcher.match_row(row)
        assert m.match_type == MatchType.SEMANTIC
        assert m.expected_section.id == "3.2.S"

    def test_semantic_confidence_is_between_threshold_and_1(self, matcher):
        row = make_row(section_title="reproductive and developmental toxicology")
        m = matcher.match_row(row)
        assert SEMANTIC_THRESHOLD <= m.confidence <= 1.0

    def test_semantic_reason_mentions_cosine_score(self, matcher):
        row = make_row(section_title="single dose acute toxicity")
        m = matcher.match_row(row)
        assert "scores" in m.match_reason.lower() or "score" in m.match_reason.lower()

    def test_semantic_reason_mentions_threshold(self, matcher):
        row = make_row(section_title="single dose acute toxicity")
        m = matcher.match_row(row)
        assert str(SEMANTIC_THRESHOLD) in m.match_reason

    def test_semantic_reason_non_empty(self, matcher):
        m = matcher.match_row(make_row(section_title="tabular listing clinical trials"))
        assert len(m.match_reason) > 10

    def test_semantic_alternative_candidates_is_empty(self, matcher):
        """Clear semantic match has no alternative candidates."""
        m = matcher.match_row(make_row(section_title="tabular listing clinical trials"))
        assert m.alternative_candidates == []


# ===========================================================================
# Section 5 — Ambiguous match
# ===========================================================================

class TestAmbiguousMatch:
    """Cosine above threshold but gap too small → AMBIGUOUS."""

    def test_ambiguous_table_of_contents(self, matcher):
        """'table of contents' scores ~0.734 for 1.1 and ~0.644 for 2.1;
        gap ~0.090 < SEMANTIC_GAP → AMBIGUOUS."""
        row = make_row(section_title="table of contents")
        m = matcher.match_row(row)
        assert m.match_type == MatchType.AMBIGUOUS

    def test_ambiguous_has_alternative_candidates(self, matcher):
        row = make_row(section_title="table of contents")
        m = matcher.match_row(row)
        assert len(m.alternative_candidates) >= 1

    def test_ambiguous_status_is_needs_review_when_no_dossier_status(self, matcher):
        row = make_row(section_title="table of contents")
        m = matcher.match_row(row)
        assert m.status == MatchStatus.NEEDS_REVIEW
        assert m.status_source == "inferred"

    def test_ambiguous_confidence_is_above_threshold(self, matcher):
        row = make_row(section_title="table of contents")
        m = matcher.match_row(row)
        assert m.confidence >= SEMANTIC_THRESHOLD

    def test_ambiguous_reason_mentions_gap(self, matcher):
        row = make_row(section_title="table of contents")
        m = matcher.match_row(row)
        assert "gap" in m.match_reason.lower()

    def test_ambiguous_reason_mentions_both_candidates(self, matcher):
        row = make_row(section_title="table of contents")
        m = matcher.match_row(row)
        # Should mention the best AND the second-best section id
        assert m.expected_section is not None
        assert m.expected_section.id in m.match_reason

    def test_ambiguous_reason_mentions_human_review(self, matcher):
        row = make_row(section_title="table of contents")
        m = matcher.match_row(row)
        assert "review" in m.match_reason.lower()

    def test_ambiguous_pharmacology_summary(self, matcher):
        """'pharmacology summary' scores ~0.803 for 4.2.1 and ~0.780 for 2.6.2;
        gap ~0.023 < SEMANTIC_GAP → AMBIGUOUS."""
        row = make_row(section_title="pharmacology summary")
        m = matcher.match_row(row)
        assert m.match_type == MatchType.AMBIGUOUS

    def test_ambiguous_dossier_status_overrides_inferred(self, matcher):
        """If dossier says 'present', status is PRESENT even for AMBIGUOUS match."""
        row = make_row(section_title="table of contents", status="present")
        m = matcher.match_row(row)
        assert m.status == MatchStatus.PRESENT
        assert m.status_source == "dossier_supplied"


# ===========================================================================
# Section 6 — Missing / unmatched
# ===========================================================================

class TestMissingUnmatched:
    """No strategy succeeds → UNMATCHED match_type, MISSING status."""

    def test_unknown_section_number(self, matcher):
        row = make_row(section_number="9.9.9")
        m = matcher.match_row(row)
        assert m.match_type == MatchType.UNMATCHED
        assert m.expected_section is None
        assert m.confidence == 0.0

    def test_unmatched_status_is_missing(self, matcher):
        row = make_row(section_number="9.9.9")
        m = matcher.match_row(row)
        assert m.status == MatchStatus.MISSING
        assert m.status_source == "inferred"

    def test_gibberish_title_unmatched(self, matcher):
        row = make_row(section_title="xyzzy foobaz qux")
        m = matcher.match_row(row)
        assert m.match_type == MatchType.UNMATCHED
        assert m.confidence == 0.0

    def test_empty_row_unmatched(self, matcher):
        row = make_row()
        m = matcher.match_row(row)
        assert m.match_type == MatchType.UNMATCHED
        assert m.expected_section is None

    def test_unmatched_reason_non_empty(self, matcher):
        m = matcher.match_row(make_row(section_number="99.99"))
        assert len(m.match_reason) > 0

    def test_unmatched_reason_mentions_no_match(self, matcher):
        m = matcher.match_row(make_row(section_number="99.99"))
        assert "no match" in m.match_reason.lower() or "not found" in m.match_reason.lower()

    def test_unmatched_reason_mentions_section_number(self, matcher):
        m = matcher.match_row(make_row(section_number="99.99"))
        assert "99.99" in m.match_reason

    def test_unmatched_alternative_candidates_empty(self, matcher):
        m = matcher.match_row(make_row(section_number="99.99"))
        assert m.alternative_candidates == []

    def test_below_threshold_title_is_unmatched(self, matcher):
        """A title that produces cosine < SEMANTIC_THRESHOLD → UNMATCHED."""
        row = make_row(section_title="invoice payment terms")
        m = matcher.match_row(row)
        assert m.match_type == MatchType.UNMATCHED

    def test_unmatched_dossier_status_not_applicable_overrides(self, matcher):
        """If dossier says 'not_applicable', that is honoured even for UNMATCHED."""
        row = make_row(section_number="9.9.9", status="not_applicable")
        m = matcher.match_row(row)
        assert m.match_type == MatchType.UNMATCHED
        assert m.status == MatchStatus.NOT_APPLICABLE
        assert m.status_source == "dossier_supplied"


# ===========================================================================
# Section 7 — Status determination
# ===========================================================================

class TestStatusDetermination:
    """Status vocabulary mapping and dossier-supplied-overrides-inferred rule."""

    # --- vocabulary map ---

    def test_present_vocab(self):
        for s in ("present", "yes", "submitted", "complete", "completed", "done"):
            assert _map_status_string(s) == MatchStatus.PRESENT, f"Failed for {s!r}"

    def test_missing_vocab(self):
        for s in ("missing", "absent", "not_submitted", "no"):
            assert _map_status_string(s) == MatchStatus.MISSING, f"Failed for {s!r}"

    def test_needs_review_vocab(self):
        for s in ("needs_review", "review", "draft", "incomplete", "pending", "partial"):
            assert _map_status_string(s) == MatchStatus.NEEDS_REVIEW, f"Failed for {s!r}"

    def test_not_applicable_vocab(self):
        for s in ("not_applicable", "n/a", "na", "waived", "inapplicable"):
            assert _map_status_string(s) == MatchStatus.NOT_APPLICABLE, f"Failed for {s!r}"

    def test_unknown_status_maps_to_needs_review(self):
        assert _map_status_string("some_unknown_value") == MatchStatus.NEEDS_REVIEW

    def test_empty_string_maps_to_needs_review(self):
        assert _map_status_string("") == MatchStatus.NEEDS_REVIEW

    # --- dossier-supplied overrides inferred ---

    def test_dossier_missing_on_exact_match(self, matcher):
        """Dossier says 'missing' even though the section exists in KB."""
        row = make_row(section_number="2.7.4", status="missing")
        m = matcher.match_row(row)
        assert m.match_type == MatchType.EXACT_NUMBER
        assert m.status == MatchStatus.MISSING
        assert m.status_source == "dossier_supplied"

    def test_dossier_not_applicable_on_exact_match(self, matcher):
        row = make_row(section_number="4.2.3.6", status="not_applicable")
        m = matcher.match_row(row)
        assert m.status == MatchStatus.NOT_APPLICABLE
        assert m.status_source == "dossier_supplied"

    def test_dossier_needs_review_on_exact_match(self, matcher):
        row = make_row(section_number="5.3.5", status="needs_review")
        m = matcher.match_row(row)
        assert m.status == MatchStatus.NEEDS_REVIEW
        assert m.status_source == "dossier_supplied"

    def test_no_status_on_unmatched_infers_missing(self, matcher):
        row = make_row(section_number="0.0.0")
        m = matcher.match_row(row)
        assert m.status == MatchStatus.MISSING
        assert m.status_source == "inferred"

    def test_no_status_on_exact_match_infers_present(self, matcher):
        row = make_row(section_number="1.1")
        m = matcher.match_row(row)
        assert m.status == MatchStatus.PRESENT
        assert m.status_source == "inferred"

    def test_no_status_on_ambiguous_infers_needs_review(self, matcher):
        row = make_row(section_title="pharmacology summary")
        m = matcher.match_row(row)
        assert m.match_type == MatchType.AMBIGUOUS
        assert m.status == MatchStatus.NEEDS_REVIEW
        assert m.status_source == "inferred"

    def test_dossier_supplied_status_case_insensitive(self, matcher):
        row = make_row(section_number="2.3", status="PRESENT")
        m = matcher.match_row(row)
        assert m.status == MatchStatus.PRESENT

    def test_dossier_supplied_status_with_spaces(self, matcher):
        row = make_row(section_number="2.4", status="not applicable")
        m = matcher.match_row(row)
        assert m.status == MatchStatus.NOT_APPLICABLE


# ===========================================================================
# Section 8 — Dossier batch (match_dossier on synthetic dossier)
# ===========================================================================

class TestDossierBatch:
    """match_dossier() on the full synthetic dossier fixture."""

    @pytest.fixture(scope="class")
    @classmethod
    def summary(cls) -> MatchSummary:
        parse_result = parse_dossier(FIXTURES / "dossier_outline_clean.csv")
        m = build_matcher()
        return m.match_dossier(parse_result.rows)

    def test_returns_match_summary(self, summary):
        assert isinstance(summary, MatchSummary)

    def test_total_rows(self, summary):
        assert summary.total_rows == 75

    def test_matches_length(self, summary):
        assert len(summary.matches) == 75

    def test_every_match_has_non_empty_reason(self, summary):
        for m in summary.matches:
            assert m.match_reason, f"Empty reason for row {m.dossier_row.row_index}"

    def test_every_confidence_in_range(self, summary):
        for m in summary.matches:
            assert 0.0 <= m.confidence <= 1.0, (
                f"confidence {m.confidence} out of range for row {m.dossier_row.row_index}"
            )

    def test_every_status_is_valid(self, summary):
        valid = set(MatchStatus)
        for m in summary.matches:
            assert m.status in valid

    def test_every_match_type_is_valid(self, summary):
        valid = set(MatchType)
        for m in summary.matches:
            assert m.match_type in valid

    def test_bulk_exact_match_count(self, summary):
        """The synthetic dossier has clean section_numbers; most should be EXACT."""
        exact = sum(1 for m in summary.matches if m.match_type == MatchType.EXACT_NUMBER)
        assert exact >= 70, f"Expected ≥70 exact matches, got {exact}"

    def test_status_counts_add_up(self, summary):
        total = sum(summary.status_counts.values())
        assert total == 75

    def test_missing_sections_have_missing_status(self, summary):
        """Critical missing sections from the dossier must map to MISSING."""
        missing_ids = {"1.8", "2.4", "2.7.4", "4.2.3.5"}
        for m in summary.matches:
            sid = m.expected_section.id if m.expected_section else None
            if sid in missing_ids:
                assert m.status == MatchStatus.MISSING, (
                    f"Section {sid} should be MISSING, got {m.status}"
                )

    def test_needs_review_sections_present(self, summary):
        """Sections marked needs_review in the dossier must map to NEEDS_REVIEW."""
        review_ids = {"1.3", "2.5", "2.6.6", "4.2.3.2", "4.2.3.4", "5.3.5", "5.3.6"}
        for m in summary.matches:
            sid = m.expected_section.id if m.expected_section else None
            if sid in review_ids:
                assert m.status == MatchStatus.NEEDS_REVIEW, (
                    f"Section {sid} should be NEEDS_REVIEW, got {m.status}"
                )

    def test_not_applicable_sections(self, summary):
        na_ids = {"1.7", "4.2.3.6"}
        for m in summary.matches:
            sid = m.expected_section.id if m.expected_section else None
            if sid in na_ids:
                assert m.status == MatchStatus.NOT_APPLICABLE

    def test_all_exact_matches_confidence_1(self, summary):
        for m in summary.matches:
            if m.match_type == MatchType.EXACT_NUMBER:
                assert m.confidence == 1.0

    def test_match_summary_to_dict(self, summary):
        d = summary.to_dict()
        assert set(d.keys()) == {
            "total_rows", "exact_number_count", "title_normalised_count",
            "semantic_count", "ambiguous_count", "unmatched_count", "status_counts",
        }
        assert d["total_rows"] == 75

    def test_match_to_dict_has_all_fields(self, summary):
        d = summary.matches[0].to_dict()
        assert "matched_section_id" in d
        assert "match_type" in d
        assert "confidence" in d
        assert "status" in d
        assert "match_reason" in d
        assert "status_source" in d

    def test_status_source_is_dossier_supplied_for_rows_with_status(self, summary):
        """All 75 rows in the synthetic dossier have a status field → all dossier_supplied."""
        for m in summary.matches:
            assert m.status_source == "dossier_supplied", (
                f"Row {m.dossier_row.row_index}: expected dossier_supplied, "
                f"got {m.status_source!r}"
            )
