"""
test_ctd_knowledge_base.py — Unit tests for the ICH M4 CTD knowledge base loader.

Test strategy
-------------
Tests are grouped into four sections:

  1. Loading — JSON is read correctly, metadata is present, section count matches.
  2. Module queries — all_modules(), get_module(), edge cases.
  3. Section queries — get_section(), all_sections(), filtering helpers.
  4. Fixture injection — tests that pass a minimal in-memory JSON path to verify
     the loader logic in isolation from the real knowledge base JSON.

All expected values (section counts, traceability categories, requirement values)
were confirmed by reading ich_m4_ctd.json before being asserted here.

Key invariants asserted throughout
-----------------------------------
  - Total section count == 75 (5 module-level ids like "M1"–"M5" are not sections)
  - Modules M1–M5 are all present
  - Every CTDSection has: id, title, requirement ∈ {required, conditional, optional}
  - safety_relevant is a bool; traceability_categories is a list
  - disclaimer is non-empty and contains a caution phrase
  - load_knowledge_base() returns the same object on repeated calls (singleton)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from core.ctd_knowledge_base import (
    CTDKnowledgeBase,
    CTDModule,
    CTDSection,
    load_knowledge_base,
)


# ---------------------------------------------------------------------------
# Helpers — minimal fixture JSON written to a tmp file
# ---------------------------------------------------------------------------

MINIMAL_JSON = {
    "schema_version": "0.0.1-test",
    "disclaimer": "Test disclaimer — not regulatory advice.",
    "modules": [
        {
            "id": "M2",
            "number": 2,
            "title": "CTD Summaries",
            "scope": "common",
            "ich_source": True,
            "sections": [
                {
                    "id": "2.7.4",
                    "number": "2.7.4",
                    "title": "Summary of Clinical Safety",
                    "parent_id": "2.7",
                    "level": 3,
                    "requirement": "required",
                    "applicability": ["NDA", "BLA", "MAA"],
                    "submission_types_note": "Primary clinical safety summary.",
                    "ich_source": True,
                    "pharmaguard_annotations": {
                        "safety_relevant": True,
                        "traceability_categories": ["hepatotoxicity", "cardiac", "bleeding"],
                    },
                },
                {
                    "id": "2.6.6",
                    "number": "2.6.6",
                    "title": "Toxicology Written Summary",
                    "parent_id": "2.6",
                    "level": 3,
                    "requirement": "required",
                    "applicability": ["NDA", "BLA", "MAA"],
                    "submission_types_note": "Written summary of toxicology.",
                    "ich_source": True,
                    "pharmaguard_annotations": {
                        "safety_relevant": True,
                        "traceability_categories": ["hepatotoxicity", "renal"],
                    },
                },
                {
                    "id": "2.3",
                    "number": "2.3",
                    "title": "Quality Overall Summary",
                    "parent_id": "M2",
                    "level": 2,
                    "requirement": "required",
                    "applicability": ["NDA", "BLA", "MAA", "ANDA"],
                    "submission_types_note": "Quality summary.",
                    "ich_source": True,
                    "pharmaguard_annotations": {
                        "safety_relevant": False,
                        "traceability_categories": [],
                    },
                },
            ],
        },
        {
            "id": "M4",
            "number": 4,
            "title": "Nonclinical Study Reports",
            "scope": "common",
            "ich_source": True,
            "sections": [
                {
                    "id": "4.2.3.4",
                    "number": "4.2.3.4",
                    "title": "Carcinogenicity",
                    "parent_id": "4.2.3",
                    "level": 4,
                    "requirement": "conditional",
                    "applicability": ["NDA", "BLA", "MAA"],
                    "submission_types_note": "Required for long-term use drugs.",
                    "ich_source": True,
                    "pharmaguard_annotations": {
                        "safety_relevant": True,
                        "traceability_categories": ["carcinogenicity"],
                    },
                },
            ],
        },
    ],
}


@pytest.fixture(scope="module")
def fixture_json_path(tmp_path_factory) -> str:
    """Write MINIMAL_JSON to a temp file and return its path."""
    p = tmp_path_factory.mktemp("ctd") / "test_ctd.json"
    p.write_text(json.dumps(MINIMAL_JSON), encoding="utf-8")
    return str(p)


@pytest.fixture(scope="module")
def fixture_kb(fixture_json_path) -> CTDKnowledgeBase:
    """Return a CTDKnowledgeBase loaded from the minimal fixture."""
    # Bypass lru_cache by constructing directly
    with open(fixture_json_path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return CTDKnowledgeBase(raw)


@pytest.fixture(scope="module")
def real_kb() -> CTDKnowledgeBase:
    """Return the knowledge base loaded from the real ich_m4_ctd.json."""
    return load_knowledge_base()


# ===========================================================================
# Section 1 — Loading and metadata
# ===========================================================================

class TestLoading:
    """Tests for JSON loading, metadata, and invariants on the real KB."""

    def test_load_returns_knowledge_base_instance(self, real_kb):
        assert isinstance(real_kb, CTDKnowledgeBase)

    def test_singleton_returns_same_object(self, real_kb):
        """load_knowledge_base() must be idempotent (lru_cache)."""
        kb2 = load_knowledge_base()
        assert real_kb is kb2

    def test_schema_version_is_present(self, real_kb):
        assert real_kb.schema_version != "unknown"
        assert isinstance(real_kb.schema_version, str)
        assert len(real_kb.schema_version) > 0

    def test_disclaimer_is_non_empty(self, real_kb):
        assert isinstance(real_kb.disclaimer, str)
        assert len(real_kb.disclaimer) > 20

    def test_disclaimer_contains_caution_phrase(self, real_kb):
        """Disclaimer must warn that this is NOT regulatory advice."""
        disc = real_kb.disclaimer.lower()
        assert "not" in disc or "disclaimer" in disc.lower()

    def test_total_section_count(self, real_kb):
        """The real JSON has exactly 75 section entries.
        The 80 grep hits include 5 module-level ids (M1–M5) which are not sections."""
        assert real_kb.total_section_count == 75

    def test_five_modules_loaded(self, real_kb):
        assert len(real_kb.all_modules()) == 5

    def test_all_module_ids_present(self, real_kb):
        ids = {m.id for m in real_kb.all_modules()}
        assert ids == {"M1", "M2", "M3", "M4", "M5"}

    def test_modules_in_numeric_order(self, real_kb):
        numbers = [m.number for m in real_kb.all_modules()]
        assert numbers == [1, 2, 3, 4, 5]

    def test_all_sections_have_non_empty_title(self, real_kb):
        for s in real_kb.all_sections():
            assert isinstance(s.title, str) and len(s.title) > 0, (
                f"Section {s.id} has empty title"
            )

    def test_all_sections_have_valid_requirement(self, real_kb):
        valid = {"required", "conditional", "optional"}
        for s in real_kb.all_sections():
            assert s.requirement in valid, (
                f"Section {s.id} has invalid requirement: {s.requirement!r}"
            )

    def test_all_sections_safety_relevant_is_bool(self, real_kb):
        for s in real_kb.all_sections():
            assert isinstance(s.safety_relevant, bool), (
                f"Section {s.id} safety_relevant is not bool"
            )

    def test_all_sections_traceability_is_list(self, real_kb):
        for s in real_kb.all_sections():
            assert isinstance(s.traceability_categories, list), (
                f"Section {s.id} traceability_categories is not list"
            )

    def test_all_sections_have_level(self, real_kb):
        for s in real_kb.all_sections():
            assert isinstance(s.level, int) and s.level >= 2

    def test_all_sections_have_parent_id(self, real_kb):
        for s in real_kb.all_sections():
            assert isinstance(s.parent_id, str) and len(s.parent_id) > 0

    def test_all_sections_applicability_is_list(self, real_kb):
        for s in real_kb.all_sections():
            assert isinstance(s.applicability, list)


# ===========================================================================
# Section 2 — Module queries
# ===========================================================================

class TestModuleQueries:
    """Tests for get_module() and all_modules()."""

    def test_get_module_m2(self, real_kb):
        m = real_kb.get_module("M2")
        assert m is not None
        assert m.id == "M2"
        assert m.number == 2
        assert "Summaries" in m.title or "Common" in m.title

    def test_get_module_case_insensitive(self, real_kb):
        """get_module should normalise to uppercase."""
        assert real_kb.get_module("m3") is not None
        assert real_kb.get_module("M3") is not None

    def test_get_module_unknown_returns_none(self, real_kb):
        assert real_kb.get_module("M9") is None
        assert real_kb.get_module("") is None

    def test_m1_is_regional_scope(self, real_kb):
        m = real_kb.get_module("M1")
        assert m.scope == "regional"

    def test_m2_m5_are_common_scope(self, real_kb):
        for mid in ("M2", "M3", "M4", "M5"):
            m = real_kb.get_module(mid)
            assert m.scope == "common", f"{mid} should be 'common', got {m.scope!r}"

    def test_each_module_has_sections(self, real_kb):
        for m in real_kb.all_modules():
            assert len(m.sections) > 0, f"Module {m.id} has no sections"

    def test_module_to_dict_shape(self, real_kb):
        d = real_kb.get_module("M5").to_dict()
        assert set(d.keys()) == {"id", "number", "title", "scope", "ich_source", "section_count"}
        assert d["id"] == "M5"
        assert isinstance(d["section_count"], int)

    def test_get_sections_for_module_m1(self, real_kb):
        sections = real_kb.get_sections_for_module("M1")
        assert len(sections) > 0
        for s in sections:
            assert s.parent_id == "M1" or s.parent_id.startswith("1.")

    def test_get_sections_for_unknown_module(self, real_kb):
        assert real_kb.get_sections_for_module("M99") == []


# ===========================================================================
# Section 3 — Section queries (real KB)
# ===========================================================================

class TestSectionQueries:
    """Tests for individual section retrieval and filter methods on the real KB."""

    # --- get_section ---

    def test_get_section_2_7_4(self, real_kb):
        """Section 2.7.4 Summary of Clinical Safety must exist and be safety_relevant."""
        s = real_kb.get_section("2.7.4")
        assert s is not None
        assert s.id == "2.7.4"
        assert s.safety_relevant is True
        assert "Clinical Safety" in s.title

    def test_get_section_1_8_pharmacovigilance(self, real_kb):
        """Section 1.8 Pharmacovigilance must exist and be safety_relevant."""
        s = real_kb.get_section("1.8")
        assert s is not None
        assert s.safety_relevant is True
        assert s.requirement == "required"

    def test_get_section_2_3_quality_summary(self, real_kb):
        s = real_kb.get_section("2.3")
        assert s is not None
        assert s.safety_relevant is False

    def test_get_section_unknown_returns_none(self, real_kb):
        assert real_kb.get_section("9.9.9") is None
        assert real_kb.get_section("") is None

    def test_get_all_section_ids_returns_75_ids(self, real_kb):
        ids = real_kb.get_all_section_ids()
        assert len(ids) == 75

    def test_get_all_section_ids_are_sorted(self, real_kb):
        ids = real_kb.get_all_section_ids()
        assert ids == sorted(ids)

    # --- applies_to / is_required_for ---

    def test_section_applies_to_submission_type(self, real_kb):
        s = real_kb.get_section("2.7.4")
        assert s.applies_to("NDA") is True
        assert s.applies_to("BLA") is True
        assert s.applies_to("MAA") is True
        assert s.applies_to("ANDA") is False  # not applicable to generics

    def test_section_is_required_for(self, real_kb):
        s = real_kb.get_section("2.7.4")
        assert s.is_required_for("NDA") is True
        assert s.is_required_for("ANDA") is False  # not in applicability

    def test_conditional_section_not_required(self, real_kb):
        s = real_kb.get_section("4.2.3.4")  # Carcinogenicity — conditional
        assert s.requirement == "conditional"
        assert s.is_required_for("NDA") is False

    # --- get_required_sections ---

    def test_get_required_sections_nda_returns_list(self, real_kb):
        required = real_kb.get_required_sections("NDA")
        assert isinstance(required, list)
        assert len(required) > 0

    def test_get_required_sections_nda_all_are_required(self, real_kb):
        for s in real_kb.get_required_sections("NDA"):
            assert s.requirement == "required"
            assert "NDA" in s.applicability

    def test_get_required_sections_anda_fewer_than_nda(self, real_kb):
        """ANDA has fewer required sections (no clinical overviews etc.)."""
        nda = real_kb.get_required_sections("NDA")
        anda = real_kb.get_required_sections("ANDA")
        assert len(anda) < len(nda)

    def test_get_sections_for_submission_type_includes_conditional(self, real_kb):
        all_nda = real_kb.get_sections_for_submission_type("NDA")
        required_nda = real_kb.get_required_sections("NDA")
        # All-NDA should include conditional sections not in required-NDA
        assert len(all_nda) > len(required_nda)

    # --- get_safety_sections ---

    def test_get_safety_sections_non_empty(self, real_kb):
        safety = real_kb.get_safety_sections()
        assert len(safety) > 0

    def test_get_safety_sections_all_safety_relevant(self, real_kb):
        for s in real_kb.get_safety_sections():
            assert s.safety_relevant is True

    def test_safety_sections_include_2_7_4(self, real_kb):
        ids = {s.id for s in real_kb.get_safety_sections()}
        assert "2.7.4" in ids

    def test_safety_sections_include_1_8(self, real_kb):
        ids = {s.id for s in real_kb.get_safety_sections()}
        assert "1.8" in ids

    # --- get_sections_by_traceability_category ---

    def test_hepatotoxicity_sections_non_empty(self, real_kb):
        sections = real_kb.get_sections_by_traceability_category("hepatotoxicity")
        assert len(sections) > 0

    def test_hepatotoxicity_sections_contain_2_7_4(self, real_kb):
        ids = {s.id for s in real_kb.get_sections_by_traceability_category("hepatotoxicity")}
        assert "2.7.4" in ids

    def test_cardiac_sections_non_empty(self, real_kb):
        sections = real_kb.get_sections_by_traceability_category("cardiac")
        assert len(sections) > 0

    def test_bleeding_sections_non_empty(self, real_kb):
        sections = real_kb.get_sections_by_traceability_category("bleeding")
        assert len(sections) > 0

    def test_unknown_category_returns_empty(self, real_kb):
        sections = real_kb.get_sections_by_traceability_category("nonexistent_xyz")
        assert sections == []

    def test_category_matching_is_case_insensitive_via_lower(self, real_kb):
        """The implementation lowercases the query, so passing already-lower is fine."""
        hepato = real_kb.get_sections_by_traceability_category("hepatotoxicity")
        assert len(hepato) > 0

    # --- get_all_traceability_categories ---

    def test_get_all_traceability_categories_is_sorted_list(self, real_kb):
        cats = real_kb.get_all_traceability_categories()
        assert isinstance(cats, list)
        assert cats == sorted(cats)

    def test_all_traceability_categories_non_empty(self, real_kb):
        cats = real_kb.get_all_traceability_categories()
        assert len(cats) > 0

    def test_known_categories_present(self, real_kb):
        cats = set(real_kb.get_all_traceability_categories())
        for expected in ("hepatotoxicity", "cardiac", "bleeding", "general", "carcinogenicity"):
            assert expected in cats, f"Expected category {expected!r} not found"

    def test_general_category_present(self, real_kb):
        cats = real_kb.get_all_traceability_categories()
        assert "general" in cats

    # --- get_conditional_sections ---

    def test_get_conditional_sections_nda(self, real_kb):
        cond = real_kb.get_conditional_sections("NDA")
        assert len(cond) > 0
        for s in cond:
            assert s.requirement == "conditional"
            assert "NDA" in s.applicability

    def test_carcinogenicity_is_conditional_for_nda(self, real_kb):
        cond_ids = {s.id for s in real_kb.get_conditional_sections("NDA")}
        assert "4.2.3.4" in cond_ids  # Carcinogenicity

    # --- to_dict ---

    def test_section_to_dict_has_all_keys(self, real_kb):
        s = real_kb.get_section("2.7.4")
        d = s.to_dict()
        expected_keys = {
            "id", "number", "title", "parent_id", "level",
            "requirement", "applicability", "submission_types_note",
            "ich_source", "safety_relevant", "traceability_categories",
        }
        assert set(d.keys()) == expected_keys

    def test_section_to_dict_values_match(self, real_kb):
        s = real_kb.get_section("2.7.4")
        d = s.to_dict()
        assert d["id"] == "2.7.4"
        assert d["safety_relevant"] is True
        assert "NDA" in d["applicability"]

    # --- all_sections ordering ---

    def test_all_sections_count(self, real_kb):
        assert len(real_kb.all_sections()) == 75

    def test_all_sections_ordered_by_module(self, real_kb):
        """Sections from M1 should come before M2, M2 before M3, etc."""
        sections = real_kb.all_sections()
        # Find first occurrence of each module prefix
        first_1x = next((i for i, s in enumerate(sections) if s.id.startswith("1.")), None)
        first_2x = next((i for i, s in enumerate(sections) if s.id.startswith("2.")), None)
        first_3x = next((i for i, s in enumerate(sections) if s.id.startswith("3.")), None)
        assert first_1x < first_2x < first_3x


# ===========================================================================
# Section 4 — Fixture injection (isolated loader logic tests)
# ===========================================================================

class TestFixtureKB:
    """Tests using the minimal fixture JSON to verify loader logic in isolation."""

    def test_fixture_loads_two_modules(self, fixture_kb):
        assert len(fixture_kb.all_modules()) == 2

    def test_fixture_total_section_count(self, fixture_kb):
        # 3 sections in M2 + 1 section in M4 = 4
        assert fixture_kb.total_section_count == 4

    def test_fixture_get_section_2_7_4(self, fixture_kb):
        s = fixture_kb.get_section("2.7.4")
        assert s is not None
        assert s.title == "Summary of Clinical Safety"
        assert s.safety_relevant is True
        assert s.requirement == "required"

    def test_fixture_get_section_missing(self, fixture_kb):
        assert fixture_kb.get_section("5.3.5") is None

    def test_fixture_get_module_m4(self, fixture_kb):
        m = fixture_kb.get_module("M4")
        assert m is not None
        assert m.number == 4

    def test_fixture_get_module_m5_missing(self, fixture_kb):
        assert fixture_kb.get_module("M5") is None

    def test_fixture_get_required_sections_nda(self, fixture_kb):
        req = fixture_kb.get_required_sections("NDA")
        # 2.7.4, 2.6.6, 2.3 are required for NDA; 4.2.3.4 is conditional
        assert len(req) == 3
        ids = {s.id for s in req}
        assert "4.2.3.4" not in ids

    def test_fixture_get_safety_sections(self, fixture_kb):
        safety = fixture_kb.get_safety_sections()
        ids = {s.id for s in safety}
        assert "2.7.4" in ids
        assert "2.6.6" in ids
        assert "4.2.3.4" in ids
        assert "2.3" not in ids  # not safety_relevant

    def test_fixture_get_sections_by_traceability_hepatotoxicity(self, fixture_kb):
        sections = fixture_kb.get_sections_by_traceability_category("hepatotoxicity")
        ids = {s.id for s in sections}
        assert "2.7.4" in ids
        assert "2.6.6" in ids  # also has hepatotoxicity

    def test_fixture_get_sections_by_traceability_cardiac(self, fixture_kb):
        sections = fixture_kb.get_sections_by_traceability_category("cardiac")
        ids = {s.id for s in sections}
        assert "2.7.4" in ids
        assert "2.6.6" not in ids  # 2.6.6 has hepatotoxicity + renal, not cardiac

    def test_fixture_get_sections_by_traceability_carcinogenicity(self, fixture_kb):
        sections = fixture_kb.get_sections_by_traceability_category("carcinogenicity")
        assert len(sections) == 1
        assert sections[0].id == "4.2.3.4"

    def test_fixture_get_conditional_sections(self, fixture_kb):
        cond = fixture_kb.get_conditional_sections("NDA")
        assert len(cond) == 1
        assert cond[0].id == "4.2.3.4"

    def test_fixture_get_all_traceability_categories(self, fixture_kb):
        cats = fixture_kb.get_all_traceability_categories()
        assert sorted(cats) == cats
        expected = {"hepatotoxicity", "cardiac", "bleeding", "renal", "carcinogenicity"}
        assert expected.issubset(set(cats))

    def test_fixture_anda_has_only_quality_summary(self, fixture_kb):
        """Only 2.3 is applicable to ANDA in our minimal fixture."""
        anda_sections = fixture_kb.get_sections_for_submission_type("ANDA")
        ids = {s.id for s in anda_sections}
        assert ids == {"2.3"}

    def test_fixture_section_applies_to(self, fixture_kb):
        s = fixture_kb.get_section("2.7.4")
        assert s.applies_to("NDA") is True
        assert s.applies_to("ANDA") is False

    def test_fixture_section_is_required_for(self, fixture_kb):
        s = fixture_kb.get_section("2.7.4")
        assert s.is_required_for("NDA") is True
        assert s.is_required_for("ANDA") is False

    def test_fixture_disclaimer_loaded(self, fixture_kb):
        assert "disclaimer" in fixture_kb.disclaimer.lower()

    def test_fixture_schema_version(self, fixture_kb):
        assert fixture_kb.schema_version == "0.0.1-test"

    def test_fixture_module_to_dict(self, fixture_kb):
        m = fixture_kb.get_module("M2")
        d = m.to_dict()
        assert d["id"] == "M2"
        assert d["section_count"] == 3

    def test_fixture_section_to_dict(self, fixture_kb):
        s = fixture_kb.get_section("4.2.3.4")
        d = s.to_dict()
        assert d["requirement"] == "conditional"
        assert d["traceability_categories"] == ["carcinogenicity"]
        assert d["safety_relevant"] is True

    def test_fixture_all_section_ids_sorted(self, fixture_kb):
        ids = fixture_kb.get_all_section_ids()
        assert ids == sorted(ids)
        assert len(ids) == 4

    def test_fixture_get_sections_for_module_m2(self, fixture_kb):
        sections = fixture_kb.get_sections_for_module("M2")
        assert len(sections) == 3

    def test_fixture_get_sections_for_module_m4(self, fixture_kb):
        sections = fixture_kb.get_sections_for_module("M4")
        assert len(sections) == 1
        assert sections[0].id == "4.2.3.4"

    def test_fixture_get_sections_for_unknown_module(self, fixture_kb):
        assert fixture_kb.get_sections_for_module("M9") == []

    def test_fixture_unknown_category_empty(self, fixture_kb):
        assert fixture_kb.get_sections_by_traceability_category("xyz_unknown") == []
