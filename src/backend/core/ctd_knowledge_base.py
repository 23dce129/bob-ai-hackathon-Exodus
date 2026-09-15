"""
ctd_knowledge_base.py — ICH M4 CTD Knowledge Base loader and query interface.

This module loads the ICH M4 CTD knowledge base from the bundled JSON file and
exposes a structured query API for use by dossier_parser.py, readiness_scorer.py,
and traceability_engine.py.

Disclaimer
----------
The underlying JSON is an interpretation of publicly available ICH M4 guidelines
for demonstration and screening purposes only.  It is NOT legal or regulatory
advice and does NOT represent the position of ICH or any regulatory authority.

Usage
-----
    from core.ctd_knowledge_base import load_knowledge_base

    kb = load_knowledge_base()              # singleton — loaded once per process
    section = kb.get_section("2.7.4")       # CTDSection object
    safety  = kb.get_safety_sections()      # list[CTDSection]
    gaps    = kb.get_required_sections("NDA")

Public API
----------
    CTDSection          — dataclass representing one leaf or sub-tree section
    CTDModule           — dataclass representing a top-level module (M1–M5)
    CTDKnowledgeBase    — main query object
    load_knowledge_base() — factory returning the singleton instance
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Path to the bundled JSON
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parent.parent / "data"
_CTD_JSON_PATH = _DATA_DIR / "ich_m4_ctd.json"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CTDSection:
    """A single section entry from the ICH M4 CTD knowledge base.

    Attributes
    ----------
    id              : unique identifier, e.g. "2.7.4"
    number          : display number (same as id for most sections, e.g. "3.2.S.4")
    title           : section title
    parent_id       : id of the parent section or module ("2.7", "3.2.S", "M4", …)
    level           : nesting depth (2 = top-level module child, 3 = sub-section, 4 = sub-sub)
    requirement     : "required" | "conditional" | "optional"
    applicability   : list of submission types this section applies to, e.g. ["NDA", "BLA"]
    submission_types_note : human-readable note on applicability nuances
    ich_source      : True if title/number come directly from ICH M4 text
    safety_relevant : True if this section directly bears on drug safety reporting
    traceability_categories : list of safety signal categories this section traces to
                              (e.g. ["hepatotoxicity", "cardiac"])
    """

    id: str
    number: str
    title: str
    parent_id: str
    level: int
    requirement: str
    applicability: list[str]
    submission_types_note: str
    ich_source: bool
    safety_relevant: bool
    traceability_categories: list[str]

    def applies_to(self, submission_type: str) -> bool:
        """Return True if this section applies to the given submission type."""
        return submission_type.upper() in self.applicability

    def is_required_for(self, submission_type: str) -> bool:
        """Return True if this section is *required* (not conditional/optional)
        for the given submission type."""
        return self.requirement == "required" and self.applies_to(submission_type)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "number": self.number,
            "title": self.title,
            "parent_id": self.parent_id,
            "level": self.level,
            "requirement": self.requirement,
            "applicability": self.applicability,
            "submission_types_note": self.submission_types_note,
            "ich_source": self.ich_source,
            "safety_relevant": self.safety_relevant,
            "traceability_categories": self.traceability_categories,
        }


@dataclass
class CTDModule:
    """A top-level CTD module (M1–M5).

    Attributes
    ----------
    id       : "M1" … "M5"
    number   : 1 … 5
    title    : module title
    scope    : "regional" or "common"
    ich_source : True for Modules 2–5; False for M1 (regional)
    sections : flat list of all CTDSection objects belonging to this module
    """

    id: str
    number: int
    title: str
    scope: str
    ich_source: bool
    sections: list[CTDSection] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "number": self.number,
            "title": self.title,
            "scope": self.scope,
            "ich_source": self.ich_source,
            "section_count": len(self.sections),
        }


# ---------------------------------------------------------------------------
# Knowledge base class
# ---------------------------------------------------------------------------

class CTDKnowledgeBase:
    """Loaded ICH M4 CTD knowledge base with structured query methods.

    Do not instantiate directly.  Use ``load_knowledge_base()`` to get the
    singleton, which is loaded once and cached for the lifetime of the process.
    """

    def __init__(self, raw: dict) -> None:
        self._disclaimer: str = raw.get("disclaimer", "")
        self._schema_version: str = raw.get("schema_version", "unknown")
        self._modules: dict[str, CTDModule] = {}
        self._sections: dict[str, CTDSection] = {}  # keyed by section id

        for mod_raw in raw.get("modules", []):
            module = CTDModule(
                id=mod_raw["id"],
                number=mod_raw["number"],
                title=mod_raw["title"],
                scope=mod_raw.get("scope", "common"),
                ich_source=mod_raw.get("ich_source", True),
            )
            for sec_raw in mod_raw.get("sections", []):
                ann = sec_raw.get("pharmaguard_annotations", {})
                section = CTDSection(
                    id=sec_raw["id"],
                    number=sec_raw["number"],
                    title=sec_raw["title"],
                    parent_id=sec_raw["parent_id"],
                    level=sec_raw["level"],
                    requirement=sec_raw["requirement"],
                    applicability=sec_raw.get("applicability", []),
                    submission_types_note=sec_raw.get("submission_types_note", ""),
                    ich_source=sec_raw.get("ich_source", True),
                    safety_relevant=ann.get("safety_relevant", False),
                    traceability_categories=ann.get("traceability_categories", []),
                )
                module.sections.append(section)
                self._sections[section.id] = section

            self._modules[module.id] = module

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def disclaimer(self) -> str:
        """Full regulatory disclaimer from the knowledge base."""
        return self._disclaimer

    @property
    def schema_version(self) -> str:
        return self._schema_version

    @property
    def total_section_count(self) -> int:
        return len(self._sections)

    # ------------------------------------------------------------------
    # Module queries
    # ------------------------------------------------------------------

    def get_module(self, module_id: str) -> Optional[CTDModule]:
        """Return the CTDModule for the given id (e.g. "M2"), or None."""
        return self._modules.get(module_id.upper())

    def all_modules(self) -> list[CTDModule]:
        """Return all modules in order M1–M5."""
        return sorted(self._modules.values(), key=lambda m: m.number)

    # ------------------------------------------------------------------
    # Section queries
    # ------------------------------------------------------------------

    def get_section(self, section_id: str) -> Optional[CTDSection]:
        """Return the CTDSection for the given id (e.g. "2.7.4"), or None."""
        return self._sections.get(section_id)

    def all_sections(self) -> list[CTDSection]:
        """Return all sections across all modules, in document order."""
        result: list[CTDSection] = []
        for module in self.all_modules():
            result.extend(module.sections)
        return result

    def get_all_section_ids(self) -> list[str]:
        """Return a sorted list of all section ids."""
        return sorted(self._sections.keys())

    def get_sections_for_module(self, module_id: str) -> list[CTDSection]:
        """Return all sections belonging to the given module id."""
        module = self.get_module(module_id)
        return module.sections if module is not None else []

    def get_required_sections(self, submission_type: str) -> list[CTDSection]:
        """Return all sections that are *required* for the given submission type.

        Parameters
        ----------
        submission_type : "NDA", "BLA", "MAA", or "ANDA"
        """
        return [s for s in self.all_sections() if s.is_required_for(submission_type)]

    def get_sections_for_submission_type(self, submission_type: str) -> list[CTDSection]:
        """Return all sections that *apply* to the given submission type
        (required + conditional + optional)."""
        return [s for s in self.all_sections() if s.applies_to(submission_type)]

    def get_safety_sections(self) -> list[CTDSection]:
        """Return all sections marked ``safety_relevant = True``."""
        return [s for s in self.all_sections() if s.safety_relevant]

    def get_sections_by_traceability_category(self, category: str) -> list[CTDSection]:
        """Return all sections whose ``traceability_categories`` includes *category*.

        Parameters
        ----------
        category : e.g. "hepatotoxicity", "cardiac", "bleeding", "general"
        """
        cat = category.lower()
        return [s for s in self.all_sections() if cat in s.traceability_categories]

    def get_conditional_sections(self, submission_type: str) -> list[CTDSection]:
        """Return sections that are *conditional* for the given submission type."""
        return [
            s for s in self.all_sections()
            if s.requirement == "conditional" and s.applies_to(submission_type)
        ]

    def get_all_traceability_categories(self) -> list[str]:
        """Return sorted list of all distinct traceability categories in the KB."""
        cats: set[str] = set()
        for s in self.all_sections():
            cats.update(s.traceability_categories)
        return sorted(cats)


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_knowledge_base(json_path: Optional[str] = None) -> CTDKnowledgeBase:
    """Load and return the CTD knowledge base singleton.

    The result is cached by ``lru_cache`` so the JSON is read only once per
    process.

    Parameters
    ----------
    json_path : optional override for the path to the JSON file.  Defaults to
                the bundled ``data/ich_m4_ctd.json``.  Pass a custom path only
                in tests that need to inject a minimal fixture JSON.
    """
    path = Path(json_path) if json_path else _CTD_JSON_PATH
    with path.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    return CTDKnowledgeBase(raw)
