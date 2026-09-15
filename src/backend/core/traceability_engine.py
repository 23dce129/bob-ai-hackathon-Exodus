"""
traceability_engine.py — Safety Signal to CTD Section Traceability Engine.

⚠️  DISCLAIMER ⚠️
This is a heuristic screening tool for internal gap-identification only.
It is NOT an FDA approval readiness assessment.
It is NOT a regulatory compliance determination.
It does NOT represent the position of the FDA, ICH, EMA, or any regulatory agency.
A detected safety signal does NOT establish causality between a drug and an adverse event.
A missing CTD section identified here does NOT mean the submission will be rejected.
All results require review by a qualified regulatory affairs professional.

Purpose
-------
Given a detected pharmacovigilance signal (drug + adverse event + statistics) and a
parsed dossier MatchSummary, identify which CTD sections are potentially relevant to
documenting that signal, and determine whether the dossier contains each of those
sections.

This module does NOT re-implement matching or scoring logic.  It reuses:
  - CTDKnowledgeBase.get_sections_by_traceability_category()  (section lookup)
  - CTDKnowledgeBase.get_sections_by_traceability_category("general") (fallback)
  - MatchSummary / SectionMatch from section_matcher                  (dossier status)
  - MatchStatus enum from section_matcher                             (status vocabulary)

Traceability score formula
--------------------------
For a single signal with N relevant CTD sections:

    score = (PRESENT_count + NEEDS_REVIEW_count × PARTIAL) / N

where PARTIAL = 0.5 (same constant as readiness_scorer).

NOT_APPLICABLE sections are excluded from N.
Score = 1.0 when N == 0 (no relevant sections → trivially documented).

Priority levels
---------------
  critical — section is safety_relevant=True AND status is MISSING
  high     — section is safety_relevant=True (regardless of status)
  normal   — section is not safety_relevant

Signal category classification
-------------------------------
classify_signal_category() maps an adverse event preferred term to one of the
categories keyed in CTDKnowledgeBase.get_all_traceability_categories().
Matching is keyword-based and deterministic.
Unrecognised terms resolve to "general".

Public API
----------
    TRACEABILITY_DISCLAIMER          — module-level constant; embedded in every result
    TRACEABILITY_PARTIAL             — partial credit for NEEDS_REVIEW (0.5)
    TraceabilityEntry                — one CTD section's traceability result
    TraceabilityResult               — all entries for one signal
    TraceabilityReport               — all results across multiple signals
    classify_signal_category()       — adverse event term → category string
    trace_signal()                   — single signal + MatchSummary → TraceabilityResult
    trace_all_signals()              — list[signal] + MatchSummary → TraceabilityReport
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from core.ctd_knowledge_base import CTDKnowledgeBase, CTDSection, load_knowledge_base
from core.section_matcher import MatchStatus, MatchSummary

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Disclaimer — embedded in every result object; cannot be omitted
# ---------------------------------------------------------------------------

TRACEABILITY_DISCLAIMER = (
    "HEURISTIC SCREENING AID ONLY — NOT an FDA approval readiness score. "
    "NOT a regulatory compliance assessment. "
    "A detected signal does NOT establish causality between a drug and an adverse event. "
    "A missing CTD section does NOT mean the submission will be rejected. "
    "Does NOT represent the position of the FDA, ICH, EMA, or any regulatory agency. "
    "All results require review by a qualified regulatory affairs professional."
)

# ---------------------------------------------------------------------------
# Scoring constant
# ---------------------------------------------------------------------------

# Partial credit awarded to NEEDS_REVIEW sections (mirrors readiness_scorer)
TRACEABILITY_PARTIAL: float = 0.5


# ---------------------------------------------------------------------------
# Keyword → category map  (deterministic; no LLM)
# ---------------------------------------------------------------------------
# Each entry: (compiled regex pattern, category_key)
# Patterns are tried in order; first match wins.
# All patterns are case-insensitive.

_CATEGORY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # hepatotoxicity / liver
    (re.compile(
        r"hepato|liver|hepat|alat|asat|alt\b|ast\b|bilirubin|jaundice|"
        r"cholestasis|cirrhosis|hepatitis|dili|drug.induced liver",
        re.IGNORECASE,
    ), "hepatotoxicity"),

    # cardiac
    (re.compile(
        r"cardiac|cardio|heart|myocardial|infarct|arrhythmia|tachycardia|"
        r"bradycardia|qt.prolong|atrial.fibrillation|ventricular|coronary|"
        r"angina|palpitation|ischaemia|ischemia",
        re.IGNORECASE,
    ), "cardiac"),

    # bleeding / haemorrhage
    (re.compile(
        r"bleed|haemorrhage|hemorrhage|hematoma|haematoma|coagulat|"
        r"thrombocytopenia|purpura|epistaxis|ecchymosis|petechiae|"
        r"gastrointestinal.bleed|gi.bleed",
        re.IGNORECASE,
    ), "bleeding"),

    # renal / kidney
    (re.compile(
        r"renal|kidney|nephro|creatinine|glomerular|proteinuria|"
        r"acute.kidney|ckd|aki\b|tubular|nephritis",
        re.IGNORECASE,
    ), "renal"),

    # neurological / CNS
    (re.compile(
        r"neurolog|neuro|seizure|convulsion|encephalopathy|stroke|"
        r"cognitive|dementia|parkinson|peripheral.neuropathy|neuropathy|"
        r"headache.severe|intracranial|cns\b|brain",
        re.IGNORECASE,
    ), "neurological"),

    # carcinogenicity / oncology
    (re.compile(
        r"carcinogen|carcino|tumour|tumor|neoplasm|malignancy|malignant|"
        r"lymphoma|leukemia|leukaemia|oncology|cancer|genotox",
        re.IGNORECASE,
    ), "carcinogenicity"),

    # reproductive / developmental toxicity
    (re.compile(
        r"reproduct|teratogen|embryo|foetal|fetal|fertility|congenital|"
        r"pregnancy|lactation|neonatal|developmental.tox",
        re.IGNORECASE,
    ), "reproductive"),

    # anaphylaxis / hypersensitivity
    (re.compile(
        r"anaphylax|hypersensitiv|allergic.reaction|angioedema|urticaria|"
        r"anaphylactic|anaphylactoid",
        re.IGNORECASE,
    ), "anaphylaxis"),

    # infection / immunosuppression
    (re.compile(
        r"infection|sepsis|septicaemia|septicemia|opportunistic|immunosuppress|"
        r"immunocompromis|pneumonia.opportunistic|tb\b|tuberculosis",
        re.IGNORECASE,
    ), "infection"),
]

_FALLBACK_CATEGORY = "general"


# ---------------------------------------------------------------------------
# Public functions — classification
# ---------------------------------------------------------------------------

def classify_signal_category(adverse_event: str) -> str:
    """Map an adverse event preferred term to a traceability category key.

    Uses deterministic keyword matching (no LLM).  The first pattern that
    matches the term wins.  Unrecognised terms resolve to ``"general"``.

    Parameters
    ----------
    adverse_event :
        The adverse event preferred term or free text (e.g. "hepatotoxicity",
        "elevated ALT", "haemorrhage", "nausea").

    Returns
    -------
    str
        One of the category keys present in the CTD knowledge base, or
        ``"general"`` if no pattern matches.

    Notes
    -----
    This function does NOT claim that the classification is correct for all
    uses.  It is a heuristic for identifying *potentially* relevant sections.
    """
    if not adverse_event:
        return _FALLBACK_CATEGORY

    term = str(adverse_event).strip()
    for pattern, category in _CATEGORY_PATTERNS:
        if pattern.search(term):
            logger.debug(
                "classify_signal_category: '%s' → '%s'", adverse_event, category
            )
            return category

    logger.debug(
        "classify_signal_category: '%s' → '%s' (fallback)", adverse_event, _FALLBACK_CATEGORY
    )
    return _FALLBACK_CATEGORY


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TraceabilityEntry:
    """Traceability result for one CTD section relative to one safety signal.

    Attributes
    ----------
    ctd_section_id : str
        Canonical KB section id, e.g. ``"2.7.4"``.
    ctd_section_title : str
        Human-readable section title from the KB.
    relevance_reason : str
        Deterministic, non-causal explanation of why this section is
        potentially relevant to the signal category.
    dossier_match : bool
        True if the dossier MatchSummary contains a PRESENT or NEEDS_REVIEW
        entry that resolves to this KB section.
    match_confidence : float
        Confidence score from the underlying SectionMatch (0.0 if absent).
    status : MatchStatus
        PRESENT / NEEDS_REVIEW / MISSING / NOT_APPLICABLE.
        Reuses the shared MatchStatus enum from section_matcher.
    priority : str
        ``"critical"`` — safety_relevant=True AND status is MISSING.
        ``"high"``     — safety_relevant=True (any status).
        ``"normal"``   — section is not safety_relevant.
    safety_relevant : bool
        Whether the KB section is flagged as safety-relevant.
    requirement : str
        The KB requirement level: ``"required"`` / ``"conditional"`` / ``"optional"``.
    """

    ctd_section_id: str
    ctd_section_title: str
    relevance_reason: str
    dossier_match: bool
    match_confidence: float
    status: MatchStatus
    priority: str
    safety_relevant: bool
    requirement: str

    def to_dict(self) -> dict:
        return {
            "ctd_section_id": self.ctd_section_id,
            "ctd_section_title": self.ctd_section_title,
            "relevance_reason": self.relevance_reason,
            "dossier_match": self.dossier_match,
            "match_confidence": round(self.match_confidence, 4),
            "status": self.status.value,
            "priority": self.priority,
            "safety_relevant": self.safety_relevant,
            "requirement": self.requirement,
        }


@dataclass
class TraceabilityResult:
    """Full traceability result for one safety signal.

    Attributes
    ----------
    drug_name : str
    adverse_event : str
    signal_category : str
        Category resolved by ``classify_signal_category()``.
    prr : Optional[float]
        PRR point estimate from the signal dict (passed through; not used in scoring).
    p_value : Optional[float]
        Statistical p-value from the signal dict (passed through; not used in scoring).
    signal_flag : bool
        Whether the PRR calculator flagged this as a potential signal.
    entries : list[TraceabilityEntry]
        One entry per relevant CTD section, ordered by section id.
    traceability_score : float
        Fraction of relevant sections with acceptable documentation.
        Formula: (PRESENT + NEEDS_REVIEW × 0.5) / N.  In [0.0, 1.0].
    critical_gaps : list[str]
        Section ids that are safety_relevant=True AND status=MISSING.
    disclaimer : str
        Hard-coded regulatory disclaimer.  Always equal to TRACEABILITY_DISCLAIMER.
    """

    drug_name: str
    adverse_event: str
    signal_category: str
    prr: Optional[float]
    p_value: Optional[float]
    signal_flag: bool
    entries: list[TraceabilityEntry] = field(default_factory=list)
    traceability_score: float = 0.0
    critical_gaps: list[str] = field(default_factory=list)
    disclaimer: str = field(default=TRACEABILITY_DISCLAIMER, init=False)

    def to_dict(self) -> dict:
        return {
            "drug_name": self.drug_name,
            "adverse_event": self.adverse_event,
            "signal_category": self.signal_category,
            "prr": self.prr,
            "p_value": self.p_value,
            "signal_flag": self.signal_flag,
            "traceability_score": round(self.traceability_score, 4),
            "critical_gaps": self.critical_gaps,
            "entries": [e.to_dict() for e in self.entries],
            "disclaimer": self.disclaimer,
        }


@dataclass
class TraceabilityReport:
    """Aggregate traceability report for a list of safety signals.

    Attributes
    ----------
    results : list[TraceabilityResult]
        One result per signal, sorted by traceability_score ascending
        (worst-documented signals first).
    overall_score : float
        Mean traceability_score across all results.  0.0 if no results.
    total_signals : int
    fully_documented : int
        Count of results with traceability_score == 1.0.
    critical_gap_count : int
        Count of results with at least one critical gap.
    disclaimer : str
        Hard-coded regulatory disclaimer.  Always equal to TRACEABILITY_DISCLAIMER.
    """

    results: list[TraceabilityResult] = field(default_factory=list)
    overall_score: float = 0.0
    total_signals: int = 0
    fully_documented: int = 0
    critical_gap_count: int = 0
    disclaimer: str = field(default=TRACEABILITY_DISCLAIMER, init=False)

    def to_dict(self) -> dict:
        return {
            "overall_score": round(self.overall_score, 4),
            "total_signals": self.total_signals,
            "fully_documented": self.fully_documented,
            "critical_gap_count": self.critical_gap_count,
            "results": [r.to_dict() for r in self.results],
            "disclaimer": self.disclaimer,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_covered_section_ids(match_summary: MatchSummary) -> dict[str, tuple[MatchStatus, float]]:
    """Return a mapping of KB section id → (status, confidence) from a MatchSummary.

    Only matches that resolved to a known KB section (expected_section is not None)
    are included.  For duplicate ids the highest-confidence match wins.

    Parameters
    ----------
    match_summary :
        Result of SectionMatcher.match_dossier().

    Returns
    -------
    dict mapping section id string → (MatchStatus, confidence float).
    """
    covered: dict[str, tuple[MatchStatus, float]] = {}
    for m in match_summary.matches:
        if m.expected_section is None:
            continue
        sec_id = m.expected_section.id
        existing = covered.get(sec_id)
        if existing is None or m.confidence > existing[1]:
            covered[sec_id] = (m.status, m.confidence)
    return covered


def _relevance_reason(section: CTDSection, category: str) -> str:
    """Build a deterministic, non-causal relevance explanation string.

    The returned text explicitly avoids causality or regulatory determination
    language.  It describes potential relevance only.
    """
    parts: list[str] = []

    # Category-specific phrasing
    cat_phrases: dict[str, str] = {
        "hepatotoxicity":  "hepatotoxicity / liver-related",
        "cardiac":         "cardiac / cardiovascular",
        "bleeding":        "bleeding / haemorrhage-related",
        "renal":           "renal / kidney-related",
        "neurological":    "neurological / CNS",
        "carcinogenicity": "carcinogenicity / oncology-related",
        "reproductive":    "reproductive / developmental toxicity",
        "anaphylaxis":     "hypersensitivity / anaphylaxis",
        "infection":       "infection / immunosuppression",
        "general":         "general safety",
    }
    cat_text = cat_phrases.get(category, category)
    parts.append(
        f"Potentially relevant to {cat_text} signals "
        f"based on the section's traceability_categories annotation"
    )

    if section.safety_relevant:
        parts.append("section is annotated as safety-relevant in the knowledge base")

    return "; ".join(parts) + "."


def _determine_priority(section: CTDSection, status: MatchStatus) -> str:
    """Return the priority level for a TraceabilityEntry.

    critical — safety_relevant=True AND MISSING
    high     — safety_relevant=True (regardless of status)
    normal   — not safety_relevant
    """
    if section.safety_relevant and status == MatchStatus.MISSING:
        return "critical"
    if section.safety_relevant:
        return "high"
    return "normal"


def _compute_traceability_score(entries: list[TraceabilityEntry]) -> float:
    """Compute the traceability score for a list of entries.

    Formula (mirrors readiness_scorer weighting):
        score = (PRESENT_count + NEEDS_REVIEW_count × TRACEABILITY_PARTIAL) / N

    NOT_APPLICABLE entries are excluded from N.
    Returns 1.0 when N == 0.
    """
    applicable = [e for e in entries if e.status != MatchStatus.NOT_APPLICABLE]
    n = len(applicable)
    if n == 0:
        return 1.0

    earned = sum(
        1.0 if e.status == MatchStatus.PRESENT
        else TRACEABILITY_PARTIAL if e.status == MatchStatus.NEEDS_REVIEW
        else 0.0
        for e in applicable
    )
    return earned / n


# ---------------------------------------------------------------------------
# Public API — trace_signal
# ---------------------------------------------------------------------------

def trace_signal(
    signal: dict,
    match_summary: MatchSummary,
    kb: Optional[CTDKnowledgeBase] = None,
) -> TraceabilityResult:
    """Trace one safety signal to potentially relevant CTD sections.

    Parameters
    ----------
    signal :
        Dict with at minimum ``"drug_name"`` and ``"adverse_event"`` keys.
        Optional keys: ``"prr"`` (float), ``"p_value"`` (float),
        ``"signal_flag"`` (bool).
    match_summary :
        Result of ``SectionMatcher.match_dossier()`` on the dossier outline.
        Used to determine which CTD sections are already present in the dossier.
    kb :
        CTD knowledge base.  If None, the singleton from ``load_knowledge_base()``
        is used.

    Returns
    -------
    TraceabilityResult

    Notes
    -----
    - This function does NOT establish causality.
    - It does NOT determine whether the submission will be accepted.
    - Relevant sections are determined solely by the ``traceability_categories``
      annotation in the knowledge base JSON.
    """
    if kb is None:
        kb = load_knowledge_base()

    drug_name = str(signal.get("drug_name", "")).strip()
    adverse_event = str(signal.get("adverse_event", "")).strip()
    prr: Optional[float] = signal.get("prr")
    p_value: Optional[float] = signal.get("p_value")
    signal_flag: bool = bool(signal.get("signal_flag", False))

    # 1. Classify the adverse event to a traceability category
    category = classify_signal_category(adverse_event)

    # 2. Fetch relevant CTD sections from the KB
    relevant_sections: list[CTDSection] = kb.get_sections_by_traceability_category(category)

    # If "general" fallback returned nothing (shouldn't happen with a well-formed KB),
    # use all safety-relevant sections as a conservative fallback.
    if not relevant_sections:
        relevant_sections = kb.get_safety_sections()
        logger.warning(
            "trace_signal: no sections found for category '%s'; "
            "falling back to all safety-relevant sections (%d sections)",
            category, len(relevant_sections),
        )

    # 3. Build covered-section lookup from the MatchSummary
    covered = _build_covered_section_ids(match_summary)

    # 4. Build TraceabilityEntry for each relevant section
    entries: list[TraceabilityEntry] = []
    for section in sorted(relevant_sections, key=lambda s: s.id):
        dossier_info = covered.get(section.id)

        if dossier_info is not None:
            status, confidence = dossier_info
            dossier_match = status in (MatchStatus.PRESENT, MatchStatus.NEEDS_REVIEW)
        else:
            # Section not found in the dossier at all
            status = MatchStatus.MISSING
            confidence = 0.0
            dossier_match = False

        priority = _determine_priority(section, status)
        reason = _relevance_reason(section, category)

        entries.append(TraceabilityEntry(
            ctd_section_id=section.id,
            ctd_section_title=section.title,
            relevance_reason=reason,
            dossier_match=dossier_match,
            match_confidence=confidence,
            status=status,
            priority=priority,
            safety_relevant=section.safety_relevant,
            requirement=section.requirement,
        ))

    # 5. Score and critical gaps
    score = _compute_traceability_score(entries)
    critical_gaps = [
        e.ctd_section_id for e in entries if e.priority == "critical"
    ]

    result = TraceabilityResult(
        drug_name=drug_name,
        adverse_event=adverse_event,
        signal_category=category,
        prr=prr,
        p_value=p_value,
        signal_flag=signal_flag,
        entries=entries,
        traceability_score=score,
        critical_gaps=critical_gaps,
    )

    logger.info(
        "trace_signal: drug='%s' event='%s' category='%s' "
        "relevant_sections=%d score=%.3f critical_gaps=%d",
        drug_name, adverse_event, category,
        len(entries), score, len(critical_gaps),
    )

    return result


# ---------------------------------------------------------------------------
# Public API — trace_all_signals
# ---------------------------------------------------------------------------

def trace_all_signals(
    signals: list[dict],
    match_summary: MatchSummary,
    kb: Optional[CTDKnowledgeBase] = None,
) -> TraceabilityReport:
    """Trace all signals and return an aggregate report.

    Parameters
    ----------
    signals :
        List of signal dicts.  Each must have ``"drug_name"`` and
        ``"adverse_event"`` keys.  See ``trace_signal()`` for full spec.
    match_summary :
        Dossier match summary from SectionMatcher.
    kb :
        CTD knowledge base singleton.  If None, the default is used.

    Returns
    -------
    TraceabilityReport
        results are sorted by traceability_score ascending (worst first).

    Notes
    -----
    An empty ``signals`` list returns an empty TraceabilityReport cleanly.
    """
    if kb is None:
        kb = load_knowledge_base()

    if not signals:
        return TraceabilityReport(
            results=[],
            overall_score=1.0,
            total_signals=0,
            fully_documented=0,
            critical_gap_count=0,
        )

    results: list[TraceabilityResult] = []
    for sig in signals:
        results.append(trace_signal(sig, match_summary, kb=kb))

    # Sort worst-documented first so reviewers see the biggest gaps at the top
    results.sort(key=lambda r: r.traceability_score)

    total = len(results)
    overall = sum(r.traceability_score for r in results) / total
    fully_documented = sum(1 for r in results if r.traceability_score >= 1.0)
    critical_gap_count = sum(1 for r in results if r.critical_gaps)

    report = TraceabilityReport(
        results=results,
        overall_score=overall,
        total_signals=total,
        fully_documented=fully_documented,
        critical_gap_count=critical_gap_count,
    )

    logger.info(
        "trace_all_signals: %d signals → overall_score=%.3f "
        "fully_documented=%d critical_gap_count=%d",
        total, overall, fully_documented, critical_gap_count,
    )

    return report
