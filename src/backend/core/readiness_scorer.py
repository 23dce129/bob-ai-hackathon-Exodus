"""
readiness_scorer.py — CTD Submission Readiness Scorer.

⚠️  DISCLAIMER ⚠️
This is a heuristic screening score for internal prioritisation and gap
identification only.
It is NOT an FDA approval readiness score.
It is NOT a regulatory compliance assessment.
It does NOT represent the position of the FDA, ICH, EMA, or any regulatory agency.
It does NOT indicate whether a submission will be accepted or approved.
All results require review by a qualified regulatory affairs professional.

Purpose
-------
Score the completeness of a CTD dossier outline against the ICH M4 knowledge
base by module and overall, so a reviewer can quickly identify which modules
have the largest gaps.

The score is entirely deterministic and arithmetical.  No language model is
used or needed.

Inputs
------
A ``MatchSummary`` produced by ``section_matcher.match_dossier()``.
Each ``SectionMatch`` carries the KB section (with its ``requirement`` and
``safety_relevant`` fields) and the resolved ``MatchStatus``.

Section weights
---------------
Each section in the KB contributes a weight to its module's possible score:

    base_weight = 1.0  if section.requirement == "required"
                  0.5  if section.requirement == "conditional"

    safety_multiplier = 1.5  if section.safety_relevant
                        1.0  otherwise

    section_weight = base_weight × safety_multiplier

Earned weight per status
------------------------
    PRESENT        → full section_weight
    NEEDS_REVIEW   → section_weight × NEEDS_REVIEW_PARTIAL (default 0.5)
    MISSING        → 0
    NOT_APPLICABLE → excluded from both numerator AND denominator

Critical-missing penalty
------------------------
A section is "critical" when:  requirement == "required"  AND  safety_relevant == True.

Each critical section that is MISSING subtracts CRITICAL_PENALTY (default 0.10)
from the raw module ratio before clamping to [0.0, 1.0].  This ensures that a
dossier which passes every non-safety section but drops a required safety section
cannot score above ~0.90 on that module.

Module score formula
--------------------
    possible = sum of section_weight for all non-NOT_APPLICABLE sections in module
    earned   = sum of earned_weight for all sections

    If possible == 0:
        raw_ratio = 1.0  (vacuously complete — no applicable sections)
    else:
        raw_ratio = earned / possible

    critical_missing = count of MISSING sections where requirement=="required"
                       AND safety_relevant==True, within this module

    module_score = clamp(raw_ratio − critical_missing × CRITICAL_PENALTY, 0.0, 1.0)

Overall score
-------------
Weighted average of the five module scores.  Module weights reflect the
regulatory importance of each module for a new drug application:

    M1 (Regional Admin)         0.10
    M2 (CTD Summaries)          0.30  ← most likely to have safety gaps
    M3 (Quality)                0.20
    M4 (Nonclinical)            0.20
    M5 (Clinical)               0.20

    overall_score = sum(module_weight × module_score) / sum(module_weight)

If a module has no applicable sections it is excluded from the weighted average.

Public API
----------
    ReadinessConfig        — configurable scoring parameters
    ModuleScore            — score result for one module
    ReadinessResult        — full result: module scores + overall score + stats
    READINESS_DISCLAIMER   — module-level constant (same text as the docstring warning)
    score_readiness()      — main entry point: MatchSummary → ReadinessResult
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from core.ctd_knowledge_base import CTDSection
from core.section_matcher import MatchStatus, MatchSummary, SectionMatch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Disclaimer — embedded in every ReadinessResult; cannot be omitted
# ---------------------------------------------------------------------------

READINESS_DISCLAIMER = (
    "HEURISTIC READINESS SCORE ONLY — NOT an FDA approval readiness score. "
    "NOT a regulatory compliance assessment. "
    "Does NOT represent the position of the FDA, ICH, EMA, or any regulatory agency. "
    "Does NOT indicate whether a submission will be accepted or approved. "
    "All results require review by a qualified regulatory affairs professional."
)

# ---------------------------------------------------------------------------
# Scoring constants — module-level so tests can inspect them
# ---------------------------------------------------------------------------

# Partial credit for NEEDS_REVIEW sections
NEEDS_REVIEW_PARTIAL: float = 0.5

# Additional penalty per critical-missing section (required + safety_relevant)
CRITICAL_PENALTY: float = 0.10

# Default module weights for overall score
DEFAULT_MODULE_WEIGHTS: dict[str, float] = {
    "M1": 0.10,
    "M2": 0.30,
    "M3": 0.20,
    "M4": 0.20,
    "M5": 0.20,
}

# Base section weights by requirement level
_BASE_WEIGHT: dict[str, float] = {
    "required":    1.0,
    "conditional": 0.5,
    "optional":    0.25,
}

# Safety multiplier
_SAFETY_MULTIPLIER: float = 1.5


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ReadinessConfig:
    """Configurable parameters for the readiness scorer.

    Attributes
    ----------
    needs_review_partial : float
        Fraction of full weight earned by a NEEDS_REVIEW section.
        Must be in [0.0, 1.0].  Default 0.5.
    critical_penalty : float
        Score deduction per critical-missing section (required + safety_relevant).
        Applied per module before clamping.  Default 0.10.
    module_weights : dict[str, float]
        Map of module id ("M1"–"M5") to weight used in the overall average.
        Weights need not sum to 1; they are normalised internally.
        Default: M1=0.10, M2=0.30, M3=0.20, M4=0.20, M5=0.20.
    """
    needs_review_partial: float = NEEDS_REVIEW_PARTIAL
    critical_penalty: float = CRITICAL_PENALTY
    module_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_MODULE_WEIGHTS)
    )

    def __post_init__(self) -> None:
        if not 0.0 <= self.needs_review_partial <= 1.0:
            raise ValueError(
                f"needs_review_partial must be in [0.0, 1.0], got {self.needs_review_partial}"
            )
        if self.critical_penalty < 0.0:
            raise ValueError(
                f"critical_penalty must be ≥ 0, got {self.critical_penalty}"
            )


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModuleScore:
    """Readiness score for one CTD module.

    Attributes
    ----------
    module_id : str
        "M1" … "M5".
    module_title : str
        Human-readable title from the KB (e.g. "Common Technical Document Summaries").
    score : float
        Completeness score ∈ [0.0, 1.0].  1.0 = fully complete.
        Incorporates critical-missing penalties.
    raw_ratio : float
        Earned weight / possible weight, before penalties.  Useful for debugging.
    possible_weight : float
        Sum of section weights for all sections that apply (excluding NOT_APPLICABLE).
    earned_weight : float
        Sum of earned weights (PRESENT=full, NEEDS_REVIEW=partial, MISSING=0).
    present_count : int
        Sections with status PRESENT.
    needs_review_count : int
        Sections with status NEEDS_REVIEW.
    missing_count : int
        Sections with status MISSING.
    not_applicable_count : int
        Sections excluded from scoring.
    critical_missing_count : int
        MISSING sections that are both required AND safety_relevant.
    penalty_applied : float
        Total penalty deducted from raw_ratio.
    section_details : list[dict]
        Per-section detail dicts for audit/display purposes.
    """
    module_id: str
    module_title: str
    score: float
    raw_ratio: float
    possible_weight: float
    earned_weight: float
    present_count: int
    needs_review_count: int
    missing_count: int
    not_applicable_count: int
    critical_missing_count: int
    penalty_applied: float
    section_details: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "module_id": self.module_id,
            "module_title": self.module_title,
            "score": round(self.score, 4),
            "score_pct": round(self.score * 100, 1),
            "raw_ratio": round(self.raw_ratio, 4),
            "possible_weight": round(self.possible_weight, 4),
            "earned_weight": round(self.earned_weight, 4),
            "present_count": self.present_count,
            "needs_review_count": self.needs_review_count,
            "missing_count": self.missing_count,
            "not_applicable_count": self.not_applicable_count,
            "critical_missing_count": self.critical_missing_count,
            "penalty_applied": round(self.penalty_applied, 4),
        }


@dataclass
class ReadinessResult:
    """Full readiness scoring result for one dossier.

    Attributes
    ----------
    overall_score : float
        Weighted average of module scores ∈ [0.0, 1.0].
    overall_score_pct : float
        overall_score × 100 (convenience).
    module_scores : dict[str, ModuleScore]
        One entry per module that had at least one applicable section.
    total_present : int
    total_needs_review : int
    total_missing : int
    total_not_applicable : int
    total_critical_missing : int
    config : ReadinessConfig
        The configuration used for this scoring run.
    disclaimer : str
        The READINESS_DISCLAIMER constant.  Set automatically; cannot be omitted.
    """
    overall_score: float
    overall_score_pct: float
    module_scores: dict[str, ModuleScore]
    total_present: int
    total_needs_review: int
    total_missing: int
    total_not_applicable: int
    total_critical_missing: int
    config: ReadinessConfig
    disclaimer: str = field(default=READINESS_DISCLAIMER, init=False)

    def to_dict(self) -> dict:
        return {
            "disclaimer": self.disclaimer,
            "overall_score": round(self.overall_score, 4),
            "overall_score_pct": round(self.overall_score_pct, 1),
            "total_present": self.total_present,
            "total_needs_review": self.total_needs_review,
            "total_missing": self.total_missing,
            "total_not_applicable": self.total_not_applicable,
            "total_critical_missing": self.total_critical_missing,
            "module_scores": {
                mid: ms.to_dict() for mid, ms in self.module_scores.items()
            },
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _section_weight(section: CTDSection) -> float:
    """Return the weight of a section based on requirement and safety relevance."""
    base = _BASE_WEIGHT.get(section.requirement, 0.25)
    multiplier = _SAFETY_MULTIPLIER if section.safety_relevant else 1.0
    return base * multiplier


def _earned_weight(
    section_weight: float,
    status: MatchStatus,
    partial: float,
) -> float:
    """Return the earned weight for a section given its status."""
    if status == MatchStatus.PRESENT:
        return section_weight
    if status == MatchStatus.NEEDS_REVIEW:
        return section_weight * partial
    # MISSING or NOT_APPLICABLE (caller excludes NOT_APPLICABLE from possible)
    return 0.0


def _module_id_for_match(match: SectionMatch) -> Optional[str]:
    """Derive the module id for a SectionMatch.

    Returns the module id string ("M1"–"M5") or None for UNMATCHED rows with
    no expected_section.  For rows with a KB section we use its parent chain;
    for UNMATCHED rows we try to parse the dossier section_number prefix.
    """
    if match.expected_section is not None:
        # Walk up to the module level via the id prefix
        sec_id = match.expected_section.id
        prefix = sec_id[0]   # "1"–"5"
        if prefix.isdigit():
            return f"M{prefix}"
        return None

    # UNMATCHED — try to guess from dossier section_number
    num = match.dossier_row.section_number
    if num and num[0].isdigit():
        return f"M{num[0]}"
    return None


# ---------------------------------------------------------------------------
# Core scoring logic
# ---------------------------------------------------------------------------

def _score_module(
    module_id: str,
    module_title: str,
    matches: list[SectionMatch],
    config: ReadinessConfig,
) -> ModuleScore:
    """Compute ``ModuleScore`` for a single module from its matched sections."""
    possible_weight = 0.0
    earned_weight_total = 0.0
    present_count = 0
    needs_review_count = 0
    missing_count = 0
    not_applicable_count = 0
    critical_missing_count = 0
    section_details: list[dict] = []

    for match in matches:
        sec = match.expected_section
        status = match.status

        if status == MatchStatus.NOT_APPLICABLE:
            not_applicable_count += 1
            section_details.append({
                "section_id": sec.id if sec else match.dossier_row.section_number,
                "section_title": sec.title if sec else match.dossier_row.section_title,
                "status": "NOT_APPLICABLE",
                "weight": 0.0,
                "earned": 0.0,
                "is_critical": False,
                "match_reason": match.match_reason,
            })
            continue

        if sec is None:
            # UNMATCHED row — treat as weight=1.0 required section, MISSING
            w = 1.0
            possible_weight += w
            missing_count += 1
            section_details.append({
                "section_id": match.dossier_row.section_number,
                "section_title": match.dossier_row.section_title,
                "status": "MISSING",
                "weight": w,
                "earned": 0.0,
                "is_critical": False,
                "match_reason": match.match_reason,
            })
            continue

        w = _section_weight(sec)
        earned = _earned_weight(w, status, config.needs_review_partial)
        is_critical = (sec.requirement == "required" and sec.safety_relevant
                       and status == MatchStatus.MISSING)

        possible_weight += w
        earned_weight_total += earned

        if status == MatchStatus.PRESENT:
            present_count += 1
        elif status == MatchStatus.NEEDS_REVIEW:
            needs_review_count += 1
        elif status == MatchStatus.MISSING:
            missing_count += 1
            if is_critical:
                critical_missing_count += 1

        section_details.append({
            "section_id": sec.id,
            "section_title": sec.title,
            "status": status.value,
            "requirement": sec.requirement,
            "safety_relevant": sec.safety_relevant,
            "weight": round(w, 4),
            "earned": round(earned, 4),
            "is_critical": is_critical,
            "match_reason": match.match_reason,
        })

    # Compute raw ratio
    if possible_weight == 0.0:
        raw_ratio = 1.0   # vacuously complete
    else:
        raw_ratio = earned_weight_total / possible_weight

    # Apply critical-missing penalty
    penalty = critical_missing_count * config.critical_penalty
    score = max(0.0, min(1.0, raw_ratio - penalty))

    return ModuleScore(
        module_id=module_id,
        module_title=module_title,
        score=score,
        raw_ratio=raw_ratio,
        possible_weight=possible_weight,
        earned_weight=earned_weight_total,
        present_count=present_count,
        needs_review_count=needs_review_count,
        missing_count=missing_count,
        not_applicable_count=not_applicable_count,
        critical_missing_count=critical_missing_count,
        penalty_applied=penalty,
        section_details=section_details,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def score_readiness(
    summary: MatchSummary,
    config: Optional[ReadinessConfig] = None,
    module_titles: Optional[dict[str, str]] = None,
) -> ReadinessResult:
    """Score the completeness of a dossier from a ``MatchSummary``.

    Parameters
    ----------
    summary : MatchSummary
        Output of ``SectionMatcher.match_dossier()``.
    config : ReadinessConfig, optional
        Scoring parameters.  Defaults to ``ReadinessConfig()`` (stock weights).
    module_titles : dict[str, str], optional
        Map of module id → title string for display.  Populated from the KB
        if not supplied.  Pass an override for tests that use a synthetic KB.

    Returns
    -------
    ReadinessResult
        Module scores, overall score, counts, and the immutable disclaimer.

    Notes
    -----
    The returned ``ReadinessResult.disclaimer`` is always equal to
    ``READINESS_DISCLAIMER`` and cannot be overridden.
    """
    if config is None:
        config = ReadinessConfig()

    # Default module titles
    _module_titles: dict[str, str] = {
        "M1": "Regional Administrative Information",
        "M2": "Common Technical Document Summaries",
        "M3": "Quality",
        "M4": "Nonclinical Study Reports",
        "M5": "Clinical Study Reports",
    }
    if module_titles:
        _module_titles.update(module_titles)

    # Group matches by module
    by_module: dict[str, list[SectionMatch]] = defaultdict(list)
    for match in summary.matches:
        mid = _module_id_for_match(match)
        if mid is not None:
            by_module[mid].append(match)
        else:
            logger.warning(
                "Row %d: could not determine module; skipping in score.",
                match.dossier_row.row_index,
            )

    # Score each module
    module_scores: dict[str, ModuleScore] = {}
    for mid in sorted(by_module.keys()):
        ms = _score_module(
            module_id=mid,
            module_title=_module_titles.get(mid, mid),
            matches=by_module[mid],
            config=config,
        )
        module_scores[mid] = ms
        logger.debug(
            "Module %s: score=%.3f raw_ratio=%.3f penalty=%.3f "
            "present=%d review=%d missing=%d critical_missing=%d",
            mid, ms.score, ms.raw_ratio, ms.penalty_applied,
            ms.present_count, ms.needs_review_count,
            ms.missing_count, ms.critical_missing_count,
        )

    # Overall weighted score
    total_weight = 0.0
    weighted_sum = 0.0
    for mid, ms in module_scores.items():
        if ms.possible_weight > 0.0:   # skip vacuously-complete empty modules
            mw = config.module_weights.get(mid, 0.0)
            weighted_sum += mw * ms.score
            total_weight += mw

    overall = weighted_sum / total_weight if total_weight > 0.0 else 0.0
    overall = max(0.0, min(1.0, overall))

    # Aggregate counts
    total_present = sum(ms.present_count for ms in module_scores.values())
    total_needs_review = sum(ms.needs_review_count for ms in module_scores.values())
    total_missing = sum(ms.missing_count for ms in module_scores.values())
    total_not_applicable = sum(ms.not_applicable_count for ms in module_scores.values())
    total_critical = sum(ms.critical_missing_count for ms in module_scores.values())

    logger.info(
        "score_readiness: overall=%.3f  present=%d  review=%d  missing=%d  "
        "not_applicable=%d  critical_missing=%d",
        overall, total_present, total_needs_review,
        total_missing, total_not_applicable, total_critical,
    )

    return ReadinessResult(
        overall_score=overall,
        overall_score_pct=overall * 100,
        module_scores=module_scores,
        total_present=total_present,
        total_needs_review=total_needs_review,
        total_missing=total_missing,
        total_not_applicable=total_not_applicable,
        total_critical_missing=total_critical,
        config=config,
    )
