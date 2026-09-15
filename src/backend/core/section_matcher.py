"""
section_matcher.py — CTD section matching engine.

Accepts a parsed dossier outline (list of ``DossierRow``) and the CTD knowledge
base, then attempts to match each submitted row to a canonical ``CTDSection``.

No LLM is used at any point.  All matching is deterministic, explainable, and
reproducible given the same inputs.  Every ``SectionMatch`` carries a non-empty
``match_reason`` that explains precisely why a candidate was chosen or why none
was found.

Matching strategy
-----------------
Three strategies are applied in priority order.  The first that produces a
confident result wins; no subsequent strategy is evaluated for that row.

1. EXACT_NUMBER
   The dossier row's ``section_number`` (after normalisation) equals a KB
   section id exactly.  Confidence = 1.0.
   Example: dossier "2.7.4" → KB "2.7.4".

2. TITLE_NORMALISED
   The dossier row's ``section_title``, after removing stopwords and collapsing
   whitespace, matches a KB section title under the same transformation.
   Confidence = 1.0 for an exact normalised match; 0.9 for a known title alias.
   Example: "Clinical Safety Summary" → normalises to same tokens as
            "Summary of Clinical Safety".

3. SEMANTIC
   TF-IDF cosine similarity between the dossier title and every KB title.
   The KB vocabulary is built once at matcher construction from all section
   titles.  A match is accepted only when:
     • cosine score ≥ SEMANTIC_THRESHOLD (0.60)
     • the top candidate score exceeds the second-best by ≥ SEMANTIC_GAP (0.10)
   If both criteria hold → MatchType.SEMANTIC, confidence = cosine score.
   If cosine ≥ threshold but gap criterion fails → MatchType.AMBIGUOUS,
   confidence = cosine score of best candidate; both candidates listed in reason.

   No match at all → MatchType.UNMATCHED.

Status assignment
-----------------
Status is determined in strict priority order:

  1. If the dossier row supplies a non-empty ``status`` field, it is normalised
     and mapped to one of the four canonical values.  This is the ONLY source of
     truth — the matcher never silently overrides it.
  2. If no status was supplied:
     • UNMATCHED rows → MISSING
     • AMBIGUOUS rows → NEEDS_REVIEW (cannot be certain without human review)
     • Matched rows → PRESENT

Status vocabulary mapping (case-insensitive, tolerant of common variants)
--------------------------------------------------------------------------
  "present", "yes", "submitted", "complete", "completed", "done"   → PRESENT
  "missing", "absent", "not submitted", "no"                        → MISSING
  "needs_review", "needs review", "review", "draft", "incomplete",
    "pending", "partial", "needs_revision"                          → NEEDS_REVIEW
  "not_applicable", "n/a", "na", "not applicable", "waived",
    "inapplicable"                                                   → NOT_APPLICABLE
  Anything else → NEEDS_REVIEW (conservative: ambiguous text triggers review)

Public API
----------
  MatchType       — enum: EXACT_NUMBER | TITLE_NORMALISED | SEMANTIC | AMBIGUOUS | UNMATCHED
  MatchStatus     — enum: PRESENT | NEEDS_REVIEW | MISSING | NOT_APPLICABLE
  SectionMatch    — dataclass: one match result per dossier row
  MatchSummary    — dataclass: all matches + aggregate stats for one dossier
  SectionMatcher  — main class; call match_dossier() or match_row()
  build_matcher() — factory that loads the real KB and returns a SectionMatcher
"""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from core.ctd_knowledge_base import CTDKnowledgeBase, CTDSection, load_knowledge_base
from core.dossier_parser import DossierRow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning constants (module-level so tests can inspect them)
# ---------------------------------------------------------------------------

SEMANTIC_THRESHOLD: float = 0.60   # minimum cosine to accept a semantic match
SEMANTIC_GAP: float = 0.10         # top-1 must exceed top-2 by at least this much

# ---------------------------------------------------------------------------
# Stopwords stripped before normalised-title comparison
# ---------------------------------------------------------------------------
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "of", "and", "or", "for", "in", "on", "at", "to",
    "with", "by", "from", "as", "is", "are", "be", "been", "its",
    "this", "that", "all", "each", "per",
})

# ---------------------------------------------------------------------------
# Known title aliases  (normalised_alias → canonical_kb_section_id)
# These are short-form or variant titles commonly used in industry.
# ---------------------------------------------------------------------------
_TITLE_ALIASES: dict[str, str] = {
    # Module 2 overviews
    "nonclinical overview":              "2.4",
    "non-clinical overview":             "2.4",
    "clinical overview":                 "2.5",
    "quality overall summary":           "2.3",
    "qos":                               "2.3",
    "clinical summary":                  "2.7",
    "nonclinical summary":               "2.6",

    # Safety summary variants
    "clinical safety summary":           "2.7.4",
    "summary clinical safety":           "2.7.4",
    "integrated safety summary":         "2.7.4",
    "iss":                               "2.7.4",
    "adverse event summary":             "2.7.4",
    "clinical efficacy summary":         "2.7.3",
    "summary clinical efficacy":         "2.7.3",
    "integrated efficacy summary":       "2.7.3",

    # Module 1 aliases
    "pharmacovigilance plan":            "1.8",
    "risk management plan":              "1.8",
    "rmp":                               "1.8",
    "rems":                              "1.8",
    "patient listings":                  "1.9",
    "crfs":                              "1.9",
    "case report forms":                 "1.9",

    # Nonclinical sub-sections
    "pk written summary":                "2.6.4",
    "pk tabulated summary":              "2.6.5",
    "tox written summary":               "2.6.6",
    "toxicology written summary":        "2.6.6",
    "tox tabulated summary":             "2.6.7",
    "pharmacology written summary":      "2.6.2",
    "pharmacology tabulated summary":    "2.6.3",

    # Module 4 tox sub-sections
    "single dose toxicity":              "4.2.3.1",
    "repeat dose toxicity":              "4.2.3.2",
    "repeated dose toxicity":            "4.2.3.2",
    "genotoxicity":                      "4.2.3.3",
    "carcinogenicity":                   "4.2.3.4",
    "reproductive toxicity":             "4.2.3.5",
    "developmental toxicity":            "4.2.3.5",
    "repro tox":                         "4.2.3.5",
    "local tolerance":                   "4.2.3.6",
    "other toxicity":                    "4.2.3.7",

    # Module 5 clinical study reports
    "biopharmaceutic studies":           "5.3.1",
    "human pk studies":                  "5.3.3",
    "human pharmacokinetic studies":     "5.3.3",
    "human pd studies":                  "5.3.4",
    "efficacy safety studies":           "5.3.5",
    "post-marketing experience":         "5.3.6",
    "post marketing experience":         "5.3.6",

    # Module 3 drug substance/product
    "drug substance":                    "3.2.s",
    "drug product":                      "3.2.p",
    "stability drug substance":          "3.2.s.7",
    "stability drug product":            "3.2.p.8",
}


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MatchType(str, Enum):
    """How the dossier row was matched to a KB section."""
    EXACT_NUMBER     = "EXACT_NUMBER"     # section_number == KB id
    TITLE_NORMALISED = "TITLE_NORMALISED" # normalised title match or alias
    SEMANTIC         = "SEMANTIC"         # TF-IDF cosine above threshold + gap
    AMBIGUOUS        = "AMBIGUOUS"        # cosine above threshold but gap too small
    UNMATCHED        = "UNMATCHED"        # nothing above threshold


class MatchStatus(str, Enum):
    """Submission readiness status for one CTD section."""
    PRESENT         = "PRESENT"
    NEEDS_REVIEW    = "NEEDS_REVIEW"
    MISSING         = "MISSING"
    NOT_APPLICABLE  = "NOT_APPLICABLE"


# ---------------------------------------------------------------------------
# Status vocabulary map
# ---------------------------------------------------------------------------

def _map_status_string(raw: str) -> MatchStatus:
    """Map a free-text status string to a canonical ``MatchStatus``.

    Conservative: anything unrecognised becomes NEEDS_REVIEW so it is never
    silently marked PRESENT.
    """
    key = raw.strip().lower().replace("-", "_").replace(" ", "_")
    if key in ("present", "yes", "submitted", "complete", "completed", "done", "provided"):
        return MatchStatus.PRESENT
    if key in ("missing", "absent", "not_submitted", "no", "not_provided"):
        return MatchStatus.MISSING
    if key in ("needs_review", "review", "draft", "incomplete", "pending",
               "partial", "needs_revision", "flag", "flagged"):
        return MatchStatus.NEEDS_REVIEW
    if key in ("not_applicable", "n/a", "na", "not_applicable", "waived",
               "inapplicable", "not_required"):
        return MatchStatus.NOT_APPLICABLE
    # Unknown → conservative
    return MatchStatus.NEEDS_REVIEW


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SectionMatch:
    """Match result for one dossier row against the CTD knowledge base.

    Attributes
    ----------
    dossier_row : DossierRow
        The parsed row this result came from.
    expected_section : Optional[CTDSection]
        The KB section we believe this row corresponds to.  None only for
        UNMATCHED results where no candidate exceeded the threshold.
    match_type : MatchType
        How the match was established.
    confidence : float
        A value in [0.0, 1.0].
          1.0 — exact number or exact normalised title match
          0.9 — alias match
          [SEMANTIC_THRESHOLD, 1.0) — TF-IDF cosine score
          0.0 — UNMATCHED
    status : MatchStatus
        Final determined status (PRESENT / NEEDS_REVIEW / MISSING / NOT_APPLICABLE).
    match_reason : str
        Non-empty human-readable explanation.  Every code path that produces a
        SectionMatch must set this field.  It is the primary audit trail.
    status_source : str
        Either "dossier_supplied" (dossier row had a status field) or
        "inferred" (matcher inferred from match result).
    alternative_candidates : list[str]
        For AMBIGUOUS only: section ids of other plausible candidates.
    """
    dossier_row: DossierRow
    expected_section: Optional[CTDSection]
    match_type: MatchType
    confidence: float
    status: MatchStatus
    match_reason: str
    status_source: str            # "dossier_supplied" | "inferred"
    alternative_candidates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "row_index": self.dossier_row.row_index,
            "dossier_section_number": self.dossier_row.section_number,
            "dossier_section_title": self.dossier_row.section_title,
            "dossier_status_raw": self.dossier_row.status,
            "matched_section_id": self.expected_section.id if self.expected_section else None,
            "matched_section_title": self.expected_section.title if self.expected_section else None,
            "match_type": self.match_type.value,
            "confidence": round(self.confidence, 4),
            "status": self.status.value,
            "status_source": self.status_source,
            "match_reason": self.match_reason,
            "alternative_candidates": self.alternative_candidates,
        }


@dataclass
class MatchSummary:
    """Aggregate results for one complete dossier match operation.

    Attributes
    ----------
    matches : list[SectionMatch]
        One entry per dossier row.
    total_rows : int
    exact_number_count : int
    title_normalised_count : int
    semantic_count : int
    ambiguous_count : int
    unmatched_count : int
    status_counts : dict[str, int]
        Keys are MatchStatus values; values are row counts.
    """
    matches: list[SectionMatch] = field(default_factory=list)
    total_rows: int = 0
    exact_number_count: int = 0
    title_normalised_count: int = 0
    semantic_count: int = 0
    ambiguous_count: int = 0
    unmatched_count: int = 0
    status_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total_rows": self.total_rows,
            "exact_number_count": self.exact_number_count,
            "title_normalised_count": self.title_normalised_count,
            "semantic_count": self.semantic_count,
            "ambiguous_count": self.ambiguous_count,
            "unmatched_count": self.unmatched_count,
            "status_counts": self.status_counts,
        }


# ---------------------------------------------------------------------------
# TF-IDF helpers  (no external ML library — pure Python)
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> list[str]:
    """Lowercase, remove punctuation, split on whitespace, drop stopwords."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t and t not in _STOPWORDS]
    return tokens


def _normalise_title(title: str) -> str:
    """Return a canonical form of a title for exact normalised comparison."""
    return " ".join(_tokenise(title))


def _build_tfidf_index(
    sections: list[CTDSection],
) -> tuple[dict[str, dict[str, float]], list[CTDSection]]:
    """Build a TF-IDF vector index over KB section titles.

    Returns
    -------
    (tfidf_vectors, ordered_sections)
        tfidf_vectors  : dict section_id → dict token → tfidf_weight
        ordered_sections : list of CTDSection in the same order as the index
    """
    # Step 1 — collect token sets per document
    doc_tokens: list[list[str]] = []
    for sec in sections:
        doc_tokens.append(_tokenise(sec.title))

    # Step 2 — IDF
    n_docs = len(doc_tokens)
    df: dict[str, int] = defaultdict(int)
    for tokens in doc_tokens:
        for tok in set(tokens):
            df[tok] += 1

    idf: dict[str, float] = {
        tok: math.log((n_docs + 1) / (count + 1)) + 1.0
        for tok, count in df.items()
    }

    # Step 3 — TF-IDF vectors (normalised L2)
    vectors: dict[str, dict[str, float]] = {}
    for sec, tokens in zip(sections, doc_tokens):
        tf: dict[str, float] = defaultdict(float)
        for tok in tokens:
            tf[tok] += 1.0
        # raw TF-IDF
        raw: dict[str, float] = {
            tok: (1.0 + math.log(tf[tok])) * idf.get(tok, 1.0)
            for tok in tf
        }
        # L2 normalise
        norm = math.sqrt(sum(v * v for v in raw.values())) or 1.0
        vectors[sec.id] = {tok: v / norm for tok, v in raw.items()}

    return vectors, sections


def _cosine(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Dot product of two L2-normalised TF-IDF vectors."""
    dot = 0.0
    for tok, weight in vec_a.items():
        if tok in vec_b:
            dot += weight * vec_b[tok]
    return dot


def _query_vector(text: str, idf: dict[str, float]) -> dict[str, float]:
    """Build a normalised TF-IDF query vector from free text."""
    tokens = _tokenise(text)
    if not tokens:
        return {}
    tf: dict[str, float] = defaultdict(float)
    for tok in tokens:
        tf[tok] += 1.0
    raw = {
        tok: (1.0 + math.log(tf[tok])) * idf.get(tok, 1.0)
        for tok in tf
    }
    norm = math.sqrt(sum(v * v for v in raw.values())) or 1.0
    return {tok: v / norm for tok, v in raw.items()}


# ---------------------------------------------------------------------------
# Main matcher class
# ---------------------------------------------------------------------------

class SectionMatcher:
    """Match dossier rows to CTD KB sections using a deterministic hybrid strategy.

    Parameters
    ----------
    kb : CTDKnowledgeBase
        Loaded knowledge base.  Typically obtained via ``load_knowledge_base()``.

    Usage
    -----
    ::

        matcher = build_matcher()
        summary = matcher.match_dossier(parse_result.rows)
        for m in summary.matches:
            print(m.to_dict())
    """

    def __init__(self, kb: CTDKnowledgeBase) -> None:
        self._kb = kb
        self._sections: list[CTDSection] = kb.all_sections()

        # Build lookup maps
        self._by_id: dict[str, CTDSection] = {s.id.lower(): s for s in self._sections}
        self._by_normalised_title: dict[str, CTDSection] = {
            _normalise_title(s.title): s for s in self._sections
        }
        # Alias map: normalised alias string → CTDSection
        self._by_alias: dict[str, CTDSection] = {}
        for alias, section_id in _TITLE_ALIASES.items():
            sec = kb.get_section(section_id)
            if sec is not None:
                self._by_alias[alias.lower()] = sec

        # Build TF-IDF index
        self._tfidf_vectors, _ = _build_tfidf_index(self._sections)

        # IDF map needed to vectorise queries
        all_tokens: list[list[str]] = [_tokenise(s.title) for s in self._sections]
        n = len(all_tokens)
        df: dict[str, int] = defaultdict(int)
        for tokens in all_tokens:
            for tok in set(tokens):
                df[tok] += 1
        self._idf: dict[str, float] = {
            tok: math.log((n + 1) / (count + 1)) + 1.0
            for tok, count in df.items()
        }

        logger.debug(
            "SectionMatcher built: %d sections, %d aliases, %d IDF tokens",
            len(self._sections), len(self._by_alias), len(self._idf),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match_row(self, row: DossierRow) -> SectionMatch:
        """Match a single ``DossierRow`` to the best KB section.

        Strategies are tried in order; first successful result is returned.
        """
        # Strategy 1 — Exact number
        result = self._try_exact_number(row)
        if result is not None:
            return result

        # Strategy 2 — Normalised title (exact + alias)
        result = self._try_normalised_title(row)
        if result is not None:
            return result

        # Strategy 3 — Semantic (TF-IDF cosine)
        result = self._try_semantic(row)
        if result is not None:
            return result

        # No match found
        return self._make_unmatched(row)

    def match_dossier(self, rows: list[DossierRow]) -> MatchSummary:
        """Match all rows in a parsed dossier and return an aggregate summary."""
        summary = MatchSummary(total_rows=len(rows))
        status_counter: dict[str, int] = defaultdict(int)

        for row in rows:
            m = self.match_row(row)
            summary.matches.append(m)
            if m.match_type == MatchType.EXACT_NUMBER:
                summary.exact_number_count += 1
            elif m.match_type == MatchType.TITLE_NORMALISED:
                summary.title_normalised_count += 1
            elif m.match_type == MatchType.SEMANTIC:
                summary.semantic_count += 1
            elif m.match_type == MatchType.AMBIGUOUS:
                summary.ambiguous_count += 1
            else:
                summary.unmatched_count += 1
            status_counter[m.status.value] += 1

        summary.status_counts = dict(status_counter)
        logger.info(
            "match_dossier: %d rows → exact=%d, normalised=%d, semantic=%d, "
            "ambiguous=%d, unmatched=%d",
            len(rows),
            summary.exact_number_count,
            summary.title_normalised_count,
            summary.semantic_count,
            summary.ambiguous_count,
            summary.unmatched_count,
        )
        return summary

    # ------------------------------------------------------------------
    # Internal matching strategies
    # ------------------------------------------------------------------

    def _try_exact_number(self, row: DossierRow) -> Optional[SectionMatch]:
        """Strategy 1: section_number matches a KB section id exactly."""
        num = row.section_number.strip().lower()
        if not num:
            return None
        sec = self._by_id.get(num)
        if sec is None:
            return None
        status, source = self._determine_status(row, MatchType.EXACT_NUMBER)
        return SectionMatch(
            dossier_row=row,
            expected_section=sec,
            match_type=MatchType.EXACT_NUMBER,
            confidence=1.0,
            status=status,
            match_reason=(
                f"Exact section number match: dossier '{row.section_number}' "
                f"== KB section '{sec.id}' (\"{sec.title}\")."
            ),
            status_source=source,
        )

    def _try_normalised_title(self, row: DossierRow) -> Optional[SectionMatch]:
        """Strategy 2: normalised title matches a KB title or a known alias."""
        title = row.section_title.strip()
        if not title:
            return None

        norm = _normalise_title(title)
        if not norm:
            return None

        # 2a — exact normalised match against KB titles
        sec = self._by_normalised_title.get(norm)
        if sec is not None:
            status, source = self._determine_status(row, MatchType.TITLE_NORMALISED)
            return SectionMatch(
                dossier_row=row,
                expected_section=sec,
                match_type=MatchType.TITLE_NORMALISED,
                confidence=1.0,
                status=status,
                match_reason=(
                    f"Normalised title exact match: \"{title}\" normalises to "
                    f"\"{norm}\", matching KB section '{sec.id}' (\"{sec.title}\")."
                ),
                status_source=source,
            )

        # 2b — alias lookup
        sec = self._by_alias.get(norm)
        if sec is not None:
            status, source = self._determine_status(row, MatchType.TITLE_NORMALISED)
            return SectionMatch(
                dossier_row=row,
                expected_section=sec,
                match_type=MatchType.TITLE_NORMALISED,
                confidence=0.9,
                status=status,
                match_reason=(
                    f"Title alias match: \"{title}\" is a known alias for "
                    f"KB section '{sec.id}' (\"{sec.title}\")."
                ),
                status_source=source,
            )

        return None

    def _try_semantic(self, row: DossierRow) -> Optional[SectionMatch]:
        """Strategy 3: TF-IDF cosine similarity against KB titles.

        Returns a SEMANTIC match if cosine ≥ threshold AND the gap between
        top-1 and top-2 is ≥ SEMANTIC_GAP.
        Returns an AMBIGUOUS match if cosine ≥ threshold but gap is too small.
        Returns None if no candidate reaches the threshold.
        """
        title = row.section_title.strip()
        if not title:
            return None

        query_vec = _query_vector(title, self._idf)
        if not query_vec:
            return None

        # Score every KB section
        scores: list[tuple[float, CTDSection]] = []
        for sec in self._sections:
            doc_vec = self._tfidf_vectors.get(sec.id, {})
            score = _cosine(query_vec, doc_vec)
            if score > 0:
                scores.append((score, sec))

        scores.sort(key=lambda x: x[0], reverse=True)

        if not scores or scores[0][0] < SEMANTIC_THRESHOLD:
            return None

        best_score, best_sec = scores[0]

        # Check disambiguation gap
        second_score = scores[1][0] if len(scores) > 1 else 0.0
        gap = best_score - second_score

        if gap < SEMANTIC_GAP:
            # Ambiguous — report both candidates, flag for review
            alts = [scores[1][1].id] if len(scores) > 1 else []
            status, source = self._determine_status(row, MatchType.AMBIGUOUS)
            return SectionMatch(
                dossier_row=row,
                expected_section=best_sec,
                match_type=MatchType.AMBIGUOUS,
                confidence=round(best_score, 4),
                status=status,
                match_reason=(
                    f"Semantic match is ambiguous: title \"{title}\" scores "
                    f"{best_score:.3f} for KB '{best_sec.id}' (\"{best_sec.title}\") "
                    f"but only {gap:.3f} above next candidate "
                    f"'{scores[1][1].id}' (\"{scores[1][1].title}\") "
                    f"(gap {gap:.3f} < required {SEMANTIC_GAP}). "
                    "Human review required."
                ),
                status_source=source,
                alternative_candidates=alts,
            )

        # Clear semantic match
        status, source = self._determine_status(row, MatchType.SEMANTIC)
        return SectionMatch(
            dossier_row=row,
            expected_section=best_sec,
            match_type=MatchType.SEMANTIC,
            confidence=round(best_score, 4),
            status=status,
            match_reason=(
                f"Semantic TF-IDF match: title \"{title}\" scores "
                f"{best_score:.3f} (threshold {SEMANTIC_THRESHOLD}) for "
                f"KB section '{best_sec.id}' (\"{best_sec.title}\"); "
                f"gap {gap:.3f} ≥ required {SEMANTIC_GAP}."
            ),
            status_source=source,
        )

    def _make_unmatched(self, row: DossierRow) -> SectionMatch:
        """Produce an UNMATCHED result when no strategy succeeded."""
        status, source = self._determine_status(row, MatchType.UNMATCHED)
        parts: list[str] = []
        if row.section_number:
            parts.append(f"section_number '{row.section_number}' not found in KB")
        if row.section_title:
            parts.append(f"title \"{row.section_title}\" did not match any KB title "
                         f"(cosine < {SEMANTIC_THRESHOLD})")
        if not parts:
            parts.append("row has neither section_number nor section_title")
        reason = "No match found: " + "; ".join(parts) + "."
        return SectionMatch(
            dossier_row=row,
            expected_section=None,
            match_type=MatchType.UNMATCHED,
            confidence=0.0,
            status=status,
            match_reason=reason,
            status_source=source,
        )

    # ------------------------------------------------------------------
    # Status determination
    # ------------------------------------------------------------------

    def _determine_status(
        self,
        row: DossierRow,
        match_type: MatchType,
    ) -> tuple[MatchStatus, str]:
        """Return (MatchStatus, source_label).

        If the dossier row supplies a non-empty status, it is mapped and used
        directly — the matcher never overrides a stated status.

        Otherwise the status is inferred from the match result:
          • UNMATCHED  → MISSING
          • AMBIGUOUS  → NEEDS_REVIEW
          • Any match  → PRESENT
        """
        if row.status:
            return _map_status_string(row.status), "dossier_supplied"

        # Infer
        if match_type == MatchType.UNMATCHED:
            return MatchStatus.MISSING, "inferred"
        if match_type == MatchType.AMBIGUOUS:
            return MatchStatus.NEEDS_REVIEW, "inferred"
        return MatchStatus.PRESENT, "inferred"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_matcher(kb: Optional[CTDKnowledgeBase] = None) -> SectionMatcher:
    """Return a ``SectionMatcher`` backed by the real (or supplied) KB.

    Parameters
    ----------
    kb : optional
        Pass an existing ``CTDKnowledgeBase`` instance (e.g. in tests).
        Defaults to the singleton returned by ``load_knowledge_base()``.
    """
    if kb is None:
        kb = load_knowledge_base()
    return SectionMatcher(kb)
