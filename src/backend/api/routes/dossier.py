"""
dossier.py — FastAPI routes for dossier parsing and CTD submission readiness.

Endpoints
---------
POST /api/dossier/check
    Upload a dossier outline file (CSV or XLSX) and receive a full readiness
    report: per-module scores, gap list, critical gaps, and match details.

GET  /api/dossier/modules
    Return the ICH M4 CTD structure as a flat JSON list, suitable for driving
    a frontend section-selection UI.

GET  /api/dossier/sample-check
    Run a readiness check on the bundled sample dossier fixture so the API can
    be exercised without uploading a file.

Design notes
------------
* All business logic lives in core modules.  Route handlers do only:
    1. Input validation / conversion
    2. Delegating to core functions
    3. Serialising the result to a plain dict
* No demo / hard-coded results anywhere.
* File uploads are accepted as multipart/form-data via FastAPI's UploadFile.
  The file is written to a temporary path, parsed by parse_dossier(), then the
  temp file is cleaned up.  This avoids holding large files in memory.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from core.ctd_knowledge_base import load_knowledge_base
from core.dossier_parser import DossierParseError, parse_dossier
from core.readiness_scorer import ReadinessConfig, score_readiness
from core.section_matcher import build_matcher

logger = logging.getLogger(__name__)

router = APIRouter()

# Bundled sample dossier fixture (committed to repo; always available)
_SAMPLE_DOSSIER = (
    Path(__file__).parent.parent.parent.parent   # src/
    / "tests" / "fixtures" / "dossier_outline_clean.csv"
)

# Supported upload extensions
_ALLOWED_EXTENSIONS = {".csv", ".xlsx"}


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------

class DossierParseMetadata(BaseModel):
    """Metadata about the parsed dossier file."""
    source_path: str
    source_format: str
    row_count: int
    parsed_rows: int
    error_count: int
    errors: list[str]
    column_map: dict[str, str]


class DossierCheckResponse(BaseModel):
    """Full readiness check response."""
    parse_metadata: DossierParseMetadata
    overall_score: float = Field(
        description="Weighted completeness score in [0.0, 1.0]."
    )
    overall_score_pct: float = Field(
        description="overall_score × 100 for display."
    )
    total_present: int
    total_needs_review: int
    total_missing: int
    total_not_applicable: int
    total_critical_missing: int
    module_scores: dict
    match_summary: dict
    disclaimer: str


class CTDSectionItem(BaseModel):
    """Single CTD section entry for the /modules endpoint."""
    id: str
    number: str
    title: str
    module_id: str
    level: int
    requirement: str
    safety_relevant: bool
    traceability_categories: list[str]
    applicability: list[str]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_readiness_check(dossier_path: Path) -> dict:
    """Core pipeline: parse → match → score → serialise.

    Returns a plain serialisable dict (FastAPI will encode to JSON).
    Raises HTTPException on file-level parse failures.
    """
    try:
        parse_result = parse_dossier(dossier_path)
    except DossierParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error parsing dossier %s", dossier_path)
        raise HTTPException(
            status_code=500, detail=f"Unexpected parse error: {exc}"
        ) from exc

    kb = load_knowledge_base()
    matcher = build_matcher(kb)
    match_summary = matcher.match_dossier(parse_result.rows)
    result = score_readiness(match_summary)

    return {
        "parse_metadata": {
            "source_path": parse_result.source_path,
            "source_format": parse_result.source_format,
            "row_count": parse_result.row_count,
            "parsed_rows": len(parse_result.rows),
            "error_count": parse_result.error_count,
            "errors": [str(e) for e in parse_result.errors],
            "column_map": parse_result.column_map,
        },
        "overall_score": round(result.overall_score, 4),
        "overall_score_pct": round(result.overall_score_pct, 2),
        "total_present": result.total_present,
        "total_needs_review": result.total_needs_review,
        "total_missing": result.total_missing,
        "total_not_applicable": result.total_not_applicable,
        "total_critical_missing": result.total_critical_missing,
        "module_scores": {
            mid: ms.to_dict() for mid, ms in result.module_scores.items()
        },
        "match_summary": match_summary.to_dict(),
        "disclaimer": result.disclaimer,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post(
    "/check",
    summary="Check a dossier outline for CTD completeness",
)
async def check_dossier(
    file: UploadFile = File(
        ...,
        description="CSV or XLSX dossier outline file.",
    ),
    drug_name: Optional[str] = Form(
        default=None,
        description="Drug name — stored in metadata only, not used in scoring.",
    ),
):
    """
    Upload a dossier outline (CSV or XLSX) and receive a full CTD readiness report.

    The file must contain at least a ``section_number`` or ``section_title`` column
    (multiple header aliases are accepted — see the dossier_parser module for the
    full alias list).

    An optional ``status`` column can carry per-row submission status:
    ``present``, ``missing``, ``needs_review``, or ``not_applicable``.
    When absent the matcher infers status from the match result.

    Returns a structured report with:
    - Per-module scores (M1–M5)
    - Overall weighted completeness score
    - Gap list (MISSING sections)
    - Critical gaps (required + safety-relevant sections that are MISSING)
    - Match details for every row

    ⚠️  HEURISTIC SCREENING ONLY — not an FDA approval readiness assessment.
    """
    # Validate extension
    suffix = Path(file.filename or "upload").suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{suffix}'. "
                f"Accepted formats: {sorted(_ALLOWED_EXTENSIONS)}"
            ),
        )

    # Write to temp file (suffix matters for parse_dossier's format detection)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        content = await file.read()
        tmp.write(content)

    try:
        response = _run_readiness_check(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    # Attach upload metadata
    response["upload"] = {
        "filename": file.filename,
        "drug_name": drug_name,
        "content_type": file.content_type,
        "size_bytes": len(content),
    }
    return response


@router.get(
    "/sample-check",
    summary="Run readiness check on the bundled sample dossier",
)
def sample_check():
    """
    Run a CTD readiness check on the committed sample dossier fixture.

    No file upload needed.  Useful for verifying the API is working and for
    frontend development without a real dossier file.

    The sample fixture is ``src/tests/fixtures/dossier_outline_clean.csv``.
    """
    if not _SAMPLE_DOSSIER.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                f"Sample dossier fixture not found at {_SAMPLE_DOSSIER}. "
                "Ensure the file exists in src/tests/fixtures/."
            ),
        )
    return _run_readiness_check(_SAMPLE_DOSSIER)


@router.get(
    "/modules",
    summary="Return the full ICH M4 CTD section structure",
    response_model=list[CTDSectionItem],
)
def get_ctd_modules():
    """
    Return every section in the ICH M4 CTD knowledge base as a flat list,
    ordered by document section number.

    Fields:
    - ``id`` / ``number`` — canonical section identifier (e.g. ``"2.7.4"``)
    - ``module_id`` — parent module (``"M1"``–``"M5"``)
    - ``requirement`` — ``"required"`` / ``"conditional"`` / ``"optional"``
    - ``safety_relevant`` — True when the section bears directly on safety
    - ``traceability_categories`` — signal categories this section covers

    This endpoint is the primary data source for the Dossier Readiness frontend page.
    """
    kb = load_knowledge_base()
    items: list[dict] = []
    for module in kb.all_modules():
        for section in module.sections:
            items.append({
                "id": section.id,
                "number": section.number,
                "title": section.title,
                "module_id": module.id,
                "level": section.level,
                "requirement": section.requirement,
                "safety_relevant": section.safety_relevant,
                "traceability_categories": section.traceability_categories,
                "applicability": section.applicability,
            })
    return items


@router.get(
    "/gap-report/{submission_type}",
    summary="Return required sections missing for a submission type",
)
def get_gap_report(submission_type: str):
    """
    Return all sections that are *required* for the given submission type
    and mark which ones are commonly omitted.

    Path parameter ``submission_type`` must be one of:
    ``NDA``, ``BLA``, ``ANDA``, ``MAA``  (case-insensitive).

    This endpoint returns the full KB requirement list — it does NOT check
    against a specific dossier.  Use ``POST /api/dossier/check`` to compare
    against an actual uploaded dossier outline.
    """
    allowed = {"NDA", "BLA", "ANDA", "MAA"}
    norm = submission_type.upper().strip()
    if norm not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"submission_type must be one of {sorted(allowed)}, got '{submission_type}'.",
        )

    kb = load_knowledge_base()
    required = kb.get_required_sections(norm)
    conditional = kb.get_conditional_sections(norm)

    return {
        "submission_type": norm,
        "required_sections": [s.to_dict() for s in required],
        "conditional_sections": [s.to_dict() for s in conditional],
        "required_count": len(required),
        "conditional_count": len(conditional),
        "safety_relevant_required_count": sum(
            1 for s in required if s.safety_relevant
        ),
    }
