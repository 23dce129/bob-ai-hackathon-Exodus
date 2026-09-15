"""
traceability.py — FastAPI routes for safety-to-submission traceability analysis.

Endpoints
---------
POST /api/traceability/trace
    Given a list of safety signals and a dossier outline file, trace each
    signal to the CTD sections that should document it, and report which
    sections are present, under review, or missing.

POST /api/traceability/trace-text
    Same as /trace but accepts signals + a plain-text section list (one
    section number per line) instead of a file upload.  Useful for quick
    programmatic use from the MCP server or CLI.

GET  /api/traceability/categories
    Return all known traceability signal categories from the KB.

Design notes
------------
* Route handlers delegate entirely to traceability_engine and section_matcher.
* Signals are supplied as a JSON body field; file upload is for the dossier.
* No hard-coded demo data anywhere.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from core.ctd_knowledge_base import load_knowledge_base
from core.dossier_parser import DossierParseError, DossierRow, parse_dossier
from core.section_matcher import MatchStatus, MatchSummary, build_matcher
from core.traceability_engine import (
    TRACEABILITY_DISCLAIMER,
    trace_all_signals,
)

logger = logging.getLogger(__name__)

router = APIRouter()

_ALLOWED_EXTENSIONS = {".csv", ".xlsx"}


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class SignalInput(BaseModel):
    """One safety signal for traceability input."""
    drug_name: str = Field(..., min_length=1, description="Drug name.")
    adverse_event: str = Field(..., min_length=1, description="Adverse event preferred term.")
    prr: Optional[float] = Field(
        default=None, description="PRR point estimate (pass-through; not used in scoring)."
    )
    p_value: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Statistical p-value."
    )
    signal_flag: bool = Field(
        default=False, description="Whether the PRR calculator flagged this pair."
    )


class TraceTextRequest(BaseModel):
    """Request body for /trace-text (no file upload)."""
    signals: list[SignalInput] = Field(
        ..., min_length=1, description="List of safety signals to trace."
    )
    section_numbers: list[str] = Field(
        default_factory=list,
        description=(
            "List of CTD section numbers present in the dossier "
            "(e.g. ['2.4', '2.7.4', '5.3.5']). "
            "Sections are treated as PRESENT with confidence 1.0."
        ),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _match_summary_from_section_numbers(
    section_numbers: list[str],
    kb,
) -> MatchSummary:
    """Build a MatchSummary from a plain list of section id strings.

    Each section id that resolves to a KB entry is marked PRESENT with
    confidence 1.0.  Unknown ids are silently ignored.
    """
    from core.section_matcher import MatchType, SectionMatch  # local import avoids circular

    matches = []
    for i, raw_id in enumerate(section_numbers):
        sec_id = raw_id.strip().lower()
        section = kb.get_section(sec_id)
        if section is None:
            # Try uppercase variant (e.g. "3.2.s" → "3.2.S")
            section = kb.get_section(raw_id.strip().upper())
        if section is None:
            continue
        row = DossierRow(
            section_number=section.id,
            section_title=section.title,
            description="",
            source="",
            status="present",
            row_index=i + 1,
        )
        matches.append(SectionMatch(
            dossier_row=row,
            expected_section=section,
            match_type=MatchType.EXACT_NUMBER,
            confidence=1.0,
            status=MatchStatus.PRESENT,
            match_reason="Supplied directly in section_numbers list",
            status_source="dossier_supplied",
        ))

    summary = MatchSummary(
        matches=matches,
        total_rows=len(matches),
        exact_number_count=len(matches),
    )
    from collections import defaultdict
    status_counter: dict[str, int] = defaultdict(int)
    for m in matches:
        status_counter[m.status.value] += 1
    summary.status_counts = dict(status_counter)
    return summary


def _run_trace(signals_raw: list[dict], match_summary: MatchSummary) -> dict:
    """Core traceability pipeline — delegates to trace_all_signals."""
    kb = load_knowledge_base()
    report = trace_all_signals(signals_raw, match_summary, kb=kb)
    return report.to_dict()


def _signals_to_dicts(signals: list[SignalInput]) -> list[dict]:
    return [
        {
            "drug_name": s.drug_name,
            "adverse_event": s.adverse_event,
            "prr": s.prr,
            "p_value": s.p_value,
            "signal_flag": s.signal_flag,
        }
        for s in signals
    ]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post(
    "/trace",
    summary="Trace safety signals to CTD dossier sections (file upload)",
)
async def trace_signals_file(
    signals_json: str = Form(
        ...,
        description=(
            "JSON array of signal objects. Each must have 'drug_name' and "
            "'adverse_event' fields. Optional: 'prr', 'p_value', 'signal_flag'."
        ),
    ),
    file: UploadFile = File(
        ...,
        description="CSV or XLSX dossier outline file.",
    ),
):
    """
    Trace a list of pharmacovigilance safety signals to their required CTD sections
    and identify which of those sections are present, under review, or missing in
    the uploaded dossier outline.

    **Request** — multipart/form-data:
    - ``signals_json``: JSON array of signal objects
    - ``file``: dossier outline (CSV or XLSX)

    **Response** per signal:
    - ``signal_category``: detected traceability category (e.g. ``"bleeding"``)
    - ``traceability_score``: fraction of relevant sections with acceptable documentation [0, 1]
    - ``critical_gaps``: section ids that are safety-relevant AND missing
    - ``entries``: per-section status, confidence, priority, and relevance reason

    ⚠️  A detected signal does NOT establish causality.
    ⚠️  A missing section does NOT mean the submission will be rejected.
    """
    # Parse signals JSON
    try:
        raw_signals = json.loads(signals_json)
        if not isinstance(raw_signals, list):
            raise ValueError("signals_json must be a JSON array")
        if len(raw_signals) == 0:
            raise ValueError("signals_json must contain at least one signal")
        # Validate each signal with the Pydantic model
        signals = [SignalInput(**s) for s in raw_signals]
    except (json.JSONDecodeError, TypeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid signals_json: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Signal validation error: {exc}",
        ) from exc

    # Validate file extension
    suffix = Path(file.filename or "upload").suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{suffix}'. "
                f"Accepted: {sorted(_ALLOWED_EXTENSIONS)}"
            ),
        )

    # Write upload to temp file
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        content = await file.read()
        tmp.write(content)

    try:
        try:
            parse_result = parse_dossier(tmp_path)
        except DossierParseError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Error parsing dossier %s", tmp_path)
            raise HTTPException(status_code=500, detail=f"Parse error: {exc}") from exc

        kb = load_knowledge_base()
        matcher = build_matcher(kb)
        match_summary = matcher.match_dossier(parse_result.rows)
        result = _run_trace(_signals_to_dicts(signals), match_summary)
    finally:
        tmp_path.unlink(missing_ok=True)

    result["dossier_metadata"] = {
        "filename": file.filename,
        "row_count": parse_result.row_count,
        "parsed_rows": len(parse_result.rows),
        "match_summary": match_summary.to_dict(),
    }
    return result


@router.post(
    "/trace-text",
    summary="Trace safety signals using a section number list (no file upload)",
)
def trace_signals_text(body: TraceTextRequest):
    """
    Trace safety signals to CTD sections without uploading a file.

    Accepts a JSON body with:
    - ``signals``: list of signal objects
    - ``section_numbers``: list of CTD section ids present in the dossier
      (e.g. ``["2.4", "2.7.4", "5.3.5"]``)

    Sections in ``section_numbers`` that match KB entries are marked PRESENT.
    Sections not in the list are treated as MISSING.

    This endpoint is convenient for the MCP server and programmatic use
    where preparing a CSV upload is impractical.
    """
    if not body.signals:
        raise HTTPException(
            status_code=422, detail="signals list must not be empty."
        )

    kb = load_knowledge_base()
    match_summary = _match_summary_from_section_numbers(body.section_numbers, kb)
    return _run_trace(_signals_to_dicts(body.signals), match_summary)


@router.get(
    "/categories",
    summary="Return all known traceability signal categories",
)
def get_categories():
    """
    Return the list of signal categories that the traceability engine recognises,
    plus the count of CTD sections mapped to each category.

    Categories correspond to ``traceability_categories`` annotations in the
    ICH M4 CTD knowledge base JSON.
    """
    kb = load_knowledge_base()
    categories = kb.get_all_traceability_categories()
    return {
        "categories": [
            {
                "name": cat,
                "section_count": len(kb.get_sections_by_traceability_category(cat)),
            }
            for cat in categories
        ],
        "total_categories": len(categories),
        "disclaimer": TRACEABILITY_DISCLAIMER,
    }
