"""
server.py — PharmaGuard AI MCP Server.

Exposes six tools that call the PharmaGuard FastAPI backend over HTTP so that
IBM Bob (or any MCP-compatible client) can perform pharmacovigilance tasks
through natural-language conversation.

Architecture
------------
This server acts as a thin adapter — it translates MCP tool calls into HTTP
requests against the running PharmaGuard backend and formats the responses for
Bob to read. All domain logic stays in the backend; nothing is duplicated here.

FastMCP 2.x note
----------------
In FastMCP 2.x, @mcp.tool() replaces the decorated function with a FunctionTool
object that is NOT directly callable. To keep the smoke test working, each tool
is implemented as a plain private function (_impl_*) that is then registered
explicitly via mcp.tool()(fn). The smoke test calls the _impl_ functions.

Transport
---------
stdio (spawned by Bob as a child process). Bob reads .bob/mcp.json and runs:
    python src/mcp-server/server.py

Prerequisites
-------------
1. The PharmaGuard backend must be running:
       cd src/backend && uvicorn main:app --port 8000
2. Python dependencies installed in this environment:
       pip install -r src/mcp-server/requirements.txt
3. Optionally set PHARMAGUARD_API_URL in your .env (defaults to localhost:8000)

Usage — Bob will call these tools automatically based on context.
Example prompts:
    "Are there any safety signals for warfarin?"
    "What is the PRR for atorvastatin with a minimum of 5 cases?"
    "Is our dossier ready for NDA submission?"
    "Show me the documentation gaps for warfarin's bleeding signal."

Testing without Bob
-------------------
    python server.py --test
Runs a built-in smoke test that calls each tool against the live backend and
prints structured results. The backend must be running on PHARMAGUARD_API_URL.
"""

from __future__ import annotations

import json
import sys
import textwrap
from typing import Optional

import httpx
from fastmcp import FastMCP

from config import mcp_settings

# ---------------------------------------------------------------------------
# FastMCP server instance
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "PharmaGuard AI",
    instructions=textwrap.dedent("""
        PharmaGuard AI is a pharmacovigilance and regulatory submission readiness
        platform. Use these tools to:
        - Detect adverse drug reaction safety signals from FDA FAERS data
        - Calculate Proportional Reporting Ratio (PRR) statistics
        - Check ICH M4 CTD dossier completeness for regulatory submission
        - Trace safety signals to required CTD documentation sections
        - Generate regulatory action briefs and gap reports

        IMPORTANT LIMITATIONS (always include in responses to users):
        - All scores and signals are HEURISTIC SCREENING AIDS only.
        - They do NOT represent FDA approval readiness.
        - A detected signal does NOT establish drug causality.
        - A missing CTD section does NOT mean the submission will be rejected.
        - All results require review by a qualified regulatory affairs professional.
    """).strip(),
)


# ---------------------------------------------------------------------------
# Shared HTTP helpers
# ---------------------------------------------------------------------------

def _get(path: str, params: dict | None = None) -> dict:
    """Make a GET request to the backend. Returns parsed JSON or raises."""
    url = mcp_settings.api_base_url.rstrip("/") + path
    with httpx.Client(timeout=mcp_settings.request_timeout) as client:
        resp = client.get(url, params=params or {})
        resp.raise_for_status()
        return resp.json()


def _post(path: str, json_body: dict) -> dict:
    """Make a POST request to the backend. Returns parsed JSON or raises."""
    url = mcp_settings.api_base_url.rstrip("/") + path
    with httpx.Client(timeout=mcp_settings.request_timeout) as client:
        resp = client.post(url, json=json_body)
        resp.raise_for_status()
        return resp.json()


def _truncate_list(items: list) -> list:
    """Limit a list to max_results for concise MCP responses."""
    return items[: mcp_settings.max_results]


def _backend_error(exc: Exception, tool_name: str) -> str:
    """Format a backend error into a safe, readable string for Bob."""
    if isinstance(exc, httpx.ConnectError):
        return (
            f"[{tool_name}] Cannot reach the PharmaGuard backend at "
            f"{mcp_settings.api_base_url}. "
            "Ensure the backend is running: "
            "cd src/backend && uvicorn main:app --port 8000"
        )
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            detail = exc.response.text[:200]
        return f"[{tool_name}] Backend returned HTTP {code}: {detail}"
    return f"[{tool_name}] Unexpected error: {exc}"


# ---------------------------------------------------------------------------
# Tool implementations — plain functions, directly callable for smoke tests
# ---------------------------------------------------------------------------

def _impl_detect_safety_signals(
    drug_name: Optional[str] = None,
    min_prr: float = 2.0,
    min_cases: int = 3,
    max_pval: float = 0.05,
) -> str:
    """
    Detect potential adverse drug reaction (ADR) safety signals from FDA FAERS
    pharmacovigilance data using Proportional Reporting Ratio (PRR) analysis.

    Inputs
    ------
    drug_name  : str, optional
        Filter results to a single drug (e.g. "warfarin", "ibuprofen").
        Omit to scan all drugs in the sample dataset.
    min_prr    : float, default 2.0
        Minimum PRR threshold. FDA/EMA standard is >= 2.0.
    min_cases  : int, default 3
        Minimum number of reports required. Standard is >= 3.
    max_pval   : float, default 0.05
        Maximum p-value (statistical significance threshold).

    Outputs
    -------
    JSON object containing:
    - flagged_signals: list of signals meeting all three thresholds, each with
        drug_name, adverse_event, case_count, prr, p_value, signal_flag,
        trend direction, and a caution disclaimer.
    - total_flagged: number of signals found.
    - total_evaluated: total drug-event pairs analysed.
    - thresholds used.

    Limitations
    -----------
    - Uses a bundled FAERS sample dataset (~500 rows), NOT the full FAERS database.
    - A flagged signal is a statistical association, NOT proof of causality.
    - Results are a screening aid for trained pharmacovigilance professionals.
    """
    try:
        data = _post("/api/signals/analyze", {
            "drug_name": drug_name,
            "min_prr": min_prr,
            "min_cases": min_cases,
            "max_pval": max_pval,
            "include_trends": True,
        })
        signals = _truncate_list(data.get("flagged_signals", []))
        return json.dumps({
            "total_evaluated": data.get("total_evaluated"),
            "total_flagged": data.get("total_flagged"),
            "drug_filter": drug_name,
            "thresholds": data.get("thresholds"),
            "flagged_signals": [
                {
                    "drug_name": s["drug_name"],
                    "adverse_event": s["adverse_event"],
                    "case_count": s.get("a"),
                    "prr": round(s["prr"], 3) if s.get("prr") else None,
                    "p_value": round(s["p_value"], 4) if s.get("p_value") else None,
                    "signal_flag": s.get("signal_flag"),
                    "trend": s.get("trend", {}).get("direction") if s.get("trend") else None,
                }
                for s in signals
            ],
            "showing_top_n": len(signals),
            "disclaimer": (
                "HEURISTIC SCREENING ONLY. Statistical association does not equal causality. "
                "Review required by a qualified pharmacovigilance professional."
            ),
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": _backend_error(exc, "detect_safety_signals")})


def _impl_calculate_prr(
    drug_name: str,
    min_prr: float = 2.0,
    min_cases: int = 3,
    max_pval: float = 0.05,
) -> str:
    """
    Calculate Proportional Reporting Ratio (PRR) for a specific drug across
    all adverse events in the sample FAERS dataset.

    The PRR measures whether a drug is reported with an adverse event more often
    than all other drugs combined. Formula:
        PRR = (cases of D with E / all cases of D)
              divided by
              (cases of other drugs with E / all cases of other drugs)

    PRR >= 2, case_count >= 3, and p-value < 0.05 is the standard EMA/FDA
    signal detection threshold combination.

    Inputs
    ------
    drug_name  : str (required)
        Drug to analyse, e.g. "warfarin", "ibuprofen", "atorvastatin".
        Case-insensitive.
    min_prr    : float, default 2.0  (must be >= 1.0)
    min_cases  : int,   default 3    (must be >= 1)
    max_pval   : float, default 0.05 (must be in (0, 1])

    Outputs
    -------
    JSON object containing:
    - flagged_signals: pairs meeting all thresholds, each with PRR point
        estimate, 95% CI, chi-squared or Fisher test result.
    - total_pairs_evaluated: total drug-event pairs checked.
    - signal_count: number flagged.

    Limitations
    -----------
    - Sample dataset only — not production-scale FAERS.
    - PRR undefined (None) when no other drug reports the same event (c=0).
    - Use Fisher's exact test interpretation when expected cell counts < 5.
    """
    try:
        data = _post("/api/signals/analyze", {
            "drug_name": drug_name,
            "min_prr": min_prr,
            "min_cases": min_cases,
            "max_pval": max_pval,
            "include_trends": False,
        })
        all_sigs = data.get("all_signals", [])
        flagged = [s for s in all_sigs if s.get("signal_flag")]
        return json.dumps({
            "drug_name": drug_name,
            "total_pairs_evaluated": len(all_sigs),
            "signal_count": len(flagged),
            "thresholds": data.get("thresholds"),
            "flagged_signals": [
                {
                    "adverse_event": s["adverse_event"],
                    "case_count": s.get("a"),
                    "prr": round(s["prr"], 3) if s.get("prr") else None,
                    "ci_lower_95": round(s["ci_lower_95"], 3) if s.get("ci_lower_95") else None,
                    "ci_upper_95": round(s["ci_upper_95"], 3) if s.get("ci_upper_95") else None,
                    "p_value": round(s["p_value"], 4) if s.get("p_value") else None,
                    "test_method": s.get("test_method"),
                }
                for s in _truncate_list(flagged)
            ],
            "disclaimer": (
                "PRR is a disproportionality measure, not a causal estimate. "
                "Requires review by a qualified pharmacovigilance professional."
            ),
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": _backend_error(exc, "calculate_prr")})


def _impl_get_signal_explanation(drug_name: str, adverse_event: str) -> str:
    """
    Retrieve a temporal trend analysis and contextual explanation for a specific
    drug-adverse event pair.

    This tool answers: "Is the reporting of this drug/event pair increasing,
    stable, or declining over time?" It provides quarter-by-quarter report
    counts, a trend score, and the direction of the trend.

    Use this AFTER detect_safety_signals or calculate_prr to investigate a
    specific flagged signal further.

    Inputs
    ------
    drug_name     : str (required)
        Drug name exactly as it appears in the signal output, e.g. "warfarin".
    adverse_event : str (required)
        Adverse event preferred term, e.g. "bleeding", "hepatotoxicity".
        Case-insensitive.

    Outputs
    -------
    JSON object containing:
    - direction: "increasing" | "decreasing" | "stable" | "insufficient_data"
    - trend_score: float in [-1.0, +1.0] (positive = increasing)
    - recent_count: reports in the most recent half of the dataset
    - previous_count: reports in the earlier half
    - pct_change: percentage change from previous to recent period
    - quarterly_counts: list of quarterly report counts
    - date_range: human-readable date range covered

    Limitations
    -----------
    - Returns error if the drug/event pair does not appear in the sample dataset.
    - The sample dataset has limited temporal range (approximately 2 years).
    - Trend direction does not imply a causal relationship.
    """
    try:
        data = _get(f"/api/signals/trend/{drug_name.strip()}/{adverse_event.strip()}")
        return json.dumps({
            "drug_name": data.get("drug_name"),
            "adverse_event": data.get("adverse_event"),
            "direction": data.get("direction"),
            "trend_score": data.get("trend_score"),
            "recent_count": data.get("recent_count"),
            "previous_count": data.get("previous_count"),
            "pct_change": data.get("pct_change"),
            "total_reports": data.get("total_reports"),
            "date_range": f"{data.get('date_range_start')} to {data.get('date_range_end')}",
            "quarterly_counts": [
                {"quarter": q["quarter"], "count": q["count"]}
                for q in data.get("quarters", [])
            ],
            "disclaimer": (
                "Trend data is descriptive only. "
                "Does not establish causality or regulatory significance."
            ),
        }, indent=2)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return json.dumps({
                "error": (
                    f"No data found for drug='{drug_name}' / event='{adverse_event}'. "
                    "Check spelling and try detect_safety_signals first to see available pairs."
                )
            })
        return json.dumps({"error": _backend_error(exc, "get_signal_explanation")})
    except Exception as exc:
        return json.dumps({"error": _backend_error(exc, "get_signal_explanation")})


def _impl_check_ctd_readiness() -> str:
    """
    Check the completeness of a regulatory dossier outline against the ICH M4
    Common Technical Document (CTD) structure (Modules 1-5).

    This tool runs a readiness check on the bundled sample dossier outline and
    returns a structured completeness report.

    No inputs required — uses the committed sample dossier fixture.
    To check a custom dossier, use the POST /api/dossier/check endpoint directly.

    Outputs
    -------
    JSON object containing:
    - overall_score: weighted completeness score in [0.0, 1.0]
    - overall_score_pct: same as percentage (0-100)
    - module_scores: per-module breakdown (M1-M5), each with:
        score, title, present, needs_review, missing, critical_missing counts.
    - total_present / total_missing / total_needs_review / total_critical_missing
    - disclaimer

    Section status values:
    - PRESENT: section is in the dossier
    - NEEDS_REVIEW: section found but requires review (low match confidence)
    - MISSING: required section not found in the dossier
    - NOT_APPLICABLE: section is not applicable to this submission

    Limitations
    -----------
    - Uses sample dossier fixture, NOT a real submission file.
    - Score is a heuristic completeness measure, NOT an FDA approval indicator.
    - Critical gaps are sections that are both required and safety-relevant.
    """
    try:
        data = _get("/api/dossier/sample-check")
        module_summary = {}
        for mid, ms in data.get("module_scores", {}).items():
            module_summary[mid] = {
                "score_pct": round(ms["score"] * 100, 1),
                "present": ms["present_count"],
                "needs_review": ms["needs_review_count"],
                "missing": ms["missing_count"],
                "critical_missing": ms["critical_missing_count"],
                "title": ms["module_title"],
            }
        return json.dumps({
            "overall_score": data.get("overall_score"),
            "overall_score_pct": data.get("overall_score_pct"),
            "total_present": data.get("total_present"),
            "total_needs_review": data.get("total_needs_review"),
            "total_missing": data.get("total_missing"),
            "total_critical_missing": data.get("total_critical_missing"),
            "module_scores": module_summary,
            "parse_metadata": data.get("parse_metadata"),
            "disclaimer": data.get("disclaimer"),
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": _backend_error(exc, "check_ctd_readiness")})


def _impl_generate_gap_report(submission_type: str = "NDA") -> str:
    """
    Generate a regulatory gap report listing all required and conditional CTD
    sections for a given submission type, with safety-relevance information.

    This report answers: "What sections are required for an NDA/BLA/ANDA/MAA
    submission, and which are safety-critical?"

    It does NOT compare against a specific dossier. Use check_ctd_readiness or
    the /api/dossier/check endpoint for dossier-specific analysis.

    Inputs
    ------
    submission_type : str, default "NDA"
        Regulatory submission type. Must be one of:
        - NDA   (New Drug Application - US FDA)
        - BLA   (Biologics License Application - US FDA)
        - ANDA  (Abbreviated New Drug Application - US FDA)
        - MAA   (Marketing Authorisation Application - EMA)

    Outputs
    -------
    JSON object containing:
    - submission_type: normalised submission type
    - required_count: number of required sections
    - conditional_count: number of conditional sections
    - safety_relevant_required_count: required sections that are safety-relevant
    - safety_relevant_required_sections: list of safety-relevant required sections
    - all_required_sections: complete list of required sections

    Limitations
    -----------
    - Based on a curated interpretation of ICH M4 guidelines for demonstration.
    - NOT legal or regulatory advice.
    - Does NOT represent the position of the FDA, ICH, EMA, or any authority.
    """
    allowed = {"NDA", "BLA", "ANDA", "MAA"}
    sub = submission_type.upper().strip()
    if sub not in allowed:
        return json.dumps({
            "error": f"submission_type must be one of {sorted(allowed)}, got '{submission_type}'."
        })
    try:
        data = _get(f"/api/dossier/gap-report/{sub}")
        safety_required = [
            s for s in data.get("required_sections", []) if s.get("safety_relevant")
        ]
        return json.dumps({
            "submission_type": data.get("submission_type"),
            "required_count": data.get("required_count"),
            "conditional_count": data.get("conditional_count"),
            "safety_relevant_required_count": data.get("safety_relevant_required_count"),
            "safety_relevant_required_sections": [
                {
                    "id": s["id"],
                    "title": s["title"],
                    "traceability_categories": s.get("traceability_categories", []),
                }
                for s in _truncate_list(safety_required)
            ],
            "all_required_sections": [
                {"id": s["id"], "title": s["title"], "safety_relevant": s.get("safety_relevant")}
                for s in data.get("required_sections", [])
            ],
            "disclaimer": (
                "Gap report based on ICH M4 guidelines (demonstration only). "
                "NOT legal or regulatory advice."
            ),
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": _backend_error(exc, "generate_gap_report")})


def _impl_generate_regulatory_action_brief(
    drug_name: str,
    adverse_events: Optional[list[str]] = None,
    dossier_sections: Optional[list[str]] = None,
) -> str:
    """
    Generate a structured regulatory action brief for a drug by combining
    signal detection with traceability analysis.

    Steps performed:
    1. Detect safety signals for the drug (PRR analysis on FAERS sample data)
    2. Trace each signal to the CTD sections that should document it
    3. Identify critical documentation gaps
    4. Produce a prioritised list of recommended actions

    This is the highest-level PharmaGuard tool, combining signal detection
    with traceability analysis into a single actionable output.

    Inputs
    ------
    drug_name        : str (required)
        Drug name, e.g. "warfarin", "ibuprofen", "atorvastatin".
    adverse_events   : list[str], optional
        Specific adverse events to trace. If omitted, detected signals are used.
        Example: ["bleeding", "hepatotoxicity"]
    dossier_sections : list[str], optional
        CTD section IDs present in the dossier (e.g. ["2.4", "2.7.4", "5.3.5"]).
        If omitted, an empty dossier is assumed (worst-case gap analysis).

    Outputs
    -------
    JSON object containing:
    - drug_name
    - signals_detected: count of flagged PRR signals for this drug
    - signals_traced: count of signals traced to CTD sections
    - overall_traceability_score: mean score across all signals [0.0, 1.0]
    - critical_gap_count: signals with one or more safety-relevant missing sections
    - signal_summary: per-signal category, score, and critical gaps
    - priority_actions: ranked list of recommended documentation actions

    Priority levels:
    - "urgent": safety_relevant section is MISSING
    - "high": signal has overall low traceability coverage

    Limitations
    -----------
    - Signal detection uses sample FAERS data only.
    - Dossier coverage based on the supplied section list, not a parsed file.
    - Does NOT establish causality or predict regulatory outcomes.
    - Requires expert pharmacovigilance and regulatory review.
    """
    try:
        # Step 1: Detect signals for this drug
        signal_data = _post("/api/signals/analyze", {
            "drug_name": drug_name,
            "min_prr": 2.0,
            "min_cases": 3,
            "max_pval": 0.05,
            "include_trends": False,
        })
        flagged = [s for s in signal_data.get("all_signals", []) if s.get("signal_flag")]

        # Step 2: Build signals for traceability
        if adverse_events:
            trace_signals = [
                {"drug_name": drug_name, "adverse_event": ae, "signal_flag": True}
                for ae in adverse_events
            ]
        elif flagged:
            trace_signals = [
                {
                    "drug_name": s["drug_name"],
                    "adverse_event": s["adverse_event"],
                    "prr": s.get("prr"),
                    "p_value": s.get("p_value"),
                    "signal_flag": True,
                }
                for s in flagged[: mcp_settings.max_results]
            ]
        else:
            return json.dumps({
                "drug_name": drug_name,
                "signals_detected": [],
                "message": (
                    f"No signals flagged for '{drug_name}' with default thresholds. "
                    "The drug may not appear in the sample dataset, or no pairs "
                    "meet all three criteria (PRR>=2, n>=3, p<0.05)."
                ),
                "disclaimer": "HEURISTIC SCREENING ONLY. Results require professional review.",
            }, indent=2)

        # Step 3: Trace signals to CTD sections
        trace_data = _post("/api/traceability/trace-text", {
            "signals": trace_signals,
            "section_numbers": dossier_sections or [],
        })

        # Step 4: Build priority actions from critical gaps
        priority_actions: list[dict] = []
        for result in trace_data.get("results", []):
            ae = result["adverse_event"]
            category = result["signal_category"]
            score = result["traceability_score"]
            for gap_id in result.get("critical_gaps", []):
                entry = next(
                    (e for e in result["entries"] if e["ctd_section_id"] == gap_id), {}
                )
                priority_actions.append({
                    "priority": "urgent",
                    "adverse_event": ae,
                    "signal_category": category,
                    "section_id": gap_id,
                    "section_title": entry.get("ctd_section_title", ""),
                    "action": (
                        f"Add required safety documentation for '{ae}' "
                        f"to CTD Section {gap_id} ({entry.get('ctd_section_title', '')})."
                    ),
                    "rationale": entry.get("relevance_reason", ""),
                })
            if score < 0.5:
                priority_actions.append({
                    "priority": "high",
                    "adverse_event": ae,
                    "signal_category": category,
                    "traceability_score": round(score, 3),
                    "action": (
                        f"Review documentation coverage for '{ae}' signals "
                        f"(current traceability score: {score:.0%})."
                    ),
                    "rationale": "Signal has low overall CTD section coverage.",
                })

        return json.dumps({
            "drug_name": drug_name,
            "signals_detected": len(flagged),
            "signals_traced": len(trace_data.get("results", [])),
            "overall_traceability_score": trace_data.get("overall_score"),
            "critical_gap_count": trace_data.get("critical_gap_count"),
            "fully_documented": trace_data.get("fully_documented"),
            "signal_summary": [
                {
                    "adverse_event": r["adverse_event"],
                    "signal_category": r["signal_category"],
                    "traceability_score": round(r["traceability_score"], 3),
                    "critical_gaps": r["critical_gaps"],
                }
                for r in trace_data.get("results", [])
            ],
            "priority_actions": priority_actions[: mcp_settings.max_results],
            "disclaimer": trace_data.get("disclaimer") or (
                "HEURISTIC SCREENING ONLY. Does not establish causality or "
                "predict regulatory outcomes. Requires expert review."
            ),
        }, indent=2)

    except Exception as exc:
        return json.dumps({"error": _backend_error(exc, "generate_regulatory_action_brief")})


# ---------------------------------------------------------------------------
# Register tools with FastMCP
# In FastMCP 2.x, mcp.tool()(fn) registers fn and returns a FunctionTool.
# We keep the _impl_ names callable for the smoke test.
# ---------------------------------------------------------------------------

mcp.tool()(_impl_detect_safety_signals)
mcp.tool()(_impl_calculate_prr)
mcp.tool()(_impl_get_signal_explanation)
mcp.tool()(_impl_check_ctd_readiness)
mcp.tool()(_impl_generate_gap_report)
mcp.tool()(_impl_generate_regulatory_action_brief)


# ---------------------------------------------------------------------------
# Built-in smoke test  (python server.py --test)
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    """Call every tool against the live backend and print results."""
    print("=" * 60)
    print("PharmaGuard MCP Server — Smoke Test")
    print(f"Backend: {mcp_settings.api_base_url}")
    print("=" * 60)

    tests = [
        ("detect_safety_signals",
         lambda: _impl_detect_safety_signals(drug_name="warfarin")),
        ("calculate_prr",
         lambda: _impl_calculate_prr(drug_name="warfarin")),
        ("get_signal_explanation",
         lambda: _impl_get_signal_explanation("warfarin", "bleeding")),
        ("check_ctd_readiness",
         lambda: _impl_check_ctd_readiness()),
        ("generate_gap_report",
         lambda: _impl_generate_gap_report("NDA")),
        ("generate_regulatory_action_brief",
         lambda: _impl_generate_regulatory_action_brief("warfarin")),
    ]

    passed = 0
    for name, fn in tests:
        print(f"\n[{name}]")
        try:
            raw = fn()
            result = json.loads(raw)
            if "error" in result:
                print(f"  FAIL: {result['error']}")
            else:
                keys = list(result.keys())[:4]
                print(f"  OK   top-level keys: {keys}")
                passed += 1
        except Exception as exc:
            print(f"  EXCEPTION: {exc}")

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{len(tests)} tools passed")
    if passed < len(tests):
        print(
            "Ensure the backend is running:\n"
            "  cd src/backend && uvicorn main:app --port 8000"
        )
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "--test" in sys.argv:
        _smoke_test()
    else:
        mcp.run()
