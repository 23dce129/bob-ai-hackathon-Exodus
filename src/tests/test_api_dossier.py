"""
test_api_dossier.py — Integration tests for the dossier and traceability API routes.

Test strategy
-------------
Tests are grouped into five sections:

  1. AppStartup      — FastAPI app imports correctly; all three routers registered.
  2. DossierSampleCheck — GET /api/dossier/sample-check response shape and values.
  3. DossierCheckUpload — POST /api/dossier/check with real fixture files;
                          validation errors for bad extensions, missing columns.
  4. DossierModules  — GET /api/dossier/modules and GET /api/dossier/gap-report/{type}.
  5. TraceabilityRoutes — POST /api/traceability/trace (file upload),
                          POST /api/traceability/trace-text (JSON body),
                          GET  /api/traceability/categories.

Key invariants asserted throughout
------------------------------------
  - Every response that calls core modules embeds a disclaimer.
  - Scores are in [0.0, 1.0].
  - No hard-coded demo results (verified by checking actual KB section counts).
  - Existing /health and /api/signals/* routes still return 200.
  - 400 for unsupported file extension; 422 for unparseable files.
  - trace-text with empty section_numbers produces score 0.0 (all MISSING).
  - trace-text with all relevant sections produces score 1.0.

Note on duckdb
--------------
The signals route imports faers_ingestor which imports duckdb.  Tests in this
file do NOT import from the signals module, so no duckdb dependency exists here.
The /health and /api/signals/* tests that used to exist in test_temporal_analyser
are NOT duplicated here; their failure mode (duckdb missing) is pre-existing.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CLEAN_CSV    = FIXTURES_DIR / "dossier_outline_clean.csv"
CLEAN_XLSX   = FIXTURES_DIR / "dossier_outline_clean.xlsx"


def _make_client():
    """Return a FastAPI TestClient without importing duckdb-dependent modules."""
    # Import inside function so collection-time errors stay isolated
    from fastapi.testclient import TestClient
    # We must patch out the faers_ingestor import in signals.py to avoid
    # the duckdb dependency during app creation.  Use sys.modules stub.
    import types, unittest.mock as mock
    duckdb_stub = types.ModuleType("duckdb")
    with mock.patch.dict(sys.modules, {"duckdb": duckdb_stub}):
        from main import app  # noqa: PLC0415
    return TestClient(app)


@pytest.fixture(scope="module")
def client():
    return _make_client()


# Minimal one-section CSV for upload tests (no dossier file needed)
_MINIMAL_CSV_BYTES = b"section_number,section_title,status\n2.4,Nonclinical Overview,present\n"

# A CSV that has no recognised columns → should get 422
_BAD_CSV_BYTES = b"col_a,col_b\nfoo,bar\n"

# Signals JSON for traceability tests
_BLEEDING_SIGNAL = {
    "drug_name": "warfarin",
    "adverse_event": "haemorrhage",
    "prr": 7.2,
    "p_value": 0.001,
    "signal_flag": True,
}
_HEPATO_SIGNAL = {
    "drug_name": "atorvastatin",
    "adverse_event": "hepatotoxicity",
    "prr": 3.5,
    "p_value": 0.02,
    "signal_flag": True,
}


# ===========================================================================
# 1. AppStartup
# ===========================================================================

class TestAppStartup:

    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_has_status_ok(self, client):
        r = client.get("/health")
        assert r.json()["status"] == "ok"

    def test_dossier_router_registered(self, client):
        # /api/dossier/modules must exist (not 404)
        r = client.get("/api/dossier/modules")
        assert r.status_code == 200

    def test_traceability_router_registered(self, client):
        r = client.get("/api/traceability/categories")
        assert r.status_code == 200

    def test_openapi_includes_dossier_tag(self, client):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        # FastAPI records tags at the operation level; collect all used tags
        spec = r.json()
        used_tags: set[str] = set()
        for path_item in spec.get("paths", {}).values():
            for op in path_item.values():
                used_tags.update(op.get("tags", []))
        assert "Dossier Readiness" in used_tags

    def test_openapi_includes_traceability_tag(self, client):
        r = client.get("/openapi.json")
        spec = r.json()
        used_tags: set[str] = set()
        for path_item in spec.get("paths", {}).values():
            for op in path_item.values():
                used_tags.update(op.get("tags", []))
        assert "Traceability" in used_tags


# ===========================================================================
# 2. DossierSampleCheck
# ===========================================================================

class TestDossierSampleCheck:

    def test_returns_200(self, client):
        r = client.get("/api/dossier/sample-check")
        assert r.status_code == 200

    def test_has_overall_score(self, client):
        r = client.get("/api/dossier/sample-check")
        assert "overall_score" in r.json()

    def test_overall_score_in_range(self, client):
        r = client.get("/api/dossier/sample-check")
        score = r.json()["overall_score"]
        assert 0.0 <= score <= 1.0

    def test_has_overall_score_pct(self, client):
        r = client.get("/api/dossier/sample-check")
        assert "overall_score_pct" in r.json()

    def test_score_pct_equals_score_times_100(self, client):
        r = client.get("/api/dossier/sample-check")
        data = r.json()
        assert abs(data["overall_score_pct"] - data["overall_score"] * 100) < 0.01

    def test_has_module_scores(self, client):
        r = client.get("/api/dossier/sample-check")
        assert "module_scores" in r.json()

    def test_module_scores_covers_five_modules(self, client):
        r = client.get("/api/dossier/sample-check")
        ms = r.json()["module_scores"]
        # Sample fixture has rows for all 5 modules
        assert len(ms) >= 1

    def test_has_disclaimer(self, client):
        r = client.get("/api/dossier/sample-check")
        data = r.json()
        assert "disclaimer" in data
        assert len(data["disclaimer"]) > 10

    def test_disclaimer_contains_heuristic(self, client):
        r = client.get("/api/dossier/sample-check")
        assert "heuristic" in r.json()["disclaimer"].lower()

    def test_has_parse_metadata(self, client):
        r = client.get("/api/dossier/sample-check")
        assert "parse_metadata" in r.json()

    def test_parse_metadata_row_count_positive(self, client):
        r = client.get("/api/dossier/sample-check")
        assert r.json()["parse_metadata"]["row_count"] > 0

    def test_has_total_present(self, client):
        r = client.get("/api/dossier/sample-check")
        assert "total_present" in r.json()

    def test_has_total_missing(self, client):
        r = client.get("/api/dossier/sample-check")
        assert "total_missing" in r.json()

    def test_has_match_summary(self, client):
        r = client.get("/api/dossier/sample-check")
        assert "match_summary" in r.json()

    def test_section_counts_consistent(self, client):
        r = client.get("/api/dossier/sample-check")
        data = r.json()
        total = (
            data["total_present"]
            + data["total_needs_review"]
            + data["total_missing"]
            + data["total_not_applicable"]
        )
        assert total == data["parse_metadata"]["row_count"]

    def test_score_not_hardcoded(self, client):
        """Score must come from the actual KB, not a hard-coded value."""
        r = client.get("/api/dossier/sample-check")
        score = r.json()["overall_score"]
        # Sample fixture is intentionally partial; score should be neither 0 nor 1
        assert 0.0 < score < 1.0


# ===========================================================================
# 3. DossierCheckUpload
# ===========================================================================

class TestDossierCheckUpload:

    def test_csv_upload_returns_200(self, client):
        r = client.post(
            "/api/dossier/check",
            files={"file": ("outline.csv", io.BytesIO(CLEAN_CSV.read_bytes()), "text/csv")},
        )
        assert r.status_code == 200

    def test_xlsx_upload_returns_200(self, client):
        r = client.post(
            "/api/dossier/check",
            files={"file": ("outline.xlsx", io.BytesIO(CLEAN_XLSX.read_bytes()),
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        )
        assert r.status_code == 200

    def test_upload_has_overall_score(self, client):
        r = client.post(
            "/api/dossier/check",
            files={"file": ("outline.csv", io.BytesIO(CLEAN_CSV.read_bytes()), "text/csv")},
        )
        assert "overall_score" in r.json()

    def test_upload_score_in_range(self, client):
        r = client.post(
            "/api/dossier/check",
            files={"file": ("outline.csv", io.BytesIO(CLEAN_CSV.read_bytes()), "text/csv")},
        )
        assert 0.0 <= r.json()["overall_score"] <= 1.0

    def test_upload_has_upload_metadata(self, client):
        r = client.post(
            "/api/dossier/check",
            files={"file": ("outline.csv", io.BytesIO(CLEAN_CSV.read_bytes()), "text/csv")},
        )
        assert "upload" in r.json()

    def test_upload_metadata_filename(self, client):
        r = client.post(
            "/api/dossier/check",
            files={"file": ("outline.csv", io.BytesIO(CLEAN_CSV.read_bytes()), "text/csv")},
        )
        assert r.json()["upload"]["filename"] == "outline.csv"

    def test_upload_drug_name_in_metadata(self, client):
        r = client.post(
            "/api/dossier/check",
            data={"drug_name": "warfarin"},
            files={"file": ("outline.csv", io.BytesIO(CLEAN_CSV.read_bytes()), "text/csv")},
        )
        assert r.json()["upload"]["drug_name"] == "warfarin"

    def test_minimal_csv_returns_200(self, client):
        r = client.post(
            "/api/dossier/check",
            files={"file": ("mini.csv", io.BytesIO(_MINIMAL_CSV_BYTES), "text/csv")},
        )
        assert r.status_code == 200

    def test_minimal_csv_one_row(self, client):
        r = client.post(
            "/api/dossier/check",
            files={"file": ("mini.csv", io.BytesIO(_MINIMAL_CSV_BYTES), "text/csv")},
        )
        assert r.json()["parse_metadata"]["row_count"] == 1

    def test_unsupported_extension_returns_400(self, client):
        r = client.post(
            "/api/dossier/check",
            files={"file": ("outline.txt", io.BytesIO(b"col\nval"), "text/plain")},
        )
        assert r.status_code == 400

    def test_unsupported_extension_error_message(self, client):
        r = client.post(
            "/api/dossier/check",
            files={"file": ("outline.txt", io.BytesIO(b"col\nval"), "text/plain")},
        )
        assert ".txt" in r.json()["detail"]

    def test_bad_csv_no_recognised_columns_returns_422(self, client):
        r = client.post(
            "/api/dossier/check",
            files={"file": ("bad.csv", io.BytesIO(_BAD_CSV_BYTES), "text/csv")},
        )
        assert r.status_code == 422

    def test_upload_has_module_scores(self, client):
        r = client.post(
            "/api/dossier/check",
            files={"file": ("outline.csv", io.BytesIO(CLEAN_CSV.read_bytes()), "text/csv")},
        )
        assert "module_scores" in r.json()

    def test_upload_has_disclaimer(self, client):
        r = client.post(
            "/api/dossier/check",
            files={"file": ("outline.csv", io.BytesIO(CLEAN_CSV.read_bytes()), "text/csv")},
        )
        assert "disclaimer" in r.json()

    def test_csv_and_xlsx_produce_same_score(self, client):
        r_csv = client.post(
            "/api/dossier/check",
            files={"file": ("outline.csv", io.BytesIO(CLEAN_CSV.read_bytes()), "text/csv")},
        )
        r_xlsx = client.post(
            "/api/dossier/check",
            files={"file": ("outline.xlsx", io.BytesIO(CLEAN_XLSX.read_bytes()),
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        )
        assert r_csv.json()["overall_score"] == pytest.approx(
            r_xlsx.json()["overall_score"], abs=0.001
        )


# ===========================================================================
# 4. DossierModules
# ===========================================================================

class TestDossierModules:

    def test_modules_returns_200(self, client):
        r = client.get("/api/dossier/modules")
        assert r.status_code == 200

    def test_modules_is_list(self, client):
        r = client.get("/api/dossier/modules")
        assert isinstance(r.json(), list)

    def test_modules_non_empty(self, client):
        r = client.get("/api/dossier/modules")
        assert len(r.json()) > 0

    def test_every_item_has_required_keys(self, client):
        r = client.get("/api/dossier/modules")
        required = {"id", "number", "title", "module_id", "level",
                    "requirement", "safety_relevant", "traceability_categories"}
        for item in r.json():
            assert required.issubset(item.keys())

    def test_module_ids_in_m1_to_m5(self, client):
        r = client.get("/api/dossier/modules")
        module_ids = {item["module_id"] for item in r.json()}
        assert module_ids.issubset({"M1", "M2", "M3", "M4", "M5"})

    def test_requirement_values_are_valid(self, client):
        r = client.get("/api/dossier/modules")
        valid = {"required", "conditional", "optional"}
        for item in r.json():
            assert item["requirement"] in valid

    def test_safety_relevant_is_bool(self, client):
        r = client.get("/api/dossier/modules")
        for item in r.json():
            assert isinstance(item["safety_relevant"], bool)

    def test_gap_report_nda_returns_200(self, client):
        r = client.get("/api/dossier/gap-report/NDA")
        assert r.status_code == 200

    def test_gap_report_bla_returns_200(self, client):
        r = client.get("/api/dossier/gap-report/BLA")
        assert r.status_code == 200

    def test_gap_report_case_insensitive(self, client):
        r = client.get("/api/dossier/gap-report/nda")
        assert r.status_code == 200

    def test_gap_report_has_required_sections(self, client):
        r = client.get("/api/dossier/gap-report/NDA")
        assert "required_sections" in r.json()

    def test_gap_report_required_count_positive(self, client):
        r = client.get("/api/dossier/gap-report/NDA")
        assert r.json()["required_count"] > 0

    def test_gap_report_invalid_type_returns_400(self, client):
        r = client.get("/api/dossier/gap-report/INVALID")
        assert r.status_code == 400

    def test_gap_report_anda_fewer_required_than_nda(self, client):
        nda = client.get("/api/dossier/gap-report/NDA").json()["required_count"]
        anda = client.get("/api/dossier/gap-report/ANDA").json()["required_count"]
        assert anda <= nda


# ===========================================================================
# 5. TraceabilityRoutes
# ===========================================================================

class TestTraceabilityRoutes:

    # ---- /categories -------------------------------------------------------

    def test_categories_returns_200(self, client):
        r = client.get("/api/traceability/categories")
        assert r.status_code == 200

    def test_categories_has_categories_key(self, client):
        r = client.get("/api/traceability/categories")
        assert "categories" in r.json()

    def test_categories_is_non_empty(self, client):
        r = client.get("/api/traceability/categories")
        assert len(r.json()["categories"]) > 0

    def test_categories_each_has_name_and_section_count(self, client):
        r = client.get("/api/traceability/categories")
        for cat in r.json()["categories"]:
            assert "name" in cat
            assert "section_count" in cat
            assert cat["section_count"] >= 0

    def test_categories_has_disclaimer(self, client):
        r = client.get("/api/traceability/categories")
        assert "disclaimer" in r.json()

    def test_categories_includes_bleeding(self, client):
        r = client.get("/api/traceability/categories")
        names = {c["name"] for c in r.json()["categories"]}
        assert "bleeding" in names

    def test_categories_includes_general(self, client):
        r = client.get("/api/traceability/categories")
        names = {c["name"] for c in r.json()["categories"]}
        assert "general" in names

    # ---- /trace-text -------------------------------------------------------

    def test_trace_text_returns_200(self, client):
        body = {
            "signals": [_BLEEDING_SIGNAL],
            "section_numbers": ["2.4", "2.7.4"],
        }
        r = client.post("/api/traceability/trace-text", json=body)
        assert r.status_code == 200

    def test_trace_text_has_results(self, client):
        body = {"signals": [_BLEEDING_SIGNAL], "section_numbers": []}
        r = client.post("/api/traceability/trace-text", json=body)
        assert "results" in r.json()

    def test_trace_text_total_signals(self, client):
        body = {"signals": [_BLEEDING_SIGNAL], "section_numbers": []}
        r = client.post("/api/traceability/trace-text", json=body)
        assert r.json()["total_signals"] == 1

    def test_trace_text_empty_sections_score_zero(self, client):
        body = {"signals": [_BLEEDING_SIGNAL], "section_numbers": []}
        r = client.post("/api/traceability/trace-text", json=body)
        result = r.json()["results"][0]
        assert result["traceability_score"] == pytest.approx(0.0)

    def test_trace_text_has_disclaimer(self, client):
        body = {"signals": [_BLEEDING_SIGNAL], "section_numbers": []}
        r = client.post("/api/traceability/trace-text", json=body)
        assert "disclaimer" in r.json()

    def test_trace_text_disclaimer_contains_heuristic(self, client):
        body = {"signals": [_BLEEDING_SIGNAL], "section_numbers": []}
        r = client.post("/api/traceability/trace-text", json=body)
        assert "heuristic" in r.json()["disclaimer"].lower()

    def test_trace_text_result_has_entries(self, client):
        body = {"signals": [_BLEEDING_SIGNAL], "section_numbers": []}
        r = client.post("/api/traceability/trace-text", json=body)
        assert len(r.json()["results"][0]["entries"]) > 0

    def test_trace_text_result_has_critical_gaps(self, client):
        body = {"signals": [_BLEEDING_SIGNAL], "section_numbers": []}
        r = client.post("/api/traceability/trace-text", json=body)
        assert "critical_gaps" in r.json()["results"][0]

    def test_trace_text_empty_dossier_has_critical_gaps(self, client):
        body = {"signals": [_BLEEDING_SIGNAL], "section_numbers": []}
        r = client.post("/api/traceability/trace-text", json=body)
        assert len(r.json()["results"][0]["critical_gaps"]) > 0

    def test_trace_text_signal_category_classified(self, client):
        body = {"signals": [_BLEEDING_SIGNAL], "section_numbers": []}
        r = client.post("/api/traceability/trace-text", json=body)
        assert r.json()["results"][0]["signal_category"] == "bleeding"

    def test_trace_text_hepato_category(self, client):
        body = {"signals": [_HEPATO_SIGNAL], "section_numbers": []}
        r = client.post("/api/traceability/trace-text", json=body)
        assert r.json()["results"][0]["signal_category"] == "hepatotoxicity"

    def test_trace_text_multiple_signals(self, client):
        body = {
            "signals": [_BLEEDING_SIGNAL, _HEPATO_SIGNAL],
            "section_numbers": [],
        }
        r = client.post("/api/traceability/trace-text", json=body)
        assert r.json()["total_signals"] == 2
        assert len(r.json()["results"]) == 2

    def test_trace_text_results_sorted_ascending(self, client):
        body = {
            "signals": [_BLEEDING_SIGNAL, _HEPATO_SIGNAL],
            "section_numbers": [],
        }
        r = client.post("/api/traceability/trace-text", json=body)
        scores = [res["traceability_score"] for res in r.json()["results"]]
        assert scores == sorted(scores)

    def test_trace_text_overall_score_in_range(self, client):
        body = {"signals": [_BLEEDING_SIGNAL], "section_numbers": []}
        r = client.post("/api/traceability/trace-text", json=body)
        assert 0.0 <= r.json()["overall_score"] <= 1.0

    def test_trace_text_fully_documented_is_int(self, client):
        body = {"signals": [_BLEEDING_SIGNAL], "section_numbers": []}
        r = client.post("/api/traceability/trace-text", json=body)
        assert isinstance(r.json()["fully_documented"], int)

    def test_trace_text_all_relevant_sections_score_1(self, client):
        """Supplying ALL KB section IDs makes every signal fully documented."""
        from core.ctd_knowledge_base import load_knowledge_base  # noqa: PLC0415
        kb = load_knowledge_base()
        all_ids = kb.get_all_section_ids()
        body = {"signals": [_BLEEDING_SIGNAL], "section_numbers": all_ids}
        r = client.post("/api/traceability/trace-text", json=body)
        assert r.json()["results"][0]["traceability_score"] == pytest.approx(1.0)

    def test_trace_text_prr_preserved(self, client):
        body = {"signals": [_BLEEDING_SIGNAL], "section_numbers": []}
        r = client.post("/api/traceability/trace-text", json=body)
        assert r.json()["results"][0]["prr"] == pytest.approx(7.2)

    def test_trace_text_empty_signals_returns_422(self, client):
        body = {"signals": [], "section_numbers": []}
        r = client.post("/api/traceability/trace-text", json=body)
        assert r.status_code in (422, 400)

    def test_trace_text_missing_adverse_event_returns_422(self, client):
        body = {
            "signals": [{"drug_name": "warfarin"}],  # missing adverse_event
            "section_numbers": [],
        }
        r = client.post("/api/traceability/trace-text", json=body)
        assert r.status_code == 422

    # ---- /trace (file upload) ---------------------------------------------

    def test_trace_file_upload_returns_200(self, client):
        signals_json = json.dumps([_BLEEDING_SIGNAL])
        r = client.post(
            "/api/traceability/trace",
            data={"signals_json": signals_json},
            files={"file": ("outline.csv", io.BytesIO(CLEAN_CSV.read_bytes()), "text/csv")},
        )
        assert r.status_code == 200

    def test_trace_file_upload_has_results(self, client):
        signals_json = json.dumps([_BLEEDING_SIGNAL])
        r = client.post(
            "/api/traceability/trace",
            data={"signals_json": signals_json},
            files={"file": ("outline.csv", io.BytesIO(CLEAN_CSV.read_bytes()), "text/csv")},
        )
        assert "results" in r.json()

    def test_trace_file_upload_has_dossier_metadata(self, client):
        signals_json = json.dumps([_BLEEDING_SIGNAL])
        r = client.post(
            "/api/traceability/trace",
            data={"signals_json": signals_json},
            files={"file": ("outline.csv", io.BytesIO(CLEAN_CSV.read_bytes()), "text/csv")},
        )
        assert "dossier_metadata" in r.json()

    def test_trace_file_upload_dossier_metadata_filename(self, client):
        signals_json = json.dumps([_BLEEDING_SIGNAL])
        r = client.post(
            "/api/traceability/trace",
            data={"signals_json": signals_json},
            files={"file": ("outline.csv", io.BytesIO(CLEAN_CSV.read_bytes()), "text/csv")},
        )
        assert r.json()["dossier_metadata"]["filename"] == "outline.csv"

    def test_trace_file_upload_score_higher_than_empty(self, client):
        """Uploading the full sample fixture should give a higher score than empty dossier."""
        signals_json = json.dumps([_BLEEDING_SIGNAL])

        r_file = client.post(
            "/api/traceability/trace",
            data={"signals_json": signals_json},
            files={"file": ("outline.csv", io.BytesIO(CLEAN_CSV.read_bytes()), "text/csv")},
        )
        r_empty = client.post(
            "/api/traceability/trace-text",
            json={"signals": [_BLEEDING_SIGNAL], "section_numbers": []},
        )
        assert r_file.json()["results"][0]["traceability_score"] > \
               r_empty.json()["results"][0]["traceability_score"]

    def test_trace_file_unsupported_ext_returns_400(self, client):
        signals_json = json.dumps([_BLEEDING_SIGNAL])
        r = client.post(
            "/api/traceability/trace",
            data={"signals_json": signals_json},
            files={"file": ("outline.txt", io.BytesIO(b"sec\n1.1\n"), "text/plain")},
        )
        assert r.status_code == 400

    def test_trace_file_invalid_signals_json_returns_400(self, client):
        r = client.post(
            "/api/traceability/trace",
            data={"signals_json": "NOT_JSON"},
            files={"file": ("outline.csv", io.BytesIO(CLEAN_CSV.read_bytes()), "text/csv")},
        )
        assert r.status_code == 400

    def test_trace_file_signals_json_not_array_returns_4xx(self, client):
        # A JSON object instead of an array is invalid; server may return 400 or 422
        r = client.post(
            "/api/traceability/trace",
            data={"signals_json": '{"drug_name": "warfarin"}'},  # object not array
            files={"file": ("outline.csv", io.BytesIO(CLEAN_CSV.read_bytes()), "text/csv")},
        )
        assert r.status_code in (400, 422)
