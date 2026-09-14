Good. I now have a complete picture of every file in the repo. Here is the full, detailed implementation plan.

---

# PharmaGuard AI — Detailed Implementation Plan

> **Nothing will be implemented until you approve.** This is planning only.

---

## Repository Map After Full Implementation

```
src/
├── .env.example                         ← expanded with new variables
├── README.md                            ← updated with src layout
│
├── backend/
│   ├── main.py                          ← FastAPI entry point
│   ├── requirements.txt
│   ├── config.py                        ← env var loading
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── signals.py               ← /api/signals/*
│   │   │   ├── dossier.py               ← /api/dossier/*
│   │   │   └── traceability.py          ← /api/traceability/*
│   │   └── models.py                    ← Pydantic schemas
│   ├── core/
│   │   ├── __init__.py
│   │   ├── faers_ingestor.py            ← Phase 2
│   │   ├── prr_calculator.py            ← Phase 3
│   │   ├── signal_clusterer.py          ← Phase 4
│   │   ├── ctd_knowledge_base.py        ← Phase 5
│   │   ├── dossier_parser.py            ← Phase 6
│   │   ├── readiness_scorer.py          ← Phase 7
│   │   ├── traceability_engine.py       ← Phase 8
│   │   └── watsonx_client.py            ← watsonx.ai Granite calls
│   └── data/
│       ├── faers_sample.csv             ← bundled FAERS sample (~500 rows)
│       └── ich_m4_structure.json        ← ICH M4 CTD knowledge base
│
├── mcp-server/
│   ├── server.py                        ← Phase 10
│   ├── requirements.txt
│   └── tools/
│       ├── __init__.py
│       ├── detect_signals.py
│       ├── check_dossier.py
│       └── trace_signal.py
│
├── frontend/
│   ├── package.json                     ← Phase 11
│   ├── vite.config.ts
│   ├── tailwind.config.ts
│   ├── index.html
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── api/client.ts                ← typed API calls to backend
│       ├── pages/
│       │   ├── SignalDetection.tsx
│       │   ├── DossierReadiness.tsx
│       │   └── Traceability.tsx
│       └── components/
│           ├── SignalCard.tsx
│           ├── ModuleProgress.tsx
│           ├── GapTable.tsx
│           └── TraceabilityMap.tsx
│
└── tests/
    ├── test_faers_ingestor.py
    ├── test_prr_calculator.py
    ├── test_signal_clusterer.py
    ├── test_ctd_knowledge_base.py
    ├── test_dossier_parser.py
    ├── test_readiness_scorer.py
    ├── test_traceability_engine.py
    └── test_api_routes.py
```

---

## Phase 1 — Repository and Development Environment

### Goal
A working Python virtual environment, a running (but empty) FastAPI server, and a confirmed development loop before any domain logic is written.

### Files to Create / Change
| File | Action |
|---|---|
| `src/backend/requirements.txt` | Create |
| `src/backend/main.py` | Create |
| `src/backend/config.py` | Create |
| `src/mcp-server/requirements.txt` | Create |
| `src/frontend/package.json` | Create (scaffold only) |
| `src/.env.example` | Update with new variables |

### Dependencies
```
# src/backend/requirements.txt
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
python-dotenv>=1.0.0
pydantic>=2.6.0
pandas>=2.2.0
scipy>=1.13.0
scikit-learn>=1.4.0
ibm-watsonx-ai>=1.0.0
httpx>=0.27.0        # for MCP server → backend HTTP calls
pytest>=8.2.0
pytest-asyncio>=0.23.0
```
```
# src/mcp-server/requirements.txt
fastmcp>=0.4.0
httpx>=0.27.0
python-dotenv>=1.0.0
```

### Implementation Steps
1. Create `src/backend/` directory structure with `__init__.py` files
2. Write `src/backend/config.py` — loads `.env` variables via `python-dotenv`, exposes a `Settings` dataclass
3. Write `src/backend/main.py` — creates a FastAPI app, mounts an `/api` router, adds a `/health` endpoint that returns `{"status": "ok"}`
4. Write `src/.env.example` with all project-specific variables (watsonx keys, API port, MCP port)
5. Scaffold frontend with `npm create vite@latest frontend -- --template react-ts` inside `src/`

### Test
```bash
cd src/backend
python -m venv .venv && .venv\Scripts\activate  # Windows
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
# In browser: GET http://localhost:8000/health → {"status":"ok"}
```

### Expected Output
- `GET /health` returns `{"status": "ok", "version": "0.1.0"}`
- FastAPI auto-docs at `http://localhost:8000/docs` load with no errors

### Common Failure Points
- **`ModuleNotFoundError`** — running `uvicorn` from the wrong directory; always run from `src/backend/`
- **`ibm-watsonx-ai` install fails on Windows** — requires `pip install ibm-watsonx-ai` with a recent pip; upgrade pip first with `python -m pip install --upgrade pip`
- **Port 8000 already in use** — change `APP_PORT` in `.env` and pass `--port $APP_PORT` to uvicorn

---

## Phase 2 — FAERS/AEMS Ingestion and Normalization

### Goal
A Python module that reads a FAERS-format CSV, validates it, normalizes column names and drug names to lowercase, and returns a clean `pandas.DataFrame` ready for statistical analysis. A bundled sample dataset (≈500 rows) is committed to the repo so the project works offline with no external API calls.

### Files to Create / Change
| File | Action |
|---|---|
| `src/backend/core/faers_ingestor.py` | Create |
| `src/backend/data/faers_sample.csv` | Create (bundled sample) |
| `src/tests/test_faers_ingestor.py` | Create |

### What FAERS Data Looks Like
The FDA publishes FAERS quarterly. The relevant columns for signal detection are:

| Column | Meaning |
|---|---|
| `primaryid` | Unique report identifier |
| `caseid` | Case identifier (may have duplicates across quarters) |
| `drug_name` | Drug involved |
| `pt` | Preferred Term (MedDRA adverse event term) |
| `outc_cod` | Outcome code (DE=death, HO=hospitalization, etc.) |
| `report_dt` | Report date |
| `age` | Patient age |
| `sex` | Patient sex |

We will build a **minimal schema** that covers exactly what PRR and clustering need — not the full FAERS schema.

### Implementation Steps
1. Define the expected schema as a Python dict: required columns, expected dtypes, allowed null columns
2. Write `load_faers(filepath: str) -> pd.DataFrame`:
   - Read CSV with `pandas.read_csv`
   - Validate required columns exist (raise `ValueError` with a clear message if not)
   - Normalize: lowercase all column names, strip whitespace from string columns, lowercase `drug_name` and `pt`
   - Parse `report_dt` to `datetime`, coerce errors to `NaT`
   - Drop rows where both `drug_name` and `pt` are null (completely unusable)
   - Return clean DataFrame
3. Write `get_drug_event_counts(df: pd.DataFrame) -> pd.DataFrame`:
   - Returns a 2-column summary: `(drug_name, pt)` pairs with their report count
   - Used by PRR calculator in Phase 3
4. Create `src/backend/data/faers_sample.csv` — a manually constructed 500-row sample with realistic drug names (aspirin, ibuprofen, metformin, warfarin, atorvastatin) and real MedDRA preferred terms (nausea, hepatotoxicity, bleeding, dizziness, rash, etc.)

### Tests (`test_faers_ingestor.py`)
```python
# What we will test:
- load_faers() with valid CSV → returns DataFrame with correct columns
- load_faers() with missing required column → raises ValueError with column name
- load_faers() with mixed-case column headers → normalizes correctly
- drug_name and pt are lowercased after load
- get_drug_event_counts() returns correct counts for known sample
- Running load_faers() on the bundled faers_sample.csv → succeeds
```

### Expected Output
```python
df = load_faers("data/faers_sample.csv")
# df.shape → (approx 480, 8) — some rows dropped if fully null
# df.columns → ['primaryid','caseid','drug_name','pt','outc_cod','report_dt','age','sex']
# df['drug_name'].unique() → ['aspirin', 'ibuprofen', 'metformin', ...]
```

### Common Failure Points
- **Mixed encoding in real FAERS files** — FAERS uses Latin-1 encoding; `read_csv(..., encoding='latin-1')` is required for real files (sample CSV we create will be clean UTF-8)
- **Column name typos** — real FAERS has columns like `drugname` (no underscore) in older formats; the normalizer must handle both
- **Empty DataFrame after filtering** — if the sample CSV has a formatting error, all rows get dropped silently; the ingestor should warn if more than 20% of rows are dropped

---

## Phase 3 — PRR and Statistical Signal Detection

### Goal
Calculate **Proportional Reporting Ratio (PRR)** for every drug–event pair in the dataset. Flag pairs where PRR ≥ 2, case count ≥ 3, and chi-squared p-value < 0.05 as potential safety signals. These three thresholds are the standard EMA/FDA criteria.

### Background (so you understand the maths)
PRR measures whether a drug is reported with an adverse event more often than all other drugs combined. For a drug D and event E:

```
          (cases of D with E / all cases of D)
PRR = ─────────────────────────────────────────────────────
      (cases of other drugs with E / all cases of other drugs)
```

It's derived from a 2×2 contingency table and is the industry-standard first-pass signal detection method used by regulatory agencies.

### Files to Create / Change
| File | Action |
|---|---|
| `src/backend/core/prr_calculator.py` | Create |
| `src/tests/test_prr_calculator.py` | Create |

### Implementation Steps
1. Write `build_contingency_table(df, drug, event) -> dict`:
   - `a` = reports of `drug` with `event`
   - `b` = reports of `drug` without `event`
   - `c` = reports of other drugs with `event`
   - `d` = reports of other drugs without `event`
2. Write `calculate_prr(a, b, c, d) -> dict`:
   - Returns `{"prr": float, "chi2": float, "p_value": float, "case_count": int}`
   - Uses `scipy.stats.chi2_contingency` for chi-squared
   - Returns `prr=None` if `(c * (a+b)) == 0` (division by zero guard)
3. Write `calculate_all_signals(df) -> pd.DataFrame`:
   - Iterates over all unique `(drug_name, pt)` pairs
   - Calls `build_contingency_table` + `calculate_prr` for each
   - Returns DataFrame with columns: `drug_name, pt, case_count, prr, chi2, p_value`
4. Write `flag_signals(signal_df, min_prr=2.0, min_cases=3, max_pval=0.05) -> pd.DataFrame`:
   - Filters to rows meeting all three criteria
   - Adds a `signal_strength` column: "strong" if PRR ≥ 5, "moderate" if PRR ≥ 2
   - Sorts by PRR descending

### Tests (`test_prr_calculator.py`)
```python
# What we will test:
- build_contingency_table() with known counts → correct a/b/c/d values
- calculate_prr() with known a/b/c/d → correct PRR (compare to hand-calculated value)
- calculate_prr() with c=0 → returns prr=None without crashing
- flag_signals() on synthetic data → only rows meeting all 3 thresholds appear
- flag_signals() with PRR=1.5 drug → NOT flagged
- flag_signals() with n=2 cases → NOT flagged even if PRR is high
- calculate_all_signals() on faers_sample.csv → returns DataFrame with no NaN in key columns
```

### Expected Output
```python
signals = flag_signals(calculate_all_signals(df))
# signals.columns → ['drug_name', 'pt', 'case_count', 'prr', 'chi2', 'p_value', 'signal_strength']
# Example row: drug_name='warfarin', pt='bleeding', case_count=15, prr=8.3, signal_strength='strong'
```

### Common Failure Points
- **Divide by zero** — when a drug has 100% reporting of an event (b=0) or no other drug has ever reported the event (c=0); both must be guarded
- **Very slow on large datasets** — iterating over all pairs is O(drugs × events); for the sample dataset this is fine, but the function should be documented as not suitable for full FAERS without vectorization
- **Chi-squared invalid** — `chi2_contingency` requires expected cell counts ≥ 5; for small counts use Fisher's exact test instead (`scipy.stats.fisher_exact`); the implementation should switch automatically

---

## Phase 4 — Adverse-Event Clustering and Trend Analysis

### Goal
Group adverse events into clusters by medical concept similarity, so that instead of showing 200 individual event terms, we surface meaningful groupings like "hepatic events", "bleeding events", "CNS effects". Also detect whether a signal is **emerging** (reports increasing over time) vs. established.

### Files to Create / Change
| File | Action |
|---|---|
| `src/backend/core/signal_clusterer.py` | Create |
| `src/tests/test_signal_clusterer.py` | Create |

### Implementation Steps
1. **System Organ Class (SOC) grouping** (primary method, no ML required):
   - Create a dictionary mapping known MedDRA Preferred Terms → their System Organ Class (SOC). E.g., "hepatotoxicity" → "Hepatobiliary disorders", "bleeding" → "Vascular disorders"
   - Write `assign_soc(pt: str) -> str`: looks up in the dictionary, returns "Other" for unknowns
   - This gives meaningful clusters without any model training
2. **K-Means clustering on TF-IDF vectors** (secondary, for unknown/new terms):
   - Write `cluster_events_tfidf(event_list: list[str], n_clusters: int = 8) -> dict[str, int]`
   - Uses `sklearn.feature_extraction.text.TfidfVectorizer` + `sklearn.cluster.KMeans`
   - Returns a mapping of event term → cluster ID
3. **Trend detection**:
   - Write `calculate_trend(df, drug, event) -> dict`:
     - Splits the dataset into two halves by `report_dt`
     - Compares report rate in first half vs. second half
     - Returns `{"trend": "emerging" | "stable" | "declining", "rate_change_pct": float}`
   - If `report_dt` is mostly null in sample data, return `"trend": "unknown"`
4. Write `enrich_signals(signal_df, raw_df) -> pd.DataFrame`:
   - Takes flagged signals from Phase 3
   - Adds `soc` (System Organ Class), `cluster_id`, and `trend` columns
   - This is the full enriched signal object

### Tests (`test_signal_clusterer.py`)
```python
# What we will test:
- assign_soc("hepatotoxicity") → "Hepatobiliary disorders"
- assign_soc("unknown_term_xyz") → "Other"
- cluster_events_tfidf() with 10 terms → returns dict with all terms as keys
- calculate_trend() with equal halves → "stable"
- calculate_trend() with increasing second half → "emerging"
- enrich_signals() adds soc, cluster_id, trend without dropping rows
```

### Expected Output
```python
enriched = enrich_signals(flagged_signals, raw_df)
# Extra columns: 'soc', 'cluster_id', 'trend'
# Example: warfarin/bleeding → soc="Vascular disorders", trend="emerging"
```

### Common Failure Points
- **TF-IDF on very short strings** — single-word medical terms produce sparse vectors; use `analyzer='char_wb'` with n-gram range `(3, 5)` for character n-grams instead of word tokens, which works much better on medical terminology
- **K-Means requires pre-specifying k** — use the elbow method or just fix k=8 (matches ICH SOC count); document this assumption
- **report_dt is NaT** — trend detection must handle missing dates gracefully; return `"unknown"` rather than crashing

---

## Phase 5 — ICH M4 CTD Requirements Knowledge Base

### Goal
Build a **machine-readable representation of the ICH M4 CTD structure** (Modules 1–5 with all required sections). This JSON file is the authoritative reference that the dossier checker (Phase 6) compares against. It is also the basis for the traceability engine (Phase 8).

This phase is **data authoring**, not programming. The output is a well-structured JSON file.

### Files to Create / Change
| File | Action |
|---|---|
| `src/backend/data/ich_m4_structure.json` | Create |
| `src/backend/core/ctd_knowledge_base.py` | Create |
| `src/tests/test_ctd_knowledge_base.py` | Create |

### JSON Structure
```json
{
  "version": "ICH M4 CTD Q1 2016",
  "modules": [
    {
      "id": "M1",
      "name": "Administrative Information and Prescribing Information",
      "required": true,
      "sections": [
        {
          "id": "1.1",
          "name": "Comprehensive Table of Contents",
          "required": true,
          "safety_relevant": false,
          "description": "Complete listing of all documents in the dossier"
        },
        {
          "id": "1.2",
          "name": "Application Form",
          "required": true,
          "safety_relevant": false,
          "description": "Regulatory submission form"
        }
        // ... all M1 sections
      ]
    },
    {
      "id": "M2",
      "name": "Common Technical Document Summaries",
      "sections": [
        { "id": "2.4", "name": "Nonclinical Overview", "required": true, "safety_relevant": true },
        { "id": "2.5", "name": "Clinical Overview", "required": true, "safety_relevant": true },
        { "id": "2.6", "name": "Nonclinical Written and Tabulated Summaries", "required": true, "safety_relevant": true },
        { "id": "2.7", "name": "Clinical Summary", "required": true, "safety_relevant": true }
      ]
    },
    {
      "id": "M3",
      "name": "Quality",
      "sections": [
        { "id": "3.1", "name": "Table of Contents", "required": true, "safety_relevant": false },
        { "id": "3.2", "name": "Body of Data", "required": true, "safety_relevant": false },
        { "id": "3.2.S", "name": "Drug Substance", "required": true, "safety_relevant": false }
        // ... subsections
      ]
    },
    {
      "id": "M4",
      "name": "Nonclinical Study Reports",
      "sections": [
        { "id": "4.1", "name": "Table of Contents", "required": true, "safety_relevant": false },
        { "id": "4.2", "name": "Study Reports", "required": true, "safety_relevant": true },
        { "id": "4.2.1", "name": "Pharmacology", "required": true, "safety_relevant": false },
        { "id": "4.2.2", "name": "Pharmacokinetics", "required": true, "safety_relevant": false },
        { "id": "4.2.3", "name": "Toxicology", "required": true, "safety_relevant": true }
      ]
    },
    {
      "id": "M5",
      "name": "Clinical Study Reports",
      "sections": [
        { "id": "5.1", "name": "Table of Contents", "required": true, "safety_relevant": false },
        { "id": "5.2", "name": "Tabular Listing of Clinical Studies", "required": true, "safety_relevant": false },
        { "id": "5.3", "name": "Clinical Study Reports", "required": true, "safety_relevant": true },
        { "id": "5.3.1", "name": "Reports of Biopharmaceutic Studies", "required": false, "safety_relevant": false },
        { "id": "5.3.5", "name": "Reports of Efficacy and Safety Studies", "required": true, "safety_relevant": true },
        { "id": "5.3.6", "name": "Reports of Post-Marketing Experience", "required": false, "safety_relevant": true }
      ]
    }
  ],
  "signal_to_section_map": {
    "hepatotoxicity": ["2.4", "2.6.6", "4.2.3.4", "5.3.5"],
    "bleeding":       ["2.4", "2.7.4", "4.2.3", "5.3.5"],
    "cardiac":        ["2.4", "2.6.6", "4.2.3", "5.3.5"],
    "renal":          ["2.4", "2.6.6", "4.2.3.6", "5.3.5"],
    "neurological":   ["2.4", "2.6.6", "4.2.3", "5.3.5"],
    "carcinogenicity":["2.4", "2.6.6", "4.2.3.7", "4.2.3.8"],
    "reproductive":   ["2.4", "2.6.6", "4.2.3.5"],
    "anaphylaxis":    ["2.4", "2.7.4", "5.3.5.4"],
    "infection":      ["2.4", "2.7.4", "5.3.5"]
  }
}
```

The `signal_to_section_map` is what enables Phase 8 (traceability). It maps **safety signal categories** (by keyword) to the CTD section IDs that must contain supporting documentation.

### Implementation Steps (`ctd_knowledge_base.py`)
1. Write `load_ctd_structure(filepath=None) -> dict` — loads and parses `ich_m4_structure.json`
2. Write `get_module(ctd, module_id: str) -> dict` — returns a module by ID (e.g., "M2")
3. Write `get_all_required_sections(ctd) -> list[dict]` — flat list of all sections where `"required": true`
4. Write `get_safety_relevant_sections(ctd) -> list[dict]` — sections where `"safety_relevant": true`
5. Write `get_sections_for_signal(ctd, signal_keyword: str) -> list[dict]` — looks up `signal_to_section_map`, returns full section objects

### Tests (`test_ctd_knowledge_base.py`)
```python
# What we will test:
- load_ctd_structure() → returns dict with "modules" key
- get_module("M2") → returns module with id "M2"
- get_all_required_sections() → all returned sections have "required": true
- get_safety_relevant_sections() → subset of all sections
- get_sections_for_signal("hepatotoxicity") → returns list including section 2.4
- get_sections_for_signal("unknown_signal") → returns empty list without crashing
```

### Expected Output
```python
ctd = load_ctd_structure()
sections = get_sections_for_signal(ctd, "hepatotoxicity")
# → [{"id": "2.4", "name": "Nonclinical Overview", ...}, {"id": "4.2.3.4", ...}, ...]
```

### Common Failure Points
- **ICH M4 numbering is not sequential** — M2 sections are numbered 2.4–2.7 (2.1–2.3 don't exist); this is correct and should not be "fixed"
- **Section IDs as strings vs. floats** — `3.2.S` is a string, not a float; always keep IDs as strings in JSON
- **Incomplete signal_to_section_map** — we cover the most common ADR categories; document that this is a curated subset, not exhaustive

---

## Phase 6 — Dossier Parsing and CTD Semantic Matching

### Goal
Accept a submitted dossier outline (as plain text or a structured list of section headings) and **identify which ICH M4 CTD sections it contains**. This is a matching problem: the submitted text rarely uses exact ICH section IDs, so we use both exact ID matching and fuzzy/keyword matching.

### Files to Create / Change
| File | Action |
|---|---|
| `src/backend/core/dossier_parser.py` | Create |
| `src/tests/test_dossier_parser.py` | Create |

### Input Formats We Accept
1. **Plain text** — a multi-line string where each line is a section heading (e.g., pasted from a table of contents)
2. **JSON list** — `[{"id": "2.4", "title": "Nonclinical Overview"}, ...]`
3. **Semi-structured text** — lines like `"Section 2.4 – Nonclinical Overview"` or `"2.4 Nonclinical Overview (see attached)"`

### Implementation Steps
1. Write `parse_dossier_outline(text: str) -> list[dict]`:
   - Split by newlines
   - For each line, try to extract a section ID using regex: `r'\b(\d+\.\d+[\w.]*)\b'`
   - Also extract the heading text after the ID
   - Returns list of `{"id": str | None, "heading": str}`
2. Write `match_to_ctd(parsed_outline, ctd) -> list[dict]`:
   - For each parsed section, try three matching strategies in order:
     1. **Exact ID match**: parsed ID == ctd section ID → high confidence
     2. **Keyword match**: heading text contains keywords from ctd section name → medium confidence
     3. **No match**: record as unrecognized
   - Returns list of `{"ctd_section_id": str, "ctd_section_name": str, "matched_by": "exact_id"|"keyword"|None, "confidence": "high"|"medium"|"low"}`
3. Write `get_covered_sections(matched_outline) -> set[str]`:
   - Returns set of CTD section IDs that were found in the dossier
4. Write `get_missing_sections(covered, ctd) -> list[dict]`:
   - Returns required CTD sections not in `covered`

### Tests (`test_dossier_parser.py`)
```python
# What we will test:
- parse_dossier_outline() with "2.4 Nonclinical Overview" → extracts id="2.4", heading="Nonclinical Overview"
- parse_dossier_outline() with line missing ID → id=None, heading=full line
- match_to_ctd() with exact ID → matched_by="exact_id", confidence="high"
- match_to_ctd() with "toxicology studies" (no ID) → matched_by="keyword", maps to 4.2.3
- get_missing_sections() with all required present → empty list
- get_missing_sections() with M5 empty → returns all M5 required sections
```

### Expected Output
```python
outline_text = """
2.4 Nonclinical Overview
2.7 Clinical Summary
5.3.5 Efficacy and Safety Studies
"""
parsed = parse_dossier_outline(outline_text)
matched = match_to_ctd(parsed, ctd)
missing = get_missing_sections(get_covered_sections(matched), ctd)
# missing → sections that are "required" but not in the dossier
```

### Common Failure Points
- **Section IDs in different formats** — "Section 3.2.S.1" vs. "3.2.S.1" vs. "3.2.s.1"; normalize to uppercase `.S`
- **False positive keyword matches** — "clinical overview" matching "nonclinical overview"; keywords must be specific enough; use multi-word phrases not single words
- **Very short outlines** — a dossier that only lists top-level modules (M1–M5) without subsections; the parser should still work and report all subsections as missing

---

## Phase 7 — Submission Readiness Scoring and Gap Reports

### Goal
Combine the output of Phase 6 with the CTD knowledge base to produce a **completeness score per module** (0–100%) and a structured **gap report** identifying missing sections, their importance, and a plain-English explanation (from watsonx.ai Granite).

### Files to Create / Change
| File | Action |
|---|---|
| `src/backend/core/readiness_scorer.py` | Create |
| `src/backend/core/watsonx_client.py` | Create |
| `src/tests/test_readiness_scorer.py` | Create |

### Implementation Steps

#### `readiness_scorer.py`
1. Write `score_module(module_id, covered_sections, ctd) -> dict`:
   - Counts required sections in the module
   - Counts how many of those are in `covered_sections`
   - Returns `{"module_id": "M2", "score": 75.0, "total_required": 4, "found": 3, "missing": [...]}`
2. Write `score_all_modules(covered_sections, ctd) -> list[dict]`:
   - Calls `score_module` for M1–M5
   - Returns list of module score objects
3. Write `generate_gap_report(module_scores, ctd) -> dict`:
   - Structures the full report: overall completeness (weighted average), per-module breakdown, list of missing sections with their descriptions
   - Adds a `"critical_gaps"` field: missing sections where `safety_relevant=True`
   - Returns a serializable dict (not watsonx output yet)
4. Write `add_narrative(gap_report, watsonx_client) -> dict`:
   - Calls `watsonx_client.generate()` with a prompt summarizing the gaps
   - Adds `"narrative"` field to the gap report
   - If watsonx is unavailable (no API key), sets narrative to a templated fallback string

#### `watsonx_client.py`
1. Write `WatsonxClient` class:
   - Constructor: loads `WATSONX_API_KEY`, `WATSONX_PROJECT_ID`, `WATSONX_URL` from env
   - `generate(prompt: str, max_tokens: int = 500) -> str`: calls Granite model via `ibm-watsonx-ai` SDK
   - `is_available() -> bool`: returns `False` if env vars are missing (graceful degradation)
2. The model to use: `ibm/granite-13b-instruct-v2` — this is a mid-sized Granite model, good at instruction following, appropriate for summaries and explanations

### Tests (`test_readiness_scorer.py`)
```python
# What we will test:
- score_module("M2") with all M2 sections covered → score=100.0
- score_module("M2") with zero M2 sections → score=0.0
- score_module("M5") with half covered → score≈50.0
- generate_gap_report() with no critical gaps → critical_gaps=[]
- generate_gap_report() with 4.2.3 missing → critical_gaps includes toxicology
- add_narrative() when watsonx unavailable → returns report with fallback narrative string (no crash)
```

### Expected Output
```python
{
  "overall_completeness": 62.5,
  "modules": [
    {"module_id": "M1", "score": 100.0, "missing": []},
    {"module_id": "M2", "score": 75.0, "missing": [{"id": "2.6.6", "name": "Toxicology Written Summary"}]},
    {"module_id": "M5", "score": 25.0, "missing": [...]}
  ],
  "critical_gaps": [
    {"id": "2.6.6", "name": "Toxicology Written Summary", "safety_relevant": true}
  ],
  "narrative": "The dossier is 62.5% complete. The most critical gap is the missing Toxicology Written Summary (Section 2.6.6), which is required to demonstrate nonclinical safety data supporting the clinical dose. Module 5 requires significant attention..."
}
```

### Common Failure Points
- **watsonx API rate limits** — the free tier has low rate limits; the `add_narrative` function must have a timeout and fall back gracefully
- **Granite prompt engineering** — the model needs a clear, structured prompt to produce a useful narrative; poorly formed prompts produce generic text; we will use a few-shot prompt template
- **Weighted scoring** — Module 5 has many more required sections than Module 1; a simple average of module scores is misleading; weight by section count

---

## Phase 8 — Safety-to-Submission Traceability

### Goal
Given one or more detected safety signals (from Phase 3/4) and a dossier outline (from Phase 6), determine:
1. Which CTD sections **should** document evidence for this signal
2. Which of those sections are **present** in the dossier
3. Which are **absent** — creating a documentation gap

This is the "third capability" that differentiates PharmaGuard AI and earns the full traceability score.

### Files to Create / Change
| File | Action |
|---|---|
| `src/backend/core/traceability_engine.py` | Create |
| `src/tests/test_traceability_engine.py` | Create |

### Implementation Steps
1. Write `classify_signal_category(signal_pt: str, ctd) -> str`:
   - Maps an adverse event preferred term (e.g., "hepatotoxicity", "elevated ALT") to a signal category key in `signal_to_section_map`
   - Uses keyword matching: "hepat*" → "hepatotoxicity", "bleed*" or "hemorrhage" → "bleeding", etc.
   - Returns the best-matching category key, or `"general"` if no match
2. Write `trace_signal(signal: dict, covered_sections: set, ctd) -> dict`:
   - `signal` = one enriched signal from Phase 3/4 (has `drug_name`, `pt`, `prr`, etc.)
   - Calls `classify_signal_category` to get the category
   - Gets expected CTD sections from `signal_to_section_map`
   - Checks which are in `covered_sections`
   - Returns:
     ```python
     {
       "drug_name": "warfarin",
       "adverse_event": "bleeding",
       "signal_category": "bleeding",
       "prr": 8.3,
       "expected_sections": ["2.4", "2.7.4", "4.2.3", "5.3.5"],
       "documented_sections": ["2.4", "5.3.5"],
       "missing_sections": [
           {"id": "2.7.4", "name": "Clinical Overview Safety Summary"},
           {"id": "4.2.3", "name": "Toxicology"}
       ],
       "traceability_score": 50.0,
       "risk_level": "high"  # because missing sections are safety_relevant
     }
     ```
3. Write `trace_all_signals(signals, covered_sections, ctd) -> list[dict]`:
   - Runs `trace_signal` for each flagged signal
   - Returns sorted by `traceability_score` ascending (worst-documented first)
4. Write `get_traceability_summary(traces) -> dict`:
   - Overall traceability score (mean across all signals)
   - Count of signals with complete documentation
   - Count of signals with critical missing documentation

### Tests (`test_traceability_engine.py`)
```python
# What we will test:
- classify_signal_category("hepatotoxicity") → "hepatotoxicity"
- classify_signal_category("elevated liver enzymes") → "hepatotoxicity"
- classify_signal_category("nausea") → "general" (no specific category)
- trace_signal() with all sections covered → missing_sections=[], traceability_score=100.0
- trace_signal() with no sections covered → all expected sections in missing_sections
- trace_all_signals() returns sorted list (lowest score first)
- risk_level="high" when any missing section is safety_relevant
```

### Expected Output
```python
trace = trace_signal(
  signal={"drug_name": "warfarin", "pt": "bleeding", "prr": 8.3},
  covered_sections={"2.4", "5.3.5"},
  ctd=ctd
)
# trace["traceability_score"] → 50.0
# trace["missing_sections"] → [{"id": "2.7.4", ...}, {"id": "4.2.3", ...}]
# trace["risk_level"] → "high"
```

### Common Failure Points
- **Adverse event terms don't match signal categories** — real FAERS preferred terms are highly specific (e.g., "Drug-induced liver injury" not "hepatotoxicity"); the classifier must use broad keyword patterns, not exact string matching
- **Signal not in `signal_to_section_map`** — must not crash; return `"general"` category with a minimal set of sections (2.4, 5.3.5)
- **Empty signal list** — `trace_all_signals([])` must return `[]` cleanly

---

## Phase 9 — Backend API

### Goal
Expose all the core engines (Phases 2–8) through a clean REST API using FastAPI. This layer is what the frontend (Phase 11) and MCP server (Phase 10) both call.

### Files to Create / Change
| File | Action |
|---|---|
| `src/backend/api/models.py` | Create |
| `src/backend/api/routes/signals.py` | Create |
| `src/backend/api/routes/dossier.py` | Create |
| `src/backend/api/routes/traceability.py` | Create |
| `src/backend/main.py` | Update (add routers) |
| `src/tests/test_api_routes.py` | Create |

### API Endpoint Design

#### Signal Detection — `/api/signals`
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/signals/analyze` | Run full signal detection on uploaded or sample FAERS data |
| `GET` | `/api/signals/sample` | Run on the bundled sample dataset (no upload needed) |
| `GET` | `/api/signals/{drug_name}` | Get all signals for a specific drug |

**Request body** (`POST /api/signals/analyze`):
```json
{
  "drug_name": "warfarin",          // optional filter; omit for all drugs
  "use_sample_data": true,          // true = use bundled CSV
  "min_prr": 2.0,
  "min_cases": 3,
  "max_pval": 0.05
}
```

**Response**:
```json
{
  "drug_name": "warfarin",
  "signals": [
    {
      "adverse_event": "bleeding",
      "case_count": 15,
      "prr": 8.3,
      "signal_strength": "strong",
      "soc": "Vascular disorders",
      "trend": "emerging",
      "explanation": "Warfarin shows a significantly elevated PRR of 8.3 for bleeding events..."
    }
  ],
  "total_signals": 3,
  "analysis_metadata": { "dataset_size": 480, "drug_total_reports": 45 }
}
```

#### Dossier Readiness — `/api/dossier`
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/dossier/check` | Check a dossier outline for CTD completeness |
| `GET` | `/api/dossier/modules` | Return the ICH M4 CTD structure (for frontend display) |

**Request body** (`POST /api/dossier/check`):
```json
{
  "outline_text": "2.4 Nonclinical Overview\n2.7 Clinical Summary\n5.3.5 Safety Studies",
  "drug_name": "warfarin"           // optional, for context in narrative
}
```

**Response**: the full gap report structure from Phase 7.

#### Traceability — `/api/traceability`
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/traceability/trace` | Trace safety signals to dossier sections |

**Request body**:
```json
{
  "drug_name": "warfarin",
  "outline_text": "2.4 Nonclinical Overview\n...",
  "use_sample_data": true
}
```

**Response**: `trace_all_signals` output + traceability summary.

### Implementation Steps
1. Write all Pydantic request/response models in `api/models.py`
2. Wire each route to the corresponding core function
3. Add proper HTTP error codes: `400` for invalid input, `422` for validation errors (FastAPI does this automatically for Pydantic), `500` for unexpected errors
4. Add CORS middleware to allow the frontend (on a different port) to call the API
5. Mount all routers in `main.py` with `/api` prefix

### Tests (`test_api_routes.py`)
```python
# Uses FastAPI's TestClient (built-in, no real server needed)
# What we will test:
- GET /health → 200 {"status": "ok"}
- GET /api/signals/sample → 200, response has "signals" key
- POST /api/signals/analyze with use_sample_data=true → 200
- POST /api/signals/analyze with invalid body → 422
- POST /api/dossier/check with outline_text → 200, has "overall_completeness"
- POST /api/dossier/check with empty string → 200, overall_completeness=0
- POST /api/traceability/trace → 200, has "traces" key
```

### Common Failure Points
- **CORS** — the frontend will be on `localhost:5173` (Vite default); without CORS middleware, every browser request will be blocked
- **Pydantic v2 breaking changes** — the `ibm-watsonx-ai` SDK may import Pydantic v1 validators; use `model_config = ConfigDict(arbitrary_types_allowed=True)` when needed
- **Startup time** — loading the FAERS CSV on every request is slow; load it once at app startup into module-level state using FastAPI's `lifespan` event

---

## Phase 10 — MCP Server and Bob Integration

### Goal
Expose PharmaGuard AI's three core capabilities as **MCP tools** that IBM Bob can call. When running, Bob users can ask natural-language questions and Bob will automatically call the right tool, pass the result back, and generate a response.

### Files to Create / Change
| File | Action |
|---|---|
| `src/mcp-server/server.py` | Create |
| `src/mcp-server/requirements.txt` | Create |
| `src/mcp-server/tools/detect_signals.py` | Create |
| `src/mcp-server/tools/check_dossier.py` | Create |
| `src/mcp-server/tools/trace_signal.py` | Create |

### How FastMCP Works (important context)
FastMCP lets you write a Python function and decorate it with `@mcp.tool()`. The function's docstring becomes the tool description that Bob reads to decide when to call it. The function's type-annotated parameters become the tool's input schema. FastMCP handles the MCP protocol entirely.

```python
# Example of what a tool looks like
from fastmcp import FastMCP

mcp = FastMCP("PharmaGuard AI")

@mcp.tool()
def detect_safety_signals(drug_name: str, min_prr: float = 2.0) -> dict:
    """
    Detect potential adverse drug reaction safety signals for a given drug
    using FDA FAERS pharmacovigilance data.
    Returns a list of flagged signals with PRR scores and explanations.
    """
    import httpx
    response = httpx.post(
        f"{BACKEND_URL}/api/signals/analyze",
        json={"drug_name": drug_name, "use_sample_data": True, "min_prr": min_prr}
    )
    return response.json()
```

### Three MCP Tools

| Tool name | Parameters | What it does |
|---|---|---|
| `detect_safety_signals` | `drug_name: str`, `min_prr: float = 2.0` | Runs signal detection on FAERS data, returns ranked signals with explanations |
| `check_submission_readiness` | `outline_text: str`, `drug_name: str = ""` | Checks a CTD dossier outline against ICH M4, returns completeness scores and gap report |
| `trace_signal_to_dossier` | `drug_name: str`, `outline_text: str` | Traces detected signals to required CTD sections, identifies documentation gaps |

### Bob Configuration
```json
// Add to Bob's MCP config (bob_mcp_config.json or equivalent)
{
  "mcpServers": {
    "pharmaguard": {
      "command": "python",
      "args": ["src/mcp-server/server.py"],
      "env": {
        "PHARMAGUARD_API_URL": "http://localhost:8000"
      }
    }
  }
}
```

### Example Bob Conversations (showing real tool use)
> **User:** "Are there any safety signals for warfarin?"
> **Bob → calls `detect_safety_signals("warfarin")` → gets structured JSON → responds:**
> "I found 3 safety signals for warfarin. The strongest is **bleeding** (PRR 8.3, 15 reports), classified as a strong signal in Vascular disorders. This signal is trending as emerging, meaning report rates have increased in recent data..."

> **User:** "Here is my dossier outline. Is it ready for submission? [pastes outline]"
> **Bob → calls `check_submission_readiness(outline_text=...)` → responds:**
> "Your dossier is 62.5% complete. The critical gap is Section 2.6.6 (Toxicology Written Summary), which is required for safety approval..."

### Implementation Steps
1. Write `server.py` — creates `FastMCP("PharmaGuard AI")` instance, registers all three tools, starts the server
2. Write each tool in its own file under `tools/` (imported by `server.py`)
3. Each tool uses `httpx.post/get` to call the FastAPI backend (not calling core functions directly — keeps MCP server stateless)
4. Add `PHARMAGUARD_API_URL` to `.env.example`
5. Write `README` instructions for how to add PharmaGuard to Bob's MCP config
6. Test that `python server.py` starts without errors and Bob can call each tool

### Tests
```bash
# Manual test: start backend, then run MCP inspector
python src/mcp-server/server.py
# In a separate terminal, use FastMCP's built-in test client or Bob directly
```

Automated unit tests for the MCP layer test only the tool functions in isolation (mock the HTTP calls to backend).

### Common Failure Points
- **MCP server must be running for Bob to call it** — both backend AND MCP server must be started; document this in setup guide
- **FastMCP version compatibility** — FastMCP is actively developed; pin the version in `requirements.txt` to avoid breaking changes
- **Tool descriptions matter** — Bob uses the docstring to decide which tool to call; vague descriptions lead to Bob using the wrong tool or not calling any tool; write precise, specific docstrings
- **Large responses** — MCP tools should return summaries, not full DataFrames; the tool layer must truncate to top N results

---

## Phase 11 — Frontend Dashboard

### Goal
A React single-page application with three pages (one per capability), styled with Tailwind CSS, calling the FastAPI backend directly over HTTP.

### Files to Create / Change
All files under `src/frontend/` (new directory).

### Page Designs

#### Page 1: Signal Detection
- Input: drug name text field + "Analyze" button
- Output: list of signal cards, each showing:
  - Drug name + adverse event name
  - PRR score (displayed as a colored badge: red=strong, orange=moderate)
  - Case count
  - System organ class
  - Trend indicator (↑ emerging, → stable, ↓ declining)
  - Expandable "Evidence" section showing the Granite-generated explanation
- Also: a bar chart (Recharts) showing PRR values across all signals

#### Page 2: Dossier Readiness
- Input: large textarea for pasting a dossier outline + drug name field + "Check" button
- Output:
  - Overall completeness percentage (large number, color-coded)
  - Five module progress bars (M1–M5) each with their score
  - Gap table: columns = Section ID, Name, Required, Safety-Relevant, Status
  - Critical gaps section (red highlight)
  - Narrative text from Granite

#### Page 3: Traceability
- Input: drug name + dossier outline textarea (same inputs as Page 2) + "Trace" button
- Output:
  - For each detected signal: a row showing the signal, its category, and a mini-table of expected/covered/missing CTD sections
  - Color coding: green = covered, red = missing
  - Overall traceability score

### Implementation Steps
1. Scaffold with Vite: `npm create vite@latest frontend -- --template react-ts`
2. Install Tailwind: `npm install -D tailwindcss postcss autoprefixer && npx tailwindcss init -p`
3. Install Recharts: `npm install recharts`
4. Write `src/api/client.ts` — typed fetch wrappers for all three backend endpoints
5. Build pages in order: SignalDetection → DossierReadiness → Traceability
6. Add a simple top navigation bar with links between pages
7. Configure Vite proxy to forward `/api/*` to `http://localhost:8000` (avoids CORS in dev)

### Tests
- Frontend tests are minimal for hackathon scope
- Verify `npm run build` succeeds (no TypeScript errors)
- Manual browser testing of each page with real backend running

### Common Failure Points
- **Vite proxy not configured** — without `server.proxy` in `vite.config.ts`, API calls from the browser will hit CORS errors even in development
- **TypeScript strictness** — API responses are `unknown` by default; write proper interface types for each response shape to avoid build errors
- **Recharts with TypeScript** — Recharts types are sometimes incomplete; use `any` casts sparingly where needed and document why
- **Long API response times** — watsonx.ai calls can take 5–10 seconds; add a loading spinner and disable the button while waiting

---

## Phase 12 — Testing

### Goal
A complete, runnable test suite that any judge can execute with a single command to verify the core logic works.

### Files to Create / Change
| File | Action |
|---|---|
| `src/tests/__init__.py` | Create |
| `src/tests/conftest.py` | Create (shared fixtures) |
| `src/tests/test_faers_ingestor.py` | Created in Phase 2 |
| `src/tests/test_prr_calculator.py` | Created in Phase 3 |
| `src/tests/test_signal_clusterer.py` | Created in Phase 4 |
| `src/tests/test_ctd_knowledge_base.py` | Created in Phase 5 |
| `src/tests/test_dossier_parser.py` | Created in Phase 6 |
| `src/tests/test_readiness_scorer.py` | Created in Phase 7 |
| `src/tests/test_traceability_engine.py` | Created in Phase 8 |
| `src/tests/test_api_routes.py` | Created in Phase 9 |

### `conftest.py` Fixtures
```python
# Shared across all tests:
- sample_df: loads faers_sample.csv once
- ctd: loads ich_m4_structure.json once
- test_client: FastAPI TestClient
- sample_outline_text: a realistic multi-section dossier outline string
- sample_signal: a single enriched signal dict (warfarin/bleeding)
```

### Commands
```bash
cd src/backend
# Run all tests
pytest ../tests/ -v

# Run a single test file
pytest ../tests/test_prr_calculator.py -v

# Run a single test function
pytest ../tests/test_prr_calculator.py::test_prr_known_values -v

# Run with coverage report
pytest ../tests/ --cov=core --cov-report=term-missing
```

### Coverage Targets
- `core/` modules: aim for ≥ 80% line coverage
- `api/routes/`: ≥ 70% (mostly integration-level tests)
- `mcp-server/`: not unit tested (relies on manual Bob integration test)

### Common Failure Points
- **Tests depend on watsonx API key** — any test that calls `WatsonxClient.generate()` must be either skipped if `WATSONX_API_KEY` is unset or mock the HTTP call with `pytest-mock`; never require a real API key to run tests
- **Test isolation** — each test must be independent; no shared mutable state between tests
- **FAERS sample CSV** — tests that load `faers_sample.csv` must use a relative path from the test file's location, or use the `conftest.py` fixture that resolves the absolute path

---

## Phase 13 — Documentation

### Goal
Fill every required submission document with real, accurate content derived from what was built.

### Files to Create / Change
| File | Action |
|---|---|
| `submission.yaml` | Fill all required fields |
| `README.md` | Replace all placeholders |
| `docs/problem-statement.md` | Write real content |
| `docs/solution-overview.md` | Write real content |
| `docs/architecture.md` | Replace diagram + fill tables |
| `docs/setup-guide.md` | Replace all placeholders |
| `src/.env.example` | Final version with all variables |

### `submission.yaml` Required Fields
```yaml
team:
  name: "Exodus"
  track: "AI"             # exact: AI | DevOps | Sustainability | Open
  lead:
    name: "[your name]"
    email: "[your email]"

submission:
  title: "PharmaGuard AI"
  tech_stack:
    languages: ["Python", "TypeScript"]
    frameworks: ["FastAPI", "React", "FastMCP"]
    ibm_technologies: ["watsonx.ai", "IBM Bob", "IBM Granite 13B"]
```

### `docs/architecture.md` Mermaid Diagram
Must be replaced with the actual PharmaGuard architecture (the diagram from the plan above).

### Setup Guide Checklist
The setup guide must be tested by following it on a fresh terminal. It must include:
- Python 3.11+ required
- `pip install -r requirements.txt`
- `cp .env.example .env` + instructions for each variable
- `uvicorn main:app --reload` command
- `python ../mcp-server/server.py` command
- `npm install && npm run dev` for frontend
- Screenshot of expected output for each step

### Common Failure Points
- **`submission.yaml` empty strings fail CI** — every `# REQUIRED` field must be non-empty; check with `yq '.' submission.yaml` locally
- **`team.track` is case-sensitive** — must be exactly `"AI"`, not `"ai"` or `"Artificial Intelligence"`
- **README still has `[Your Team Name]` or `[Your Project Title Here]`** — CI explicitly checks for these strings and fails if present
- **`demo/demo-video-link.txt` line 1** — must not contain `your-demo-video-link-here`; the CI reads only line 1

---

## Phase 14 — Demo and Presentation

### Goal
Record a 3–5 minute demo video and prepare a slide deck, completing the submission checklist.

### Files to Create / Change
| File | Action |
|---|---|
| `demo/demo-video-link.txt` | Replace placeholder URL with real video link |
| `demo/live-demo-url.txt` | Add deployed URL or write `NOT DEPLOYED` |
| `demo/screenshots/01-signal-detection.png` | Screenshot of signal detection results |
| `demo/screenshots/02-dossier-readiness.png` | Screenshot of dossier gap report |
| `demo/screenshots/03-traceability.png` | Screenshot of traceability view |
| `demo/screenshots/04-bob-integration.png` | Screenshot of Bob calling MCP tool |
| `presentation/slides.pdf` | Slide deck |

### Demo Video Script (3–5 minutes)
1. **(0:00–0:30)** — Open PharmaGuard AI dashboard, briefly show three tabs
2. **(0:30–1:30)** — Type "warfarin" in Signal Detection, click Analyze, walk through the results: PRR scores, signal strength, trend, Granite explanation
3. **(1:30–2:30)** — Switch to Dossier Readiness, paste a sample CTD outline, click Check, walk through the gap report (point out a critical missing section)
4. **(2:30–3:30)** — Switch to Traceability, show how the warfarin/bleeding signal is linked to missing CTD sections
5. **(3:30–4:30)** — Switch to IBM Bob terminal, ask "Are there safety signals for warfarin?", show Bob calling the MCP tool and generating a response
6. **(4:30–5:00)** — Brief summary of what makes this submission strong

### Slide Deck Structure (5–7 slides)
1. **Problem** — pharmacovigilance teams miss emerging signals; regulatory affairs teams spend days manually checking CTD completeness
2. **Solution** — PharmaGuard AI: three capabilities, one integrated system
3. **Architecture** — the Mermaid diagram, annotated with IBM Bob and watsonx.ai integration points
4. **Demo highlights** — 2–3 screenshots from the running app
5. **IBM Technology Integration** — specifically how Bob + Granite are load-bearing (not decorative)
6. **Impact** — what this could become beyond the hackathon

### Common Failure Points
- **Loom/YouTube requires "anyone with link" access** — private videos fail CI indirectly (judges can't watch them)
- **Video link on line 2 of `demo-video-link.txt`** — CI reads only line 1; put the URL on the very first line
- **Not enough screenshots** — minimum 3 required; don't forget one showing Bob/MCP integration (this directly supports the IBM Bob scoring criterion)
- **Slides as `.pptx` on Mac opened on Windows** — fonts may render incorrectly; export to PDF before submitting

---

## Dependency Map (phases that must complete before others start)

```
Phase 1  ─────────────────────────────────────────────────────────────────────────┐
Phase 2  (needs Phase 1)                                                           │
Phase 3  (needs Phase 2)                                                           │
Phase 4  (needs Phase 3)                                                           │
Phase 5  (independent after Phase 1)                                               │
Phase 6  (needs Phase 5)                                                           │
Phase 7  (needs Phase 6)                                                           │
Phase 8  (needs Phase 4 + Phase 7)                                                 │
Phase 9  (needs Phase 2–8)                                                         ├─ Phase 12
Phase 10 (needs Phase 9)                                                           │
Phase 11 (needs Phase 9)                                                           │
Phase 12 (tests written alongside each phase, run after Phase 9)                  │
Phase 13 (needs all phases complete)                                               │
Phase 14 (needs Phase 13 + running application)                                   ┘
```

**Fastest independent parallel track:** Phases 5, 6, 7 can be developed in parallel with Phases 2, 3, 4 since they don't share code — only converge at Phase 8.

---

This is the complete implementation plan for all 14 phases. No code has been written. Please review and let me know:

1. Are you ready to start with **Phase 1**?
2. Do you have a watsonx.ai API key, or should I design the watsonx integration to degrade gracefully when no key is present (so you can demo without one)?
3. Any phases you want to adjust in scope before we begin?