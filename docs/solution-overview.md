# Solution Overview

## What We Built

PharmaGuard AI is a full-stack pharmacovigilance and regulatory submission readiness platform. It ingests FDA FAERS adverse event data, automatically detects safety signals using established statistical methods, scores and ranks those signals by clinical priority, checks ICH M4 CTD dossier completeness, and traces each signal to the specific regulatory sections that must document it — all through a React dashboard and an IBM Bob MCP integration.

**All core detection and scoring logic is entirely deterministic.** No language model is required for the pipeline to work. IBM watsonx.ai is an optional enhancement that can generate narrative explanations; it is never a dependency for the quantitative results.

## How It Works

1. **Data ingestion** — The `faers_ingestor` module loads a FAERS quarterly CSV file, validates the schema, normalises drug names (stripping salt/formulation suffixes such as "hydrochloride"), and maps MedDRA preferred terms to a consistent internal format. The bundled `faers_sample.csv` dataset is used automatically with no setup required.

2. **PRR signal detection** — The `prr_calculator` module constructs a 2×2 contingency table for every drug–adverse-event pair and computes the Proportional Reporting Ratio (PRR) using the Evans 2001 log-normal method, including 95% confidence intervals. Statistical significance is confirmed by chi-squared test (when all expected cell counts ≥ 5) or Fisher's exact test. A signal is flagged when all three EMA/WHO criteria are met simultaneously: PRR ≥ 2.0, case count ≥ 3, and p-value ≤ 0.05. All thresholds are configurable.

3. **Temporal trend analysis** — The `temporal_analyser` module assigns each report to a calendar quarter, splits the time series into a previous and recent half, and computes a trend score in [−1.0, +1.0]. Direction is classified as *increasing*, *decreasing*, *stable*, or *insufficient_data*. Quarterly counts and percentage change are returned alongside the direction.

4. **Priority scoring** — The `priority_scorer` module combines six evidence dimensions into a single 0–100 heuristic screening score per flagged signal:
   - PRR strength (max 30 pts)
   - Report volume (max 20 pts)
   - Statistical strength / p-value (max 20 pts)
   - Serious outcome rate — Death, Life-threatening, Hospitalisation, Disability, Congenital anomaly, Required intervention (max 15 pts)
   - Reporting trend direction (max 10 pts)
   - DBSCAN cluster strength (max 5 pts)

5. **Signal clustering** — The `signal_clusterer` module applies DBSCAN (density-based clustering from scikit-learn) to the full report set, grouping reports by drug, event, outcome, and demographic features. Cluster membership informs the cluster-strength component of the priority score.

6. **CTD dossier readiness** — The `dossier_parser` and `section_matcher` modules parse a free-text or structured dossier outline and match it against the `CTDKnowledgeBase` — a curated knowledge base of all required and conditional ICH M4 sections across Modules 1–5. Each section receives a status: `PRESENT`, `NEEDS_REVIEW`, `MISSING`, or `NOT_APPLICABLE`. The `readiness_scorer` then computes a weighted completeness score per module and overall, applying a 1.5× safety multiplier to safety-relevant sections.

7. **Signal-to-CTD traceability** — The `traceability_engine` maps each flagged safety signal to the CTD sections that must document it. It classifies each signal by adverse event category (e.g., hepatotoxicity, bleeding, cardiotoxicity) and retrieves the corresponding CTD section list from the knowledge base. For each section it reports whether it is present, missing, or needs review in the dossier, and computes a traceability score. Critical gaps — safety-relevant sections that are missing — are surfaced as priority actions.

8. **React dashboard** — The frontend (React + Vite + Tailwind CSS) provides six pages:
   - **Dashboard** — summary stats and quick navigation
   - **Signal Detection** — run PRR analysis, browse all signals, filter by drug
   - **Signal Details** — full explanation for a single drug–event pair (PRR, trend, priority score, narrative paragraphs)
   - **Submission Readiness** — upload or paste a dossier outline and see module-level completeness scores
   - **Gap Report** — browse required and conditional CTD sections by submission type (NDA, BLA, ANDA, MAA)
   - **Action Brief** — combined signal + traceability + gap analysis for a specific drug

9. **IBM Bob MCP integration** — The `src/mcp-server/server.py` registers PharmaGuard AI as an MCP server in Bob's `.bob/mcp.json`. It exposes six tools that call the FastAPI backend over HTTP:
   - `detect_safety_signals` — run full PRR signal detection
   - `calculate_prr` — PRR for a specific drug
   - `get_signal_explanation` — temporal trend for a drug–event pair
   - `check_ctd_readiness` — dossier completeness check
   - `generate_gap_report` — required CTD sections by submission type
   - `generate_regulatory_action_brief` — combined signal + traceability + gap analysis

## Key Design Decisions

| Decision | Rationale |
|---|---|
| All quantitative calculations are deterministic, no LLM | Reproducibility is essential in a regulated domain. Results must be identical on every run and auditable without AI opacity. |
| watsonx.ai is optional, not required | The platform is fully functional without credentials. watsonx.ai adds optional narrative explanation generation but never gates the core pipeline. |
| MCP server as a thin HTTP adapter | All domain logic stays in the FastAPI backend. The MCP server is a thin translation layer — it makes no domain decisions and duplicates no logic. |
| PRR with chi-squared / Fisher selection | Fisher's exact test is used automatically when expected cell counts are < 5, which is common in small pharmacovigilance datasets. This follows EMA guidelines. |
| DBSCAN for clustering, not KMeans | DBSCAN does not require a pre-specified number of clusters, handles noise/outliers naturally, and is appropriate for sparse, high-dimensional drug–event feature vectors. |
| DuckDB for in-process data storage | DuckDB provides SQL aggregation performance on the bundled sample dataset without requiring a separate database server, keeping local setup to a single command. |

## IBM Technologies Used

- **IBM Bob:** The entire pharmacovigilance pipeline is accessible as a conversational AI tool through IBM Bob's MCP integration. Bob users can type "Are there any safety signals for warfarin?" or "Is our NDA dossier complete?" and receive structured, evidence-grounded answers directly in the Bob chat interface.

- **IBM Bob MCP Server (FastMCP):** The MCP server (`src/mcp-server/`) is built with FastMCP and registered in `.bob/mcp.json`. It is spawned by Bob as a child process over stdio transport and calls the FastAPI backend over HTTP.

- **IBM watsonx.ai (optional):** When `WATSONX_API_KEY` and `WATSONX_PROJECT_ID` are configured in `.env`, the `/api/signals/explain/{drug}/{event}` endpoint can augment its deterministic output with AI-generated narrative paragraphs. The `ibm-watsonx-ai` Python SDK (v1.7.2) is installed in the backend virtual environment. When credentials are not configured, the endpoint falls back gracefully to the deterministic explanation pipeline — no error is raised.

## Important Limitations

All scores, signal flags, traceability results, and dossier readiness scores produced by PharmaGuard AI are **heuristic screening aids only**. They:

- Do NOT establish causality between a drug and an adverse event.
- Do NOT constitute an FDA approval readiness assessment.
- Do NOT represent the position of the FDA, ICH, EMA, or any regulatory authority.
- Require review by a qualified pharmacovigilance professional and regulatory affairs specialist before any clinical or regulatory action is taken.

The bundled FAERS dataset (`src/backend/data/faers_sample.csv`) is a synthetic demonstration dataset. It is not drawn from the live FDA FAERS database.
