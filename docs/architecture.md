# Architecture

## System Architecture

```mermaid
graph TD
    subgraph Browser ["Browser (port 5173 in dev)"]
        UI[React + Vite + Tailwind CSS<br/>6 pages: Dashboard · Signal Detection ·<br/>Signal Details · Readiness · Gap Report · Action Brief]
    end

    subgraph Backend ["FastAPI Backend (port 8000)"]
        API[FastAPI app<br/>main.py]
        SR[/api/signals<br/>signals.py]
        DR[/api/dossier<br/>dossier.py]
        TR[/api/traceability<br/>traceability.py]
        API --> SR
        API --> DR
        API --> TR
    end

    subgraph Core ["Core Processing Layer (src/backend/core/)"]
        FI[faers_ingestor<br/>CSV → normalised DataFrame]
        PRR[prr_calculator<br/>2×2 contingency + PRR + CI]
        TA[temporal_analyser<br/>quarterly trend scoring]
        PS[priority_scorer<br/>6-dimension 0–100 score]
        SC[signal_clusterer<br/>DBSCAN clustering]
        DP[dossier_parser<br/>free-text outline → sections]
        SM[section_matcher<br/>sections → PRESENT/MISSING/NEEDS_REVIEW]
        RS[readiness_scorer<br/>weighted M1–M5 completeness]
        KB[ctd_knowledge_base<br/>ICH M4 section registry]
        TE[traceability_engine<br/>signal → CTD section gaps]

        SR --> FI
        SR --> PRR
        SR --> TA
        SR --> PS
        SR --> SC
        DR --> DP
        DR --> SM
        DR --> RS
        DR --> KB
        TR --> TE
        TE --> KB
        SM --> KB
        PS --> TA
        PS --> SC
    end

    subgraph Data ["Data Layer"]
        CSV[(faers_sample.csv<br/>src/backend/data/)]
        FI --> CSV
    end

    subgraph MCP ["MCP Server (src/mcp-server/, port 8001)"]
        SRV[server.py<br/>FastMCP — 6 tools]
        CFG[config.py<br/>PHARMAGUARD_API_URL]
        SRV --> CFG
    end

    subgraph IBM ["IBM Technologies"]
        BOB[IBM Bob<br/>reads .bob/mcp.json]
        WX[IBM watsonx.ai<br/>optional — narrative generation<br/>ibm-watsonx-ai SDK]
    end

    UI -->|REST /api/*| API
    BOB -->|stdio MCP protocol| SRV
    SRV -->|HTTP GET/POST| API
    SR -.->|optional SDK call| WX
```

## Components

| Component | Technology | Responsibility |
|---|---|---|
| **React Frontend** | React 19, Vite 8, Tailwind CSS 3, TypeScript, Recharts, react-router-dom | Dashboard UI — signal browsing, dossier readiness, gap reports, action briefs. Proxies all `/api/*` requests to the FastAPI backend in development. |
| **FastAPI Backend** | Python 3.12, FastAPI, Uvicorn, Pydantic | REST API entry point. Mounts three routers (`/api/signals`, `/api/dossier`, `/api/traceability`). Reads `.env` for watsonx.ai credentials. |
| **faers_ingestor** | pandas, DuckDB, pyarrow | Loads, validates, and normalises FAERS quarterly CSV data. Supports multiple column aliases, encodings, and delimiters. Exports to Parquet or DuckDB for large datasets. |
| **prr_calculator** | scipy (chi2_contingency, fisher_exact), pandas | Builds 2×2 contingency tables and computes PRR + 95% CI (log-normal method, Evans 2001) + statistical significance test for every drug–event pair. |
| **temporal_analyser** | pandas | Groups reports by calendar quarter and computes trend score, direction, and percentage change for each drug–event pair. |
| **priority_scorer** | Pure Python arithmetic | Combines PRR strength, report volume, p-value, serious outcome rate, trend, and cluster context into a 0–100 screening score. |
| **signal_clusterer** | scikit-learn (DBSCAN), pandas | One-hot encodes drug, event, outcome, and sex features and applies DBSCAN to group related adverse-event reports into clusters. |
| **dossier_parser** | Pure Python | Parses free-text or structured dossier outlines into normalised section identifiers. |
| **section_matcher** | Pure Python | Matches parsed section identifiers against the CTD knowledge base and assigns PRESENT / NEEDS_REVIEW / MISSING / NOT_APPLICABLE status. |
| **readiness_scorer** | Pure Python arithmetic | Computes a weighted per-module (M1–M5) and overall completeness score, applying a 1.5× multiplier to safety-relevant sections. |
| **ctd_knowledge_base** | Pure Python data class | Registry of all ICH M4 CTD sections across Modules 1–5 with requirement level, safety relevance, and traceability category. |
| **traceability_engine** | Pure Python | Maps a flagged adverse event to its relevant CTD sections by keyword-based category classification, computes traceability score, and surfaces critical gaps. |
| **MCP Server** | Python, FastMCP 2.x, httpx | Thin HTTP adapter that translates IBM Bob MCP tool calls into REST requests against the FastAPI backend. No domain logic. |
| **IBM Bob** | IBM Bob (.bob/mcp.json) | Spawns the MCP server as a child process over stdio and makes all six pharmacovigilance tools available via natural-language conversation. |
| **IBM watsonx.ai** | ibm-watsonx-ai SDK 1.7.2 | Optional — generates AI narrative paragraphs in `/api/signals/explain/{drug}/{event}`. Graceful fallback when credentials are absent. |
| **Bundled dataset** | CSV (faers_sample.csv) | Synthetic FAERS sample (~500 rows) committed to `src/backend/data/`. Used by all backend endpoints with no external data source required. |

## Data Flow

### Signal Detection Flow

```
User (Dashboard or Bob)
  │
  ▼
POST /api/signals/analyze  (or GET /api/signals/sample)
  │
  ├─► faers_ingestor.load_faers(faers_sample.csv)
  │       → normalised DataFrame (drug_name, adverse_event, outcome_code, report_date, …)
  │
  ├─► prr_calculator.calculate_all_signals(df)
  │       → list[SignalResult]  (PRR, 95% CI, p-value, test_method, signal_flag)
  │
  ├─► temporal_analyser.analyse_all_trends(df)
  │       → list[TrendResult]  (direction, trend_score, pct_change, quarterly_counts)
  │
  ├─► signal_clusterer.cluster_reports(df)
  │       → list[ClusterResult]  (cluster_id, top_drugs, top_events, purity)
  │
  └─► priority_scorer.score_all_signals(flagged, df, trends, clusters)
          → list[PriorityScore]  (final_score, component breakdown)
```

### Dossier Readiness Flow

```
User (Readiness page or Bob)
  │
  ▼
POST /api/dossier/check  { drug_name, outline_text }
  │
  ├─► dossier_parser.parse_outline(outline_text)
  │       → list[str]  (section identifiers extracted from outline)
  │
  ├─► section_matcher.match_dossier(section_ids, knowledge_base)
  │       → MatchSummary  (per-section PRESENT/MISSING/NEEDS_REVIEW/NOT_APPLICABLE)
  │
  └─► readiness_scorer.score_readiness(match_summary)
          → ReadinessReport  (overall_score, per-module M1–M5 scores, critical_missing)
```

### Traceability Flow

```
User (Action Brief page or Bob)
  │
  ▼
POST /api/traceability/trace-text  { signals, section_numbers }
  │
  ├─► traceability_engine.classify_signal_category(adverse_event)
  │       → category string  (e.g. "hepatotoxicity", "bleeding", "general")
  │
  ├─► ctd_knowledge_base.get_sections_by_traceability_category(category)
  │       → list[CTDSection]  (the CTD sections that must document this signal)
  │
  └─► traceability_engine.trace_signal(signal, match_summary)
          → TraceabilityResult  (per-section status, traceability_score, critical_gaps, priority_actions)
```

## Security Considerations

- API keys and credentials are loaded exclusively from environment variables via `pydantic-settings`. They are never hardcoded in source files.
- `.env` is in `.gitignore` and is never committed. Only `.env.example` (with dummy values) is committed.
- No authentication middleware is implemented in this hackathon prototype. All API endpoints are public. Production deployment would require API key or OAuth token validation.
- The bundled FAERS dataset contains no real patient data — it is synthetic.
- CORS is configured to allow only `localhost:5173`, `localhost:3000`, and `localhost:8000` in development.

## Scalability Notes

The current implementation is sized for a hackathon prototype with the bundled ~500-row sample dataset. A production deployment would address the following:

- **Large FAERS datasets:** The `faers_ingestor` already supports export to Parquet and DuckDB. The `prr_calculator` notes that a DuckDB-backed aggregation path is the intended upgrade for full quarterly FAERS data (~2 million rows).
- **Backend scaling:** The FastAPI application is stateless. Multiple instances can run behind a load balancer. Uvicorn's worker count (`--workers N`) can be increased for higher throughput.
- **watsonx.ai throughput:** The `/explain` endpoint makes a synchronous watsonx.ai SDK call. For production, this should be moved to an async task queue (e.g., Celery + Redis) to avoid blocking API workers.
- **Frontend:** The Vite build produces a static bundle that can be served from any CDN or object storage (IBM Cloud Object Storage, S3, Cloudflare Pages).
