# PharmaGuard AI — Source Code

This directory contains all source code for the PharmaGuard AI project.

## Directory Structure

```
src/
├── backend/                  # FastAPI backend — all domain logic and REST API
│   ├── api/
│   │   └── routes/
│   │       ├── signals.py    # /api/signals — PRR detection, trends, priority, explain
│   │       ├── dossier.py    # /api/dossier — CTD readiness, gap reports
│   │       └── traceability.py  # /api/traceability — signal-to-CTD traceability
│   ├── core/                 # Domain logic (no FastAPI dependencies)
│   │   ├── faers_ingestor.py    # FAERS CSV loading and normalisation
│   │   ├── prr_calculator.py    # PRR + 95% CI + chi-squared/Fisher test
│   │   ├── temporal_analyser.py # Quarterly trend scoring
│   │   ├── priority_scorer.py   # 6-dimension 0–100 priority score
│   │   ├── signal_clusterer.py  # DBSCAN clustering of adverse-event reports
│   │   ├── dossier_parser.py    # Free-text dossier outline → section list
│   │   ├── section_matcher.py   # Section list → PRESENT/MISSING/NEEDS_REVIEW
│   │   ├── readiness_scorer.py  # Weighted M1–M5 completeness scoring
│   │   ├── ctd_knowledge_base.py  # ICH M4 section registry
│   │   ├── traceability_engine.py # Signal → CTD section gap analysis
│   │   └── schemas.py           # Shared Pydantic data models
│   ├── data/
│   │   └── faers_sample.csv  # Bundled synthetic FAERS dataset (~500 rows)
│   ├── config.py             # Environment variable settings (pydantic-settings)
│   ├── main.py               # FastAPI app entry point, CORS, router mounting
│   └── requirements.txt      # Python dependencies
│
├── frontend/                 # React + Vite + Tailwind CSS dashboard
│   ├── src/
│   │   ├── api/
│   │   │   ├── client.ts    # axios API client (all backend endpoints)
│   │   │   └── types.ts     # TypeScript interfaces for API responses
│   │   ├── components/      # Shared UI: Badge, Card, Layout, Spinner, ProgressBar, DisclaimerBanner
│   │   ├── pages/           # Route pages
│   │   │   ├── Dashboard.tsx
│   │   │   ├── SignalDetection.tsx
│   │   │   ├── SignalDetails.tsx
│   │   │   ├── SubmissionReadiness.tsx
│   │   │   ├── GapReport.tsx
│   │   │   └── ActionBrief.tsx
│   │   ├── App.tsx          # React Router setup
│   │   └── main.tsx         # React entry point
│   ├── vite.config.ts       # Vite config — dev proxy /api/* → localhost:8000
│   ├── tailwind.config.js
│   ├── package.json
│   └── tsconfig.json
│
├── mcp-server/               # IBM Bob MCP server
│   ├── server.py             # FastMCP app — 6 pharmacovigilance tools
│   ├── config.py             # MCP server settings (PHARMAGUARD_API_URL, timeout, max_results)
│   └── requirements.txt      # MCP server Python dependencies (fastmcp, httpx, pydantic-settings)
│
├── tests/                    # pytest test suite (956 tests)
│   ├── conftest.py           # Shared fixtures
│   ├── fixtures/             # Test data files
│   ├── test_faers_ingestor.py
│   ├── test_prr_calculator.py
│   ├── test_prr_deterministic.py
│   ├── test_temporal_analyser.py
│   ├── test_priority_scorer.py
│   ├── test_signal_clusterer.py
│   ├── test_ctd_knowledge_base.py
│   ├── test_dossier_parser.py
│   ├── test_section_matcher.py
│   ├── test_readiness_scorer.py
│   ├── test_traceability_engine.py
│   └── test_api_dossier.py
│
├── .env.example              # Environment variable template — copy to .env
└── README.md                 # This file
```

## Quick Start

Full setup instructions are in [`docs/setup-guide.md`](../docs/setup-guide.md). Short version:

### Backend

```bash
cd src/backend
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

### Frontend

```bash
cd src/frontend
npm install
npm run dev
# App: http://localhost:5173
```

### Tests

```bash
cd src/backend
python -m pytest ../tests/ -q
# Expected: 956 passed, 0 failed
```

## Environment Variables

Copy `src/.env.example` to `src/.env`. All variables have sensible defaults — only `WATSONX_API_KEY` and `WATSONX_PROJECT_ID` are needed for optional AI-generated explanations. See [`docs/setup-guide.md`](../docs/setup-guide.md) for the full reference.

## Key Design Notes

- **No database server required.** The backend uses DuckDB in-process on the bundled `faers_sample.csv`.
- **No watsonx.ai credentials required.** All API endpoints work deterministically without them.
- **MCP server is a thin adapter.** It calls the FastAPI backend over HTTP. All domain logic is in `src/backend/core/`.
- **Tests run from `src/backend/`.** The `conftest.py` adds `src/backend` to `sys.path` so imports resolve correctly.
