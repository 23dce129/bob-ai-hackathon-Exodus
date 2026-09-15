# PharmaGuard AI

> AI-powered pharmacovigilance platform for adverse drug reaction signal detection, ICH M4 CTD dossier readiness checking, and regulatory traceability.

---

## 👥 Team

| Field | Value |
|---|---|
| **Team Name** | Exodus |
| **Track** | AI |
| **Team Lead** | Vansh Vyas — 23dce129@charusat.edu.in |

---

## 🎯 Problem Statement

Pharmacovigilance teams manually sift through thousands of FDA FAERS reports to detect adverse drug reaction signals — a process that is slow, error-prone, and requires deep specialist expertise. Drug safety scientists and regulatory affairs professionals lack an integrated tool that combines statistical signal detection, dossier completeness checking, and CTD section traceability in a single workflow.

---

## 💡 Solution

PharmaGuard AI is a full-stack pharmacovigilance platform that automatically detects adverse drug reaction safety signals from FDA FAERS data using Proportional Reporting Ratio (PRR) analysis, scores and ranks signals by clinical priority, checks ICH M4 CTD dossier completeness, and traces each signal to the required regulatory submission sections — all through a React dashboard and an IBM Bob-integrated MCP server.

---

## ✨ Key Features

- **PRR Signal Detection:** Proportional Reporting Ratio analysis with 95% CI (log-normal method), chi-squared/Fisher test, and configurable FDA/EMA-standard thresholds (PRR ≥ 2, n ≥ 3, p < 0.05).
- **Priority Scoring:** Multi-dimensional scoring combining PRR strength (30 pts), report volume (20 pts), statistical strength (20 pts), serious outcome rate (15 pts), reporting trend (10 pts), and cluster context (5 pts).
- **CTD Dossier Readiness:** ICH M4 CTD completeness checker with weighted per-module (M1–M5) scoring, identifying missing, present, and needs-review sections.
- **Signal Traceability:** Engine that links each flagged ADR signal to the specific CTD regulatory sections that must document it, with gap analysis and priority actions.
- **IBM Bob MCP Integration:** All pharmacovigilance pipeline capabilities exposed as IBM Bob-callable MCP tools — signal detection, PRR calculation, trend analysis, dossier checking, and regulatory gap reports.

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| **Languages** | Python, TypeScript |
| **Frameworks** | FastAPI, React, Vite, Tailwind CSS, Pydantic |
| **IBM Technologies** | IBM watsonx.ai, IBM Bob, IBM Bob MCP Server |
| **Databases** | DuckDB |
| **Other** | Uvicorn, pytest, pandas, scikit-learn, scipy, Recharts, react-router-dom |

---

## 📁 Repository Structure

```
├── src/
│   ├── backend/              # FastAPI application
│   │   ├── api/routes/       # signals.py, dossier.py, traceability.py
│   │   ├── core/             # PRR calculator, temporal analyser, priority scorer,
│   │   │                     # traceability engine, dossier parser, CTD knowledge base
│   │   ├── data/             # faers_sample.csv (bundled dataset)
│   │   └── main.py           # Application entry point
│   ├── frontend/             # React + Vite dashboard
│   │   └── src/
│   │       ├── pages/        # Dashboard, SignalDetection, SignalDetails,
│   │       │                 # SubmissionReadiness, GapReport, ActionBrief
│   │       └── components/   # Shared UI components
│   ├── mcp-server/           # IBM Bob MCP server
│   └── tests/                # pytest test suite (956 tests)
├── docs/                     # Written documentation
├── demo/                     # Screenshots and demo video link
├── presentation/             # Slide deck
└── submission.yaml           # Structured submission metadata
```

---

## ⚡ How to Run

### Backend (FastAPI)

```bash
cd src/backend

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment (watsonx.ai credentials optional)
cp ../env.example ../.env
# Edit .env with your WATSONX_API_KEY and WATSONX_PROJECT_ID if available

# Start the API server
uvicorn main:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
```

### Frontend (React)

```bash
cd src/frontend

npm install
npm run dev
# App: http://localhost:5173
```

### MCP Server (IBM Bob integration)

```bash
cd src/mcp-server

pip install -r requirements.txt
python server.py
# MCP server: http://localhost:8001
```

### Run Tests

```bash
cd src/backend
.venv\Scripts\python -m pytest ../tests/ -q
```

---

## 🖥️ Demo

| Artifact | Link |
|---|---|
| 📹 Demo Video | [See demo/demo-video-link.txt](demo/demo-video-link.txt) |
| 🖼️ Screenshots | [See demo/screenshots/](demo/screenshots/) |
| 📊 Presentation | [See presentation/](presentation/) |

---

## ⚠️ Known Limitations

- Uses a bundled synthetic FAERS sample dataset (~500 rows) — not a live FDA FAERS database connection.
- IBM watsonx.ai narrative generation is optional — all API endpoints work deterministically without credentials.
- No user authentication is implemented.
- The frontend is a hackathon prototype and is not production-hardened.

---

## 🏅 What We're Most Proud Of

The end-to-end signal traceability engine that links a detected statistical ADR signal directly to the specific ICH M4 CTD regulatory sections that must document it, and the IBM Bob MCP integration that makes every pipeline capability available as a conversational tool directly inside Bob — turning a traditionally manual regulatory workflow into an AI-assisted one.

---
