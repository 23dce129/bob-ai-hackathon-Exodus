# Setup Guide

> **This file is read by the automated evaluation pipeline. Be precise and complete.**

## Prerequisites

Ensure the following are installed before you begin:

- **Python 3.11 or 3.12** — [Download](https://www.python.org/downloads/)
  - Verify: `python --version`
- **Node.js 18 or 20** — [Download](https://nodejs.org/)
  - Verify: `node --version` and `npm --version`
- **Git** — [Download](https://git-scm.com/)

> IBM watsonx.ai credentials are **optional**. All backend endpoints and the MCP server work without them. Credentials are only needed for the AI-generated narrative explanation feature at `GET /api/signals/explain/{drug}/{event}`.

---

## 1. Clone the Repository

```bash
git clone https://github.com/23dce129/bob-ai-hackathon-Exodus.git
cd bob-ai-hackathon-Exodus
```

---

## 2. Backend Setup

### 2.1 Create and Activate a Virtual Environment

```bash
cd src/backend

# Create virtual environment
python -m venv .venv

# Activate — Windows (PowerShell)
.venv\Scripts\activate

# Activate — macOS / Linux
# source .venv/bin/activate
```

### 2.2 Install Dependencies

```bash
pip install -r requirements.txt
```

This installs FastAPI, Uvicorn, pandas, scipy, scikit-learn, DuckDB, pydantic, and the IBM watsonx.ai SDK (~60 packages total).

### 2.3 Configure Environment Variables

```bash
# From the repo root:
cp src/.env.example src/.env
```

Edit `src/.env`. The minimum required content for local development (no watsonx.ai):

```dotenv
APP_PORT=8000
APP_ENV=development
PHARMAGUARD_API_URL=http://localhost:8000
MCP_SERVER_PORT=8001
```

To enable optional watsonx.ai narrative generation, also set:

```dotenv
WATSONX_API_KEY=your_api_key_here
WATSONX_PROJECT_ID=your_project_id_here
WATSONX_URL=https://us-south.ml.cloud.ibm.com
```

Get these values from [IBM Cloud](https://cloud.ibm.com/) → your watsonx.ai project → Manage → Credentials.

### 2.4 Start the Backend

```bash
# From src/backend/ with the virtual environment active:
uvicorn main:app --reload --port 8000
```

**Verify:** Open [http://localhost:8000/health](http://localhost:8000/health) — you should see:

```json
{"status": "ok", "version": "0.1.0", "environment": "development", "watsonx_configured": false}
```

**Swagger UI** (full interactive API docs): [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 3. Frontend Setup

Open a **new terminal** (keep the backend running).

```bash
cd src/frontend

npm install
npm run dev
```

**Verify:** Open [http://localhost:5173](http://localhost:5173) — the PharmaGuard AI dashboard should load.

> The Vite dev server automatically proxies all `/api/*` requests to `http://127.0.0.1:8000`, so the frontend and backend communicate without CORS issues.

### Frontend Production Build (optional)

```bash
cd src/frontend
npm run build
# Output: src/frontend/dist/
```

---

## 4. MCP Server Setup (IBM Bob Integration)

The MCP server is only needed if you want to use PharmaGuard AI tools through IBM Bob. The backend and frontend work without it.

### 4.1 Install MCP Server Dependencies

```bash
cd src/mcp-server
pip install -r requirements.txt
```

This installs FastMCP, httpx, pydantic-settings, and python-dotenv into whatever Python environment is currently active. You can reuse the backend virtual environment:

```bash
# If reusing the backend venv:
cd src/backend
.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r ../mcp-server/requirements.txt
```

### 4.2 Register with IBM Bob

The MCP server is already registered in `.bob/mcp.json`:

```json
{
  "mcpServers": {
    "pharmaguard": {
      "command": "python",
      "args": ["<absolute-path-to-repo>/src/mcp-server/server.py"],
      "env": {
        "PHARMAGUARD_API_URL": "http://localhost:8000",
        "PHARMAGUARD_MCP_TIMEOUT": "30",
        "PHARMAGUARD_MCP_MAX_RESULTS": "10"
      }
    }
  }
}
```

> **Important:** Update the `args` path to match your local clone location if it differs from the path already in `.bob/mcp.json`.

### 4.3 Use with IBM Bob

With the backend running, open IBM Bob. The following prompts will call the MCP tools:

- `"Are there any safety signals for warfarin?"` → calls `detect_safety_signals`
- `"What is the PRR for ibuprofen?"` → calls `calculate_prr`
- `"Is our NDA dossier complete?"` → calls `check_ctd_readiness`
- `"Show me documentation gaps for warfarin's bleeding signal."` → calls `generate_regulatory_action_brief`

### 4.4 Smoke Test Without Bob

```bash
cd src/mcp-server
python server.py --test
# Requires backend running at PHARMAGUARD_API_URL (default: http://localhost:8000)
```

---

## 5. Running the Test Suite

```bash
cd src/backend
.venv\Scripts\python -m pytest ../tests/ -v
# Windows (if venv is active): python -m pytest ../tests/ -v
# macOS/Linux: python -m pytest ../tests/ -v
```

Expected output: **956 passed**, 0 failed, 8 warnings (deprecation warnings from third-party libraries only).

To run a specific test file:

```bash
python -m pytest ../tests/test_prr_calculator.py -v
```

---

## 6. Environment Variables Reference

All variables are read from `src/.env` (which is never committed). Copy from `src/.env.example`:

| Variable | Default | Required | Description |
|---|---|---|---|
| `WATSONX_API_KEY` | _(empty)_ | No | IBM watsonx.ai API key. Enables `/explain` narratives. |
| `WATSONX_PROJECT_ID` | _(empty)_ | No | IBM watsonx.ai project ID. |
| `WATSONX_URL` | `https://us-south.ml.cloud.ibm.com` | No | watsonx.ai regional endpoint. |
| `APP_PORT` | `8000` | No | Backend server port. |
| `APP_ENV` | `development` | No | Environment label shown in `/health`. |
| `PHARMAGUARD_API_URL` | `http://localhost:8000` | No | URL the MCP server uses to reach the backend. |
| `MCP_SERVER_PORT` | `8001` | No | Port for the MCP server (informational only; server uses stdio transport). |
| `PHARMAGUARD_MCP_MAX_RESULTS` | `10` | No | Max signals/sections returned per MCP tool call. |
| `PHARMAGUARD_MCP_TIMEOUT` | `30` | No | HTTP timeout (seconds) for MCP → backend calls. |

---

## 7. Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `uvicorn: command not found` | Virtual environment not active | Run `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (macOS/Linux) |
| `ModuleNotFoundError: No module named 'fastapi'` | Dependencies not installed | Run `pip install -r requirements.txt` from `src/backend/` |
| Frontend shows "Network Error" on API calls | Backend not running | Start the backend: `uvicorn main:app --reload --port 8000` |
| Frontend loads but API data is empty | CORS / proxy misconfiguration | Use `npm run dev` (Vite dev server), not a static file server. Vite proxies `/api/*` to port 8000. |
| `/health` returns 500 | `faers_sample.csv` missing | Verify `src/backend/data/faers_sample.csv` exists |
| Bob shows "Cannot reach PharmaGuard backend" | Backend not running when Bob called the tool | Start the backend before using Bob's pharmacovigilance tools |
| watsonx.ai 401 error | Invalid API key or project ID | Check `WATSONX_API_KEY` and `WATSONX_PROJECT_ID` in `src/.env`. The endpoint works without watsonx credentials. |
| `pytest` reports import errors | Running pytest from wrong directory | Run from `src/backend/` with the virtual environment active |
