"""
main.py — PharmaGuard AI FastAPI application entry point.

Run with:
    uvicorn main:app --reload --port 8000

Swagger UI available at:
    http://localhost:8000/docs
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings


# ---------------------------------------------------------------------------
# Lifespan: runs once at startup and once at shutdown.
# Heavy resources (data loading, model init) go here in later phases.
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print(f"PharmaGuard AI v{settings.app_version} starting up...")
    print(f"Environment : {settings.app_env}")
    print(f"watsonx.ai  : {'available' if settings.watsonx_available else 'not configured (will use fallback)'}")
    yield
    # Shutdown
    print("PharmaGuard AI shutting down.")


# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------
app = FastAPI(
    title="PharmaGuard AI",
    description=(
        "Pharmacovigilance and regulatory submission readiness platform. "
        "Detects adverse drug reaction safety signals from FDA FAERS data, "
        "checks ICH M4 CTD dossier completeness, and traces safety signals "
        "to required regulatory submission sections."
    ),
    version=settings.app_version,
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — allow the React frontend (Vite default port 5173) in development
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://localhost:3000",   # fallback for CRA / other
        "http://localhost:8000",   # same-origin (e.g. Swagger UI)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers — uncommented phase by phase as routes are implemented
# ---------------------------------------------------------------------------
# from api.routes.signals       import router as signals_router       # Phase 9
# from api.routes.dossier       import router as dossier_router       # Phase 9
# from api.routes.traceability  import router as traceability_router  # Phase 9

# app.include_router(signals_router,      prefix="/api/signals",      tags=["Signal Detection"])
# app.include_router(dossier_router,      prefix="/api/dossier",      tags=["Dossier Readiness"])
# app.include_router(traceability_router, prefix="/api/traceability", tags=["Traceability"])


# ---------------------------------------------------------------------------
# Core endpoints
# ---------------------------------------------------------------------------
@app.get("/health", tags=["System"])
def health_check():
    """
    Liveness probe. Returns the application version and current environment.
    Use this to verify the server is running before running any other command.
    """
    return {
        "status": "ok",
        "version": settings.app_version,
        "environment": settings.app_env,
        "watsonx_configured": settings.watsonx_available,
    }
