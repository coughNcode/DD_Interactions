"""
backend/main.py — DDI-Guard FastAPI backend.

Endpoints:
  GET  /drugs    -> sorted list of ~1,700 drug display names for the frontend dropdown
  POST /analyze  -> full 3-panel analysis: ADMET + GraphSAGE risk + DDInter evidence

Design:
  • All heavy resources (GNN model, ADMET, DDInter CSV, drug catalogue) are loaded
    ONCE at application startup via the FastAPI lifespan, not per-request.
  • Every /analyze call is fully stateless — results are always freshly computed
    from the request body, eliminating Bug 1 (stale cached results from Streamlit).
  • CORS enabled for http://localhost:8501 (Streamlit default port).
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure the backend package is importable regardless of working directory
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import inference as _inf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("ddi_guard.api")


# ── Startup / shutdown lifespan ───────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all heavy resources once at startup; release on shutdown."""
    logger.info("Loading DDI-Guard resources (GraphSAGE, ADMET-AI, DDInter, catalogue)...")
    _inf._load_all_resources()
    logger.info("All resources loaded. Server ready.")
    yield
    logger.info("DDI-Guard server shutting down.")


# ── FastAPI application ───────────────────────────────────────────────────────

app = FastAPI(
    title="DDI-Guard API",
    description=(
        "Drug-Drug Interaction risk assessment backend. "
        "Combines GraphSAGE GNN predictions, ADMET-AI pharmacokinetics, "
        "and DDInter clinical evidence into a single /analyze endpoint."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",   # Streamlit default
        "http://127.0.0.1:8501",
        "http://localhost:3000",   # future React frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    candidate_name:  str = Field(..., description="Human-readable candidate drug name (e.g. 'Aspirin').")
    candidate_smiles: str = Field(..., description="SMILES string for the candidate drug.")
    target_drug:     str = Field(..., description="Display name of the target drug (must match a key from GET /drugs).")


class AdmetProfile(BaseModel):
    herg:   float | str
    bbb:    float | str
    cyp3a4: float | str
    note:   Optional[str] = None
    error:  Optional[str] = None


class InteractionRisk(BaseModel):
    score:        Optional[float] = None
    status_label: Optional[str]  = None
    error:        Optional[str]  = None


class ClinicalEvidence(BaseModel):
    found:              bool
    reason:             str
    reason_text:        Optional[str]  = None
    severity:           Optional[str]  = None
    mechanism_text:     Optional[str]  = None
    resolved_candidate: Optional[str]  = None
    resolved_target:    Optional[str]  = None
    resolution_methods: Optional[dict] = None


class AnalyzeResponse(BaseModel):
    candidate_name:  str
    target_drug:     str
    target_smiles:   Optional[str]   = None
    admet_profile:   AdmetProfile
    interaction_risk: InteractionRisk
    clinical_evidence: ClinicalEvidence


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/drugs", response_model=list[str], summary="List all available target drugs")
async def list_drugs() -> list[str]:
    """
    Returns a sorted list of human-readable drug names for the frontend dropdown.
    Each name can be passed directly as `target_drug` in POST /analyze.
    """
    catalogue = _inf.get_drug_catalogue()
    if not catalogue:
        raise HTTPException(status_code=503, detail="Drug catalogue not loaded. Check server startup logs.")
    return sorted(catalogue.keys())


@app.post("/analyze", response_model=AnalyzeResponse, summary="Full 3-panel DDI analysis")
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Runs all three analysis panels for a candidate↔target drug pair.

    - **admet_profile**: ADMET-AI pharmacokinetic properties of the candidate SMILES.
    - **interaction_risk**: GraphSAGE predicted DDI risk score (0–1) with risk label.
    - **clinical_evidence**: DDInter record lookup with diagnostic reason codes.

    Every call is fully stateless — results are always freshly computed from the
    request body. No Streamlit session_state or server-side caching involved.
    """
    catalogue = _inf.get_drug_catalogue()

    # Validate target drug
    if request.target_drug not in catalogue:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown target drug '{request.target_drug}'. Use GET /drugs to see valid options.",
        )

    _db_id, target_smiles = catalogue[request.target_drug]

    logger.info(
        "Analyzing: candidate='%s' target='%s'",
        request.candidate_name, request.target_drug,
    )

    # Panel 1: ADMET
    admet_raw = _inf.get_admet_profile(request.candidate_smiles)

    # Panel 2: GraphSAGE
    risk_raw = _inf.predict_ddi_risk(request.candidate_smiles, target_smiles)

    # Panel 3: DDInter clinical evidence
    evidence_raw = _inf.get_clinical_evidence(request.candidate_name, request.target_drug)

    return AnalyzeResponse(
        candidate_name   = request.candidate_name,
        target_drug      = request.target_drug,
        target_smiles    = target_smiles,
        admet_profile    = AdmetProfile(**admet_raw),
        interaction_risk = InteractionRisk(**risk_raw),
        clinical_evidence= ClinicalEvidence(**evidence_raw),
    )


@app.get("/health", summary="Health check")
async def health():
    """Quick liveness probe — confirms the server is up and resources are loaded."""
    catalogue = _inf.get_drug_catalogue()
    return {
        "status": "ok",
        "drug_catalogue_size": len(catalogue),
        "ddinter_loaded": _inf._ddinter_df is not None,
        "gnn_loaded":     _inf._gnn_model is not None,
        "admet_loaded":   _inf._admet_model is not None,
    }
