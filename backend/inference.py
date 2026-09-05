"""
backend/inference.py — DDI-Guard inference engine (stateless, startup-loaded resources).

Key design decisions vs. app/inference.py:
  • All heavy resources (GraphSAGE model, ADMET-AI, DDInter DataFrame, drug catalogue)
    are loaded ONCE at module-import time via module-level singletons.  Every /analyze
    call reuses them — no per-request disk I/O or model instantiation (fixes Bug 1).

  • DDInter lookup now uses DrugBank canonical names (via drugbank_id_to_name.csv) as
    the primary match surface, with a built-in synonym alias table as secondary fallback
    and explicit diagnostic codes returned when no match is found (fixes Bug 2).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger("ddi_guard.inference")

# ── Path resolution — anchored to this file's location ────────────────────────
_BACKEND_DIR  = Path(__file__).resolve().parent
_ROOT         = _BACKEND_DIR.parent
_MODEL_PATH   = _ROOT / "models" / "graphsage_mvp.pt"
_EVIDENCE_CSV = _ROOT / "ddinter_evidence.csv"        # raw 156k-row DDInter master
_TDC_CSV      = _ROOT / "data" / "processed" / "tdc_drugbank_ddi.csv"
_NAME_MAP_CSV = _ROOT / "data" / "drugbank_id_to_name.csv"

# ── 1. GraphSAGE architecture (identical to train_graphsage.py) ───────────────
try:
    from torch_geometric.nn import SAGEConv
    _PYGEOM_OK = True
except ImportError:
    _PYGEOM_OK = False
    logger.warning("torch_geometric not available — GraphSAGE risk scoring disabled.")


if _PYGEOM_OK:
    class GraphSAGEEncoder(torch.nn.Module):
        def __init__(self, in_channels: int = 1024, hidden_channels: int = 128, out_channels: int = 64):
            super().__init__()
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, out_channels)

        def forward(self, x, edge_index):
            x = F.relu(self.conv1(x, edge_index))
            return F.relu(self.conv2(x, edge_index))

    class EdgeDecoder(torch.nn.Module):
        def __init__(self, in_channels: int = 64, hidden_channels: int = 32):
            super().__init__()
            self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
            self.lin2 = torch.nn.Linear(hidden_channels, 1)

        def forward(self, z, edge_index):
            x = F.relu(self.lin1(z[edge_index[0]] * z[edge_index[1]]))
            return torch.sigmoid(self.lin2(x)).squeeze()

    class DDIModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = GraphSAGEEncoder()
            self.decoder = EdgeDecoder()

        def forward(self, x, edge_index_msg, edge_index_pred):
            z = self.encoder(x, edge_index_msg)
            return self.decoder(z, edge_index_pred)


# ── 2. Module-level singletons (populated by _load_all_resources) ─────────────
_gnn_model = None
_admet_model = None
_ddinter_df: pd.DataFrame | None = None
_ddinter_names_lower: set = set()
_drug_catalogue: dict = {}
_canonical_alias: dict = {}


def _build_alias_table(ddinter_df: pd.DataFrame) -> dict:
    """
    Build {alias_lower -> canonical_cased_DDInter_name}.

    Step 1: every DDInter drug name maps to itself (case-insensitive identity).
    Step 2: hardcoded INN/trade name aliases for high-frequency mismatches.

    This directly fixes Bug 2: users type 'Aspirin' but DDInter stores
    'Acetylsalicylic acid'. The alias table bridges that gap transparently.
    """
    all_names = (
        set(ddinter_df["Drug_A"].dropna().unique()) |
        set(ddinter_df["Drug_B"].dropna().unique())
    )
    alias = {n.lower().strip(): n for n in all_names}

    MANUAL_ALIASES = {
        "aspirin":              "Acetylsalicylic acid",
        "acetylsalicylate":     "Acetylsalicylic acid",
        "tylenol":              "Acetaminophen",
        "paracetamol":          "Acetaminophen",
        "coumadin":             "Warfarin",
        "motrin":               "Ibuprofen",
        "advil":                "Ibuprofen",
        "tegretol":             "Carbamazepine",
        "dilantin":             "Phenytoin",
        "glucophage":           "Metformin",
        "lipitor":              "Atorvastatin",
        "zocor":                "Simvastatin",
        "prozac":               "Fluoxetine",
        "zoloft":               "Sertraline",
        "plavix":               "Clopidogrel",
        "prilosec":             "Omeprazole",
        "nexium":               "Esomeprazole",
        "amoxil":               "Amoxicillin",
        "cipro":                "Ciprofloxacin",
        "deltasone":            "Prednisone",
        "lasix":                "Furosemide",
        "lanoxin":              "Digoxin",
        "xanax":                "Alprazolam",
        "valium":               "Diazepam",
        "ambien":               "Zolpidem",
        "synthroid":            "Levothyroxine",
        "hctz":                 "Hydrochlorothiazide",
    }

    for alias_key, canonical in MANUAL_ALIASES.items():
        if canonical.lower().strip() in alias:
            alias[alias_key.lower().strip()] = canonical

    return alias


def _load_all_resources() -> None:
    """Load every heavy resource exactly once. Called from FastAPI lifespan startup."""
    global _gnn_model, _admet_model, _ddinter_df
    global _ddinter_names_lower, _drug_catalogue, _canonical_alias

    # GraphSAGE model
    if _PYGEOM_OK and _MODEL_PATH.exists():
        model = DDIModel()
        model.load_state_dict(
            torch.load(str(_MODEL_PATH), map_location="cpu", weights_only=True)
        )
        model.eval()
        _gnn_model = model
        logger.info("GraphSAGE model loaded from %s", _MODEL_PATH)
    else:
        logger.warning(
            "GraphSAGE model NOT loaded (pygeom_ok=%s, model_exists=%s)",
            _PYGEOM_OK, _MODEL_PATH.exists(),
        )

    # ADMET-AI
    try:
        from admet_ai import ADMETModel
        _admet_model = ADMETModel()
        logger.info("ADMET-AI model loaded.")
    except Exception as exc:
        logger.warning("ADMET-AI not available: %s", exc)

    # DDInter clinical evidence
    if _EVIDENCE_CSV.exists():
        _ddinter_df = pd.read_csv(_EVIDENCE_CSV)
        _ddinter_names_lower = {
            n.lower().strip()
            for col in ("Drug_A", "Drug_B")
            for n in _ddinter_df[col].dropna().unique()
        }
        _canonical_alias = _build_alias_table(_ddinter_df)
        logger.info(
            "DDInter loaded: %d rows, %d unique drug names, %d alias entries.",
            len(_ddinter_df), len(_ddinter_names_lower), len(_canonical_alias),
        )
    else:
        logger.warning("DDInter evidence CSV not found at %s", _EVIDENCE_CSV)

    # Drug catalogue
    _drug_catalogue = _build_drug_catalogue()
    logger.info("Drug catalogue loaded: %d entries.", len(_drug_catalogue))


# ── 3. Drug catalogue ──────────────────────────────────────────────────────────

def _build_drug_catalogue() -> dict:
    """Returns {display_name: (drugbank_id, smiles)} for all ~1,705 TDC drugs."""
    if not _TDC_CSV.exists():
        return {
            "Acetylsalicylic acid": ("DB00945", "CC(=O)OC1=CC=CC=C1C(=O)O"),
            "Imatinib": ("DB00619", "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CC=CC=N5"),
        }

    name_map: dict = {}
    if _NAME_MAP_CSV.exists():
        df_nm = pd.read_csv(_NAME_MAP_CSV)
        name_map = dict(zip(df_nm["drugbank_id"].astype(str), df_nm["name"].astype(str)))

    df = pd.read_csv(_TDC_CSV)
    id_to_smiles: dict = {}
    for _, row in df.iterrows():
        for id_col, smi_col in (("Drug1_ID", "Drug1"), ("Drug2_ID", "Drug2")):
            db_id = str(row[id_col])
            smiles = str(row[smi_col])
            if db_id not in id_to_smiles and smiles and smiles != "nan":
                id_to_smiles[db_id] = smiles

    catalogue: dict = {}
    for db_id, smiles in id_to_smiles.items():
        label = name_map.get(db_id, db_id)
        catalogue[label] = (db_id, smiles)
    return catalogue


# ── 4. Public inference functions ─────────────────────────────────────────────

def get_drug_catalogue() -> dict:
    """Return the startup-loaded drug catalogue."""
    return _drug_catalogue


def smiles_to_fp(smiles: str, radius: int = 2, n_bits: int = 1024) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits), dtype=np.float32)


def get_admet_profile(smiles: str) -> dict:
    """
    Panel 1 — ADMET pharmacokinetics.
    Uses startup-loaded singleton — never touches disk per request.
    """
    if _admet_model is None:
        return {"herg": "N/A", "bbb": "N/A", "cyp3a4": "N/A", "note": "ADMET-AI not available"}
    try:
        result = _admet_model.predict(smiles)
        preds = result.iloc[0].to_dict() if isinstance(result, pd.DataFrame) else result
        return {
            "herg":   round(float(preds.get("hERG",             0.0)), 3),
            "bbb":    round(float(preds.get("BBB_Martins",      0.0)), 3),
            "cyp3a4": round(float(preds.get("CYP3A4_Inhibitor", 0.0)), 3),
        }
    except Exception as exc:
        logger.error("ADMET prediction failed: %s", exc)
        return {"herg": "Error", "bbb": "Error", "cyp3a4": "Error", "error": str(exc)}


def predict_ddi_risk(smiles_a: str, smiles_b: str) -> dict:
    """
    Panel 2 — GraphSAGE DDI risk score.
    Bug 1 fix: uses startup-loaded _gnn_model — zero disk I/O per request.
    """
    if _gnn_model is None:
        return {"error": "GraphSAGE model not loaded."}

    fp_a = smiles_to_fp(smiles_a)
    fp_b = smiles_to_fp(smiles_b)

    x               = torch.tensor(np.array([fp_a, fp_b]), dtype=torch.float)
    edge_index_msg  = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_index_pred = torch.tensor([[0], [1]], dtype=torch.long)

    with torch.no_grad():
        score = float(_gnn_model(x, edge_index_msg, edge_index_pred).item())

    label = "Low Risk" if score < 0.4 else "Moderate Risk" if score < 0.7 else "High Risk"
    return {"score": round(score, 4), "status_label": label}


def _resolve_to_ddinter_name(name: str) -> tuple:
    """
    Map a user-supplied name to a canonical DDInter name.
    Returns (canonical_or_None, method_string).
    """
    key = name.lower().strip()
    if key in _canonical_alias:
        canonical = _canonical_alias[key]
        method = "exact" if key == canonical.lower().strip() else "alias"
        return canonical, method
    return None, "not_in_vocabulary"


def get_clinical_evidence(candidate_name: str, target_name: str) -> dict:
    """
    Panel 3 — DDInter clinical evidence lookup.

    Bug 2 fixes:
      • Case-insensitive, bidirectional matching.
      • Alias resolution: 'Aspirin' -> 'Acetylsalicylic acid', trade names, etc.
      • Diagnostic reason codes distinguish vocabulary misses from genuine absences.
    Always returns a dict with 'found: bool' — never returns None.
    """
    if _ddinter_df is None:
        return {
            "found": False,
            "reason": "ddinter_unavailable",
            "reason_text": "DDInter evidence CSV not loaded.",
        }

    canon_a, method_a = _resolve_to_ddinter_name(candidate_name)
    canon_b, method_b = _resolve_to_ddinter_name(target_name)

    logger.debug(
        "DDInter resolution: '%s'->(%s,%s) | '%s'->(%s,%s)",
        candidate_name, canon_a, method_a,
        target_name, canon_b, method_b,
    )

    # All vocabulary misses share a single reason code so the frontend only branches on two cases.
    # resolved_candidate/resolved_target carry whichever name(s) did resolve.
    if canon_a is None or canon_b is None:
        # Identify the name(s) that failed to resolve for the frontend message
        missing = []
        if canon_a is None:
            missing.append(f"'{candidate_name}'")
        if canon_b is None:
            missing.append(f"'{target_name}'")
        missing_str = " and ".join(missing)
        return {
            "found": False,
            "reason": "not_in_vocabulary",
            "reason_text": (
                f"{missing_str} {'is' if len(missing) == 1 else 'are'} not in DDInter's clinical "
                f"coverage (~1,194 of 1,705 catalogued drugs). DDInter simply does not have "
                f"data for {'this drug' if len(missing) == 1 else 'these drugs'}."
            ),
            "resolved_candidate": canon_a,   # None if unresolved, string if resolved
            "resolved_target":    canon_b,   # None if unresolved, string if resolved
            "missing_names":      missing,   # list of display names that failed
        }

    # Bidirectional case-insensitive match
    a_ser  = _ddinter_df["Drug_A"].str.strip().str.lower()
    b_ser  = _ddinter_df["Drug_B"].str.strip().str.lower()
    ca_low = canon_a.lower()
    cb_low = canon_b.lower()

    mask  = ((a_ser == ca_low) & (b_ser == cb_low)) | ((a_ser == cb_low) & (b_ser == ca_low))
    match = _ddinter_df[mask]

    if match.empty:
        return {
            "found": False,
            "reason": "no_record",
            "reason_text": (
                f"Both drugs in DDInter vocabulary (candidate='{canon_a}', "
                f"target='{canon_b}'), but no interaction record between them. "
                "Genuine 'no evidence on file'."
            ),
            "resolved_candidate": canon_a,
            "resolved_target": canon_b,
        }

    row       = match.iloc[0]
    level_col = next((c for c in _ddinter_df.columns if c.lower() in ("level", "severity")), None)
    severity  = str(row[level_col]).strip() if level_col else "Documented"

    return {
        "found": True,
        "reason": "found",
        "severity": severity,
        "mechanism_text": (
            "No mechanism text in this DDInter dataset. "
            "See DDInter online for full pharmacokinetic details."
        ),
        "resolved_candidate": canon_a,
        "resolved_target": canon_b,
        "resolution_methods": {"candidate": method_a, "target": method_b},
    }
