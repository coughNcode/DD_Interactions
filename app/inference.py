import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem

# ── Paths anchored to this file's location ─────────────────────────────────────
_APP_DIR      = Path(__file__).resolve().parent
_ROOT         = _APP_DIR.parent
_MODEL_PATH   = _ROOT / "models" / "graphsage_mvp.pt"
_EVIDENCE_CSV = _ROOT / "data" / "processed" / "ddinter_evidence.csv"
_TDC_CSV      = _ROOT / "data" / "processed" / "tdc_drugbank_ddi.csv"
_NAME_MAP_CSV = _ROOT / "data" / "drugbank_id_to_name.csv"

try:
    from torch_geometric.nn import SAGEConv
    _PYGEOM_OK = True
except ImportError:
    _PYGEOM_OK = False

try:
    from admet_ai import ADMETModel
    admet_model = ADMETModel()
except ImportError:
    admet_model = None


# ── Model Architecture (must match train_graphsage.py exactly) ─────────────────
if _PYGEOM_OK:
    class GraphSAGEEncoder(torch.nn.Module):
        def __init__(self, in_channels=1024, hidden_channels=128, out_channels=64):
            super().__init__()
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, out_channels)

        def forward(self, x, edge_index):
            x = F.relu(self.conv1(x, edge_index))
            return F.relu(self.conv2(x, edge_index))


    class EdgeDecoder(torch.nn.Module):
        def __init__(self, in_channels=64, hidden_channels=32):
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


# ── Panel 1: ADMET ──────────────────────────────────────────────────────────────
def get_admet_profile(smiles: str):
    if admet_model is None:
        return {"hERG Toxicity": "N/A", "Blood-Brain Barrier": "N/A", "CYP3A4 Inhibition": "N/A"}
    try:
        # admet_model.predict() returns a DataFrame — take first row as dict
        result = admet_model.predict(smiles)
        if isinstance(result, pd.DataFrame):
            preds = result.iloc[0].to_dict()
        else:
            preds = result  # legacy dict path
        return {
            "hERG Toxicity":        round(float(preds.get("hERG", 0.0)), 3),
            "Blood-Brain Barrier":  round(float(preds.get("BBB_Martins", 0.0)), 3),
            "CYP3A4 Inhibition":    round(float(preds.get("CYP3A4_Inhibitor", 0.0)), 3),
        }
    except Exception:
        return {"hERG Toxicity": "Error", "Blood-Brain Barrier": "Error", "CYP3A4 Inhibition": "Error"}


# ── Panel 2: GraphSAGE Risk Score ───────────────────────────────────────────────
def predict_ddi_risk(smiles_a: str, smiles_b: str, model_path: Path = _MODEL_PATH):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not _PYGEOM_OK:
        raise ImportError("torch_geometric not available")

    model = DDIModel()
    model.load_state_dict(torch.load(str(model_path), map_location="cpu", weights_only=True))
    model.eval()

    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)

    fp_a = np.array(AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, 1024), dtype=np.float32) if mol_a else np.zeros(1024, dtype=np.float32)
    fp_b = np.array(AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, 1024), dtype=np.float32) if mol_b else np.zeros(1024, dtype=np.float32)

    x = torch.tensor(np.array([fp_a, fp_b]), dtype=torch.float)
    edge_index_msg  = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_index_pred = torch.tensor([[0], [1]], dtype=torch.long)

    with torch.no_grad():
        return float(model(x, edge_index_msg, edge_index_pred).item())


# ── Panel 3: Clinical Evidence (DDInter — best-effort by drug name) ─────────────
def get_clinical_evidence(drug_a_name: str, drug_b_name: str,
                          csv_path: Path = _EVIDENCE_CSV):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)

    da = str(drug_a_name).lower().strip()
    db = str(drug_b_name).lower().strip()

    # DDInter columns: DDInterID_A, Drug_A, DDInterID_B, Drug_B, Level
    name_a_col  = next((c for c in df.columns if c.lower() in ["drug_a", "drug1"]), df.columns[1])
    name_b_col  = next((c for c in df.columns if c.lower() in ["drug_b", "drug2"]), df.columns[3])
    level_col   = next((c for c in df.columns if c.lower() in ["level", "severity"]), None)

    a_ser = df[name_a_col].astype(str).str.lower().str.strip()
    b_ser = df[name_b_col].astype(str).str.lower().str.strip()

    match = df[
        ((a_ser == da) & (b_ser == db)) |
        ((a_ser == db) & (b_ser == da))
    ]

    if not match.empty:
        row = match.iloc[0]
        return {
            "Level":     str(row[level_col]) if level_col else "Documented Interaction",
            "Mechanism": "No mechanism details available in DDInter dataset.",
        }
    return None  # "no evidence found" — reported by caller, not silently hidden


# ── DrugBank ID → human-readable name mapping ──────────────────────────────────
def load_drugbank_name_map(csv_path: Path = _NAME_MAP_CSV) -> dict:
    """Returns {drugbank_id: human_name}. Falls back to raw ID if CSV missing."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    return dict(zip(df["drugbank_id"].astype(str), df["name"].astype(str)))


# ── Drug catalogue for UI dropdown ─────────────────────────────────────────────
def load_tdc_drug_catalogue(
    tdc_csv: Path = _TDC_CSV,
    name_map_csv: Path = _NAME_MAP_CSV,
) -> dict:
    """
    Returns {display_label: (drugbank_id, smiles)} for all 1,705 TDC DrugBank drugs.
    display_label = human-readable drug name (e.g. 'Aspirin'); falls back to raw DB ID.
    Callers must unpack the tuple to get the DB ID (for evidence lookup) and SMILES
    (for GraphSAGE prediction).
    """
    tdc_csv = Path(tdc_csv)
    if not tdc_csv.exists():
        return {
            "Aspirin": ("DB00945", "CC(=O)OC1=CC=CC=C1C(=O)O"),
            "Imatinib": ("DB00619", "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CC=CC=N5"),
        }

    name_map = load_drugbank_name_map(name_map_csv)

    df = pd.read_csv(tdc_csv)
    id_to_smiles: dict = {}
    for _, row in df.iterrows():
        d1, s1 = str(row["Drug1_ID"]), str(row["Drug1"])
        d2, s2 = str(row["Drug2_ID"]), str(row["Drug2"])
        if d1 not in id_to_smiles and s1 and s1 != "nan":
            id_to_smiles[d1] = s1
        if d2 not in id_to_smiles and s2 and s2 != "nan":
            id_to_smiles[d2] = s2

    catalogue: dict = {}
    for db_id, smiles in id_to_smiles.items():
        label = name_map.get(db_id, db_id)  # fallback: raw DB ID for 5 unresolved
        catalogue[label] = (db_id, smiles)

    return catalogue