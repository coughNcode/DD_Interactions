import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.nn import SAGEConv

# ── Paths anchored to this file's location ─────────────────────────────────────
_APP_DIR  = Path(__file__).resolve().parent
_ROOT     = _APP_DIR.parent
_MODEL_PATH  = _ROOT / "models" / "graphsage_mvp.pt"
_EVIDENCE_CSV = _ROOT / "data" / "processed" / "ddinter_evidence.csv"

try:
    from admet_ai import ADMETModel
    admet_model = ADMETModel()
except ImportError:
    admet_model = None

# ── Model Architecture (must match train_graphsage.py exactly) ─────────────────
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

# Panel 1: ADMET
def get_admet_profile(smiles: str):
    if admet_model is None:
        return {"hERG": "N/A", "BBB": "N/A", "CYP3A4": "N/A"}
    try:
        preds = admet_model.predict(smiles) # Returns a dictionary for single SMILES
        return {
            "hERG Toxicity": round(preds.get("hERG", 0.0), 3),
            "Blood-Brain Barrier": round(preds.get("BBB_Martins", 0.0), 3),
            "CYP3A4 Inhibition": round(preds.get("CYP3A4_Inhibitor", 0.0), 3)
        }
    except Exception:
        return {"hERG Toxicity": "Error", "Blood-Brain Barrier": "Error", "CYP3A4 Inhibition": "Error"}

# Panel 2: GraphSAGE Risk Score
def predict_ddi_risk(smiles_a: str, smiles_b: str, model_path: str = "models/graphsage_mvp.pt"):
    if not os.path.exists(model_path): raise FileNotFoundError
    model = DDIModel()
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    
    fp_a = np.array(AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, 1024), dtype=np.float32) if mol_a else np.zeros(1024, dtype=np.float32)
    fp_b = np.array(AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, 1024), dtype=np.float32) if mol_b else np.zeros(1024, dtype=np.float32)
    
    x = torch.tensor(np.array([fp_a, fp_b]), dtype=torch.float)
    edge_index_msg = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_index_pred = torch.tensor([[0], [1]], dtype=torch.long)
    
    with torch.no_grad():
        return float(model(x, edge_index_msg, edge_index_pred).item())

# Panel 3: Clinical Evidence
def get_clinical_evidence(drug_a_name: str, drug_b_name: str, csv_path: str = "data/processed/merged_ddi.csv"):
    if not os.path.exists(csv_path): return None
    df = pd.read_csv(csv_path)
    
    da, db = str(drug_a_name).lower(), str(drug_b_name).lower()
    
    # Locate column names dynamically
    id1_col = next((c for c in df.columns if c.lower() in ['drug1_id', 'drug_a']), df.columns[0])
    id2_col = next((c for c in df.columns if c.lower() in ['drug2_id', 'drug_b']), df.columns[1])
    level_col = next((c for c in df.columns if c.lower() in ['level', 'severity']), None)
    mech_col = next((c for c in df.columns if c.lower() in ['mechanism', 'description']), None)
    
    match = df[
        ((df[id1_col].astype(str).str.lower() == da) & (df[id2_col].astype(str).str.lower() == db)) |
        ((df[id1_col].astype(str).str.lower() == db) & (df[id2_col].astype(str).str.lower() == da))
    ]
    
    if not match.empty:
        row = match.iloc[0]
        return {
            "Level": str(row[level_col]) if level_col else "Documented Interaction",
            "Mechanism": str(row[mech_col]) if mech_col else "No mechanism details provided in dataset."
        }
    return None