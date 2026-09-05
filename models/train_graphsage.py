import os
import time
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import SAGEConv
from sklearn.metrics import roc_auc_score

# ── Paths anchored to this file's location ─────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT       = SCRIPT_DIR.parent
CSV_PATH   = ROOT / "data" / "processed" / "tdc_training_edges.csv"
MODEL_PATH = ROOT / "models" / "graphsage_mvp.pt"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)


def smiles_to_fp(smiles: str, radius: int = 2, n_bits: int = 1024) -> np.ndarray:
    """Convert SMILES to Morgan fingerprint; returns zero vector on failure."""
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return np.zeros(n_bits, dtype=np.float32)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp, dtype=np.float32)
    except Exception:
        return np.zeros(n_bits, dtype=np.float32)


def load_and_build_graph(csv_path: Path = CSV_PATH, max_edges: int = 40_000):
    print(f"Loading TDC training edges from {csv_path}...")

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing {csv_path}. Run the data pipeline script first."
        )

    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df):,} rows | columns: {list(df.columns)}")

    # ── Column mapping (rigid — tdc_training_edges.csv has fixed schema) ─────
    id1_col     = "drug1_id"
    id2_col     = "drug2_id"
    smiles1_col = "drug1_smiles"
    smiles2_col = "drug2_smiles"
    label_col   = "label"

    for col in [id1_col, id2_col, smiles1_col, smiles2_col, label_col]:
        if col not in df.columns:
            raise ValueError(
                f"Expected column '{col}' not found. Columns: {list(df.columns)}"
            )

    # Use only positive edges for graph structure; RandomLinkSplit adds negatives.
    df_pos = df[df[label_col] == 1].copy()
    if len(df_pos) > max_edges:
        df_pos = df_pos.sample(n=max_edges, random_state=42).reset_index(drop=True)
    print(f"  Positive edges used for graph: {len(df_pos):,}")

    # ── Build node index + fingerprints ─────────────────────────────────────
    smiles_cache: dict = {}
    for _, row in df_pos.iterrows():
        smiles_cache[str(row[id1_col])] = str(row[smiles1_col])
        smiles_cache[str(row[id2_col])] = str(row[smiles2_col])

    unique_drugs: dict = {}
    node_features: list = []
    edge_list: list = []
    fp_failures = 0

    print("  Generating Morgan fingerprints for unique drugs...")
    for _, row in df_pos.iterrows():
        id1 = str(row[id1_col])
        id2 = str(row[id2_col])

        if id1 not in unique_drugs:
            fp = smiles_to_fp(smiles_cache[id1])
            if fp.sum() == 0:
                fp_failures += 1
            unique_drugs[id1] = len(unique_drugs)
            node_features.append(fp)

        if id2 not in unique_drugs:
            fp = smiles_to_fp(smiles_cache[id2])
            if fp.sum() == 0:
                fp_failures += 1
            unique_drugs[id2] = len(unique_drugs)
            node_features.append(fp)

        edge_list.append([unique_drugs[id1], unique_drugs[id2]])

    print(f"  Unique drugs processed: {len(unique_drugs):,}")
    print(f"  Fingerprint failures (zero vectors): {fp_failures}")

    x = torch.tensor(np.array(node_features), dtype=torch.float)
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    data = Data(x=x, edge_index=edge_index)
    print(f"  Graph: {data.num_nodes:,} nodes, {data.num_edges:,} edges")
    return data


# ── Model Architecture ──────────────────────────────────────────────────────────
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
        edge_feat = z[edge_index[0]] * z[edge_index[1]]  # Hadamard product
        x = F.relu(self.lin1(edge_feat))
        return torch.sigmoid(self.lin2(x)).squeeze()


class DDIModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = GraphSAGEEncoder()
        self.decoder = EdgeDecoder()

    def forward(self, x, edge_index_msg, edge_index_pred):
        z = self.encoder(x, edge_index_msg)
        return self.decoder(z, edge_index_pred)


# ── Training ────────────────────────────────────────────────────────────────────
def train():
    graph_data = load_and_build_graph()

    # LEAKAGE PREVENTION: strict train/val/test split on edges
    transform = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=True,
    )
    train_data, val_data, test_data = transform(graph_data)

    # Device detection — explicit report (M1-aware, no CUDA check)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\nDevice: CUDA — {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("\nDevice: MPS (Apple Silicon M1) — torch.device('mps')")
        print("NOTE: Some torch_geometric scatter ops may lack MPS kernels; will fall back to CPU if NotImplementedError is raised.")
    else:
        device = torch.device("cpu")
        print("\nDevice: CPU (neither CUDA nor MPS available)")

    def _run_training(device):
        """Inner training loop — lower LR, validate every epoch, save BEST checkpoint."""
        nonlocal model, optimizer
        # LR lowered 0.005 → 0.001 for smoother convergence
        model = DDIModel().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        td = train_data.to(device)
        vd = val_data.to(device)

        print(f"\nActual device used for training: {device}")
        print(f"LR: 0.001 | Epochs: 30 | Checkpoint: best val ROC-AUC")
        print(f"Training split:   {td.edge_label_index.shape[1]:,} edges")
        print(f"Validation split: {vd.edge_label_index.shape[1]:,} edges")
        print(f"Test split:       {test_data.edge_label_index.shape[1]:,} edges")
        print("\nStarting training (30 epochs, saving best val AUC checkpoint)...")

        best_val_roc  = -1.0
        best_epoch    = -1
        best_state    = None

        t0 = time.time()
        for epoch in range(1, 31):
            # ── Train step ────────────────────────────────────────────────────
            model.train()
            optimizer.zero_grad()
            pred = model(td.x, td.edge_index, td.edge_label_index)
            loss = F.binary_cross_entropy(pred, td.edge_label.float())
            loss.backward()
            optimizer.step()

            # ── Validate every epoch ──────────────────────────────────────────
            model.eval()
            with torch.no_grad():
                val_pred = model(vd.x, vd.edge_index, vd.edge_label_index)
                val_roc  = roc_auc_score(vd.edge_label.cpu(), val_pred.cpu())

            # ── Track best ───────────────────────────────────────────────────
            if val_roc > best_val_roc:
                best_val_roc = val_roc
                best_epoch   = epoch
                # Deep-copy state dict to CPU so we keep it safe
                best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            elapsed = time.time() - t0
            marker = " ★ BEST" if epoch == best_epoch else ""
            print(f"  Epoch {epoch:03d} | Loss: {loss:.4f} | Val ROC-AUC: {val_roc:.4f} | {elapsed:.1f}s{marker}")

        total_time = time.time() - t0
        print(f"\nTraining complete in {total_time:.1f}s")
        print(f"Best checkpoint: Epoch {best_epoch:03d} | Val ROC-AUC = {best_val_roc:.4f}")

        # Restore best weights into model so caller saves the best, not last
        model.load_state_dict(best_state)

    model = optimizer = None
    try:
        _run_training(device)
    except NotImplementedError as e:
        if device.type == "mps":
            print(f"\n⚠️  MPS NotImplementedError: {e}")
            print("   Falling back to device = torch.device('cpu') and restarting training.")
            device = torch.device("cpu")
            _run_training(device)
        else:
            raise

    # Save BEST checkpoint (model.state_dict() was restored to best inside _run_training)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Best checkpoint saved → {MODEL_PATH}")

    # Verify reload
    print("\nVerifying saved checkpoint...")
    sd = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    print("State dict keys and shapes:")
    for k, v in sd.items():
        print(f"  {k}: {tuple(v.shape)}")


if __name__ == "__main__":
    train()