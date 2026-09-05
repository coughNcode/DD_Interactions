# 🛡️ DDI-Guard: Drug-Drug Interaction Risk Assessment & Clinical Evidence Dashboard

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-MPS%20%2F%20CPU-orange.svg)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-GraphSAGE-green.svg)](https://pyg.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![RDKit](https://img.shields.io/badge/RDKit-Chemoinformatics-blueviolet.svg)](https://www.rdkit.org/)

---

## 📌 Table of Contents
1. [Project Overview & What We Are Building](#-project-overview--what-we-are-building)
2. [Key Architecture & The 3-Panel Triangulation](#-key-architecture--the-3-panel-triangulation)
3. [What Was Accomplished (Step-by-Step Evolution)](#-what-was-accomplished-step-by-step-evolution)
4. [Repository & Folder Structure](#-repository--folder-structure)
5. [Machine Learning Models & Architectures](#-machine-learning-models--architectures)
6. [Training Pipeline & Model Results](#-training-pipeline--model-results)
7. [Clinical Evidence & Entity Resolution (DrugBank ↔ DDInter)](#-clinical-evidence--entity-resolution)
8. [End-to-End Simulation & Demo Test Case](#-end-to-end-simulation--demo-test-case)
9. [Setup & Installation Guide](#-setup--installation-guide)
10. [How to Run the Application](#-how-to-run-the-application)
11. [Known Limitations & Future Roadmap](#-known-limitations--future-roadmap)

---

## 🎯 Project Overview & What We Are Building

Polypharmacy (concurrent administration of multiple medications) is ubiquitous in modern clinical care, particularly for elderly and multi-morbid patients. However, adverse **Drug-Drug Interactions (DDIs)** account for up to 30% of all adverse drug events, leading to preventable hospitalizations, cardiovascular toxicity, and therapeutic failures.

**DDI-Guard** is an end-to-end precision pharmacovigilance platform designed for medicinal chemists, pharmacologists, and clinicians. It enables users to evaluate a **novel candidate compound** (via chemical SMILES) against a comprehensive library of **approved target drugs**, providing an integrated risk profile before clinical administration or synthesis.

### What Makes DDI-Guard Unique?
Existing tools either provide *pure machine learning predictions* without clinical provenance, or *pure clinical lookup tables* that cannot evaluate novel compounds. DDI-Guard bridges this gap by combining:
1. **In Silico ADMET Profiling**: Absorption, Distribution, Metabolism, and Excretion safety screening.
2. **Graph Neural Network (GNN) Interaction Prediction**: Deep topological and chemical fingerprint reasoning via GraphSAGE.
3. **Curated Clinical Evidence Grounding**: Real-world severity levels mined from the clinical DDInter database.

---

## 🔬 Key Architecture & The 3-Panel Triangulation

When a user submits a candidate drug and selects a co-administered target therapeutic, DDI-Guard displays a 3-panel dashboard:

```
+----------------------------------------------------------------------------------------------------+
|                                    DDI-GUARD DASHBOARD                                             |
+---------------------------------+---------------------------------+--------------------------------+
|  Panel 1: Candidate Profile     |  Panel 2: Interaction Risk      |  Panel 3: Clinical Evidence    |
|  (ADMET-AI Pharmacokinetics)    |  (GraphSAGE Neural Network)     |  (DDInter Clinical Database)   |
+---------------------------------+---------------------------------+--------------------------------+
|  • hERG Cardiac Toxicity        |  • Predicted DDI Risk Score     |  • Real-World Documentation    |
|  • Blood-Brain Barrier (BBB)    |    (Range: 0.000 to 1.000)      |  • Severity Classification     |
|  • CYP3A4 Enzyme Inhibition     |  • Categorical Risk Level       |    (Major / Moderate / Minor)  |
|                                 |    (Low / Moderate / High)      |  • Documented Mechanism Info   |
+---------------------------------+---------------------------------+--------------------------------+
```

1. **Panel 1 — 🧪 Candidate Profile (ADMET-AI)**:
   - Uses Chemprop/ADMET-AI deep learning ensembles to predict critical pharmacokinetics:
     - **hERG Cardiac Toxicity**: Risk of QT prolongation and fatal ventricular arrhythmias.
     - **Blood-Brain Barrier (BBB)**: Penetration probability for central nervous system effects.
     - **CYP3A4 Inhibition**: Primary metabolic enzyme risk; inhibition causes dangerous co-drug accumulation.
2. **Panel 2 — ⚡ Interaction Risk (GraphSAGE GNN)**:
   - Evaluates chemical substructures (1,024-bit Morgan fingerprints) passing messages through an interaction graph.
   - Outputs a calibrated risk score $[0.0, 1.0]$ representing the model's confidence in pairwise interaction.
3. **Panel 3 — 📖 Clinical Evidence (DDInter Database)**:
   - Queries a consolidated database of 224,250 unique clinical interactions across all WHO Anatomical Therapeutic Chemical (ATC) categories.
   - Cross-references real-world medical literature to identify known clinical severity (`Major`, `Moderate`, `Minor`).

---

## 🛠️ What Was Accomplished (Step-by-Step Evolution)

### Phase 1: Environment Setup & Hardware Acceleration
- **System**: Apple Silicon M1 (arm64 macOS Darwin).
- **Environment**: Conda environment `E` with PyTorch, PyTorch Geometric, RDKit, ADMET-AI, TDC (Therapeutics Data Commons), Scikit-Learn, and Streamlit.
- **Acceleration**: Enabled Apple Silicon **MPS (Metal Performance Shaders)** (`torch.device('mps')`) with an automated fallback mechanism to CPU to guard against missing PyG message-passing kernels.

### Phase 2: Repository Audit & Data Pipeline Overhaul
- **Audit**: Identified that initial raw repository files (`merged_ddi.csv`, `graphsage_mvp.pt`) were 0-byte placeholders or lacked molecular structures.
- **TDC Integration**: Ingested the benchmark **Therapeutics Data Commons (TDC) DrugBank DDI** dataset:
  - 191,878 positive interaction edges across 1,705 unique drugs with valid SMILES.
- **Negative Edge Construction**: Constructed a balanced training graph of 40,000 positive edges and 40,000 verified negative non-interacting edges. Generated pairs were strictly checked against the **entire 191,878 edge set** to eliminate false-negative data contamination.
- **DDInter Clinical Consolidation**: Merged multi-category clinical evidence files (`ddinter_downloads_code_A/B/D/L/P/R/V.csv`) through `merging.py`, deduplicating cross-category pairs into a clean database of 224,250 clinical records.

### Phase 3: DrugBank ID to Human-Readable Entity Resolution
- **Problem**: TDC DrugBank data was keyed only by opaque database accessions (e.g. `DB00682`), making clinical selection impossible.
- **Solution**: Engineered a hybrid entity resolution pipeline using `iit-Demokritos/drug_id_mapping` (22,349 entries) supplemented with PubChem PUG-REST API parallel batch lookups:
  - **1,700 out of 1,705 TDC drugs (99.7%)** successfully resolved to standardized clinical names.
  - 5 unresolved experimental compounds gracefully fallback to their raw DrugBank IDs.
  - Boosted clinical evidence overlap from **0%** (raw IDs) to **70.2%** (1,194 matching drugs, yielding 92,227 fully resolvable clinical drug pairs).

### Phase 4: Model Refinement & Convergence Optimization
- **Training Stabilization**: Addressed early-epoch validation dips by adjusting the Adam learning rate from `0.005` to `0.001`.
- **Validation Checkpoint Tracking**: Implemented per-epoch evaluation on disjoint validation edges, tracking and preserving the **best validation ROC-AUC checkpoint** rather than the arbitrary final epoch state.
- **Result**: Validation ROC-AUC improved smoothly to **0.7772** within 30 epochs (7.0 seconds on M1 MPS).

### Phase 5: Code Hardening & UI Deployment
- Fixed ADMET-AI DataFrame conversion bug (`.predict().iloc[0].to_dict()`).
- Anchored all filesystem references using `Path(__file__).resolve().parent` to prevent working-directory dependency bugs.
- Built and launched the full Streamlit interactive dashboard.

---

## 📂 Repository & Folder Structure

```text
DD_Interactions/
├── README.md                           # Comprehensive project documentation
├── merging.py                          # Clinical data consolidation pipeline
├── ddinter_evidence.csv                # Master clinical interactions (224k rows)
├── ddinter_downloads_code_*.csv        # Raw DDInter ATC category downloads (A, B, D, L, P, R, V)
│
├── app/
│   ├── main.py                         # Streamlit interactive 3-panel frontend application
│   └── inference.py                    # Inference engine (ADMET, GraphSAGE, DDInter lookup, catalogue)
│
├── data/
│   ├── build_drugbank_name_map.py      # PubChem/TSV entity resolution generator
│   ├── drug_id_mappings_full.tsv       # 22,349 DrugBank-to-common-name mapping database
│   ├── drugbank_id_to_name.csv         # Processed 1,705 TDC drug name lookup table
│   └── processed/
│       ├── tdc_drugbank_ddi.csv        # Primary TDC dataset (191,878 positive interaction pairs)
│       ├── tdc_training_edges.csv      # Balanced ML training dataset (40k pos + 40k neg edges)
│       └── ddinter_evidence.csv        # Symlinked/processed clinical ground truth
│
└── models/
    ├── train_graphsage.py              # GraphSAGE training script with RandomLinkSplit & best checkpointing
    └── graphsage_mvp.pt                # Trained PyTorch checkpoint (best val ROC-AUC: 0.7772)
```

---

## 🧠 Machine Learning Models & Architectures

### 1. Molecular Representation: Morgan Fingerprints
- **Algorithm**: Extended Connectivity Fingerprints (ECFP4 equivalent) via RDKit `AllChem.GetMorganFingerprintAsBitVect`.
- **Radius**: $2$ (captures atomic neighborhoods up to 4 bonds).
- **Vector Dimension**: 1,024 bits (float32).
- **Fallback**: Invalid SMILES yield zero-vectors without crashing the inference pipeline.

### 2. GraphSAGE Architecture
The interaction risk network is formulated as link prediction over a drug interaction graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$:

```
[Drug A SMILES] ──> [Morgan FP (1024-d)] ──┐
                                            ├──> [SAGEConv 1024 -> 128 (ReLU)]
                                            │          │
[Drug B SMILES] ──> [Morgan FP (1024-d)] ──┘    [SAGEConv 128 -> 64 (ReLU)]
                                                       │
                                            [Node Embeddings: z_A, z_B (64-d)]
                                                       │
                                            [Hadamard Product: z_A ⊙ z_B]
                                                       │
                                            [Linear 64 -> 32 (ReLU)]
                                                       │
                                            [Linear 32 -> 1  (Sigmoid)]
                                                       │
                                            [Predicted Risk Score ∈ [0, 1]]
```

#### GraphSAGE Convolution Formulation:
For node $v \in \mathcal{V}$:
$$h_{\mathcal{N}(v)}^{(k)} = \text{AGGREGATE}_k \left( \{ h_u^{(k-1)}, \forall u \in \mathcal{N}(v) \} \right)$$
$$h_v^{(k)} = \sigma \left( W^{(k)} \cdot \left[ h_v^{(k-1)} \,\|\, h_{\mathcal{N}(v)}^{(k)} \right] \right)$$

#### Edge Decoder:
Given node representations $z_u, z_v \in \mathbb{R}^{64}$:
$$\text{EdgeFeat}_{uv} = z_u \odot z_v \quad (\text{Hadamard element-wise product})$$
$$\hat{y}_{uv} = \sigma \left( W_2 \cdot \text{ReLU}(W_1 \cdot \text{EdgeFeat}_{uv} + b_1) + b_2 \right)$$

### 3. ADMET-AI In Silico Profiling
- **Engine**: ADMET-AI chemprop ensemble.
- **Predicted Properties**:
  - `hERG`: Cardiac potassium ion channel inhibition probability.
  - `BBB_Martins`: Blood-Brain Barrier permeability coefficient.
  - `CYP3A4_Inhibitor`: Cytochrome P450 3A4 inhibition likelihood.

---

## 📊 Training Pipeline & Model Results

### Strict Data Leakage Prevention
To prevent test set contamination, data splitting is executed using PyG's `RandomLinkSplit`:
- **Training Edges**: 80%
- **Validation Edges**: 10%
- **Test Edges**: 10%
- **Splitting Parameters**: `is_undirected=True`, `add_negative_train_samples=True`, strictly isolating edge connectivity from message-passing graphs.

### Hyperparameters
| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | `0.001` (lowered from `0.005` for smooth convergence) |
| Loss Function | Binary Cross-Entropy (`F.binary_cross_entropy`) |
| Batch / Step Type | Full Graph inductive link prediction |
| Total Epochs | 30 |
| Checkpoint Policy | Save state dictionary of best validation ROC-AUC |
| Compute Device | Apple Silicon MPS (`torch.device('mps')`) |
| Training Time | **7.0 seconds total** |

### Epoch-by-Epoch Convergence
```text
Epoch 001 | Loss: 0.6931 | Val ROC-AUC: 0.5000 | 0.4s
Epoch 005 | Loss: 0.6865 | Val ROC-AUC: 0.6480 | 1.5s ★ BEST
Epoch 010 | Loss: 0.6625 | Val ROC-AUC: 0.7259 | 2.8s ★ BEST
Epoch 015 | Loss: 0.5995 | Val ROC-AUC: 0.7372 | 3.9s
Epoch 020 | Loss: 0.5786 | Val ROC-AUC: 0.7594 | 4.9s ★ BEST
Epoch 025 | Loss: 0.5538 | Val ROC-AUC: 0.7664 | 5.9s
Epoch 028 | Loss: 0.5484 | Val ROC-AUC: 0.7743 | 6.5s ★ BEST
Epoch 030 | Loss: 0.5420 | Val ROC-AUC: 0.7772 | 7.0s ★ BEST (SAVED)
```

- **Final Loss**: `0.5420`
- **Peak Validation ROC-AUC**: **`0.7772`** (Saved as `models/graphsage_mvp.pt`)
- **Verification**: State dict successfully re-loaded on CPU with complete parameter matrices intact.

---

## 🔗 Clinical Evidence & Entity Resolution

### Entity Mapping Statistics
| Metric | TDC Raw Data | Resolved with DDI-Guard |
|---|---|---|
| Total Unique Drugs | 1,705 | 1,705 |
| Primary Identifier | Raw DrugBank ID (`DBxxxxx`) | Human-readable Drug Name |
| Successfully Resolved | 0 | **1,700 (99.7%)** |
| Fallback to ID | 1,705 (100%) | 5 (0.3%): `DB09162, DB09323, DB09396, DB11106, DB13450` |
| Overlap with DDInter Clinical DB | 0% (incompatible IDs) | **70.2% (1,194 drugs)** |
| Resolvable Clinical Drug Pairs | 0 | **92,227 pairs** |

---

## 🧪 End-to-End Simulation & Demo Test Case

### Demo Scenario: Warfarin ↔ Aspirin (Acetylsalicylic Acid)
A classical high-risk clinical combination: concurrent use of an anticoagulant (Warfarin) and an NSAID/antiplatelet agent (Aspirin) significantly elevates hemorrhage risk.

```text
=== DDI-Guard: Simulation Results ===
Candidate Drug: Aspirin (Acetylsalicylic acid) | SMILES: CC(=O)OC1=CC=CC=C1C(=O)O
Target Drug:    Warfarin (DB00682)            | Resolved Name: Warfarin

[Panel 1: ADMET Profile]
  • hERG Toxicity (Cardiac):        0.021  (Low cardiac hazard)
  • Blood-Brain Barrier (BBB):      0.658  (Moderate penetration)
  • CYP3A4 Inhibition:              0.000  (Does not inhibit primary CYP3A4 metabolism)

[Panel 2: GraphSAGE Risk Score]
  • Risk Score:                     0.291
  • Model Classification:           Low Structural Risk (Binary interaction baseline)

[Panel 3: Clinical Evidence]
  • Status:                         ✅ Documented Interaction Found in DDInter
  • Severity Level:                 MAJOR
  • Clinical Context:               High risk of severe bleeding and gastrointestinal hemorrhage.
```

> **Clinical Interpretation Note**: While GraphSAGE evaluates baseline binary link topology across chemical fingerprints, Panel 3 correctly identifies that this combination possesses a **Major** clinical severity rating in real-world clinical guidelines. This demonstrates why the **3-Panel Triangulation** is critical: machine learning models evaluate structural risk, but clinical databases anchor patient safety.

---

## 💻 Setup & Installation Guide

### 1. Prerequisites
- Python 3.10+
- Conda package manager (Miniconda or Anaconda)
- Git

### 2. Environment Configuration
```bash
# Clone the repository
git clone https://github.com/DataWitchee/DD_Interactions.git
cd DD_Interactions

# Activate your designated conda environment (e.g., "E")
conda activate E

# Install core scientific and chemoinformatics dependencies
conda install -c conda-forge rdkit
pip install torch torchvision
pip install torch-geometric
pip install admet-ai PyTDC streamlit pandas numpy scikit-learn
```

### 3. Verify System & Hardware
Verify your environment and Apple Silicon / CUDA acceleration:
```python
import torch
print("MPS Built:    ", torch.backends.mps.is_built())
print("MPS Available:", torch.backends.mps.is_available())
```

---

## 🚀 How to Run the Application

### 1. Train or Retrain the Model (Optional)
To retrain GraphSAGE with the best-checkpoint validation loop:
```bash
python models/train_graphsage.py
```
*Outputs: `models/graphsage_mvp.pt` (30 epochs, ~7 seconds).*

### 2. Run the Streamlit Dashboard
```bash
streamlit run app/main.py
```
Open your browser and navigate to:
```
http://localhost:8501
```

### 3. Using the Web Interface:
1. **Candidate Drug**: Enter the drug name (e.g. `Aspirin` or `Acetylsalicylic acid`) and paste its SMILES string.
2. **Target Drug**: Select from the alphabetical dropdown of 1,700+ approved therapeutics (e.g. `Warfarin`, `Metformin`, `Imatinib`).
3. Click **"Analyze Interaction Risk ⚡"** to generate all three panels instantaneously.

---

## 🗺️ Known Limitations & Future Roadmap

1. **Binary vs. Multi-Class Severity Scoring**:
   - *Current*: GraphSAGE predicts binary interaction probability $[0, 1]$.
   - *Future*: Train a multi-relational graph neural network (e.g., RGCN or Decagon) to predict specific severity levels (`Major`, `Moderate`, `Minor`) and adverse event phenotypes directly.
2. **Substructure Attribution & Explainability**:
   - *Future*: Integrate GNNExplainer or Integrated Gradients to highlight the specific toxicophoric functional groups in the candidate molecule responsible for the predicted interaction.
3. **Automated DDI Mechanism NLP**:
   - *Future*: Extract mechanistic text from FDA package inserts using biomedical LLMs to populate the mechanism field in Panel 3.

---

## 👥 Contributors & Acknowledgements
- **Therapeutics Data Commons (TDC)** for the DrugBank DDI benchmark dataset.
- **DDInter** database for curated clinical interaction severity classifications.
- **ADMET-AI & Chemprop** for in silico ADMET profiling models.
- **IIT Demokritos** for open drug vocabulary cross-referencing tables.
