import streamlit as st
import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.inference import get_admet_profile, predict_ddi_risk, get_clinical_evidence

st.set_page_config(page_title="DDI-Guard", page_icon="🛡️", layout="wide")

@st.cache_data
def load_local_drugs(csv_path="data/processed/merged_ddi.csv"):
    """Populates dropdown from the local merged CSV to avoid TDC downloads."""
    if not os.path.exists(csv_path):
        return {"DB00945 (Aspirin)": "CC(=O)OC1=CC=CC=C1C(=O)O", "DB00619 (Imatinib)": "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CC=CC=N5"}
    
    df = pd.read_csv(csv_path)
    drug_map = {}
    
    # Smart column mapping for UI Dropdown
    id1_col = next((c for c in df.columns if c.lower() in ['drug1_id', 'drug_a']), df.columns[0])
    id2_col = next((c for c in df.columns if c.lower() in ['drug2_id', 'drug_b']), df.columns[1])
    s1_col = next((c for c in df.columns if c.lower() in ['drug1', 'smiles_a', 'smiles1']), df.columns[2])
    s2_col = next((c for c in df.columns if c.lower() in ['drug2', 'smiles_b', 'smiles2']), df.columns[3])
    
    # Grab first 1000 unique drugs to keep dropdown responsive
    for _, row in df.head(5000).iterrows():
        drug_map[str(row[id1_col])] = str(row[s1_col])
        drug_map[str(row[id2_col])] = str(row[s2_col])
    return drug_map

def main():
    st.title("🛡️ DDI-Guard Dashboard")
    st.markdown("Predict Drug-Drug Interaction Risk and View Clinical Evidence.")
    
    drug_map = load_local_drugs()
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        candidate_name = st.text_input("Candidate Drug Name", value="New_Compound_X", help="Used for Evidence Lookup")
    with col2:
        candidate_smiles = st.text_input("Candidate SMILES", value="CC(=O)OC1=CC=CC=C1C(=O)O")
    with col3:
        target_drug_id = st.selectbox("Target Drug", options=list(drug_map.keys()))
        target_smiles = drug_map[target_drug_id]
        
    analyze_btn = st.button("Analyze Interaction Risk ⚡", type="primary", use_container_width=True)
    st.markdown("---")
    
    if analyze_btn:
        with st.spinner("Analyzing profile and interaction risk..."):
            admet_data = get_admet_profile(candidate_smiles)
            evidence = get_clinical_evidence(candidate_name, target_drug_id)
            
            try:
                risk_score = predict_ddi_risk(candidate_smiles, target_smiles)
            except FileNotFoundError:
                risk_score = None

            # Render UI 
            panel1, panel2, panel3 = st.columns(3)
            
            # PANEL 1: Candidate Profile
            with panel1:
                st.markdown("### 🧪 Candidate Profile")
                st.caption("Powered by ADMET-AI")
                st.metric(label="hERG Toxicity (Cardiac)", value=admet_data.get("hERG Toxicity", "N/A"))
                st.metric(label="Blood-Brain Barrier (BBB)", value=admet_data.get("Blood-Brain Barrier", "N/A"))
                st.metric(label="CYP3A4 Inhibition", value=admet_data.get("CYP3A4 Inhibition", "N/A"))
                
            # PANEL 2: Interaction Risk
            with panel2:
                st.markdown("### ⚡ Interaction Risk")
                st.caption("Powered by GraphSAGE + TDC")
                
                if risk_score is not None:
                    if risk_score < 0.4:
                        risk_level, alert = "Low Risk", st.success
                    elif risk_score < 0.7:
                        risk_level, alert = "Moderate Risk", st.warning
                    else:
                        risk_level, alert = "High Risk", st.error
                        
                    st.metric(label=f"Risk Score with {target_drug_id}", value=f"{risk_score:.3f}")
                    st.progress(risk_score)
                    alert(f"**Status:** {risk_level}")
                else:
                    st.error("⚠️ Model weights not found. Run training script first.")
                    
            # PANEL 3: Clinical Evidence
            with panel3:
                st.markdown("### 📖 Clinical Evidence")
                st.caption("Powered by Local Database")
                
                if evidence:
                    st.success("✅ Documented Interaction Found")
                    st.markdown(f"**Level of Interaction:** {evidence['Level']}")
                    st.info(f"**Mechanism:** {evidence['Mechanism']}")
                else:
                    st.warning("⚠️ No clinical evidence found. Risk score is purely predictive.")

if __name__ == "__main__":
    main()