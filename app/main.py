import sys
import os

# sys.path must be set BEFORE importing from app package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
from app.inference import (
    get_admet_profile,
    predict_ddi_risk,
    get_clinical_evidence,
    load_tdc_drug_catalogue,
)

st.set_page_config(page_title="DDI-Guard", page_icon="🛡️", layout="wide")


@st.cache_data
def _load_drug_catalogue() -> dict:
    """
    Returns {display_name: (drugbank_id, smiles)}.
    1,700 of 1,705 drugs show human-readable names; 5 unresolved show raw DB IDs.
    """
    return load_tdc_drug_catalogue()


def main():
    st.title("🛡️ DDI-Guard Dashboard")
    st.markdown("Predict Drug-Drug Interaction Risk and View Clinical Evidence.")

    catalogue = _load_drug_catalogue()
    # Sorted by display name for a clean alphabetical dropdown
    sorted_labels = sorted(catalogue.keys())

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        candidate_name = st.text_input(
            "Candidate Drug Name",
            value="Aspirin",
            help="Enter the drug's common name. Used for DDInter clinical evidence lookup (best-effort).",
        )
    with col2:
        candidate_smiles = st.text_input(
            "Candidate SMILES",
            value="CC(=O)OC1=CC=CC=C1C(=O)O",
        )
    with col3:
        # Dropdown shows human-readable drug NAMES (not raw DBxxxxx IDs)
        target_label = st.selectbox(
            "Target Drug",
            options=sorted_labels,
            help="1,700 TDC DrugBank drugs with resolved names; 5 shown as raw DB IDs.",
        )
        target_db_id, target_smiles = catalogue[target_label]
        st.caption(f"DrugBank ID: `{target_db_id}`")

    analyze_btn = st.button(
        "Analyze Interaction Risk ⚡", type="primary", use_container_width=True
    )
    st.markdown("---")

    if analyze_btn:
        with st.spinner("Analyzing profile and interaction risk..."):
            admet_data = get_admet_profile(candidate_smiles)

            # GraphSAGE risk score — uses SMILES
            try:
                risk_score = predict_ddi_risk(candidate_smiles, target_smiles)
            except FileNotFoundError:
                risk_score = None
            except Exception as e:
                risk_score = None
                st.warning(f"Risk scoring error: {e}")

            # DDInter clinical evidence — best-effort by drug name
            # Pass human-readable names to both sides for the best match chance.
            # target_label is already the resolved human name (e.g. "Warfarin").
            evidence = get_clinical_evidence(candidate_name, target_label)

        # ── Render UI ─────────────────────────────────────────────────────────
        panel1, panel2, panel3 = st.columns(3)

        # PANEL 1: Candidate Profile (ADMET)
        with panel1:
            st.markdown("### 🧪 Candidate Profile")
            st.caption("Powered by ADMET-AI")
            st.metric("hERG Toxicity (Cardiac)",   admet_data.get("hERG Toxicity",       "N/A"))
            st.metric("Blood-Brain Barrier (BBB)",  admet_data.get("Blood-Brain Barrier",  "N/A"))
            st.metric("CYP3A4 Inhibition",          admet_data.get("CYP3A4 Inhibition",   "N/A"))

        # PANEL 2: Interaction Risk (GraphSAGE)
        with panel2:
            st.markdown("### ⚡ Interaction Risk")
            st.caption(f"GraphSAGE · {candidate_name} ↔ {target_label}")

            if risk_score is not None:
                if risk_score < 0.4:
                    risk_level, alert = "Low Risk",      st.success
                elif risk_score < 0.7:
                    risk_level, alert = "Moderate Risk", st.warning
                else:
                    risk_level, alert = "High Risk",     st.error

                st.metric(f"Risk Score", f"{risk_score:.3f}")
                st.progress(risk_score)
                alert(f"**Status:** {risk_level}")
            else:
                st.error("⚠️ Model weights not found. Run training script first.")

        # PANEL 3: Clinical Evidence (DDInter — best-effort name match)
        with panel3:
            st.markdown("### 📖 Clinical Evidence")
            st.caption("DDInter database · best-effort name match")

            if evidence:
                st.success("✅ Documented Interaction Found")
                st.markdown(f"**Severity Level:** {evidence['Level']}")
                st.info(f"**Mechanism:** {evidence['Mechanism']}")
            else:
                st.warning(
                    "⚠️ No clinical evidence found in DDInter for this pair. "
                    "Risk score is purely model-predicted."
                )


if __name__ == "__main__":
    main()