"""
frontend/app.py — DDI-Guard Streamlit frontend.

Consumes the FastAPI backend at http://localhost:8000.
No ML / RDKit / inference logic here — pure API calls + Plotly rendering.
"""

from __future__ import annotations

import hashlib
import json

import plotly.graph_objects as go
import requests
import streamlit as st

# ── Constants ─────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"

# Real training convergence data from the project's own GraphSAGE training run
# (30 epochs, Adam LR=0.001, best val ROC-AUC checkpoint saved)
_TRAINING_EPOCHS = [1,  5,    10,    15,    20,    25,    28,    30]
_TRAINING_AUCS   = [0.5, 0.6480, 0.7259, 0.7372, 0.7594, 0.7664, 0.7743, 0.7772]

# Severity colour mapping for Clinical Evidence badge
_SEVERITY_COLORS = {
    "major":    ("#C0392B", "#FDEDEC"),   # (text, bg)
    "moderate": ("#D35400", "#FEF5E7"),
    "minor":    ("#B7950B", "#FEFBD8"),
}


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DDI-Guard: Drug-Drug Interaction Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Minimal CSS polish (light theme only, no dark overrides) ──────────────────
st.markdown("""
<style>
/* Top-level header */
.ddi-header {
    padding: 1.2rem 0 0.4rem 0;
    border-bottom: 2px solid #E8ECF0;
    margin-bottom: 1.6rem;
}
.ddi-header h1 {
    font-size: 1.85rem;
    font-weight: 700;
    color: #1A1A2E;
    margin: 0;
}
.ddi-header p {
    color: #555;
    margin: 0.25rem 0 0 0;
    font-size: 0.95rem;
}

/* Panel cards */
.panel-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.4rem 1.2rem 1rem 1.2rem;
    height: 100%;
}
.panel-title {
    font-size: 1rem;
    font-weight: 600;
    color: #1A1A2E;
    margin-bottom: 0.1rem;
    letter-spacing: 0.01em;
}
.panel-subtitle {
    font-size: 0.75rem;
    color: #888;
    margin-bottom: 1rem;
}

/* Severity badge */
.severity-badge {
    display: inline-block;
    padding: 0.35rem 0.9rem;
    border-radius: 20px;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.03em;
    margin-bottom: 0.5rem;
}

/* Evidence resolution caption */
.resolution-caption {
    font-size: 0.75rem;
    color: #888;
    margin-top: 0.5rem;
    line-height: 1.5;
}

/* Analyze button row */
div[data-testid="stButton"] > button[kind="primary"] {
    background: #3B5BDB;
    color: white;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.6rem 2rem;
    border: none;
    width: 100%;
    transition: background 0.15s;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #2F4CBF;
}

/* Expander header */
div[data-testid="stExpander"] summary {
    font-weight: 600;
    color: #3B5BDB;
}
</style>
""", unsafe_allow_html=True)


# ── Backend helpers ───────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def fetch_drug_list() -> list[str]:
    """GET /drugs — cached for 1 hour, refreshed on server restart."""
    try:
        resp = requests.get(f"{API_BASE}/drugs", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.error(f"⚠️ Could not reach backend at {API_BASE}. Is the FastAPI server running?\n\n`{exc}`")
        return []


def analyze(candidate_name: str, candidate_smiles: str, target_drug: str) -> dict | None:
    """POST /analyze — always fresh, result keyed in session_state by input hash."""
    cache_key = hashlib.sha256(
        json.dumps([candidate_name, candidate_smiles, target_drug], sort_keys=True).encode()
    ).hexdigest()

    if cache_key in st.session_state:
        return st.session_state[cache_key]

    try:
        resp = requests.post(
            f"{API_BASE}/analyze",
            json={
                "candidate_name":   candidate_name,
                "candidate_smiles": candidate_smiles,
                "target_drug":      target_drug,
            },
            timeout=60,
        )
        if resp.status_code >= 500:
            st.error(f"Backend error {resp.status_code}: {resp.text[:300]}")
            return None
        if resp.status_code == 422:
            detail = resp.json().get("detail", resp.text)
            st.error(f"Validation error: {detail}")
            return None
        resp.raise_for_status()
        result = resp.json()
        st.session_state[cache_key] = result
        return result
    except requests.exceptions.ConnectionError:
        st.error(
            f"⚠️ Cannot connect to backend at `{API_BASE}`. "
            "Start it with:\n```\nuvicorn backend.main:app --host 0.0.0.0 --port 8000\n```"
        )
        return None
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
        return None


# ── Plotly charts ─────────────────────────────────────────────────────────────

def gauge_chart(score: float, label: str) -> go.Figure:
    """Plotly gauge for interaction risk score 0-1."""
    if score < 0.4:
        bar_color = "#27AE60"
    elif score < 0.7:
        bar_color = "#E67E22"
    else:
        bar_color = "#C0392B"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 42, "color": bar_color}, "valueformat": ".3f"},
        title={"text": f"<b>{label}</b>", "font": {"size": 15, "color": "#1A1A2E"}},
        gauge={
            "axis": {
                "range": [0, 1],
                "tickwidth": 1,
                "tickcolor": "#CCC",
                "tickfont": {"size": 11},
            },
            "bar": {"color": bar_color, "thickness": 0.3},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0.0, 0.4], "color": "#EAFAF1"},
                {"range": [0.4, 0.7], "color": "#FEF9E7"},
                {"range": [0.7, 1.0], "color": "#FDEDEC"},
            ],
            "threshold": {
                "line": {"color": bar_color, "width": 3},
                "thickness": 0.8,
                "value": score,
            },
        },
    ))
    fig.update_layout(
        template="plotly_white",
        height=280,
        margin=dict(l=20, r=20, t=50, b=10),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
    )
    return fig


def radar_chart(herg: float, bbb: float, cyp3a4: float) -> go.Figure:
    """Plotly polar/radar chart for the 3 ADMET properties."""
    # Clamp values to [0,1] for safe radar rendering
    def safe(v):
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.0

    values = [safe(herg), safe(bbb), safe(cyp3a4)]
    categories = ["hERG Toxicity", "BBB Penetration", "CYP3A4 Inhibition"]

    fig = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(59, 91, 219, 0.12)",
        line=dict(color="#3B5BDB", width=2),
        marker=dict(color="#3B5BDB", size=7),
        hovertemplate="%{theta}: %{r:.3f}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_white",
        polar=dict(
            bgcolor="#FFFFFF",
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont={"size": 9, "color": "#AAA"},
                gridcolor="#E8ECF0",
                linecolor="#E8ECF0",
                tickvals=[0.25, 0.5, 0.75, 1.0],
                ticktext=["0.25", "0.50", "0.75", "1.0"],
            ),
            angularaxis=dict(
                tickfont={"size": 11, "color": "#333"},
                gridcolor="#E8ECF0",
                linecolor="#CCC",
            ),
        ),
        height=280,
        margin=dict(l=40, r=40, t=30, b=10),
        paper_bgcolor="#FFFFFF",
        showlegend=False,
    )
    return fig


def training_convergence_chart() -> go.Figure:
    """Static Plotly line chart of GraphSAGE training convergence."""
    fig = go.Figure()

    # Shaded zone: above 0.7 = good performance region
    fig.add_hrect(y0=0.7, y1=1.0, fillcolor="rgba(39,174,96,0.07)", line_width=0)

    fig.add_trace(go.Scatter(
        x=_TRAINING_EPOCHS,
        y=_TRAINING_AUCS,
        mode="lines+markers",
        name="Val ROC-AUC",
        line=dict(color="#3B5BDB", width=2.5),
        marker=dict(
            size=8,
            color=_TRAINING_AUCS,
            colorscale=[[0, "#E8ECF0"], [1, "#3B5BDB"]],
            line=dict(color="white", width=1.5),
        ),
        hovertemplate="Epoch %{x}: AUC = %{y:.4f}<extra></extra>",
    ))

    # Annotate final point
    fig.add_annotation(
        x=30, y=0.7772,
        text="<b>0.7772</b>",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#3B5BDB",
        arrowwidth=1.5,
        font=dict(size=12, color="#3B5BDB"),
        bgcolor="#FFFFFF",
        bordercolor="#3B5BDB",
        borderwidth=1,
        borderpad=4,
        yshift=12,
    )

    fig.add_hline(
        y=0.7,
        line_dash="dot",
        line_color="#27AE60",
        annotation_text="AUC = 0.70 threshold",
        annotation_font_color="#27AE60",
        annotation_font_size=10,
    )

    fig.update_layout(
        template="plotly_white",
        title=dict(
            text="Validation ROC-AUC across 30 training epochs — final: <b>0.7772</b>",
            font=dict(size=13, color="#1A1A2E"),
        ),
        xaxis=dict(
            title="Epoch",
            tickmode="array",
            tickvals=_TRAINING_EPOCHS,
            gridcolor="#F0F2F5",
            linecolor="#CCC",
        ),
        yaxis=dict(
            title="Val ROC-AUC",
            range=[0.45, 0.82],
            gridcolor="#F0F2F5",
            linecolor="#CCC",
        ),
        height=340,
        margin=dict(l=50, r=30, t=60, b=50),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        legend=dict(x=0.02, y=0.05, bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# ── Clinical Evidence renderer ─────────────────────────────────────────────────

def render_evidence(evidence: dict) -> None:
    found          = evidence.get("found", False)
    severity       = (evidence.get("severity") or "").strip()
    reason         = evidence.get("reason", "")
    reason_text    = evidence.get("reason_text", "")
    resolved_cand  = evidence.get("resolved_candidate")
    resolved_targ  = evidence.get("resolved_target")

    if not found:
        if reason == "not_in_vocabulary":
            # One or both drugs absent from DDInter's ~1,194-drug coverage.
            # missing_names carries the original user-typed display names that failed to resolve.
            missing = evidence.get("missing_names", [])
            drug_display = missing[0].strip("'") if missing else "One or both drugs"
            st.warning(
                f"**{drug_display}** is not in DDInter's clinical coverage "
                f"(~1,194 of 1,705 catalogued drugs). Clinical evidence unavailable "
                f"for this drug — GraphSAGE structural prediction above is the only "
                f"signal available."
            )
        elif reason == "no_record":
            # Both drugs resolved to DDInter vocab; this specific pair has no entry.
            cand = resolved_cand or "the candidate"
            targ = resolved_targ or "the target drug"
            st.info(
                f"No documented interaction found between **{cand}** and **{targ}** "
                f"in DDInter. This does not confirm safety — it means this specific "
                f"pair has not been studied/catalogued."
            )
        elif reason == "ddinter_unavailable":
            st.error(f"❌ {reason_text}")
        else:
            st.warning(f"No clinical evidence found. {reason_text or reason}")
        return

    # Found — render severity badge
    sev_key = severity.lower()
    text_color, bg_color = _SEVERITY_COLORS.get(sev_key, ("#555555", "#F0F2F5"))

    if sev_key == "unknown" or not sev_key:
        badge_label = severity or "Documented"
        caption_note = "Documented interaction — severity not specified in source data."
        text_color, bg_color = "#555555", "#F0F2F5"
    else:
        badge_label = severity
        caption_note = None

    st.markdown(
        f"<div style='margin-bottom:0.5rem'>"
        f"<span class='severity-badge' "
        f"style='color:{text_color};background:{bg_color};border:1.5px solid {text_color}44'>"
        f"✓ {badge_label}"
        f"</span></div>",
        unsafe_allow_html=True,
    )

    if caption_note:
        st.caption(caption_note)

    # Resolved name captions — shows how names were matched
    if resolved_cand or resolved_targ:
        parts = []
        if resolved_cand:
            parts.append(f"Candidate → **{resolved_cand}**")
        if resolved_targ:
            parts.append(f"Target → **{resolved_targ}**")
        st.markdown(
            "<div class='resolution-caption'>Matched as: " + " · ".join(parts) + "</div>",
            unsafe_allow_html=True,
        )


# ── Main app ─────────────────────────────────────────────────────────────────

def main() -> None:
    # Header
    st.markdown("""
    <div class="ddi-header">
        <h1>🛡️ DDI-Guard: Drug-Drug Interaction Dashboard</h1>
        <p>Enter a candidate compound and a target drug to assess interaction risk, ADMET safety, and clinical evidence.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Input row ────────────────────────────────────────────────────────────
    drug_list = fetch_drug_list()

    col_name, col_smiles, col_target = st.columns([1.2, 2, 1.5])

    with col_name:
        candidate_name = st.text_input(
            "Candidate Drug Name",
            value="Aspirin",
            placeholder="e.g. Aspirin",
            help="Free-text name used for DDInter clinical evidence lookup. Common aliases (e.g. 'Aspirin') are resolved automatically.",
        )

    with col_smiles:
        candidate_smiles = st.text_input(
            "Candidate SMILES",
            value="CC(=O)OC1=CC=CC=C1C(=O)O",
            placeholder="e.g. CC(=O)OC1=CC=CC=C1C(=O)O",
            help="SMILES string used for GraphSAGE risk scoring and ADMET profiling.",
        )

    with col_target:
        if drug_list:
            # Default to Warfarin for the canonical demo case
            default_idx = drug_list.index("Warfarin") if "Warfarin" in drug_list else 0
            target_drug = st.selectbox(
                "Target Drug",
                options=drug_list,
                index=default_idx,
                help=f"{len(drug_list):,} DrugBank-approved drugs with resolved names.",
            )
        else:
            target_drug = st.selectbox("Target Drug", options=["(backend unavailable)"])

    # ── Analyze button ────────────────────────────────────────────────────────
    btn_disabled = not drug_list or not candidate_smiles.strip()
    analyze_clicked = st.button(
        "Analyze Interaction Risk ⚡",
        type="primary",
        disabled=btn_disabled,
        use_container_width=True,
    )

    if analyze_clicked:
        with st.spinner("Running all three analysis panels..."):
            result = analyze(candidate_name.strip(), candidate_smiles.strip(), target_drug)
            if result:
                # Store last result key for the display block below
                st.session_state["_last_result"] = result

    st.markdown("---")

    # ── Results panels ────────────────────────────────────────────────────────
    result = st.session_state.get("_last_result")

    if result is None:
        st.markdown(
            "<div style='text-align:center;color:#AAA;padding:3rem 0;font-size:1rem'>"
            "Enter drug details above and click <b>Analyze Interaction Risk ⚡</b> to see results."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        admet    = result.get("admet_profile", {})
        risk     = result.get("interaction_risk", {})
        evidence = result.get("clinical_evidence", {})

        p1, p2, p3 = st.columns(3, gap="medium")

        # ── Panel 1: Candidate Profile (ADMET radar) ──────────────────────────
        with p1:
            st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
            st.markdown("<div class='panel-title'>🧪 Candidate Profile</div>", unsafe_allow_html=True)
            st.markdown("<div class='panel-subtitle'>ADMET-AI · Pharmacokinetic safety</div>", unsafe_allow_html=True)
            st.caption(
                "ADMET values reflect the candidate molecule's intrinsic properties "
                "and do not change based on the selected target drug."
            )

            herg   = admet.get("herg",   0)
            bbb    = admet.get("bbb",    0)
            cyp3a4 = admet.get("cyp3a4", 0)

            if any(v == "Error" for v in [herg, bbb, cyp3a4]):
                st.warning("ADMET prediction encountered an error. Check the backend logs.")
            elif any(v == "N/A" for v in [herg, bbb, cyp3a4]):
                st.info("ADMET-AI not available on this server.")
            else:
                fig_radar = radar_chart(herg, bbb, cyp3a4)
                st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})

                # Raw values as small metrics below the chart
                m1, m2, m3 = st.columns(3)
                m1.metric("hERG",   f"{float(herg):.3f}",   help="Cardiac toxicity risk (↑ = higher risk)")
                m2.metric("BBB",    f"{float(bbb):.3f}",    help="Blood-brain barrier penetration probability")
                m3.metric("CYP3A4", f"{float(cyp3a4):.3f}", help="CYP3A4 inhibition probability")

            st.markdown("</div>", unsafe_allow_html=True)

        # ── Panel 2: Interaction Risk (GraphSAGE gauge) ───────────────────────
        with p2:
            st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
            st.markdown("<div class='panel-title'>⚡ Interaction Risk</div>", unsafe_allow_html=True)
            cand_lbl = result.get("candidate_name", candidate_name)
            targ_lbl = result.get("target_drug",    target_drug)
            st.markdown(
                f"<div class='panel-subtitle'>GraphSAGE GNN · {cand_lbl} ↔ {targ_lbl}</div>",
                unsafe_allow_html=True,
            )

            if "error" in risk and risk["error"]:
                st.error(f"Model error: {risk['error']}")
            else:
                score = risk.get("score", 0.0)
                label = risk.get("status_label", "Unknown")

                fig_gauge = gauge_chart(score, label)
                st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

                # Risk interpretation note
                if score < 0.4:
                    st.success("Low structural interaction signal from the GNN.")
                elif score < 0.7:
                    st.warning("Moderate structural interaction signal — review clinical evidence.")
                else:
                    st.error("High structural interaction signal — clinical review recommended.")

            st.markdown("</div>", unsafe_allow_html=True)

        # ── Panel 3: Clinical Evidence (DDInter) ──────────────────────────────
        with p3:
            st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
            st.markdown("<div class='panel-title'>📖 Clinical Evidence</div>", unsafe_allow_html=True)
            st.markdown("<div class='panel-subtitle'>DDInter database · real-world documentation</div>", unsafe_allow_html=True)
            render_evidence(evidence)
            st.markdown("</div>", unsafe_allow_html=True)

    # ── Model Performance expander ────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📈 Model Performance — GraphSAGE Training Convergence"):
        st.markdown(
            "GraphSAGE trained for **30 epochs** on 40k positive + 40k negative DrugBank edges. "
            "Adam optimiser, LR=0.001, checkpoint saved at best validation ROC-AUC. "
            "Trained in **7.0 seconds** on Apple Silicon MPS.",
        )
        fig_train = training_convergence_chart()
        st.plotly_chart(fig_train, use_container_width=True, config={"displayModeBar": False})

        st.markdown(
            "<div style='font-size:0.8rem;color:#888;margin-top:0.5rem'>"
            "Model: 2-layer GraphSAGE (1024→128→64) + Hadamard edge decoder (64→32→1). "
            "Features: Morgan fingerprints (radius=2, 1024-bit). "
            "Evaluation: <code>RandomLinkSplit</code> with 10% val / 10% test, "
            "strict leakage prevention."
            "</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
