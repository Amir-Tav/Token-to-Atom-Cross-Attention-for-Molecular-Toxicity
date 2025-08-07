import streamlit as st
from utils import (
    predict_and_explain_all_labels,
    summarize_prediction,
    generate_toxicity_radar,
    generate_mechanistic_report,
)

st.set_page_config(page_title="Tox21 Drug Toxicity Predictor", layout="centered")
st.title("💊 Tox21 Drug Toxicity Predictor")
st.markdown(
    "Enter a **SMILES string** to predict toxicological endpoints and receive domain-aware mechanistic explanations."
)

# ── SMILES Input ──────────────────────────────────────────
user_input = st.text_input("SMILES (e.g., CC(=O)OC1=CC=CC=C1C(=O)O):")

if user_input:
    smiles = user_input.strip()
    try:
        # === Prediction + SHAP fusion ===
        result = predict_and_explain_all_labels(smiles)

        # === Summary ===
        st.markdown("### 🧠 Mechanistic Summary")
        st.markdown(summarize_prediction(result), unsafe_allow_html=True)

        # === Per-label mechanistic reports ===
        for label in result["predicted_labels"]:
            expl   = result["explanations"][label]
            shap_df, prob, thr = expl["shap_df"], expl["prob"], expl["threshold"]
            st.markdown(
                generate_mechanistic_report(label, shap_df, prob, thr, smiles),
                unsafe_allow_html=True,
            )

        # === Radar Plot ===
        st.markdown("### 🧭 Toxicity Fingerprint")
        radar_fig = generate_toxicity_radar(smiles, result)
        st.plotly_chart(radar_fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
