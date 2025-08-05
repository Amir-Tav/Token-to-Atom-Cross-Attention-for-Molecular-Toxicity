import streamlit as st
from utils import (
    predict_and_explain_all_labels,
    summarize_prediction,
    generate_toxicity_radar,
    highlight_toxicophores
)

st.set_page_config(page_title="Tox21 Drug Toxicity Predictor", layout="centered")
st.title("💊 Tox21 Drug Toxicity Predictor")
st.markdown("Enter a **SMILES string** to predict toxicological endpoints.")

# --- SMILES Input ---
user_input = st.text_input("SMILES:")  # Default: Aspirin

if user_input:
    smiles = user_input.strip()
    try:
        # === Prediction + SHAP Explanation ===
        result = predict_and_explain_all_labels(smiles)
        st.markdown(summarize_prediction(result), unsafe_allow_html=True)

        # === Radar Plot ===
        radar_fig = generate_toxicity_radar(smiles, result)
        st.markdown("### 🧭 Toxicity Fingerprint")
        st.plotly_chart(radar_fig, use_container_width=True)

        # === SMARTS Toxicophore Detection ===
        st.markdown("### 🧬 Detected Toxicophores")
        matched, img = highlight_toxicophores(smiles)

        if matched:
            st.markdown("**☣️ Matched SMARTS Toxicophores:**")
            for rule in matched:
                st.markdown(f"- {rule}")
            st.image(img, caption="Highlighted Toxicophore Regions", use_column_width=True)
        else:
            st.info("✅ No toxicophoric substructures detected.")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
