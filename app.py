"""
app.py – Streamlit front‑end for the Tox21 predictor
Last updated: 2025‑07‑28
"""

import base64

import streamlit as st

import utils

st.set_page_config(page_title="Tox21 predictor", layout="wide")
st.title("🧪 Tox21 multi‑endpoint toxicity classifier (v0.5)")

smiles = st.text_input("Paste a SMILES string here", "")

if st.button("Predict", disabled=not smiles):
    try:
        out = utils.predict(smiles)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    st.markdown(out["sentence"])

    # probabilities
    with st.expander("Prediction probabilities"):
        st.dataframe(
            out["pred_df"].style.format({"Probability": "{:.2f}"}),
            hide_index=True,
            use_container_width=True,
        )

    # SHAP token contributions (NEW)
    if not out["shap_df"].empty:
        with st.expander("Explainability – top token contributions"):
            st.dataframe(
                out["shap_df"].style.format({"SHAP": "{:.3f}"}),
                hide_index=True,
            )

    # phys‑chem
    with st.expander("Physico‑chemical descriptors"):
        st.dataframe(out["physchem_df"], hide_index=True)

    # PubChem molecule data
    if not out["pubchem_df"].empty:
        with st.expander("PubChem molecular data"):
            st.dataframe(out["pubchem_df"], hide_index=True)

    # assays
    if not out["assay_df"].empty:
        with st.expander("PubChem assay activity"):
            st.dataframe(out["assay_df"], hide_index=True)

    # toxicophores
    if not out["tox_df"].empty:
        with st.expander("Toxicophore alerts"):
            st.dataframe(out["tox_df"], hide_index=True)

    # radar plot
    st.image(out["radar_png"].getvalue(), caption="Phys‑chem radar plot")

    # PDF download
    with open(out["report_path"], "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f'<a href="data:application/pdf;base64,{b64}" '
        'download="Tox21_report.pdf">📄 Download full PDF report</a>',
        unsafe_allow_html=True,
    )
