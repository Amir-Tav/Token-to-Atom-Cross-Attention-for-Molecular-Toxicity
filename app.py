# =============================
# app.py – UI for full feature set
# =============================
import streamlit as st
from utils import predict
from pathlib import Path

st.set_page_config(page_title="Tox21 multi‑endpoint predictor", layout="wide")
st.title("🧪 Tox21 Multi‑endpoint Predictor")

smiles = st.text_input("Enter SMILES", "CCOc1ccc2nc(S(N)(=O)=O)sc2c1")
if st.button("Predict") and smiles:
    with st.spinner("Crunching …"):
        out = predict(smiles)

    # verdict sentence + SHAP‑coloured molecule
    st.markdown(out["sentence"], unsafe_allow_html=True)
    st.image(out["mol_svg"], use_column_width=False)

    # probability table
    with st.expander("Probability table"):
        st.dataframe(out["table"], hide_index=True)

    # phys‑chem descriptors + radar
    with st.expander("Phys‑chem descriptors"):
        col1, col2 = st.columns([1, 1])
        col1.dataframe(out["physchem_df"], hide_index=True)
        col2.image(f"data:image/png;base64,{out['radar_png']}")

    # toxicophore SMARTS
    if not out["toxic_df"].empty:
        with st.expander("Toxicophore SMARTS"):
            st.dataframe(out["toxic_df"], hide_index=True)

    # PubChem assays
    if not out["assay_df"].empty:
        with st.expander("PubChem assays"):
            st.dataframe(out["assay_df"], hide_index=True)

    # ChEMBL targets
    if not out["chembl_df"].empty:
        with st.expander("ChEMBL target enrichment"):
            st.dataframe(out["chembl_df"], hide_index=True)

    # PDF report download
    report_path = Path(out["report_path"])
    with report_path.open("rb") as fh:
        st.download_button("Download PDF report", fh, file_name=report_path.name, mime="application/pdf")
