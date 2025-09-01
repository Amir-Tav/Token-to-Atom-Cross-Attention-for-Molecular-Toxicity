import json
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw

from utils import (
    label_cols,
    thresholds,
    validate_smiles,
    resolve_compound_name,
    predict_and_explain_all_labels,
    generate_toxicity_radar,
    generate_mechanistic_report,
    match_toxicophores_with_explanations,
    highlight_toxicophores,
    plot_contrib_bar,
    explain_near_misses,
    platform_diagnostics,
    get_settings,
    set_use_standardization,
)

st.set_page_config(page_title="Tox21 Concept Explainer (ChemBERTa + TCAV)", layout="wide")

settings = get_settings()

st.title("🧪 Tox21 Concept Explainer (ChemBERTa + TCAV)")
st.caption("Using finalized thresholds from **implementation/v3/eval/thresholds_selected_v3.json** "
           "and TCAV v3 from **implementation/v3/stats/** (presence-gated).")
st.write(
    "Input a **SMILES** string. The ChemBERTa model predicts toxicity across 12 endpoints and "
    "explains its decision with **concept-level TCAV evidence**."
)

# --------------------------
# Sidebar controls
# --------------------------
st.sidebar.header("Settings")

# Debug & diagnostics
debug = st.sidebar.checkbox("Debug mode (show detailed errors)", value=False)
with st.sidebar.expander("Diagnostics", expanded=False):
    st.code(json.dumps(platform_diagnostics(), indent=2))

# Canonicalization toggle
use_std = st.sidebar.checkbox(
    "Use standardized/canonical SMILES for inference",
    value=bool(settings.get("use_standardization", True))
)
set_use_standardization(use_std)

# Concept filters
st.sidebar.subheader("Concept Evidence Filters")
min_tcav = st.sidebar.slider("Min TCAV (keep ≥)", 0.5, 1.0, float(settings["tcav"]["min_tcav"]), 0.01)
max_p = st.sidebar.select_slider("Max p-value (keep ≤)",
                                 options=[0.10, 0.05, 0.01, 0.001],
                                 value=float(settings["tcav"]["max_p"]))
top_k_concepts = st.sidebar.number_input("Max concepts to display", min_value=3, max_value=15,
                                         value=int(settings["tcav"]["top_k"]), step=1)

# Near-miss control
top_features_near_miss = st.sidebar.number_input(
    "Top negative concepts to show (near-miss)",
    min_value=1, max_value=20, value=5, step=1
)

# --------------------------
# Input + trigger
# --------------------------
smiles = st.text_input("SMILES", placeholder="e.g., C1=CC=CC=C1 (benzene)")

new_input_submitted = smiles and st.session_state.get("last_submitted_smiles") != smiles
run_prediction = st.button("Run prediction") or new_input_submitted

if run_prediction and smiles:
    ok, msg = validate_smiles(smiles)
    if not ok:
        st.error(msg); st.stop()
    with st.spinner("Running ChemBERTa and assembling concept evidence..."):
        try:
            result = predict_and_explain_all_labels(
                smiles, min_tcav=float(min_tcav), max_p=float(max_p), top_k=int(top_k_concepts)
            )
        except Exception as e:
            if debug: st.exception(e)
            else: st.error("Prediction failed. Enable 'Debug mode' for details.")
            st.stop()
    st.session_state["last_result"] = result
    st.session_state["last_submitted_smiles"] = smiles

# --------------------------
# Show results
# --------------------------
if "last_result" in st.session_state and smiles:
    result = st.session_state["last_result"]

    # Summary
    st.subheader("Summary")
    compound_name = resolve_compound_name(smiles)
    st.markdown(f"**SMILES:** `{smiles}`")
    st.markdown(f"**Compound name:** {compound_name}")

    st.markdown("### Model Predictions")
    if result["predicted_labels"]:
        st.write(", ".join(result["predicted_labels"]))
    else:
        st.write("_No endpoints predicted positive at current thresholds._")

    # Radar
    try:
        st.plotly_chart(generate_toxicity_radar(smiles, result), use_container_width=True)
    except Exception as e:
        if debug: st.exception(e)

    # Endpoint details (sorted by prob)
    selected_labels = sorted(
        result["predicted_labels"],
        key=lambda lb: result["explanations"][lb]["prob"],
        reverse=True
    )

    if selected_labels:
        st.subheader("Endpoint Details")
        for lb in selected_labels:
            exp = result["explanations"][lb]
            prob = float(exp["prob"]); thr = float(exp["threshold"])
            contrib_df = exp["shap_df"]  # holds presence-gated TCAV contributions
            margin_val = prob - thr

            with st.expander(f"{lb} — Prob {prob:.3f} | Thr {thr:.3f} | Margin {margin_val:.3f}", expanded=False):
                try:
                    st.markdown(generate_mechanistic_report(lb, contrib_df, prob, thr, smiles))
                except Exception as e:
                    if debug: st.exception(e)
                try:
                    st.plotly_chart(plot_contrib_bar(contrib_df), use_container_width=True)
                except Exception as e:
                    if debug: st.exception(e)

                # 2D structure
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        st.image(Draw.MolToImage(mol, size=(420, 280)),
                                 caption="2D molecular structure", use_container_width=False)
                except Exception as e:
                    if debug: st.exception(e)

                # Optional SMARTS toxicophores (if rules present)
                try:
                    matches = match_toxicophores_with_explanations(smiles, lb)
                    if matches:
                        st.markdown("**Toxicophores detected**")
                        try:
                            _, img = highlight_toxicophores(smiles)
                            st.image(img, caption="Matched substructures")
                        except Exception as e:
                            if debug: st.exception(e)
                        for m in matches:
                            st.markdown(f"- **{m['name']}** — {m['explanation']}")
                except Exception as e:
                    if debug: st.exception(e)

    # Near-miss explainer
    with st.expander("Why not other endpoints? (largest negative concept drivers)", expanded=False):
        try:
            rows = explain_near_misses(smiles, top_n=len(label_cols), top_features=int(top_features_near_miss))
            if rows:
                for lb, prob, thr, neg_df in rows:
                    st.markdown(f"### {lb} — **Prob {prob:.3f} | Thr {thr:.3f} | Margin {prob - thr:.3f}**")
                    c1, c2 = st.columns([1, 2], gap="large")
                    with c1:
                        gauge_fig = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=float(prob),
                            title={'text': "Prob vs Threshold"},
                            delta={'reference': float(thr), 'increasing': {'color': "crimson"}},
                            gauge={
                                'axis': {'range': [0, 1]},
                                'bar': {'color': "darkred"},
                                'steps': [
                                    {'range': [0, float(thr)], 'color': "lightgray"},
                                    {'range': [float(thr), 1], 'color': "lightpink"}
                                ],
                                'threshold': {'line': {'color': "red", 'width': 4}, 'value': float(thr)}
                            }
                        ))
                        gauge_fig.update_layout(height=300, font=dict(size=18))
                        st.plotly_chart(gauge_fig, use_container_width=True)
                    with c2:
                        st.dataframe(neg_df.reset_index(drop=True), use_container_width=True)
        except Exception as e:
            if debug: st.exception(e)















# import json
# import streamlit as st
# import plotly.graph_objects as go
# import numpy as np
# import pandas as pd
# from pathlib import Path
# from rdkit import Chem
# from rdkit.Chem import Draw

# from utils import (
#     label_cols,
#     thresholds,
#     validate_smiles,
#     resolve_compound_name,
#     predict_and_explain_all_labels,
#     generate_toxicity_radar,
#     generate_mechanistic_report,
#     match_toxicophores_with_explanations,
#     highlight_toxicophores,
#     plot_contrib_bar,              
#     explain_near_misses,
#     platform_diagnostics,
#     get_settings,
#     set_use_standardization,
# )

# # --- Optional built-in background set for faithfulness checks (fallback) ---
# DEFAULT_BG_SMILES = [
#     "c1ccccc1",                  # benzene
#     "c1ccc(cc1)O",               # phenol
#     "CC(=O)Oc1ccccc1C(=O)O",     # aspirin
#     "O=[N+]([O-])c1ccccc1",      # nitrobenzene
#     "Clc1ccccc1",                # chlorobenzene
#     "CCOC(=O)C",                 # ethyl acetate
#     "CC(=O)C",                   # acetone
#     "CC1=CC(=O)OC2=CC=CC=C12",   # coumarin-like
#     "C=CC(=O)C=C",               # enone (Michael acceptor-like)
#     "O=C(O)C(O)O",               # (di)carboxylic acid-like
#     "CCN(CC)CC",                 # trialkylamine-like
#     "C1=CC=C2C(=C1)C=CC=C2"      # naphthalene
# ]

# st.set_page_config(page_title="Tox21 Concept Explainer (ChemBERTa + TCAV)", layout="wide")

# settings = get_settings()

# st.title("🧪 Tox21 Concept Explainer (ChemBERTa + TCAV)")
# st.write(
#     "Input a **SMILES** string. The ChemBERTa model predicts toxicity across 12 endpoints and "
#     "explains its decision with **concept-level TCAV evidence**."
# )

# # --------------------------
# # Sidebar controls
# # --------------------------
# st.sidebar.header("Settings")

# # Debug & diagnostics
# debug = st.sidebar.checkbox("Debug mode (show detailed errors)", value=False)
# with st.sidebar.expander("Diagnostics", expanded=False):
#     st.code(json.dumps(platform_diagnostics(), indent=2))

# # Canonicalization toggle
# use_std = st.sidebar.checkbox(
#     "Use standardized/canonical SMILES for inference",
#     value=bool(settings.get("use_standardization", True))
# )
# set_use_standardization(use_std)

# # Concept filters
# st.sidebar.subheader("Concept Evidence Filters")
# min_tcav = st.sidebar.slider("Min TCAV (keep ≥)", 0.5, 1.0, float(settings["tcav"]["min_tcav"]), 0.01)
# max_p = st.sidebar.select_slider("Max p-value (keep ≤)", options=[0.10, 0.05, 0.01, 0.001], value=float(settings["tcav"]["max_p"]))
# top_k_concepts = st.sidebar.number_input("Max concepts to display", min_value=3, max_value=15, value=int(settings["tcav"]["top_k"]), step=1)

# # Near-miss control
# top_features_near_miss = st.sidebar.number_input(
#     "Top negative concepts to show (near-miss)",
#     min_value=1, max_value=20, value=5, step=1
# )

# # --------------------------
# # Input + trigger
# # --------------------------
# smiles = st.text_input("SMILES", placeholder="e.g., C1=CC=CC=C1 (benzene)")

# new_input_submitted = smiles and st.session_state.get("last_submitted_smiles") != smiles
# run_prediction = st.button("Run prediction") or new_input_submitted

# if run_prediction and smiles:
#     ok, msg = validate_smiles(smiles)
#     if not ok:
#         st.error(msg); st.stop()
#     with st.spinner("Running ChemBERTa and assembling concept evidence..."):
#         try:
#             result = predict_and_explain_all_labels(
#                 smiles, min_tcav=float(min_tcav), max_p=float(max_p), top_k=int(top_k_concepts)
#             )
#         except Exception as e:
#             if debug: st.exception(e)
#             else: st.error("Prediction failed. Enable 'Debug mode' for details.")
#             st.stop()
#     st.session_state["last_result"] = result
#     st.session_state["last_submitted_smiles"] = smiles

# # --------------------------
# # Show results
# # --------------------------
# if "last_result" in st.session_state and smiles:
#     result = st.session_state["last_result"]

#     # Summary
#     st.subheader("Summary")
#     compound_name = resolve_compound_name(smiles)
#     st.markdown(f"**SMILES:** `{smiles}`")
#     st.markdown(f"**Compound name:** {compound_name}")

#     st.markdown("### Model Predictions")
#     if result["predicted_labels"]:
#         st.write(", ".join(result["predicted_labels"]))
#     else:
#         st.write("_No endpoints predicted positive at current thresholds._")

#     # Radar
#     try:
#         st.plotly_chart(generate_toxicity_radar(smiles, result), use_container_width=True)
#     except Exception as e:
#         if debug: st.exception(e)

#     # Endpoint details
#     selected_labels = sorted(
#         result["predicted_labels"],
#         key=lambda lb: result["explanations"][lb]["prob"],
#         reverse=True
#     )

#     if selected_labels:
#         st.subheader("Endpoint Details")
#         for lb in selected_labels:
#             exp = result["explanations"][lb]
#             prob = float(exp["prob"]); thr = float(exp["threshold"])
#             contrib_df = exp["shap_df"]  # now holds contrib_score
#             margin_val = prob - thr

#             with st.expander(f"{lb} — Prob {prob:.3f} | Thr {thr:.3f} | Margin {margin_val:.3f}", expanded=False):
#                 try:
#                     st.markdown(generate_mechanistic_report(lb, contrib_df, prob, thr, smiles))
#                 except Exception as e:
#                     if debug: st.exception(e)
#                 try:
#                     st.plotly_chart(plot_contrib_bar(contrib_df), use_container_width=True)
#                 except Exception as e:
#                     if debug: st.exception(e)

#                 # 2D structure
#                 try:
#                     mol = Chem.MolFromSmiles(smiles)
#                     if mol:
#                         st.image(Draw.MolToImage(mol, size=(420, 280)),
#                                  caption="2D molecular structure", use_container_width=False)
#                 except Exception as e:
#                     if debug: st.exception(e)

#                 # Optional SMARTS toxicophores (if rules present)
#                 try:
#                     matches = match_toxicophores_with_explanations(smiles, lb)
#                     if matches:
#                         st.markdown("**Toxicophores detected**")
#                         try:
#                             _, img = highlight_toxicophores(smiles)
#                             st.image(img, caption="Matched substructures")
#                         except Exception as e:
#                             if debug: st.exception(e)
#                         for m in matches:
#                             st.markdown(f"- **{m['name']}** — {m['explanation']}")
#                 except Exception as e:
#                     if debug: st.exception(e)

#     # Near-miss explainer
#     with st.expander("Why not other endpoints? (largest negative concept drivers)", expanded=False):
#         try:
#             rows = explain_near_misses(smiles, top_n=len(label_cols), top_features=int(top_features_near_miss))
#             if rows:
#                 for lb, prob, thr, neg_df in rows:
#                     st.markdown(f"### {lb} — **Prob {prob:.3f} | Thr {thr:.3f} | Margin {prob - thr:.3f}**")
#                     c1, c2 = st.columns([1, 2], gap="large")
#                     with c1:
#                         gauge_fig = go.Figure(go.Indicator(
#                             mode="gauge+number+delta",
#                             value=float(prob),
#                             title={'text': "Prob vs Threshold"},
#                             delta={'reference': float(thr), 'increasing': {'color': "crimson"}},
#                             gauge={
#                                 'axis': {'range': [0, 1]},
#                                 'bar': {'color': "darkred"},
#                                 'steps': [
#                                     {'range': [0, float(thr)], 'color': "lightgray"},
#                                     {'range': [float(thr), 1], 'color': "lightpink"}
#                                 ],
#                                 'threshold': {'line': {'color': "red", 'width': 4}, 'value': float(thr)}
#                             }
#                         ))
#                         gauge_fig.update_layout(height=300, font=dict(size=18))
#                         st.plotly_chart(gauge_fig, use_container_width=True)
#                     with c2:
#                         st.dataframe(neg_df.reset_index(drop=True), use_container_width=True)
#         except Exception as e:
#             if debug: st.exception(e)






# # import json
# # import streamlit as st
# # import plotly.graph_objects as go
# # import numpy as np
# # import pandas as pd
# # from pathlib import Path
# # from rdkit import Chem
# # from rdkit.Chem import Draw

# # from utils import (
# #     label_cols,
# #     thresholds,
# #     validate_smiles,
# #     resolve_compound_name,
# #     predict_and_explain_all_labels,
# #     generate_toxicity_radar,
# #     generate_mechanistic_report,
# #     match_toxicophores_with_explanations,
# #     highlight_toxicophores,
# #     plot_shap_bar,
# #     explain_near_misses,
# #     platform_diagnostics,
# #     get_settings,
# #     set_use_standardization,
# # )

# # st.set_page_config(page_title="Tox21 Concept Explainer (ChemBERTa + TCAV)", layout="wide")

# # settings = get_settings()

# # st.title("🧪 Tox21 Concept Explainer (ChemBERTa + TCAV)")
# # st.write(
# #     "Input a **SMILES** string. The ChemBERTa model predicts toxicity across 12 endpoints and "
# #     "explains its decision with **concept-level TCAV evidence**."
# # )

# # # --------------------------
# # # Sidebar controls
# # # --------------------------
# # st.sidebar.header("Settings")

# # # Debug & diagnostics
# # debug = st.sidebar.checkbox("Debug mode (show detailed errors)", value=False)
# # with st.sidebar.expander("Diagnostics", expanded=False):
# #     st.code(json.dumps(platform_diagnostics(), indent=2))

# # # Canonicalization toggle
# # use_std = st.sidebar.checkbox(
# #     "Use standardized/canonical SMILES for inference",
# #     value=bool(settings.get("use_standardization", True))
# # )
# # set_use_standardization(use_std)

# # # Concept filters
# # st.sidebar.subheader("Concept Evidence Filters")
# # min_tcav = st.sidebar.slider("Min TCAV (keep ≥)", 0.5, 1.0, float(settings["tcav"]["min_tcav"]), 0.01)
# # max_p = st.sidebar.select_slider("Max p-value (keep ≤)", options=[0.10, 0.05, 0.01, 0.001], value=float(settings["tcav"]["max_p"]))
# # top_k_concepts = st.sidebar.number_input("Max concepts to display", min_value=3, max_value=15, value=int(settings["tcav"]["top_k"]), step=1)

# # # Near-miss control
# # top_features_near_miss = st.sidebar.number_input(
# #     "Top negative concepts to show (near-miss)",
# #     min_value=1, max_value=20, value=5, step=1
# # )

# # # --------------------------
# # # Input + trigger
# # # --------------------------
# # smiles = st.text_input("SMILES", placeholder="e.g., C1=CC=CC=C1 (benzene)")

# # new_input_submitted = smiles and st.session_state.get("last_submitted_smiles") != smiles
# # run_prediction = st.button("Run prediction") or new_input_submitted

# # if run_prediction and smiles:
# #     ok, msg = validate_smiles(smiles)
# #     if not ok:
# #         st.error(msg); st.stop()
# #     with st.spinner("Running ChemBERTa and assembling concept evidence..."):
# #         try:
# #             result = predict_and_explain_all_labels(
# #                 smiles, min_tcav=float(min_tcav), max_p=float(max_p), top_k=int(top_k_concepts)
# #             )
# #         except Exception as e:
# #             if debug: st.exception(e)
# #             else: st.error("Prediction failed. Enable 'Debug mode' for details.")
# #             st.stop()
# #     st.session_state["last_result"] = result
# #     st.session_state["last_submitted_smiles"] = smiles

# # # --------------------------
# # # Show results
# # # --------------------------
# # if "last_result" in st.session_state and smiles:
# #     result = st.session_state["last_result"]

# #     # Summary
# #     st.subheader("Summary")
# #     compound_name = resolve_compound_name(smiles)
# #     st.markdown(f"**SMILES:** `{smiles}`")
# #     st.markdown(f"**Compound name:** {compound_name}")

# #     st.markdown("### Model Predictions")
# #     if result["predicted_labels"]:
# #         st.write(", ".join(result["predicted_labels"]))
# #     else:
# #         st.write("_No endpoints predicted positive at current thresholds._")

# #     # Radar
# #     try:
# #         st.plotly_chart(generate_toxicity_radar(smiles, result), use_container_width=True)
# #     except Exception as e:
# #         if debug: st.exception(e)

# #     # Endpoint details
# #     selected_labels = sorted(
# #         result["predicted_labels"],
# #         key=lambda lb: result["explanations"][lb]["prob"],
# #         reverse=True
# #     )

# #     if selected_labels:
# #         st.subheader("Endpoint Details")
# #         for lb in selected_labels:
# #             exp = result["explanations"][lb]
# #             prob = float(exp["prob"]); thr = float(exp["threshold"])
# #             shap_df = exp["shap_df"]; margin_val = prob - thr

# #             with st.expander(f"{lb} — Prob {prob:.3f} | Thr {thr:.3f} | Margin {margin_val:.3f}", expanded=False):
# #                 try:
# #                     st.markdown(generate_mechanistic_report(lb, shap_df, prob, thr, smiles))
# #                 except Exception as e:
# #                     if debug: st.exception(e)
# #                 try:
# #                     st.plotly_chart(plot_shap_bar(shap_df), use_container_width=True)
# #                 except Exception as e:
# #                     if debug: st.exception(e)

# #                 # 2D structure
# #                 try:
# #                     mol = Chem.MolFromSmiles(smiles)
# #                     if mol:
# #                         st.image(Draw.MolToImage(mol, size=(420, 280)),
# #                                  caption="2D molecular structure", use_container_width=False)
# #                 except Exception as e:
# #                     if debug: st.exception(e)

# #                 # Optional SMARTS toxicophores (if rules present)
# #                 try:
# #                     matches = match_toxicophores_with_explanations(smiles, lb)
# #                     if matches:
# #                         st.markdown("**Toxicophores detected**")
# #                         try:
# #                             _, img = highlight_toxicophores(smiles)
# #                             st.image(img, caption="Matched substructures")
# #                         except Exception as e:
# #                             if debug: st.exception(e)
# #                         for m in matches:
# #                             st.markdown(f"- **{m['name']}** — {m['explanation']}")
# #                 except Exception as e:
# #                     if debug: st.exception(e)

# #     # Near-miss explainer
# #     with st.expander("Why not other endpoints? (largest negative concept drivers)", expanded=False):
# #         try:
# #             rows = explain_near_misses(smiles, top_n=len(label_cols), top_features=int(top_features_near_miss))
# #             if rows:
# #                 for lb, prob, thr, neg_df in rows:
# #                     st.markdown(f"### {lb} — **Prob {prob:.3f} | Thr {thr:.3f} | Margin {prob - thr:.3f}**")
# #                     c1, c2 = st.columns([1, 2], gap="large")
# #                     with c1:
# #                         gauge_fig = go.Figure(go.Indicator(
# #                             mode="gauge+number+delta",
# #                             value=float(prob),
# #                             title={'text': "Prob vs Threshold"},
# #                             delta={'reference': float(thr), 'increasing': {'color': "crimson"}},
# #                             gauge={
# #                                 'axis': {'range': [0, 1]},
# #                                 'bar': {'color': "darkred"},
# #                                 'steps': [
# #                                     {'range': [0, float(thr)], 'color': "lightgray"},
# #                                     {'range': [float(thr), 1], 'color': "lightpink"}
# #                                 ],
# #                                 'threshold': {'line': {'color': "red", 'width': 4}, 'value': float(thr)}
# #                             }
# #                         ))
# #                         gauge_fig.update_layout(height=300, font=dict(size=18))
# #                         st.plotly_chart(gauge_fig, use_container_width=True)
# #                     with c2:
# #                         st.dataframe(neg_df.reset_index(drop=True), use_container_width=True)
# #         except Exception as e:
# #             if debug: st.exception(e)
