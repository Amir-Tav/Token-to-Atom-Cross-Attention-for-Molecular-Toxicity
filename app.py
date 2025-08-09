import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
# RDKit for 2D molecular depiction
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
    plot_shap_bar,
    explain_near_misses,   # supports top_features param if you added earlier
)

st.set_page_config(page_title="Tox21 Meta-Explainer", layout="wide")

st.title("🧪 Tox21 Meta-Explainer")
st.write(
    "Input a **SMILES** string. The model predicts toxicity across 12 endpoints, "
    "then explains its decision with SHAP descriptors and SMARTS (toxicophore) matches."
)

# --------------------------
# Sidebar: controls
# --------------------------
st.sidebar.header("Settings")

# Near-miss control
top_features_near_miss = st.sidebar.number_input(
    "Top negative features to show (near-miss)",
    min_value=1, max_value=20, value=5, step=1
)

# Analysis Filters
st.sidebar.subheader("Analysis Filters")
max_feat_analysis = st.sidebar.slider(
    "Number of top features that influenced the model's predictions",
    min_value=5, max_value=100, value=20, step=1
)

# --------------------------
# Input + Trigger
# --------------------------
smiles = st.text_input("SMILES", placeholder="e.g., C1=CC=CC=C1 (benzene)")

# Check if new SMILES entered (Enter key or change)
new_input_submitted = (
    smiles
    and st.session_state.get("last_submitted_smiles") != smiles
)

# Prediction trigger
run_prediction = st.button("Run prediction") or new_input_submitted

if run_prediction and smiles:
    # Validate
    ok, msg = validate_smiles(smiles)
    if not ok:
        st.error(msg)
        st.stop()

    with st.spinner("Computing descriptors, running model, and generating explanations..."):
        try:
            result = predict_and_explain_all_labels(smiles)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    # Save to session_state so we can reuse without recomputing
    st.session_state["last_result"] = result
    st.session_state["last_submitted_smiles"] = smiles

# If we have a saved result, use it
if "last_result" in st.session_state and smiles:
    result = st.session_state["last_result"]

    # =========================
    # Summary
    # =========================
    st.subheader("Summary")
    compound_name = resolve_compound_name(smiles)
    st.markdown(f"**SMILES:** `{smiles}`")
    st.markdown(f"**Compound name:** {compound_name}")

    st.markdown("### Model Predictions")
    if result["predicted_labels"]:
        st.write(", ".join(result["predicted_labels"]))
    else:
        st.write("_No endpoints predicted positive at current thresholds._")

    st.plotly_chart(generate_toxicity_radar(smiles, result), use_container_width=True)

    # =========================
    # Endpoint details
    # =========================
    selected_labels = sorted(
        result["predicted_labels"],
        key=lambda lb: result["explanations"][lb]["prob"],
        reverse=True
    )

    if selected_labels:
        st.subheader("Endpoint Details")
        for lb in selected_labels:
            exp = result["explanations"][lb]
            prob = float(exp["prob"])
            thr = float(exp["threshold"])
            shap_df = exp["shap_df"]
            margin_val = prob - thr

            with st.expander(f"{lb} — Prob {prob:.3f} | Thr {thr:.3f} | Margin {margin_val:.3f}", expanded=False):
                st.markdown(generate_mechanistic_report(lb, shap_df, prob, thr, smiles))
                st.plotly_chart(plot_shap_bar(shap_df), use_container_width=True)

                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        st.image(Draw.MolToImage(mol, size=(420, 280)),
                                 caption="2D molecular structure", use_container_width=False)
                except Exception:
                    pass

                matches = match_toxicophores_with_explanations(smiles, lb)
                if matches:
                    st.markdown("**Toxicophores detected**")
                    _, img = highlight_toxicophores(smiles)
                    st.image(img, caption="Matched substructures")
                    for m in matches:
                        st.markdown(f"- **{m['name']}** — {m['explanation']}")

    # =========================
    # Near-miss explainer
    # =========================
    with st.expander("Why not other endpoints? (largest negative drivers)", expanded=False):
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
                    st.dataframe(neg_df[["feature", "shap_value"]].reset_index(drop=True),
                                 use_container_width=True)

    # =========================
    # Analysis
    # =========================
    st.subheader("Analysis")
    if selected_labels:
        selected_graph = "Feature coverage curve (how fast top features explain the prediction)"
        for lb in selected_labels:
            exp = result["explanations"][lb]
            shap_df = exp["shap_df"].copy()
            st.markdown(f"### {lb}")

            desc_df = shap_df[~shap_df["feature"].str.startswith("TOXICOPHORE_")].copy()
            if not desc_df.empty:
                desc_df["abs_shap"] = np.abs(pd.to_numeric(desc_df["shap_value"], errors="coerce"))
                desc_df = desc_df.sort_values("abs_shap", ascending=False)

                total_abs = desc_df["abs_shap"].sum() + 1e-12
                desc_df["cumulative_share"] = desc_df["abs_shap"].cumsum() / total_abs

                kmax = min(max_feat_analysis, len(desc_df))
                coverage = desc_df.head(kmax)
                feat_names = coverage["feature"].astype(str).tolist()

                cov_fig = go.Figure()
                cov_fig.add_trace(go.Scatter(
                    x=list(range(1, kmax + 1)),
                    y=coverage["cumulative_share"],
                    mode="lines+markers",
                    hovertext=feat_names,
                    hovertemplate="<b>%{hovertext}</b><br>Top #%{x}<br>Cumulative |SHAP|: %{y:.2f}<extra></extra>"
                ))
                cov_fig.update_layout(
                    title="Feature coverage curve (how fast top features explain the prediction)",
                    xaxis_title="Top features (ordered by |SHAP|)",
                    yaxis_title="Cumulative |SHAP| share",
                    yaxis=dict(range=[0, 1]),
                    height=340,
                    margin=dict(l=10, r=10, t=40, b=10),
                    xaxis=dict(
                        tickmode="array",
                        tickvals=list(range(1, kmax + 1)),
                        ticktext=[f if len(f) <= 16 else f[:14] + "…" for f in feat_names],
                        tickangle=45
                    )
                )
                st.plotly_chart(cov_fig, use_container_width=True)














# import os
# import json
# import numpy as np
# import pandas as pd
# import streamlit as st
# import plotly.graph_objects as go

# # RDKit for 2D molecular depiction
# from rdkit import Chem
# from rdkit.Chem import Draw

# # Import ONLY things that exist in your utils.py
# from utils import (
#     label_cols,
#     thresholds,
#     feature_masks,
#     feature_names,
#     validate_smiles,
#     resolve_compound_name,
#     compute_descriptors,
#     predict_and_explain_all_labels,
#     generate_toxicity_radar,
#     generate_mechanistic_report,
#     match_toxicophores_with_explanations,
#     highlight_toxicophores,
#     plot_shap_bar,
#     explain_near_misses,
#     postprocess_predictions,
#     _cached_model,
# )

# st.set_page_config(page_title="Tox21 Meta-Explainer", layout="wide")

# st.title("🧪 Tox21 Meta-Explainer")
# st.write(
#     "Input a **SMILES** string. The model predicts toxicity across 12 endpoints, "
#     "then explains its decision with SHAP descriptors and SMARTS (toxicophore) matches."
# )

# # --------------------------
# # Controls
# # --------------------------
# top_neg = st.sidebar.number_input(
#     "Top negative features to show (near-miss)",
#     min_value=1, max_value=20, value=3, step=1
# )

# # --------------------------
# # Input + Trigger
# # --------------------------
# smiles = st.text_input(
#     "SMILES",
#     placeholder="e.g., C1=CC=CC=C1 (benzene)"
# )

# # Detect if user pressed enter with a new SMILES
# new_input_submitted = (
#     smiles
#     and st.session_state.get("last_submitted_smiles") != smiles
# )

# # Prediction is triggered if:
# # - Button clicked
# # - OR new SMILES entered and Enter pressed
# run_prediction = st.button("Run prediction") or new_input_submitted

# # Store this SMILES in session state so it won't re-trigger unnecessarily
# if smiles:
#     st.session_state["last_submitted_smiles"] = smiles

# # --------------------------
# # Predict
# # --------------------------
# if run_prediction and smiles:
#     # Early validation before heavy compute
#     ok, msg = validate_smiles(smiles)
#     if not ok:
#         st.error(msg)
#         st.stop()

#     with st.spinner("Computing descriptors, running model, and generating explanations..."):
#         try:
#             result = predict_and_explain_all_labels(smiles)
#         except Exception as e:
#             st.error(f"Prediction failed: {e}")
#             st.stop()

#     # =========================
#     # Summary
#     # =========================
#     st.subheader("Summary")

#     # Show SMILES and resolved compound name
#     compound_name = resolve_compound_name(smiles)
#     st.markdown(f"**SMILES:** `{smiles}`")
#     st.markdown(f"**Compound name:** {compound_name}")

#     # Quick access list of predicted classes
#     st.markdown("### Model Predictions")
#     if result["predicted_labels"]:
#         st.write(", ".join(result["predicted_labels"]))
#     else:
#         st.write("_No endpoints predicted positive at current thresholds._")

#     st.write("")  # spacer

#     # Radar chart (always visible)
#     st.plotly_chart(generate_toxicity_radar(smiles, result), use_container_width=True)

#     # =========================
#     # Endpoint details
#     # =========================
#     selected_labels = sorted(
#         result["predicted_labels"],
#         key=lambda lb: result["explanations"][lb]["prob"],
#         reverse=True
#     )

#     if not selected_labels:
#         st.info("No endpoints passed the thresholds.")
#     else:
#         st.subheader("Endpoint Details")
#         for lb in selected_labels:
#             expl = result["explanations"].get(lb)
#             if not expl:
#                 continue

#             prob = float(expl["prob"])
#             thr = float(expl["threshold"])
#             shap_df = expl["shap_df"]
#             margin_val = prob - thr

#             with st.expander(f"{lb} — Prob {prob:.3f} | Thr {thr:.3f} | Margin {margin_val:.3f}", expanded=False):

#                 # Mechanistic report
#                 st.markdown(generate_mechanistic_report(lb, shap_df, prob, thr, smiles))

#                 # SHAP bar chart
#                 st.plotly_chart(plot_shap_bar(shap_df), use_container_width=True)

#                 # 2D molecule depiction
#                 try:
#                     mol = Chem.MolFromSmiles(smiles)
#                     if mol is not None:
#                         img2d = Draw.MolToImage(mol, size=(420, 280))
#                         st.image(img2d, caption="2D molecular structure", use_container_width=False)
#                 except Exception:
#                     pass

#                 # Toxicophores
#                 matches = match_toxicophores_with_explanations(smiles, lb)
#                 if matches:
#                     st.markdown("**Toxicophores detected**")
#                     _, img = highlight_toxicophores(smiles)
#                     st.image(img, caption="Matched substructures")
#                     for m in matches:
#                         st.markdown(f"- **{m['name']}** — {m['explanation']}")

#     # =========================
#     # Near-miss explainer
#     # =========================
#     with st.expander(" Why not other endpoints? (largest negative drivers)", expanded=False):
#         rows = explain_near_misses(smiles, top_n=len(label_cols))
#         if not rows:
#             st.caption("No near-misses to explain (or SHAP unavailable).")
#         else:
#             for lb, prob, thr, neg_df in rows:
#                 st.markdown(f"### {lb}  —  **Prob {prob:.3f} | Thr {thr:.3f} | Margin {prob - thr:.3f}**")

#                 c1, c2 = st.columns([1, 2], gap="large")

#                 with c1:
#                     # Gauge plot
#                     gauge_fig = go.Figure(go.Indicator(
#                         mode="gauge+number+delta",
#                         value=float(prob),
#                         title={'text': "Prob vs Threshold"},
#                         delta={'reference': float(thr), 'increasing': {'color': "crimson"}},
#                         gauge={
#                             'axis': {'range': [0, 1]},
#                             'bar': {'color': "darkred"},
#                             'steps': [
#                                 {'range': [0, float(thr)], 'color': "lightgray"},
#                                 {'range': [float(thr), 1], 'color': "lightpink"}
#                             ],
#                             'threshold': {
#                                 'line': {'color': "red", 'width': 4},
#                                 'thickness': 0.75,
#                                 'value': float(thr)
#                             }
#                         }
#                     ))
#                     gauge_fig.update_layout(
#                         autosize=True,
#                         height=300,  # smaller height
#                         font=dict(size=20)  # smaller fonts overall
#                     )
#                     st.plotly_chart(gauge_fig, use_container_width=True)

#                 with c2:
#                     st.caption("Largest negative SHAP contributors (pushing probability down)")
#                     st.dataframe(neg_df.reset_index(drop=True), use_container_width=True)
