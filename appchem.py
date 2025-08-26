import streamlit as st
import plotly.graph_objects as go
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw

# ---- utils imports (robust: some may be missing if you haven't enabled them) ----
import chemutils as U

# graceful getters (avoid crashes if helpers are absent)
def _get(name, default=None):
    return getattr(U, name, default)

label_cols = _get("label_cols", [
    "NR-AR","NR-AR-LBD","NR-AhR","NR-Aromatase","NR-ER",
    "NR-ER-LBD","NR-PPAR-gamma","SR-ARE","SR-ATAD5",
    "SR-HSE","SR-MMP","SR-p53"
])

thresholds = _get("thresholds", {lb: 0.5 for lb in label_cols})
validate_smiles = _get("validate_smiles", lambda s: (True, "ok"))
resolve_compound_name = _get("resolve_compound_name", lambda s: "(unknown)")

# LightGBM path (may be None if you don't use it)
predict_and_explain_all_labels   = _get("predict_and_explain_all_labels")
get_smarts_matches_for_label     = _get("get_smarts_matches_for_label")
plot_shap_waterfall              = _get("plot_shap_waterfall")
plot_shap_force                  = _get("plot_shap_force")
plot_feature_coverage_curve      = _get("plot_feature_coverage_curve")

# ChemBERTa path (new)
predict_all_labels_chemberta     = _get("predict_all_labels_chemberta")
explain_tokens_chemberta         = _get("explain_tokens_chemberta")
generate_token_heatmap           = _get("generate_token_heatmap")

# generic radar (works for both models). Fallback to LightGBM-specific if present.
generate_toxicity_radar_from_probs = _get("generate_toxicity_radar_from_probs")
generate_toxicity_radar             = _get("generate_toxicity_radar")  # legacy (LightGBM)

st.set_page_config(page_title="Tox21 Toxicity Predictor", layout="wide")
st.title("Tox21 Toxicity Prediction")
st.caption("Toggle between LightGBM (SHAP + SMARTS) and ChemBERTa (token attribution).")


# ---- small helper for per-label gauges (reusable) ----
def render_gauge_column(col, label, prob, thr):
    with col:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=float(prob),
            title={"text": label},
            delta={"reference": float(thr),
                   "increasing":{"color":"#00A884"},
                   "decreasing":{"color":"#E45755"}},
            gauge={"axis":{"range":[0,1]},
                   "threshold":{"line":{"color":"#636EFA","width":2},
                                "thickness":0.75,"value":float(thr)}}
        ))
        gauge.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(gauge, use_container_width=True)


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.subheader("Settings")
    model_choice = st.selectbox(
        "Model",
        ["LightGBM (SHAP + SMARTS)", "ChemBERTa (Token attribution)"],
        index=0 if predict_and_explain_all_labels else 1  # default to available
    )
    smiles = st.text_input("SMILES", value="CC(=O)Oc1ccccc1C(=O)O")  # Aspirin
    top_n_labels = st.slider("Top-N endpoints to display", 1, len(label_cols), 5)
    show_advanced = st.checkbox("Show advanced sections", value=True)

# =========================
# Main
# =========================
if st.button("Predict", use_container_width=True):
    ok, msg = validate_smiles(smiles)
    if not ok:
        st.error(f"Invalid SMILES: {msg}")
        st.stop()

    # 2D structure
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        st.image(Draw.MolToImage(mol, size=(420, 280)))

    compound_name = resolve_compound_name(smiles)

    # ---- run BOTH models (gracefully handle missing ones) ----
    lb_res = None
    dl_res = None

    if predict_and_explain_all_labels:
        try:
            lb_res = predict_and_explain_all_labels(smiles)     # LightGBM
        except Exception as e:
            st.warning(f"LightGBM inference failed: {e}")

    if predict_all_labels_chemberta:
        try:
            dl_res = predict_all_labels_chemberta(smiles)       # ChemBERTa
        except Exception as e:
            st.warning(f"ChemBERTa inference failed: {e}")

    # ---- summary header ----
    st.subheader("Summary")
    st.markdown(f"**SMILES:** `{smiles}`")
    st.markdown(f"**Compound name:** {compound_name}")

    # ---- side-by-side columns: LightGBM (left) vs ChemBERTa (right) ----
    cL, cR = st.columns(2, gap="large")

    # LEFT: LightGBM
    with cL:
        st.markdown("### LightGBM (SHAP + SMARTS)")
        if lb_res is None:
            st.info("LightGBM pipeline unavailable.")
        else:
            # radar
            if generate_toxicity_radar_from_probs:
                st.plotly_chart(
                    generate_toxicity_radar_from_probs(lb_res["probabilities"], thresholds, title="LightGBM Toxicity Radar"),
                    use_container_width=True
                )

            # predictions list
            if lb_res.get("predicted_labels"):
                st.write("**Predicted positive:** " + ", ".join(lb_res["predicted_labels"]))
            else:
                st.write("_No endpoints predicted positive at current thresholds._")

            # per-label gauges (top-N by prob)
            st.markdown("**Per-label probabilities**")
            lp = lb_res["probabilities"]
            lb_sorted = sorted(lp.items(), key=lambda kv: kv[1], reverse=True)[:top_n_labels]
            cols = st.columns(min(5, len(lb_sorted)))
            for i, (lb, pr) in enumerate(lb_sorted):
                render_gauge_column(cols[i % len(cols)], lb, pr, thresholds[lb])

    # RIGHT: ChemBERTa
    with cR:
        st.markdown("### ChemBERTa (Token attribution)")
        if dl_res is None:
            st.info("ChemBERTa pipeline unavailable.")
        else:
            # radar
            if generate_toxicity_radar_from_probs:
                st.plotly_chart(
                    generate_toxicity_radar_from_probs(dl_res["probabilities"], dl_res["thresholds"], title="ChemBERTa Toxicity Radar"),
                    use_container_width=True
                )

            # predictions list
            if dl_res.get("predicted_labels"):
                st.write("**Predicted positive:** " + ", ".join(dl_res["predicted_labels"]))
            else:
                st.write("_No endpoints predicted positive at current thresholds._")

            # per-label gauges (top-N by prob)
            st.markdown("**Per-label probabilities**")
            dp = dl_res["probabilities"]
            dt = dl_res["thresholds"]
            dl_sorted = sorted(dp.items(), key=lambda kv: kv[1], reverse=True)[:top_n_labels]
            cols = st.columns(min(5, len(dl_sorted)))
            for i, (lb, pr) in enumerate(dl_sorted):
                render_gauge_column(cols[i % len(cols)], lb, pr, dt[lb])

    # ---- agreement summary (only if both models ran) ----
    if lb_res is not None and dl_res is not None:
        st.markdown("---")
        st.subheader("Agreement summary")

        set_lb = set(lb_res.get("predicted_labels", []))
        set_dl = set(dl_res.get("predicted_labels", []))
        inter  = set_lb & set_dl
        union  = set_lb | set_dl
        jacc   = (len(inter) / len(union)) if union else 1.0

        c1, c2, c3 = st.columns(3)
        c1.metric("Both positive (count)", len(inter))
        c2.metric("Either positive (count)", len(union))
        c3.metric("Jaccard index", f"{jacc:.2f}")

        # simple comparison table
        rows = []
        for lb in label_cols:
            rows.append({
                "Label": lb,
                "LightGBM prob": float(lb_res["probabilities"].get(lb, 0.0)) if lb_res else None,
                "ChemBERTa prob": float(dl_res["probabilities"].get(lb, 0.0)) if dl_res else None,
                "LB ≥ thr": int(lb_res and lb_res["probabilities"].get(lb, 0.0) >= thresholds[lb]),
                "DL ≥ thr": int(dl_res and dl_res["probabilities"].get(lb, 0.0) >= (dl_res["thresholds"].get(lb, 0.5) if dl_res else 0.5)),
            })
        st.dataframe(rows, use_container_width=True)
        
    # =========================
    # CHEMBERTA PATH
    # =========================
    else:
        if predict_all_labels_chemberta is None:
            st.error("ChemBERTa pipeline not available in utils.py.")
            st.stop()

        result = predict_all_labels_chemberta(smiles)

        # --- summary ---
        st.subheader("Summary")
        st.markdown(f"**SMILES:** `{smiles}`")
        st.markdown(f"**Compound name:** {compound_name}")
        st.markdown("### Model Predictions (ChemBERTa)")
        if result["predicted_labels"]:
            st.write(", ".join(result["predicted_labels"]))
        else:
            st.write("_No endpoints predicted positive at current thresholds._")

        # Radar (generic)
        probs_dict = result["probabilities"]
        thr_dict   = result["thresholds"]
        if generate_toxicity_radar_from_probs:
            st.plotly_chart(
                generate_toxicity_radar_from_probs(probs_dict, thr_dict, title="ChemBERTa Toxicity Radar"),
                use_container_width=True
            )

        # Per-label gauges
        st.subheader("Per-label probabilities")
        probs_sorted = sorted(probs_dict.items(), key=lambda kv: kv[1], reverse=True)[:top_n_labels]
        cols = st.columns(min(5, len(probs_sorted)))
        for i, (lb, pr) in enumerate(probs_sorted):
            with cols[i % len(cols)]:
                thr = thr_dict[lb]
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=float(pr),
                    title={"text": lb},
                    delta={"reference": float(thr),
                           "increasing":{"color":"#00A884"},
                           "decreasing":{"color":"#E45755"}},
                    gauge={"axis":{"range":[0,1]},
                           "threshold":{"line":{"color":"#636EFA","width":2},"thickness":0.75,"value":float(thr)}}
                ))
                gauge.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(gauge, use_container_width=True)

        # Token attribution for the top label (if available)
        if show_advanced and explain_tokens_chemberta and generate_token_heatmap:
            top_label_idx = int(np.argmax(result["raw_prob_array"]))
            top_label = label_cols[top_label_idx]
            st.subheader(f"Token-level attribution — {top_label}")
            tokens, scores = explain_tokens_chemberta(smiles, label_idx=top_label_idx, n_steps=32)
            st.plotly_chart(
                generate_token_heatmap(tokens, scores, title=f"{top_label} (ChemBERTa IG)"),
                use_container_width=True
            )
            st.caption("Note: attributions are over SMILES tokens (not atoms).")

