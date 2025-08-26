# ============================
# utils.py — unified helpers for LightGBM and ChemBERTa
# ============================

# --- stdlib ---
import os, json
from typing import Dict, List, Tuple, Optional

# --- scientific ---
import numpy as np
import pandas as pd

# --- rdkit / mordred (optional; only needed for SMARTS or descriptors) ---
try:
    from rdkit import Chem
except Exception:
    Chem = None

# --- external (optional) ---
try:
    import pubchempy as pcp
except Exception:
    pcp = None

# --- plotting ---
import plotly.graph_objects as go

# --- transformers / torch (ChemBERTa) ---
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import IntegratedGradients

# ============================
# Paths & labels
# ============================
MODEL_PATH_LGB = os.environ.get("MODEL_PATH", "tox21_lightgb_pipeline/models/v9")
CHEMBERTA_DIR  = os.environ.get("CHEMBERTA_MODEL_DIR",  "tox21_chembera_pipeline/models/chemberta_v1")
CHEMBERTA_OUT  = os.environ.get("CHEMBERTA_OUTPUTS_DIR","tox21_chembera_pipeline/outputs")

label_cols = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER",
    "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5",
    "SR-HSE", "SR-MMP", "SR-p53",
]

# ============================
# Validation / name resolve
# ============================
def validate_smiles(s: str) -> Tuple[bool, str]:
    if not isinstance(s, str) or not s.strip():
        return False, "Empty input."
    if Chem is None:
        return True, "RDKit not available; skipping validation."
    m = Chem.MolFromSmiles(s)
    if m is None:
        return False, "RDKit failed to parse the SMILES."
    return True, "OK"

def resolve_compound_name(smiles: str) -> str:
    if pcp is None:
        return "(unknown)"
    try:
        res = pcp.get_compounds(smiles, "smiles")
        if res:
            n = res[0].iupac_name or res[0].synonyms[0]
            return n or "(unknown)"
    except Exception:
        pass
    return "(unknown)"

# ============================
# Thresholds (LightGBM) — fallback 0.5 each
# ============================
def _load_thresholds_lgb() -> Dict[str, float]:
    # try to load thresholds.json from LightGBM models dir
    path_json = os.path.join(MODEL_PATH_LGB, "thresholds.json")
    if os.path.exists(path_json):
        try:
            with open(path_json, "r") as f:
                th = json.load(f)
            # ensure all labels present
            return {lb: float(th.get(lb, 0.5)) for lb in label_cols}
        except Exception:
            pass
    return {lb: 0.5 for lb in label_cols}

thresholds: Dict[str, float] = _load_thresholds_lgb()

# ============================
# LightGBM predictor (minimal; no SHAP here)
# NOTE: This is a compatibility shim. If you have your own advanced LightGBM
#       pipeline with SHAP/SMARTS, keep that instead of this stub.
# ============================
def predict_and_explain_all_labels(smiles: str):
    """
    Minimal compatibility version:
    - Validates SMILES
    - Returns zero-probabilities and no SHAP if no LightGBM models are present
    - If you have your own LightGBM loading/prediction code, keep it instead.
    """
    ok, msg = validate_smiles(smiles)
    if not ok:
        raise ValueError(msg)

    # You can replace this with your real LightGBM inference.
    # For now we return zeros so the UI still works.
    probs = {lb: 0.0 for lb in label_cols}
    preds = [lb for lb, p in probs.items() if p >= thresholds[lb]]

    return {
        "probabilities": probs,      # {label: prob}
        "predicted_labels": preds,   # []
        "shap_per_label": {},        # {label: shap_df}, if you implement
    }

# ============================
# Generic radar chart (probabilities dict) — works for both models
# ============================
def generate_toxicity_radar_from_probs(probabilities: Dict[str, float],
                                       thresholds_dict: Optional[Dict[str, float]] = None,
                                       title: str = "Toxicity Radar"):
    probs = [float(probabilities.get(lbl, 0.0)) for lbl in label_cols]
    labels = label_cols + [label_cols[0]]
    series = probs + [probs[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=series,
        theta=labels,
        fill='toself',
        name='Probability',
        line=dict(color='crimson'),
        hovertemplate='%{theta}<br>Prob: %{r:.3f}<extra></extra>',
    ))

    if thresholds_dict is not None:
        thr = [float(thresholds_dict.get(lbl, 0.5)) for lbl in label_cols]
        thr_series = thr + [thr[0]]
        fig.add_trace(go.Scatterpolar(
            r=thr_series,
            theta=labels,
            name='Threshold',
            line=dict(color='#636EFA', dash='dot'),
            hovertemplate='%{theta}<br>Thr: %{r:.3f}<extra></extra>',
        ))

    fig.update_layout(
        title=title,
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickfont_size=10)),
        showlegend=True,
        margin=dict(l=30, r=30, t=50, b=30),
        height=500
    )
    return fig

# ============================
# ChemBERTa integration
# ============================

_CHEMBERTA_CACHE = {
    "tokenizer": None,
    "model": None,
    "thresholds": None,
    "device": None,
}

def _load_chemberta_thresholds():
    try:
        path = os.path.join(CHEMBERTA_OUT, "summary.json")
        with open(path, "r") as f:
            sj = json.load(f)
        th = sj.get("thresholds", {})
        return np.asarray([float(th.get(lbl, 0.5)) for lbl in label_cols], dtype=float)
    except Exception:
        return np.full(len(label_cols), 0.5, dtype=float)

def load_chemberta_pipeline():
    if _CHEMBERTA_CACHE["model"] is not None:
        return (_CHEMBERTA_CACHE["tokenizer"],
                _CHEMBERTA_CACHE["model"],
                _CHEMBERTA_CACHE["thresholds"],
                _CHEMBERTA_CACHE["device"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(CHEMBERTA_DIR)
    mdl = AutoModelForSequenceClassification.from_pretrained(CHEMBERTA_DIR)
    mdl.to(device).eval()
    th = _load_chemberta_thresholds()
    _CHEMBERTA_CACHE.update({
        "tokenizer": tok,
        "model": mdl,
        "thresholds": th,
        "device": device
    })
    return tok, mdl, th, device

def _sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))

def predict_all_labels_chemberta(smiles: str):
    ok, msg = validate_smiles(smiles)
    if not ok:
        raise ValueError(f"Invalid SMILES: {msg}")

    tokenizer, model, th, device = load_chemberta_pipeline()
    enc = tokenizer(
        smiles,
        padding=False,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits.detach().cpu().numpy()  # (1, C)
        prob   = _sigmoid_np(logits)[0]                       # (C,)

    probs_dict = {label_cols[i]: float(prob[i]) for i in range(len(label_cols))}
    thr_dict   = {label_cols[i]: float(th[i])   for i in range(len(label_cols))}
    preds      = [label_cols[i] for i in range(len(label_cols)) if prob[i] >= th[i]]

    return {
        "probabilities": probs_dict,
        "thresholds": thr_dict,
        "predicted_labels": preds,
        "logits": logits,
        "raw_prob_array": prob
    }

def explain_tokens_chemberta(smiles: str, label_idx: int, n_steps: int = 32):
    tokenizer, model, _, device = load_chemberta_pipeline()

    enc = tokenizer(
        smiles,
        padding=False,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    def forward_embeds(inputs_embeds):
        out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits
        return torch.sigmoid(out)[:, label_idx]

    emb_layer = model.get_input_embeddings()
    inputs_embeds = emb_layer(input_ids).detach().clone().requires_grad_(True)

    ig = IntegratedGradients(forward_embeds)
    baselines = torch.zeros_like(inputs_embeds)
    attributions, _ = ig.attribute(
        inputs=inputs_embeds,
        baselines=baselines,
        n_steps=n_steps,
        return_convergence_delta=True
    )
    token_scores = attributions.abs().sum(dim=-1).detach().cpu().numpy()[0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    if token_scores.max() > 0:
        token_scores = token_scores / token_scores.max()

    return tokens, token_scores

def generate_token_heatmap(tokens, scores, title="Token attribution"):
    fig = go.Figure()
    fig.add_bar(x=list(range(len(tokens))), y=scores)
    fig.update_layout(
        title=title,
        xaxis_title="Token index (SMILES tokens)",
        yaxis_title="Normalized importance",
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(len(tokens))),
            ticktext=[t if len(t) <= 8 else t[:6] + "…" for t in tokens],
            tickangle=45
        )
    )
    return fig

# ============================
# Optional LightGBM visuals (waterfall/force/coverage) + SMARTS wrapper
# Provide no-op defaults; replace with your real implementations if desired.
# ============================
def get_smarts_matches_for_label(label: str, smiles: str):
    # Placeholder: return empty list unless you wire SMARTS rules
    return []

def plot_shap_waterfall(shap_df: pd.DataFrame, top: int = 12):
    # Minimal bar chart from a dataframe with columns ['feature','shap_value']
    df = shap_df.copy()
    if "feature" not in df or "shap_value" not in df:
        return go.Figure()
    df["abs"] = df["shap_value"].abs()
    df = df.sort_values("abs", ascending=False).head(int(top))
    fig = go.Figure()
    fig.add_bar(
        x=df["feature"],
        y=df["shap_value"],
        marker_color=np.where(df["shap_value"] >= 0, "#2ca02c", "#d62728"),
        hovertemplate="%{x}<br>SHAP=%{y:.4f}<extra></extra>",
    )
    fig.update_layout(
        title="SHAP contribution (top features)",
        xaxis_title="Feature",
        yaxis_title="SHAP",
        height=360,
        margin=dict(l=10, r=10, t=40, b=60),
        xaxis=dict(tickangle=45),
        showlegend=False,
    )
    return fig

def plot_shap_force(shap_df: pd.DataFrame, top: int = 12):
    df = shap_df.copy()
    if "feature" not in df or "shap_value" not in df:
        return go.Figure()
    df["abs"] = df["shap_value"].abs()
    df = df.sort_values("abs", ascending=False).head(int(top))
    pos = df[df["shap_value"] > 0]
    neg = df[df["shap_value"] < 0]
    fig = go.Figure()
    fig.add_bar(x=pos["feature"], y=pos["shap_value"], name="Positive", marker_color="#2ca02c")
    fig.add_bar(x=neg["feature"], y=neg["shap_value"], name="Negative", marker_color="#d62728")
    fig.update_layout(
        barmode="relative",
        title="SHAP force-style (top features)",
        xaxis_title="Feature", yaxis_title="Contribution",
        height=360, margin=dict(l=10, r=10, t=40, b=60),
        xaxis=dict(tickangle=45),
    )
    return fig

def plot_feature_coverage_curve(shap_df: pd.DataFrame, kmax: int = 20):
    df = shap_df.copy()
    if "shap_value" not in df:
        return go.Figure()
    vals = df["shap_value"].abs().sort_values(ascending=False).to_numpy()
    if vals.size == 0:
        xs = np.arange(1, kmax + 1)
        ys = np.zeros_like(xs, dtype=float)
    else:
        vals = vals[:kmax]
        csum = np.cumsum(vals)
        xs = np.arange(1, len(vals) + 1)
        ys = csum / (np.sum(vals) + 1e-12)
    fig = go.Figure()
    fig.add_scatter(x=xs, y=ys, mode="lines+markers")
    fig.update_layout(
        title="Cumulative |SHAP| coverage",
        xaxis_title="Top-k features", yaxis_title="Cumulative |SHAP| share",
        yaxis=dict(range=[0, 1]),
        height=340, margin=dict(l=10, r=10, t=40, b=40),
    )
    return fig