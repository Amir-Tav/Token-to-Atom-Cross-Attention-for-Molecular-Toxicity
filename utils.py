# --- Standard Library ---
import os, io, json, joblib, functools

# --- Scientific Computing ---
import numpy as np
import pandas as pd

# --- Visualization ---
import plotly.graph_objects as go
from PIL import Image

# --- RDKit (Cheminformatics) ---
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")  # silence RDKit warnings

# --- Mordred (Descriptors) ---
from mordred import Calculator, descriptors

# --- SHAP (Explainability) ---
import shap  # works with LightGBM tree boosters

# --- External Lookup (Optional) ---
import pubchempy as pcp  # optional; used for name resolution

# ============================
# Paths & Globals
# ============================
# Default to v9; override via env var MODEL_PATH (e.g., models/v8)
MODEL_PATH = os.environ.get("MODEL_PATH", "tox21_lightgb_pipeline/models/v9")
META_PATH  = "tox21_lightgb_pipeline/Data_v6/meta_explainer"

label_cols = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER",
    "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5",
    "SR-HSE", "SR-MMP", "SR-p53",
]

# ============================
# Load artifacts
# ============================
with open(os.path.join(MODEL_PATH, "feature_names.txt")) as f:
    feature_names = f.read().splitlines()

thresholds     = joblib.load(os.path.join(MODEL_PATH, "thresholds.pkl"))    # for better classifications i could improve the threshold later...
feature_masks  = joblib.load(os.path.join(MODEL_PATH, "feature_masks.pkl"))

# label_mask not required at inference; load if present (kept for completeness)
try:
    label_mask = np.load(os.path.join(MODEL_PATH, "label_mask.npy"), allow_pickle=True)
except Exception:
    label_mask = None

with open(os.path.join(META_PATH, "meta_explanations_plain.json")) as f:
    META_EXPLAIN_DICT = json.load(f)
with open(os.path.join(META_PATH, "smarts_rules_final.json")) as f:
    SMARTS_RULES = json.load(f)

# Mordred calculator (2D only, as in training)
calc = Calculator(descriptors, ignore_3D=True)

# ── endpoint-specific mechanistic tails (edit as you like) ─────────────
ENDPOINT_TAIL = {
    "SR-MMP":   "enhanced zinc-binding capacity and reactivity at metalloprotein active sites — both linked to MMP interference",
    "SR-ARE":   "electrophilic / oxidative stress that triggers the ARE antioxidant pathway",
    "SR-ATAD5": "replication-fork stress and activation of the ATAD5 DNA-repair mechanism",
    "SR-HSE":   "protein-misfolding stress that induces the heat-shock response",
    "SR-p53":   "DNA-damage signalling that stabilises p53 and promotes cell-cycle arrest or apoptosis",
    "NR-AR":          "androgen-receptor activation and downstream androgenic gene expression",
    "NR-AR-LBD":      "high-affinity occupation of the AR ligand-binding domain and altered AR signalling",
    "NR-AhR":         "planar aromatic binding to AhR and dysregulated xenobiotic response genes",
    "NR-Aromatase":   "CYP19 aromatase inhibition via heme-iron coordination and reduced estrogen synthesis",
    "NR-ER":          "estrogen-receptor binding and transcriptional activation of ER target genes",
    "NR-ER-LBD":      "ligand-binding-domain engagement within ER, altering co-activator recruitment",
    "NR-PPAR-gamma":  "PPAR-γ activation that modulates adipogenic and metabolic gene regulation",
}

# ============================
# Caching helpers
# ============================

@functools.lru_cache(maxsize=128)
def _cached_model(label: str):
    """Fast model loader (cached)."""
    return joblib.load(os.path.join(MODEL_PATH, f"{label}.pkl"))

@functools.lru_cache(maxsize=512)
def _cached_descriptors_from_smiles(smiles: str):
    """Cache descriptor vector for repeated queries."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        desc = calc(mol).asdict()
        df = pd.DataFrame([desc])[feature_names]
        df = df.replace([np.inf, -np.inf], np.nan).fillna(-1)
        return df.values.flatten().tolist()
    except Exception:
        return None

# ============================
# Helper functions
# ============================
def validate_smiles(smiles: str) -> tuple[bool, str]:
    """Return (is_valid, message). Lightweight check used by the app before inference."""
    if not isinstance(smiles, str) or not smiles.strip():
        return False, "Please enter a non-empty SMILES string."
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, "Invalid SMILES string. Try a simple example like 'CCO' (ethanol)."
    return True, ""

def compute_descriptors(smiles: str):
    """Compute Mordred descriptors aligned to training order; NaN/inf -> -1 (cached)."""
    return _cached_descriptors_from_smiles(smiles)

@functools.lru_cache(maxsize=256)
def resolve_compound_name(smiles: str) -> str:
    """
    Resolve a human-friendly compound name from SMILES via PubChem.
    - Tries IUPAC first, then common name/synonym.
    - Gracefully falls back to 'Unknown compound' if offline/not found.
    """
    try:
        # primary lookup by SMILES
        comps = pcp.get_compounds(smiles, namespace="smiles")
        if not comps:
            return "Unknown compound"
        c = comps[0]
        # Prefer IUPAC name; otherwise use title or first synonym
        if getattr(c, "iupac_name", None):
            return c.iupac_name
        if getattr(c, "synonyms", None):
            return c.synonyms[0]
        if getattr(c, "title", None):
            return c.title
        return "Unknown compound"
    except Exception:
        return "Unknown compound"

def match_toxicophores_with_explanations(smiles: str, label: str | None = None):
    """Return list of {'name','explanation'} toxicophores that match the SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    label_rules = SMARTS_RULES.get(label, []) if label else [
        rule for rules in SMARTS_RULES.values() for rule in rules
    ]
    hits = []
    for rule in label_rules:
        patt = Chem.MolFromSmarts(rule["smarts"])
        if patt and mol.HasSubstructMatch(patt):
            hits.append({"name": rule["name"], "explanation": rule["explanation"]})
    return hits

def highlight_toxicophores(smiles: str):
    """Draw molecule with highlighted toxicophore substructures across all labels."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES structure.")
    AllChem.Compute2DCoords(mol)
    highlight_atoms = set()
    matched = []
    all_rules = [r for rules in SMARTS_RULES.values() for r in rules]
    for rule in all_rules:
        patt = Chem.MolFromSmarts(rule["smarts"])
        if not patt:
            continue
        matches = mol.GetSubstructMatches(patt)
        if matches:
            matched.append(f"☣️ **{rule['name']}**: {rule['explanation']}")
            for m in matches:
                highlight_atoms.update(m)
    drawer = rdMolDraw2D.MolDraw2DCairo(500, 300)
    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(highlight_atoms),
        legend="Matched Toxicophores" if matched else "No toxicophores found"
    )
    drawer.FinishDrawing()
    img = Image.open(io.BytesIO(drawer.GetDrawingText()))
    return matched, img

def build_one_liner(label: str, shap_df: pd.DataFrame, smiles: str) -> str:
    """Short textual rationale: top descriptor + first toxicophore + mechanistic tail."""
    desc_df = shap_df[~shap_df["feature"].str.startswith("TOXICOPHORE_")].copy()
    desc_df = desc_df.reindex(desc_df["shap_value"].abs().sort_values(ascending=False).index)
    if desc_df.empty:
        return f"Predicted {label} toxicity lacks a dominant molecular driver."
    top_desc = desc_df.iloc[0]
    d_name = top_desc["feature"]
    d_dir  = "high " if top_desc["shap_value"] > 0 else "low "
    tox = match_toxicophores_with_explanations(smiles, label)
    tox_part = f"and the presence of a {tox[0]['name']}" if tox else ""
    tail = ENDPOINT_TAIL.get(label, "mechanistic activity")
    return f"Predicted {label} toxicity may be due to {d_dir}{d_name} {tox_part}, suggesting {tail}."

# ============================
# Post-processing (display-only)
# ============================
def postprocess_predictions(probs_dict: dict, thresholds_dict: dict, top_k: int | None = None, margin: float | None = None):
    """
    Display-only filter: keep labels whose prob >= threshold, optionally limit to top_k,
    and optionally require within 'margin' of the top probability.
    Raw model outputs are NOT modified upstream; this just filters what we show.
    """
    passed = {lbl: p for lbl, p in probs_dict.items() if p >= thresholds_dict.get(lbl, 0.5)}
    if top_k is not None:
        passed = dict(sorted(passed.items(), key=lambda x: x[1], reverse=True)[:top_k])
    if margin is not None and len(passed) > 1:
        top_prob = max(passed.values())
        passed = {lbl: p for lbl, p in passed.items() if p >= top_prob * margin}
    return passed

# ============================
# Core prediction + SHAP
# ============================
def predict_and_explain_all_labels(smiles: str):
    """
    Uses LightGBM boosters in MODEL_PATH + per-label thresholds.
    Generates SHAP explanations and injects SMARTS pseudo-features.
    Includes runtime safeguards: if SHAP fails, falls back to tree feature importances.
    """
    desc_values = compute_descriptors(smiles)
    if desc_values is None:
        raise ValueError("Invalid SMILES")

    X_full = np.array([desc_values])
    results = {}
    predicted_labels = []

    for label in label_cols:
        booster = _cached_model(label)
        kept_idx = feature_masks[label]  # boolean mask
        X_input = X_full[:, kept_idx]
        features = [feature_names[i] for i, keep in enumerate(kept_idx) if keep]

        # Probability for binary task
        prob = float(booster.predict(X_input)[0])
        thr = float(thresholds[label])
        pred = int(prob >= thr)

        if pred == 1:
            predicted_labels.append(label)

            # SHAP explanation with fallback
            shap_df = None
            try:
                explainer = shap.Explainer(booster)
                ex = explainer(X_input)
                shap_vals = np.array(ex.values[0], dtype=float)
                shap_df = pd.DataFrame({
                    "feature": features,
                    "shap_value": shap_vals,
                    "feature_value": X_input[0]
                })
            except Exception:
                # Fallback: normalized split gain as proxy
                imp = booster.feature_importance(importance_type="gain")
                shap_df = pd.DataFrame({
                    "feature": features,
                    "shap_value": (imp[:len(features)] / (np.sum(imp) + 1e-8)).astype(float),
                    "feature_value": X_input[0]
                })

            # Inject SMARTS pseudo-features
            for match in match_toxicophores_with_explanations(smiles, label):
                shap_df = pd.concat([
                    shap_df,
                    pd.DataFrame([{
                        "feature": f"TOXICOPHORE_{match['name']}",
                        "shap_value": 0.01,
                        "feature_value": 1.0
                    }])
                ], ignore_index=True)

            results[label] = {
                "prob": prob,
                "threshold": thr,
                "pred_score": prob,
                "shap_df": shap_df,
                "top_features": (
                    shap_df[["feature", "shap_value"]]
                    .reindex(shap_df["shap_value"].abs().sort_values(ascending=False).index)
                    .head(4)
                    .values
                    .tolist()
                )
            }

    return {"smiles": smiles, "predicted_labels": predicted_labels, "explanations": results}

# ============================
# Visualization helpers
# ============================
def plot_shap_bar(shap_df: pd.DataFrame, top: int = 10):
    """
    Horizontal bar chart of top-|SHAP| descriptors.
    Excludes SMARTS pseudo-features so this reflects descriptor contribution only.
    """
    import plotly.express as px
    df = shap_df.copy()
    df = df[~df["feature"].str.startswith("TOXICOPHORE_")]
    df = df.reindex(df["shap_value"].abs().sort_values(ascending=False).index).head(top)
    fig = px.bar(df, x="shap_value", y="feature", orientation="h")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
    return fig

def explain_near_misses(smiles: str, top_n: int = 3, top_features: int = 3):
    """
    For the non-predicted labels with the highest probabilities (closest to threshold),
    return tuples of (label, prob, thr, neg_df) where neg_df lists the top negative SHAP drivers.
    top_features controls how many negative contributors to include per label.
    """
    desc_values = compute_descriptors(smiles)
    if desc_values is None:
        return []
    X_full = np.array([desc_values])
    rows = []
    for label in label_cols:
        booster = _cached_model(label)
        kept = feature_masks[label]
        X_in = X_full[:, kept]
        prob = float(booster.predict(X_in)[0])
        thr = float(thresholds[label])
        if prob < thr:
            try:
                ex = shap.Explainer(booster)(X_in)
                shap_vals = np.array(ex.values[0], dtype=float)
            except Exception:
                imp = booster.feature_importance(importance_type="gain")
                shap_vals = - (imp[:int(kept.sum())] / (np.sum(imp) + 1e-8))
            feats = [feature_names[i] for i, k in enumerate(kept) if k]
            df = pd.DataFrame({"feature": feats, "shap_value": shap_vals})
            neg = df.sort_values("shap_value").head(int(top_features))  # most negative contributors
            rows.append((label, prob, thr, neg))
    rows.sort(key=lambda t: t[1], reverse=True)
    return rows[:int(top_n)]
def generate_toxicity_radar(smiles: str, results: dict):
    """Radar over all 12 endpoints showing probabilities (0 for non-predicted)."""
    probs = [results["explanations"].get(lbl, {}).get("prob", 0.0) for lbl in label_cols]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=probs + [probs[0]],
        theta=label_cols + [label_cols[0]],
        fill='toself',
        name='Predicted Toxicity',
        line=dict(color='crimson'),
        marker=dict(symbol='circle'),
        hovertemplate='%{theta}<br>Prob: %{r:.2f}<extra></extra>',
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickfont_size=10)),
        showlegend=False,
        title=f"Toxicity Radar for: {smiles}",
        margin=dict(l=30, r=30, t=50, b=30),
        height=500
    )
    return fig

def generate_mechanistic_report(
    label: str,
    shap_df: pd.DataFrame,
    prob: float,
    threshold: float,
    smiles: str,
    top_k: int = 4,
    shap_cutoff: float = 0.01,
):
    """Markdown block: confidence + one-liner + contributing descriptors."""
    lines = [
        f"### 🔍 {label} — Mechanistic Report\n",
        f"✅ **Prediction confidence**: `{prob:.2f}` (threshold = `{threshold:.2f}`)",
        "",
        build_one_liner(label, shap_df, smiles),
        "",
        "📊 **Contributing Molecular Descriptors:**"
    ]
    desc_df = shap_df[~shap_df["feature"].str.startswith("TOXICOPHORE_")].copy()
    desc_df = desc_df.reindex(desc_df["shap_value"].abs().sort_values(ascending=False).index)
    desc_df = desc_df[desc_df["shap_value"].abs() > shap_cutoff].head(top_k)

    if desc_df.empty:
        lines.append("- No dominant molecular descriptors detected.")
    else:
        for _, row in desc_df.iterrows():
            fname = row["feature"]
            shap_val = float(row["shap_value"])
            direction = "↑ increase" if shap_val > 0 else "↓ decrease"
            expl = META_EXPLAIN_DICT.get(fname, "no biological annotation")
            lines.append(f"- **{fname}**: {expl} ({direction}, SHAP={shap_val:.3f})")

    lines.append("\n---")
    return "\n".join(lines)

def summarize_prediction(result: dict):
    """Original brief summary sentence (kept for completeness)."""
    smiles = result["smiles"]
    predicted = result["predicted_labels"]
    if not predicted:
        return f"🔬 The drug (SMILES: `{smiles}`) is predicted **not to exhibit significant toxicity endpoints.**"
    ordered = sorted(predicted, key=lambda lb: result["explanations"][lb]["pred_score"], reverse=True)
    return f"🔬 The drug (SMILES: `{smiles}`) is predicted to be toxic in: **{', '.join(ordered)}**."