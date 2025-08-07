# --- Standard Library ---
import os, io, json, joblib

# --- Scientific Computing ---
import numpy as np
import pandas as pd

# --- Visualization ---
import plotly.graph_objects as go
from PIL import Image

# --- RDKit (Cheminformatics) ---
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")        # silence RDKit warnings

# --- Mordred (Descriptors) ---
from mordred import Calculator, descriptors

# --- SHAP (Explainability) ---
import shap

# --- External Lookup (Optional) ---
import pubchempy as pcp              # optional

# --- Paths & Globals -------------------------------------------------
MODEL_PATH = "tox21_lightgb_pipeline/models/v7"
SAVE_DIR   = "tox21_lightgb_pipeline/Data_v6/processed"

label_cols = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER",
    "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5",
    "SR-HSE", "SR-MMP", "SR-p53",
]

# --------------------------------------------------------------------
#  Load feature names, meta-explanations, and per-endpoint SMARTS rules
# --------------------------------------------------------------------
with open(os.path.join(SAVE_DIR, "feature_names.txt")) as f:
    feature_names = f.read().splitlines()

with open("tox21_lightgb_pipeline/Data_v6/meta_explainer/meta_explanations_plain.json") as f:
    META_EXPLAIN_DICT = json.load(f)

with open("tox21_lightgb_pipeline/Data_v6/meta_explainer/smarts_rules_final.json") as f:
    SMARTS_RULES = json.load(f)

thresholds     = joblib.load(os.path.join(MODEL_PATH, "thresholds.pkl"))
feature_masks  = joblib.load(os.path.join(MODEL_PATH, "feature_masks.pkl"))
calc           = Calculator(descriptors, ignore_3D=True)


# ── endpoint-specific mechanistic tails (edit as you like) ─────────────
ENDPOINT_TAIL = {
    # Stress-response (SR) endpoints ────────────────────────────────
    "SR-MMP":   "enhanced zinc-binding capacity and reactivity at metalloprotein active sites — both linked to MMP interference",
    "SR-ARE":   "electrophilic / oxidative stress that triggers the ARE antioxidant pathway",
    "SR-ATAD5": "replication-fork stress and activation of the ATAD5 DNA-repair mechanism",
    "SR-HSE":   "protein-misfolding stress that induces the heat-shock response",
    "SR-p53":   "DNA-damage signalling that stabilises p53 and promotes cell-cycle arrest or apoptosis",

    # Nuclear-receptor (NR) endpoints ──────────────────────────────
    "NR-AR":          "androgen-receptor activation and downstream androgenic gene expression",
    "NR-AR-LBD":      "high-affinity occupation of the AR ligand-binding domain and altered AR signalling",
    "NR-AhR":         "planar aromatic binding to AhR and dysregulated xenobiotic response genes",
    "NR-Aromatase":   "CYP19 aromatase inhibition via heme-iron coordination and reduced estrogen synthesis",
    "NR-ER":          "estrogen-receptor binding and transcriptional activation of ER target genes",
    "NR-ER-LBD":      "ligand-binding-domain engagement within ER, altering co-activator recruitment",
    "NR-PPAR-gamma":  "PPAR-γ activation that modulates adipogenic and metabolic gene regulation",
}


## for the one line paraph 
def build_one_liner(label: str, shap_df: pd.DataFrame, smiles: str) -> str:
    """Return a sentence: descriptor + toxicophore + tail."""
    # 1) top descriptor
    desc_df = shap_df[~shap_df["feature"].str.startswith("TOXICOPHORE_")].copy()
    desc_df = desc_df.reindex(desc_df["shap_value"].abs().sort_values(ascending=False).index)
    if desc_df.empty:
        return f"Predicted {label} toxicity lacks a dominant molecular driver."

    top_desc = desc_df.iloc[0]
    d_name   = top_desc["feature"]
    d_dir    = "high " if top_desc["shap_value"] > 0 else "low "
    desc_part = f"{d_dir}{d_name}"

    # 2) first matching toxicophore, if any
    tox_matches = match_toxicophores_with_explanations(smiles, label)
    if tox_matches:
        tox_part = f"and the presence of a {tox_matches[0]['name']}"
    else:
        tox_part = ""

    # 3) tail
    tail = ENDPOINT_TAIL.get(label, "mechanistic activity")

    return f"Predicted {label} toxicity may be due to {desc_part} {tox_part}, suggesting {tail}."


def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        desc = calc(mol).asdict()
        return [desc.get(f, 0.0) for f in feature_names]
    except:
        return None


def match_toxicophores_with_explanations(smiles, label=None):
    """
    Match SMARTS toxicophores based on the user's SMILES and toxicity label.
    If label is None, all rules are checked.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    matches = []

    # Get all SMARTS for the given label, or all labels if label=None
    label_smarts = SMARTS_RULES.get(label, []) if label else [
        rule for all_rules in SMARTS_RULES.values() for rule in all_rules
    ]

    for rule in label_smarts:
        patt = Chem.MolFromSmarts(rule["smarts"])
        if patt and mol.HasSubstructMatch(patt):
            matches.append({
                "name": rule["name"],
                "explanation": rule["explanation"]
            })

    return matches


def highlight_toxicophores(smiles):
    """
    Draw molecule with highlighted toxicophores across all labels.
    Returns matched toxicophores (with explanations) and the image.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES structure.")

    AllChem.Compute2DCoords(mol)
    highlight_atoms = set()
    matched_toxophores = []

    # Flatten all SMARTS rules
    all_rules = [rule for sublist in SMARTS_RULES.values() for rule in sublist]

    for rule in all_rules:
        patt = Chem.MolFromSmarts(rule["smarts"])
        if not patt:
            continue
        matches = mol.GetSubstructMatches(patt)
        if matches:
            matched_toxophores.append(f"☣️ **{rule['name']}**: {rule['explanation']}")
            for match in matches:
                highlight_atoms.update(match)

    # Draw molecule with highlighted atoms
    drawer = rdMolDraw2D.MolDraw2DCairo(500, 300)
    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(highlight_atoms),
        legend="Matched Toxicophores" if matched_toxophores else "No toxicophores found"
    )
    drawer.FinishDrawing()
    img_data = drawer.GetDrawingText()
    img = Image.open(io.BytesIO(img_data))

    return matched_toxophores, img


def predict_and_explain_all_labels(smiles):
    """
    Predict all 12 toxicity labels using trained models and enrich explanation using SHAP + SMARTS rules.
    """
    desc_values = compute_descriptors(smiles)
    if desc_values is None:
        raise ValueError("Invalid SMILES")

    X_full = np.array([desc_values])
    results = {}
    predicted_labels = []

    for label in label_cols:
        model = joblib.load(os.path.join(MODEL_PATH, f"{label}.pkl"))
        threshold = thresholds[label]
        kept_indices = feature_masks[label]

        X_input = X_full[:, kept_indices]
        features = [feature_names[i] for i in kept_indices]

        prob = model.predict_proba(X_input)[0, 1]
        pred = int(prob >= threshold)

        if pred == 1:
            predicted_labels.append(label)
            explainer = shap.Explainer(model)
            shap_values = explainer(X_input)

            shap_df = pd.DataFrame({
                "feature": features,
                "shap_value": shap_values.values[0],
                "feature_value": X_input[0]
            })

            # Inject SMARTS pseudo-features specific to this label
            smarts_matches = match_toxicophores_with_explanations(smiles, label)
            for match in smarts_matches:
                to_add = pd.DataFrame([{
                    "feature": f"TOXICOPHORE_{match['name']}",
                    "shap_value": 0.01,
                    "feature_value": 1.0
                }])
                shap_df = pd.concat([shap_df, to_add], ignore_index=True)

            results[label] = {
                "prob": prob,
                "threshold": threshold,
                "pred_score": prob,
                "shap_df": shap_df,
                "top_features": shap_df[["feature", "shap_value"]].head(4).values.tolist()
            }

    return {
        "smiles": smiles,
        "predicted_labels": predicted_labels,
        "explanations": results
    }


## textual justification and explanations 
def generate_mechanistic_report(
    label,
    shap_df,
    prob,
    threshold,
    smiles,
    top_k: int = 4,
    shap_cutoff: float = 0.01,
):
    """
    Returns markdown: one-liner summary + descriptor table.
    """
    lines = [
        f"### 🔍 {label} — Mechanistic Report\n",
        f"✅ **Prediction confidence**: `{prob:.2f}` (threshold = `{threshold:.2f}`)",
    ]

    # ── NEW one-liner summary ───────────────────────────────────────────
    lines.append("\n" + build_one_liner(label, shap_df, smiles) + "\n")

    # ── Descriptor Features (unchanged) ────────────────────────────────
    lines.append("\n📊 **Contributing Molecular Descriptors:**")
    desc_df = shap_df[~shap_df["feature"].str.startswith("TOXICOPHORE_")].copy()
    desc_df = desc_df.reindex(desc_df["shap_value"].abs().sort_values(ascending=False).index)
    desc_df = desc_df[desc_df["shap_value"].abs() > shap_cutoff].head(top_k)

    if desc_df.empty:
        lines.append("- No dominant molecular descriptors detected.")
    else:
        for _, row in desc_df.iterrows():
            fname     = row["feature"]
            shap_val  = row["shap_value"]
            direction = "↑ increase" if shap_val > 0 else "↓ decrease"
            expl      = META_EXPLAIN_DICT.get(fname, "no biological annotation")
            lines.append(f"- **{fname}**: {expl} ({direction}, SHAP={shap_val:.3f})")

    lines.append("\n---")
    return "\n".join(lines)

def summarize_prediction(result):
    smiles = result["smiles"]
    predicted = result["predicted_labels"]
    if not predicted:
        return f"🔬 The drug (SMILES: `{smiles}`) is predicted **not to exhibit significant toxicity endpoints.**"

    sorted_labels = sorted(
        predicted, 
        key=lambda label: result["explanations"][label]["pred_score"], 
        reverse=True
    )

    text = f"🔬 The drug (SMILES: `{smiles}`) is predicted to be toxic in: **{', '.join(sorted_labels)}**."
    return text


def generate_toxicity_radar(smiles, results):
    labels = results["predicted_labels"]
    explanations = results["explanations"]
    probs = []
    for label in label_cols:
        prob = explanations.get(label, {}).get("prob", 0.0)
        probs.append(prob)
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
