# --- Standard Library ---
import os
import io
import json
import joblib

# --- Scientific Computing ---
import numpy as np
import pandas as pd

# --- Visualization ---
import plotly.graph_objects as go
from PIL import Image

# --- RDKit (Cheminformatics) ---
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # Silence RDKit warnings

# --- Mordred (Descriptors) ---
from mordred import Calculator, descriptors

# --- SHAP (Explainability) ---
import shap

# --- External Lookup (Optional) ---
import pubchempy as pcp  # Only needed if you plan to fetch extra info from PubChem


MODEL_PATH = "tox21_lightgb_pipeline/models/v7"  # ⬅️ New model path
SAVE_DIR = "tox21_lightgb_pipeline/Data_v6/processed"

label_cols = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER",
    "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5",
    "SR-HSE", "SR-MMP", "SR-p53"
]

# Define SMARTS patterns for common toxicophores
TOXICOPHORE_SMARTS = {
    "Aromatic amine": "[NX3][cR]",                     # e.g., Aniline
    "Nitro group": "[NX3](=O)=O",                      # e.g., Nitrobenzene
    "Halogen": "[F,Cl,Br,I]",                          # Halogen atoms
    "Thiophene ring": "c1ccsc1",                       # 5-membered sulfur heterocycles
    "Alkyl halide": "[CX4][F,Cl,Br,I]",                # R-Cl, R-Br, etc.
    "Epoxide": "[C;r3]1[O;r3][C;r3]1",                 # 3-membered ring with O
    "Michael acceptor": "C=CC=O",                      # α,β-unsaturated carbonyl
    "Imine": "C=N",                                    # C=N double bond
    "Quinone": "O=C1C=CC(=O)C=C1",                     # e.g., Benzoquinone
    "Hydrazine": "NN",                                 # R-NH-NH-R
}

# Load feature names and model-specific masks
with open(os.path.join(SAVE_DIR, "feature_names.txt")) as f:
    feature_names = f.read().splitlines()

# Load meta explanations for generate_textual_explanation()
with open("tox21_lightgb_pipeline/Data_v6/meta_explainer/meta_explanations.json") as f:
    META_EXPLAIN_DICT = json.load(f)

# Load SMARTS toxicophore mapping rules
with open("tox21_lightgb_pipeline/Data_v6/meta_explainer/smarts_rules.json") as f:
    SMARTS_RULES = json.load(f)




thresholds = joblib.load(os.path.join(MODEL_PATH, "thresholds.pkl"))
feature_masks = joblib.load(os.path.join(MODEL_PATH, "feature_masks.pkl"))

# Mordred calculator
calc = Calculator(descriptors, ignore_3D=True)


##############################################################----SHAPES----#####

###--------------------------------generate_toxicity_radar()
def generate_toxicity_radar(smiles, results):
    """
    Generate a radar chart showing toxicity probabilities across all 12 labels.
    """
    labels = results["predicted_labels"]
    explanations = results["explanations"]

    probs = []
    active = []

    for label in label_cols:
        if label in explanations:
            prob = explanations[label]["prob"]
        else:
            prob = 0.0
        probs.append(prob)
        active.append(label in labels)

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=probs + [probs[0]],  # Close the loop
        theta=label_cols + [label_cols[0]],
        fill='toself',
        name='Predicted Toxicity',
        line=dict(color='crimson'),
        marker=dict(symbol='circle'),
        hovertemplate='%{theta}<br>Prob: %{r:.2f}<extra></extra>',
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickfont_size=10),
        ),
        showlegend=False,
        title=f"Toxicity Radar for: {smiles}",
        margin=dict(l=30, r=30, t=50, b=30),
        height=500
    )

    return fig

###-------------------------------highlight_toxicophores()
def highlight_toxicophores(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    highlighted_mols = []
    legend_list = []

    for tox_label, smarts_list in TOXICOPHORE_SMARTS.items():
        match_found = False
        for smarts in smarts_list:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                match_found = True
                break

        if match_found:
            match_mol = Chem.Mol(mol)
            AllChem.Compute2DCoords(match_mol)

            # Highlight all matching atoms from all SMARTS under this label
            highlight_atoms = set()
            for smarts in smarts_list:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern:
                    matches = match_mol.GetSubstructMatches(pattern)
                    for match in matches:
                        highlight_atoms.update(match)

            for idx in highlight_atoms:
                match_mol.GetAtomWithIdx(idx).SetProp('atomNote', '*')

            highlighted_mols.append(match_mol)
            legend_list.append(f"{tox_label} toxicophore")

    if not highlighted_mols:
        return None

    img = Draw.MolsToGridImage(
        [mol] + highlighted_mols,
        legends=["Original"] + legend_list,
        useSVG=False
    )
    return img




def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        desc = calc(mol).asdict()
        return [desc.get(f, 0.0) for f in feature_names]
    except:
        return None

def predict_and_explain_all_labels(smiles):
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
            }).sort_values(by="shap_value", key=np.abs, ascending=False)

            results[label] = {
            "prob": prob,
            "threshold": threshold,
            "pred_score": prob,  # ← Added for sorting in summarize_prediction()
            "shap_df": shap_df,
            "top_features": shap_df[["feature", "shap_value"]].head(4).values.tolist()
        }
    return {
        "smiles": smiles,
        "predicted_labels": predicted_labels,
        "explanations": results
    }
###-------------------------------SMART Meta explainer
def match_toxicophores_with_explanations(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    matches = []
    for name, rule in SMARTS_RULES.items():
        patt = Chem.MolFromSmarts(rule["smarts"])
        if patt and mol.HasSubstructMatch(patt):
            matches.append({
                "name": name,
                "explanation": rule["explanation"]
            })
    return matches


def highlight_toxicophores(smiles):
    """
    Returns a list of matched toxicophores and an RDKit image with highlights.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES structure.")

    # Generate 2D coordinates for better visualization
    AllChem.Compute2DCoords(mol)

    highlight_atoms = set()
    matched_toxophores = []

    for name, rule in TOXICOPHORE_SMARTS.items():
        smarts = rule["smarts"]
        explanation = rule["explanation"]

        pattern = Chem.MolFromSmarts(smarts)
        if not pattern:
            continue

        matches = mol.GetSubstructMatches(pattern)
        if matches:
            matched_toxophores.append(f"☣️ **{name}**: {explanation}")
            for match in matches:
                highlight_atoms.update(match)

    # Draw molecule with highlights
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

# ###--------------------------------generate_textual_explanation()
def generate_textual_explanation(label, shap_df, max_features=4):
    """
    Generate explanation text for a predicted toxicity label.
    Combines:
    - SMARTS toxicophore matches (if available in SHAP df)
    - SHAP feature importance and domain-aware meta explanations
    """
    lines = []

    # === 1. SMARTS-based toxicophore explanation (if exists) ===
    smarts_row = shap_df[shap_df["feature"].str.startswith("TOXICOPHORE_")]
    if not smarts_row.empty:
        lines.append("Matched Toxicophores:")
        for _, row in smarts_row.iterrows():
            smarts_key = row["feature"].replace("TOXICOPHORE_", "")
            rule = SMARTS_RULES.get(smarts_key, "unclassified toxicophore")
            lines.append(f"☣️ {smarts_key}: {rule}")
        lines.append("")  # Add spacing

    # === 2. SHAP-ranked features and role-based explanations ===
    # Filter non-toxicophores, sort, and get top features
    top_feats = shap_df[~shap_df["feature"].str.startswith("TOXICOPHORE_")].head(max_features)
    
    for _, row in top_feats.iterrows():
        fname = row["feature"]
        shap_val = row["shap_value"]
        direction = "↑ increase" if shap_val > 0 else "↓ decrease"
        explanation = META_EXPLAIN_DICT.get(
            fname, "miscellaneous descriptor — no mapped biological role found"
        )
        lines.append(
            f"- **{fname}** ({explanation}): contributes to toxicity via {direction} effect (SHAP = {shap_val:.3f})"
        )

    # Final output formatting
    if lines:
        return f"💡 **{label} Explanation**:\n\n" + "\n".join(lines)
    else:
        return f"ℹ️ For **{label}**, no dominant features or toxicophores were found."


###-------------------------------summarize_prediction()
def summarize_prediction(result):
    """
    Generate a full summary text for the predicted toxicity profile,
    showing toxic classes sorted by model confidence, along with
    textual + SMARTS-based SHAP explanations.
    """
    smiles = result["smiles"]
    predicted = result["predicted_labels"]
    if not predicted:
        return f"🔬 The drug (SMILES: `{smiles}`) is predicted **not to exhibit significant toxicity endpoints.**"

    # Sort predicted labels by model prediction score
    sorted_labels = sorted(
        predicted, 
        key=lambda label: result["explanations"][label]["pred_score"], 
        reverse=True
    )

    # Header
    text = f"🔬 The drug (SMILES: `{smiles}`) is predicted to be toxic in: **{', '.join(sorted_labels)}**.\n\n"

    # For each predicted toxicity class, show SHAP + SMARTS explanation
    for label in sorted_labels:
        shap_df = result["explanations"][label]["shap_df"]
        explanation = generate_textual_explanation(label, shap_df)
        text += explanation + "\n\n"

    return text.strip()


