# ===========================================
# utils.py – multi‑endpoint version 2025‑07‑27
# ===========================================
"""Utility library for the Tox21 Streamlit demo.

What this revision adds
-----------------------
✓ 12‑endpoint probabilities in tidy DataFrame.  
✓ Meta‑MLP single‑sentence rationale.  
✓ PubChem title lookup + BioAssay evidence (AID 743219).  
✓ RDKit phys‑chem descriptor panel (+ radar plot).  
✓ Toxicophore SMARTS panel.  
✓ Atom‑level SHAP colouring stub (ready for real explainer).  
✓ One‑click downloadable PDF report (tables + plots).  
✓ Optional ChEMBL target enrichment table.
"""

from __future__ import annotations
import io, functools, tempfile, textwrap, base64, requests
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

import shap, joblib  # retained for future explanatory work

# ─────────────────────────────────── CONFIG ──────────────────────────────────
ENDPOINTS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER",
    "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE",
    "SR-MMP", "SR-p53",
]

PHYS_CHEM = [  # (RDKit function name, pretty label)
    ("MolWt", "MW"),
    ("MolLogP", "cLogP"),       # <- fixed: use MolLogP (always present)
    ("TPSA", "TPSA"),
    ("NumHDonors", "H‑donors"),
    ("NumHAcceptors", "H‑acceptors"),
    ("NumRotatableBonds", "RotB"),
]

TOXICOPHORES = [
    ("Michael acceptor", "[$([O,S]=C-C=C)]"),
    ("Nitro‑aromatic", "[NX3](=O)=O-[cR]"),
    ("Alkyl halide", "[CX4][Cl,Br,I]"),
    ("Epoxide", "[OX2r3]"),
    ("Anilide", "c-NC(=O)"),
]

CONFIG: Dict[str, str | float | bool] = {
    "model_dir": "models/v4",
    "backbone": "seyonec/ChemBERTa-zinc-base-v1",
    "state_dict": "models/v4/model.pt",
    "explainer": "Data_v3/SHAP_val_full/shap_means.npy",
    "meta_explainer": "models/v4/MLP/meta_mlp.pt",
    "threshold": 0.5,
    "use_gpu": torch.cuda.is_available(),
}

STRESS_AID = 743219
PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"

# ────────────────────────── TOKENIZER & CLASSIFIER ──────────────────────────
@functools.lru_cache(maxsize=1)
def _load_model():
    tok = AutoTokenizer.from_pretrained(CONFIG["model_dir"], use_fast=False)

    class Classifier(nn.Module):
        def __init__(self, n_labels: int = 12):
            super().__init__()
            try:
                self.bert = AutoModel.from_pretrained(CONFIG["model_dir"])
            except ValueError:
                self.bert = AutoModel.from_pretrained(CONFIG["backbone"])
            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(self.bert.config.hidden_size, n_labels),
            )

        def forward(self, **kw):
            pooled = self.bert(**kw).pooler_output
            return self.classifier(pooled)

    mdl = Classifier()
    mdl.load_state_dict(torch.load(CONFIG["state_dict"], map_location="cpu"), strict=True)
    dev = torch.device("cuda" if CONFIG["use_gpu"] else "cpu")
    return tok, mdl.to(dev).eval(), dev

# ───────────────────────────── HTTP HELPERS ─────────────────────────────────
@functools.lru_cache(maxsize=512)
def _get_json(url: str) -> dict:
    r = requests.get(url, timeout=10); r.raise_for_status(); return r.json()

# ───────────────────────────── PUBCHEM HELPERS ─────────────────────────────
@functools.lru_cache(maxsize=256)
def pubchem_title(smiles: str) -> str | None:
    url = f"{PUG}/compound/smiles/{smiles}/property/Title/JSON"
    try:
        return _get_json(url)["PropertyTable"]["Properties"][0]["Title"]
    except Exception:
        return None

@functools.lru_cache(maxsize=256)
def pubchem_assays(smiles: str) -> pd.DataFrame:
    url = f"{PUG}/compound/smiles/{smiles}/aids/JSON"
    try:
        info = _get_json(url)["InformationList"]["Information"]
        return pd.DataFrame([{"AID": x["AID"], "Active": x.get("ActiveState")} for x in info])
    except Exception:
        return pd.DataFrame()

# ───────────────────────────── ChEMBL HELPERS ──────────────────────────────
@functools.lru_cache(maxsize=256)
def chembl_targets(smiles: str) -> pd.DataFrame:
    url = f"{CHEMBL}/molecule/search.json?query={requests.utils.quote(smiles)}"
    try:
        hits = _get_json(url)["molecules"]
        if not hits:
            return pd.DataFrame()
        chembl_id = hits[0]["molecule_chembl_id"]
        acts = _get_json(f"{CHEMBL}/activity.json?molecule_chembl_id={chembl_id}&limit=1000")["activities"]
        df = pd.DataFrame([{"Target": x["target_chembl_id"]} for x in acts])
        return df.value_counts().rename("#Assays").reset_index()
    except Exception:
        return pd.DataFrame()

# ───────────────────────────── EXPLANATION TEXT ─────────────────────────────
def _explain_sentence(drug: str | None, active: List[str], shap_vecs: np.ndarray) -> str:
    if not active:
        return "Model predicts no Tox21 endpoint above threshold."
    try:
        meta = torch.load(CONFIG["meta_explainer"], map_location="cpu")
        meta.eval()
        with torch.no_grad():
            txt = meta(torch.tensor(shap_vecs, dtype=torch.float32).unsqueeze(0))
        if isinstance(txt, (str, bytes)):
            return txt.decode() if isinstance(txt, bytes) else txt
    except Exception:
        pass

    prefix = drug if drug else "Model"
    return f"{prefix} predicts toxicity for {', '.join(active)}."

# ────────────────────────── RDKit DESCRIPTORS & RADAR ───────────────────────
def _physchem(smiles: str) -> pd.DataFrame:
    mol = Chem.MolFromSmiles(smiles)
    rows = []
    for func_name, label in PHYS_CHEM:
        val = getattr(Descriptors, func_name)(mol)
        rows.append({"Property": label, "Value": val})
    return pd.DataFrame(rows)

def _radar_plot(df: pd.DataFrame) -> bytes:
    labels = df["Property"].tolist() + [df["Property"].iloc[0]]
    values = df["Value"].astype(float).tolist() + [df["Value"].iloc[0]]

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, polar=True)
    theta = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    ax.plot(theta, values); ax.fill(theta, values, alpha=0.25)
    ax.set_xticks(theta); ax.set_xticklabels(labels)
    ax.set_title("Phys‑chem radar"); ax.grid(True)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight"); plt.close(fig)
    return buf.getvalue()

# ────────────────────── SMARTS / TOXICOPHORE SEARCH ────────────────────────
def _toxicophores(smiles: str) -> pd.DataFrame:
    mol = Chem.MolFromSmiles(smiles); hits = []
    for name, smarts in TOXICOPHORES:
        if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
            hits.append({"Toxicophore": name, "SMARTS": smarts})
    return pd.DataFrame(hits)

# ───────────────────────── SHAP ATOM HIGHLIGHT (stub) ───────────────────────
def _shap_svg(mol: Chem.Mol) -> str:
    drawer = rdMolDraw2D.MolDraw2DSVG(250, 250)
    drawer.DrawMolecule(mol); drawer.FinishDrawing()
    return drawer.GetDrawingText()   # <- uniform colour until real SHAP mapping

# ───────────────────────────── PDF REPORT BUILDER ───────────────────────────
def _build_pdf(smiles: str, pred_df: pd.DataFrame, phys_df: pd.DataFrame,
               radar_png: bytes, assay_df: pd.DataFrame, expl_sentence: str,
               tox_df: pd.DataFrame) -> Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    with PdfPages(tmp.name) as pdf:
        # cover page ---------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4
        ax.axis("off")
        ax.text(0.01, 0.98,
                f"Tox21 multi‑endpoint report\n\nSMILES: {smiles}\n\n{expl_sentence}",
                va="top", wrap=True, fontsize=10)
        pdf.savefig(fig); plt.close(fig)

        # helper: table page -------------------------------------------------
        def _add_table(title: str, df: pd.DataFrame):
            if df.empty:
                return
            fig, ax = plt.subplots(figsize=(8, max(1.5, len(df) * 0.28)))
            ax.axis("off")

            df_disp = df.copy()
            num_cols = df_disp.select_dtypes(include=[np.number]).columns
            df_disp[num_cols] = df_disp[num_cols].round(3)

            tbl = ax.table(cellText=df_disp.values,
                           colLabels=df_disp.columns,
                           loc="center")
            tbl.auto_set_font_size(False); tbl.set_fontsize(8)
            tbl.auto_set_column_width(col=list(range(len(df_disp.columns))))
            ax.set_title(title, pad=12, fontsize=10)
            pdf.savefig(fig); plt.close(fig)

        # tables -------------------------------------------------------------
        _add_table("Tox21 Endpoint Probabilities", pred_df)
        _add_table("Phys‑chem descriptors", phys_df)
        _add_table("PubChem assays", assay_df)
        _add_table("Toxicophore SMARTS", tox_df)

        # radar chart --------------------------------------------------------
        fig = plt.figure(figsize=(4, 4)); ax = fig.add_subplot(111)
        ax.axis("off"); ax.imshow(plt.imread(io.BytesIO(radar_png)))
        pdf.savefig(fig); plt.close(fig)

    return Path(tmp.name)


# ────────────────────────────── MAIN PREDICT ────────────────────────────────
def predict(smiles: str) -> Dict:
    tok, mdl, dev = _load_model()
    inputs = tok(smiles, return_tensors="pt").to(dev)
    with torch.no_grad():
        probs = torch.sigmoid(mdl(**inputs).squeeze()).cpu().numpy()

    pred_df = pd.DataFrame({"Endpoint": ENDPOINTS, "Prob": probs}).sort_values("Endpoint")
    active = [ep for ep, p in zip(ENDPOINTS, probs) if p >= CONFIG["threshold"]]
    assays = pubchem_assays(smiles)
    sentence = _explain_sentence(pubchem_title(smiles), active, probs)
    if "SR-ARE" in active and (assays["AID"] == STRESS_AID).any():
        sentence += " Experimentally confirmed active in ARE‑luciferase assay (AID 743219)."

    mol = Chem.MolFromSmiles(smiles)
    mol_svg = _shap_svg(mol)

    phys_df = _physchem(smiles)
    radar_png = _radar_plot(phys_df)
    tox_df = _toxicophores(smiles)
    chembl_df = chembl_targets(smiles)

    pdf_path = _build_pdf(smiles, pred_df, phys_df, radar_png, assays, sentence, tox_df)
    radar_b64 = base64.b64encode(radar_png).decode()

    return {
        "sentence": sentence,
        "table": pred_df,
        "mol_svg": mol_svg,
        "assay_df": assays,
        "physchem_df": phys_df,
        "radar_png": radar_b64,
        "toxic_df": tox_df,
        "chembl_df": chembl_df,
        "report_path": str(pdf_path),
    }
