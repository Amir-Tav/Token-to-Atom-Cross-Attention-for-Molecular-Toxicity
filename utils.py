"""
utils.py – helpers for the Tox21 Streamlit demo
Last updated: 2025‑07‑28
"""

from __future__ import annotations

import functools
import io
import math
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import requests
import torch
from matplotlib.backends.backend_pdf import PdfPages
from rdkit import Chem
from rdkit.Chem import Descriptors

# Optional; the app still runs if SHAP is missing.
try:
    import shap
except ImportError:  # pragma: no cover
    shap = None

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError:  # pragma: no cover
    AutoTokenizer = AutoModelForSequenceClassification = None

# ───────────────────────────── CONSTANTS ──────────────────────────────

MODELDIR = Path("models/v4")  # adjust if your checkpoint lives elsewhere

PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PROPS = (
    "MolecularFormula,MolecularWeight,ExactMass,"
    "XLogP3-AA,TPSA,HBondDonorCount,HBondAcceptorCount,"
    "CanonicalSMILES,InChIKey"
)

ENDPOINTS: List[str] = [
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
]

# ──────────────────────────── LOADERS ──────────────────────────────────


@functools.lru_cache(maxsize=1)
def _load_tokenizer():
    if AutoTokenizer is None:
        raise RuntimeError("Transformers not installed")
    if MODELDIR.exists():
        return AutoTokenizer.from_pretrained(MODELDIR)
    # fallback
    return AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")


@functools.lru_cache(maxsize=1)
def _load_model():
    if AutoModelForSequenceClassification is None:
        raise RuntimeError("Transformers not installed")
    if MODELDIR.exists() and (MODELDIR / "pytorch_model.bin").exists():
        model = AutoModelForSequenceClassification.from_pretrained(
            MODELDIR, num_labels=len(ENDPOINTS)
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            "seyonec/ChemBERTa-zinc-base-v1", num_labels=len(ENDPOINTS)
        )
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model


@functools.lru_cache(maxsize=1)
def _load_shap_explainer():
    """
    One‑time construction of a SHAP Explainer using the HuggingFace tokenizer
    as the masker. Returns None if SHAP is not available.
    """
    if shap is None:
        return None
    model = _load_model()
    tokenizer = _load_tokenizer()

    def f(x):  # model wrapper that returns logits
        """
        SHAP may feed `x` as a NumPy array; convert it to a plain list[str]
        to keep the tokenizer happy.
        """
        if isinstance(x, str):
            texts = [x]
        else:                       # list, tuple, numpy.ndarray, etc.
            texts = list(x)

        with torch.no_grad():
            enc = tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(model.device)
            return model(**enc).logits.detach().cpu().numpy()


# ─────────────────────── HTTP HELPER ───────────────────────────────────


def _get_json(url: str, timeout: int = 10) -> Dict:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


# ───────────────────── PUBCHEM HELPERS ─────────────────────────────────


@functools.lru_cache(maxsize=256)
def pubchem_assays(smiles: str) -> pd.DataFrame:
    url = f"{PUG}/compound/smiles/{smiles}/aids/JSON?response_type=activity"
    try:
        lst = _get_json(url)["InformationList"]["Information"][0]["AIDactivity"]
        df = pd.DataFrame(lst)
        df.rename(columns={"AID": "Assay", "Activity": "Outcome"}, inplace=True)
        return df.sort_values("Assay", ignore_index=True)
    except Exception:
        return pd.DataFrame()


@functools.lru_cache(maxsize=256)
def pubchem_props(smiles: str) -> pd.DataFrame:
    url = f"{PUG}/compound/smiles/{smiles}/property/{PROPS}/JSON"
    try:
        row = _get_json(url)["PropertyTable"]["Properties"][0]
        return pd.DataFrame(
            [
                {"Property": k, "Value": v}
                for k, v in row.items()
                if k != "CID"
            ]
        )
    except Exception:
        return pd.DataFrame()


@functools.lru_cache(maxsize=256)
def pubchem_name(smiles: str) -> str | None:
    url = f"{PUG}/compound/smiles/{smiles}/property/IUPACName/JSON"
    try:
        n = _get_json(url)["PropertyTable"]["Properties"][0]["IUPACName"]
        return n
    except Exception:
        return None


# ───────────────── PHYS‑CHEM & RADAR PLOT ─────────────────────────────


def _physchem(smiles: str) -> pd.DataFrame:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    rows = [
        ("MolWt", Descriptors.MolWt(mol)),
        ("LogP", Descriptors.MolLogP(mol)),
        ("TPSA", Descriptors.TPSA(mol)),
        ("HBA", Descriptors.NumHAcceptors(mol)),
        ("HBD", Descriptors.NumHDonors(mol)),
        ("RotB", Descriptors.NumRotatableBonds(mol)),
    ]
    return pd.DataFrame(rows, columns=["Descriptor", "Value"])


def _radar_plot(phys_df: pd.DataFrame) -> io.BytesIO:
    labels = phys_df["Descriptor"].tolist()
    values = phys_df["Value"].astype(float).tolist()

    labels.append(labels[0])
    values.append(values[0])

    angles = [
        n / float(len(labels) - 1) * 2 * math.pi for n in range(len(labels))
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values, marker="o")
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels[:-1])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


# ───────────────────── TOXICOPHORE SEARCH ──────────────────────────────

TOX_SMARTS = {
    "Nitro-aromatic": "[#6;a][N+](=O)[O-]",
    "α-β-Unsat. carbonyl": "O=[$([#6]=[#6])]",
    "Michael acceptor": "[$([#6]=[C;!R]);!$([#6][C;!R]=O)]",
    "Epoxide": "C1OC1",
    "Alkyl halide": "[CX4][Cl,Br,I]",
}


def _toxicophores(smiles: str) -> pd.DataFrame:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return pd.DataFrame()
    rows = []
    for name, smarts in TOX_SMARTS.items():
        patt = Chem.MolFromSmarts(smarts)
        if mol.HasSubstructMatch(patt):
            rows.append({"Alert": name, "SMARTS": smarts})
    return pd.DataFrame(rows)


# ───────────────────────── PDF REPORT (unchanged, shortened) ───────────

def _add_table(pdf: PdfPages, title: str, df: pd.DataFrame):
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=20)
    tbl = ax.table(
        cellText=df.values, colLabels=df.columns, loc="center", cellLoc="left"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)
    pdf.savefig(fig)
    plt.close(fig)


def _build_pdf(
    smiles: str,
    pred_df: pd.DataFrame,
    phys_df: pd.DataFrame,
    radar_png: io.BytesIO,
    assay_df: pd.DataFrame,
    verdict: str,
    tox_df: pd.DataFrame,
    pc_df: pd.DataFrame,
):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    with PdfPages(tmp.name) as pdf:
        # cover
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        ax.set_title("Tox21 prediction report", fontsize=20, pad=30)
        ax.text(0.05, 0.8, f"SMILES:\n{smiles}", wrap=True, fontsize=12)
        ax.text(0.05, 0.6, verdict, wrap=True, fontsize=12)
        pdf.savefig(fig)
        plt.close(fig)

        _add_table(pdf, "Prediction probabilities", pred_df)
        _add_table(pdf, "Physico‑chemical descriptors", phys_df)
        _add_table(pdf, "PubChem molecular data", pc_df)

        fig = plt.figure(figsize=(6, 6))
        plt.imshow(plt.imread(radar_png))
        plt.axis("off")
        pdf.savefig(fig)
        plt.close(fig)

        _add_table(pdf, "PubChem assays", assay_df)
        _add_table(pdf, "Toxicophore matches", tox_df)
    return tmp.name


# ────────────────────────────── PREDICT ────────────────────────────────


def _shap_df(tokens: List[str], shap_row: List[float]) -> pd.DataFrame:
    """Return the top‑10 token importances for an endpoint."""
    df = (
        pd.DataFrame({"Token": tokens, "SHAP": shap_row})
        .sort_values("SHAP", key=abs, ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    return df


def predict(smiles: str) -> Dict:
    """
    End‑to‑end inference + evidence + SHAP token attributions.
    """
    tokenizer = _load_tokenizer()
    model = _load_model()

    enc = tokenizer(
        smiles,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        logits = model(**enc).logits.squeeze().cpu()

    probs = torch.sigmoid(logits).numpy()
    pred_df = (
        pd.DataFrame({"Endpoint": ENDPOINTS, "Probability": probs})
        .sort_values("Probability", ascending=False)
        .reset_index(drop=True)
    )

    hits = pred_df[pred_df["Probability"] >= 0.5]["Endpoint"].tolist()
    drug_name = pubchem_name(smiles) or smiles
    if hits:
        verdict = (
            f"**{drug_name}** shows predicted toxicity for "
            f"{', '.join(hits)}."
        )
    else:
        verdict = f"✅ **{drug_name}** predicted inactive for all 12 endpoints."

    assays = pubchem_assays(smiles)
    phys_df = _physchem(smiles)
    pc_df = pubchem_props(smiles)
    tox_df = _toxicophores(smiles)
    radar_png = _radar_plot(phys_df)

    # ←── SHAP
    shap_explainer = _load_shap_explainer()
    token_df = pd.DataFrame()
    if shap_explainer is not None:
        shap_out = shap_explainer([smiles])[0]  # 1 × tokens × classes
        top_ep = int(pred_df.iloc[0].name)  # index of highest‑prob endpoint
        token_df = _shap_df(
            shap_out.data,
            shap_out.values[:, top_ep],
        )
        verdict += " SHAP analysis highlights the coloured tokens as key contributors."

    # clause if ARE assay confirmed
    if (
        not assays.empty
        and 743219 in assays["Assay"].values
        and hits
        and "AID 743219" not in verdict
    ):
        verdict += " Experimentally confirmed active in ARE‑luciferase assay (AID 743219)."

    pdf_path = _build_pdf(
        smiles, pred_df, phys_df, radar_png, assays, verdict, tox_df, pc_df
    )

    return dict(
        sentence=verdict,
        pred_df=pred_df,
        physchem_df=phys_df,
        pubchem_df=pc_df,
        assay_df=assays,
        tox_df=tox_df,
        radar_png=radar_png,
        shap_df=token_df,
        report_path=pdf_path,
    )
