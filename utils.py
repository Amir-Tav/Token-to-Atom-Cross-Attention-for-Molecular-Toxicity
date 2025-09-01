# utils.py — ChemBERTa v2 + final thresholds + TCAV v3 (presence-gated) + app helpers
# Strict paths:
#   • thresholds: implementation/v3/eval/thresholds_selected_v3.json
#   • TCAV:       implementation/v3/stats/tcav_summary_v3.(csv|json)

import os, io, json, math, hashlib, logging, random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

# =========================
# Basic setup
# =========================
ROOT = Path("implementation")
V3_DIR = ROOT / "v3"
V3_META  = V3_DIR / "metadata"
V3_STATS = V3_DIR / "stats"
V3_EVAL  = V3_DIR / "eval"

MODEL_DIR = ROOT / "models" / "chemberta_v2" / "v2_best"
META_DIR  = ROOT / "models" / "chemberta_v2" / "metadata"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 128  # token length for SMILES

# App/user settings (used by sidebar)
SETTINGS: Dict = {
    "max_len": MAX_LEN,
    "use_standardization": True,
    "enable_pubchem": False,
    "tcav": {"min_tcav": 0.60, "max_p": 0.05, "top_k": 6},
}

random.seed(42); np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tox21.utils")

def _file_sig(p: Path) -> str:
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:12]
    except Exception:
        return "missing"

# =========================
# Model load (ChemBERTa v2)
# =========================
_tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
_cfg = AutoConfig.from_pretrained(str(MODEL_DIR))
_mdl = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR), config=_cfg).to(DEVICE).eval()
if hasattr(_mdl.config, "use_cache"):  # avoid warning spam
    _mdl.config.use_cache = False

id2label = _mdl.config.id2label
label2id = _mdl.config.label2id
label_cols = [id2label[i] for i in range(len(id2label))]

# =========================
# Strict thresholds (NO fallbacks)
# =========================
THRESH_PATH = Path("implementation/v3/eval/thresholds_selected_v3.json")
if not THRESH_PATH.exists():
    raise FileNotFoundError(
        f"Missing finalized thresholds file: {THRESH_PATH}"
    )
thresholds: Dict[str, float] = json.load(open(THRESH_PATH, "r"))

# =========================
# Calibration (optional, used if files exist)
# =========================
CAL_METHODS_PATH = META_DIR / "calibration_methods.json"
PLATT_PATH       = META_DIR / "platt_params.json"
ISO_PATH         = META_DIR / "isotonic_params.json"
TEMP_PATH        = META_DIR / "temperature.npy"

CAL_METHOD = json.load(open(CAL_METHODS_PATH)) if CAL_METHODS_PATH.exists() else {}
PLATT      = json.load(open(PLATT_PATH)) if PLATT_PATH.exists() else {}
ISO        = json.load(open(ISO_PATH)) if ISO_PATH.exists() else {}
TEMP       = float(np.load(TEMP_PATH)[0]) if TEMP_PATH.exists() else 1.0

def _apply_isotonic_scalar(p: float, X: List[float], Y: List[float]) -> float:
    if not X or not Y:
        return float(p)
    return float(np.interp(float(p), np.asarray(X, float), np.asarray(Y, float)))

def _calibrate(lbl: str, logit: float, prob: float) -> float:
    """Apply per-label calibration if available."""
    method = CAL_METHOD.get(lbl, {}).get("method", "none")
    if method == "temp":
        t = max(TEMP, 1e-6)
        return float(1.0 / (1.0 + math.exp(-float(logit) / t)))
    if method == "platt":
        pars = PLATT.get(lbl)
        if not pars: return float(prob)
        A, B = float(pars.get("A", 1.0)), float(pars.get("B", 0.0))
        return float(1.0 / (1.0 + math.exp(-(A * float(logit) + B))))
    if method == "iso":
        pars = ISO.get(lbl)
        if not pars: return float(prob)
        return _apply_isotonic_scalar(float(prob), pars.get("X", []), pars.get("Y", []))
    return float(prob)

# =========================
# TCAV (strict v3 file; NO fallbacks)
# =========================
def _pick_tcav_summary_file() -> Path:
    p_csv  = V3_STATS / "tcav_summary_v3.csv"
    p_json = V3_STATS / "tcav_summary_v3.json"
    if p_csv.exists():  return p_csv
    if p_json.exists(): return p_json
    raise FileNotFoundError(
        "TCAV summary not found. Expected one of:\n"
        "  • implementation/v3/stats/tcav_summary_v3.csv\n"
        "  • implementation/v3/stats/tcav_summary_v3.json"
    )

TCAV_SUMMARY_FILE = _pick_tcav_summary_file()
if TCAV_SUMMARY_FILE.suffix.lower() == ".json":
    _tcav_df = pd.DataFrame(json.load(open(TCAV_SUMMARY_FILE, "r")))
else:
    _tcav_df = pd.read_csv(TCAV_SUMMARY_FILE)

# normalize p / q columns
if "p_value" not in _tcav_df.columns:
    if "p_emp_null" in _tcav_df.columns:
        _tcav_df["p_value"] = _tcav_df["p_emp_null"]
    elif "p_value_ttest" in _tcav_df.columns:
        _tcav_df["p_value"] = _tcav_df["p_value_ttest"]
    elif "p_value_binom" in _tcav_df.columns:
        _tcav_df["p_value"] = _tcav_df["p_value_binom"]
    else:
        _tcav_df["p_value"] = 1.0

def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    if p.size == 0: return p
    m = p.size
    order = np.argsort(p)
    ranks = np.arange(1, m+1)
    q_sorted = (p[order] * m / ranks).clip(0, 1)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q = np.empty_like(p); q[order] = q_sorted
    return q

if "q_value" not in _tcav_df.columns:
    _tcav_df["q_value"] = _bh_fdr(_tcav_df["p_value"].values)

# =========================
# Concepts (SMARTS & knowledge)
# =========================
# Prefer v3 concepts map if present; otherwise build from v2 SMARTS rules.
CONCEPTS_V3 = V3_META / "concepts_v3.json"
SMARTS_V2   = Path("implementation/v2/smarts_lib/smarts_rules_final.json")  # your vetted file
CONCEPT_TO_SMARTS: Dict[str, List[str]] = {}

if CONCEPTS_V3.exists():
    items = json.load(open(CONCEPTS_V3, "r"))
    for it in items:
        nm = str(it.get("name", "")).strip()
        sm = str(it.get("smarts", "")).strip()
        if nm and sm:
            CONCEPT_TO_SMARTS.setdefault(nm, []).append(sm)
elif SMARTS_V2.exists():
    rules = json.load(open(SMARTS_V2, "r"))
    tmp = {}
    for _, lst in rules.items():
        for r in lst:
            nm = str(r.get("name", "")).strip()
            sm = str(r.get("smarts", "")).strip()
            if nm and sm:
                tmp.setdefault(nm, set()).add(sm)
    CONCEPT_TO_SMARTS = {k: sorted(list(v)) for k, v in tmp.items()}
else:
    # If you want to force this to exist too, swap to raising FileNotFoundError.
    CONCEPT_TO_SMARTS = {}

# Optional concept knowledge for nicer text (label->concept->string)
CK_CANDIDATES = [
    Path("implementation/concept_Knowledge/concept_knowledge_auto_v2.json"),
    Path("implementation/v2/smarts_lib/concept_knowledge_auto_v2.json"),
]
CONCEPT_KNOWLEDGE: Dict[str, Dict[str, str]] = {}
for p in CK_CANDIDATES:
    if p.exists():
        try:
            raw = json.load(open(p, "r", encoding="utf-8"))
            if isinstance(raw, dict):
                for lbl, d in raw.items():
                    if isinstance(d, dict):
                        CONCEPT_KNOWLEDGE[lbl] = d
                    elif isinstance(d, list):
                        # Some files are list-of-objects per label
                        CONCEPT_KNOWLEDGE[lbl] = {it.get("name",""): it.get("explanation","") for it in d if "name" in it}
        except Exception:
            pass

# Problematic labels (for warnings) from your balanced cohorts
BALANCED_JSON = V3_STATS / "near_threshold_balanced_v3.json"
PROBLEM_LABELS = set()
if BALANCED_JSON.exists():
    bal = json.load(open(BALANCED_JSON, "r"))
    for row in bal.get("cohorts", []):
        n_pos = int(row.get("n_pos_used", 0))
        n_neg = int(row.get("n_neg_used", 0))
        n_tot = int(row.get("n_total_used", 0))
        if n_tot < 100 or n_pos == 0 or n_neg == 0:
            PROBLEM_LABELS.add(row["label"])

# =========================
# Settings API for app
# =========================
def get_settings() -> dict:
    return {
        "max_len": MAX_LEN,
        "use_standardization": SETTINGS.get("use_standardization", True),
        "enable_pubchem": SETTINGS.get("enable_pubchem", False),
        "tcav": SETTINGS.get("tcav", {"min_tcav":0.6,"max_p":0.05,"top_k":6})
    }

def set_use_standardization(flag: bool):
    SETTINGS["use_standardization"] = bool(flag)

def set_enable_pubchem(flag: bool):
    SETTINGS["enable_pubchem"] = bool(flag)

# =========================
# Prediction helpers
# =========================
def _smiles_pre(smi: str) -> str:
    return smi.strip()

def canonical_smiles(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    try: return Chem.MolToSmiles(mol, canonical=True)
    except Exception: return smiles

def validate_smiles(smiles: str) -> Tuple[bool, str]:
    if not isinstance(smiles, str) or not smiles.strip():
        return False, "Please enter a non-empty SMILES string."
    if Chem.MolFromSmiles(smiles) is None:
        return False, "Invalid SMILES string. Try a simple example like 'CCO' (ethanol)."
    return True, ""

def _prep_smiles_for_model(smiles: str) -> str:
    if SETTINGS.get("use_standardization", True):
        return canonical_smiles(smiles) or smiles
    return smiles

@torch.inference_mode()
def _logits_for(smiles_batch: List[str], max_len: int = MAX_LEN) -> np.ndarray:
    enc = _tok(smiles_batch, truncation=True, padding=True, max_length=max_len, return_tensors="pt").to(DEVICE)
    out = _mdl(**enc).logits  # [B, L]
    return out.detach().cpu().numpy()

def predict_probs(smiles: str, max_len: int = MAX_LEN) -> Dict[str, float]:
    """Return calibrated per-label probabilities for one SMILES."""
    smi = _prep_smiles_for_model(smiles)
    logits = _logits_for([smi], max_len=max_len)[0]  # (L,)
    probs = 1.0 / (1.0 + np.exp(-logits))
    out = {}
    for i, lbl in enumerate(label_cols):
        out[lbl] = float(_calibrate(lbl, logits[i], probs[i]))
    return out

def predict_probs_batch(smiles_list: List[str], max_len: int = MAX_LEN) -> List[Dict[str, float]]:
    """Return list of dicts (one per SMILES)."""
    smiles_proc = [_prep_smiles_for_model(s) for s in smiles_list]
    logits = _logits_for(smiles_proc, max_len=max_len)        # (B, L)
    probs  = 1.0 / (1.0 + np.exp(-logits))                    # (B, L)
    out: List[Dict[str, float]] = []
    for b in range(probs.shape[0]):
        d = {}
        for i, lbl in enumerate(label_cols):
            d[lbl] = float(_calibrate(lbl, logits[b, i], probs[b, i]))
        out.append(d)
    return out

def predict_probs_matrix(smiles_list: List[str], max_len: int = MAX_LEN) -> np.ndarray:
    """Return (N, L) matrix in the order of label_cols."""
    dicts = predict_probs_batch(smiles_list, max_len=max_len)
    return np.stack([[d[lbl] for lbl in label_cols] for d in dicts], axis=0)

def label_hits_from_probs(probs: Dict[str, float], thr: Dict[str, float]) -> List[str]:
    return [lbl for lbl, p in probs.items() if p >= thr.get(lbl, 0.5)]

# Optional: external name resolver
try:
    import pubchempy as pcp
except Exception:
    pcp = None

def resolve_compound_name(smiles: str) -> str:
    if not SETTINGS.get("enable_pubchem", False) or pcp is None:
        return "Unknown compound"
    try:
        comps = pcp.get_compounds(smiles, namespace="smiles")
        if not comps: return "Unknown compound"
        c = comps[0]
        for key in ("iupac_name", "title"):
            val = getattr(c, key, None)
            if val: return val
        if getattr(c, "synonyms", None):
            return c.synonyms[0]
        return "Unknown compound"
    except Exception:
        return "Unknown compound"

# =========================
# Presence-gated TCAV utilities
# =========================
def _concept_present_in_smiles(smiles: str, concept: str) -> Tuple[bool, int]:
    """Returns (present?, count) for a concept in this SMILES using SMARTS list."""
    patt_list = CONCEPT_TO_SMARTS.get(concept, [])
    if not patt_list:
        return False, 0
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return False, 0
    total = 0
    for sm in patt_list:
        q = Chem.MolFromSmarts(sm)
        if q:
            total += len(m.GetSubstructMatches(q))
    return (total > 0), total

def _apply_tcav_contrib(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "q_value" not in d.columns:
        d["q_value"] = 1.0
    d["q_value"] = d["q_value"].clip(0, 1)
    d["contrib_score"] = (d["tcav_mean"] - 0.5) * (1.0 - d["q_value"])
    d["direction"] = np.where(d["contrib_score"] >= 0, "↑", "↓")
    return d

def _concept_importance_for_label(label: str,
                                  min_tcav: float,
                                  max_p: float,
                                  top_k: int,
                                  include_negative: bool = True,
                                  neg_tcav: float = 0.40,
                                  smiles_for_presence: Optional[str] = None,
                                  drop_absent: bool = False,
                                  boost_present: float = 0.5) -> pd.DataFrame:
    """
    If smiles_for_presence is given, add instance-aware flags and compute final_score:
      final_score = contrib_score * (1 + boost_present * present)
    If drop_absent=True, remove concepts not detected in this molecule.
    """
    sdf = _tcav_df[_tcav_df["label"] == label].copy()
    if sdf.empty:
        return pd.DataFrame(columns=[
            "feature","contrib_score","p_value","q_value","tcav_mean",
            "concept","direction","ci95","present","present_count","final_score"
        ])

    pos = sdf[(sdf["tcav_mean"] >= float(min_tcav)) & (sdf["p_value"] <= float(max_p))].copy()
    neg = pd.DataFrame()
    if include_negative:
        neg = sdf[(sdf["tcav_mean"] <= float(neg_tcav)) & (sdf["p_value"] <= float(max_p))].copy()

    keep = pd.concat([pos, neg], ignore_index=True)
    if keep.empty:
        return pd.DataFrame(columns=[
            "feature","contrib_score","p_value","q_value","tcav_mean",
            "concept","direction","ci95","present","present_count","final_score"
        ])

    keep["feature"] = "CONCEPT_" + keep["concept"].astype(str)
    keep = _apply_tcav_contrib(keep)

    pres = []
    pres_cnt = []
    if smiles_for_presence:
        for c in keep["concept"].astype(str).tolist():
            p, n = _concept_present_in_smiles(smiles_for_presence, c)
            pres.append(bool(p)); pres_cnt.append(int(n))
    else:
        pres = [False]*len(keep); pres_cnt = [0]*len(keep)
    keep["present"] = pres
    keep["present_count"] = pres_cnt

    keep["final_score"] = keep["contrib_score"] * (1.0 + (boost_present if smiles_for_presence else 0.0) * keep["present"].astype(float))

    if drop_absent and smiles_for_presence:
        keep = keep[keep["present"]]

    keep = keep.sort_values(["direction","final_score","tcav_mean"], ascending=[True, False, False])

    top_pos = keep[keep["direction"]=="↑"].head(top_k)
    top_neg = keep[keep["direction"]=="↓"].head(2) if include_negative else pd.DataFrame(columns=keep.columns)
    out = pd.concat([top_pos, top_neg], ignore_index=True)

    return out.reset_index(drop=True)[
        ["feature","contrib_score","p_value","q_value","tcav_mean",
         "concept","direction","ci95","present","present_count","final_score"]
    ]

def concept_coverage_summary(label: str,
                             min_tcav: float = SETTINGS["tcav"]["min_tcav"],
                             max_p: float = SETTINGS["tcav"]["max_p"],
                             neg_tcav: float = 0.40) -> Dict[str, int]:
    df = _concept_importance_for_label(label, min_tcav=min_tcav, max_p=max_p, top_k=999, include_negative=True, neg_tcav=neg_tcav, smiles_for_presence=None)
    if df.empty: return {"drivers": 0, "counters": 0}
    return {"drivers": int((df["contrib_score"] > 0).sum()),
            "counters": int((df["contrib_score"] < 0).sum())}

def _kb_phrase(label: str, concept: str) -> str:
    # prefer label-specific text; fallback to generic if present
    if label in CONCEPT_KNOWLEDGE and concept in CONCEPT_KNOWLEDGE[label]:
        return CONCEPT_KNOWLEDGE[label][concept]
    if "_generic" in CONCEPT_KNOWLEDGE and concept in CONCEPT_KNOWLEDGE["_generic"]:
        return CONCEPT_KNOWLEDGE["_generic"][concept]
    return concept

ENDPOINT_TAIL = {
    "SR-MMP":   "mitochondrial depolarization consistent with ΔΨm collapse",
    "SR-ARE":   "electrophilic/oxidative stress consistent with Nrf2/ARE activation",
    "SR-ATAD5": "replication stress linked to DNA repair/PCNA unloading",
    "SR-HSE":   "proteotoxic stress consistent with heat-shock response",
    "SR-p53":   "DNA-damage signalling leading to p53 stabilization",
    "NR-AR":          "androgen receptor engagement at the LBD",
    "NR-AR-LBD":      "high-affinity occupation of the AR ligand-binding domain",
    "NR-AhR":         "planar aromatic binding to AhR and xenobiotic response",
    "NR-Aromatase":   "CYP19 (aromatase) perturbation of estrogen synthesis",
    "NR-ER":          "estrogen receptor modulation and downstream transcription",
    "NR-ER-LBD":      "ligand-binding-domain engagement within ER",
    "NR-PPAR-gamma":  "PPAR-γ activation affecting metabolic gene programs",
}

def _build_one_liner(label: str, contrib_df: pd.DataFrame, smiles: str) -> str:
    if contrib_df is None or contrib_df.empty:
        return f"Predicted {label} toxicity lacks clear concept evidence."
    df = contrib_df.copy()
    df_pos = df[(df["final_score"] > 0)].copy()
    df_pos_present = df_pos[df_pos["present"]].sort_values("final_score", ascending=False)
    if not df_pos_present.empty:
        pick = df_pos_present.iloc[0]
    elif not df_pos.empty:
        pick = df_pos.sort_values("final_score", ascending=False).iloc[0]
    else:
        df_neg_present = df[(df["final_score"] < 0) & (df["present"])].sort_values("final_score")
        pick = (df_neg_present.iloc[0] if not df_neg_present.empty
                else df.sort_values("final_score", ascending=False).iloc[0])

    cname = str(pick["concept"])
    direction = "driven by" if float(pick["final_score"]) > 0 else "countered by"
    tail = ENDPOINT_TAIL.get(label, "mechanistic activity")
    phrase = _kb_phrase(label, cname)
    pres_note = " (detected in molecule)" if bool(pick.get("present", False)) else ""
    return f"{direction} **{cname}**{pres_note} — {phrase}; overall consistent with {tail}."

def _format_concept_bullet(row, label: str) -> str:
    fname = row["feature"]; cname = fname.replace("CONCEPT_", "")
    sgn = "↑" if row["contrib_score"] >= 0 else "↓"
    tcav_mean = float(row.get("tcav_mean", 0.5))
    # informal strength category
    d = abs(tcav_mean - 0.5)
    if d >= 0.40:  strength = "very strong"
    elif d >= 0.25: strength = "strong"
    elif d >= 0.15: strength = "moderate"
    else: strength = "weak"
    pval = float(row.get("p_value", 1.0))
    qval = float(row.get("q_value", np.nan)) if "q_value" in row else None
    ptxt = "<1e-12" if pval < 1e-12 else (f"{pval:.1e}" if pval < 1e-6 else f"{pval:.3f}")
    qtxt = ("<1e-6" if (qval is not None and qval < 1e-6) else (f"{qval:.3f}" if qval is not None and np.isfinite(qval) else "nan"))
    desc = _kb_phrase(label, cname)
    ci = row.get("ci95", None)
    ci_txt = ""
    try:
        if isinstance(ci, str):
            ci_parsed = json.loads(ci.replace("'", '"'))
            if isinstance(ci_parsed, list) and len(ci_parsed) == 2:
                ci_txt = f" [{ci_parsed[0]:.2f}–{ci_parsed[1]:.2f}]"
    except Exception:
        pass
    pres = " • **present**" if bool(row.get("present", False)) else ""
    cnt  = int(row.get("present_count", 0))
    pres += f"×{cnt}" if cnt > 1 else ""
    pq = f"p={ptxt}"
    if qval is not None and np.isfinite(qval):
        pq += f", q={qtxt}"
    return f"- **{cname}** ({sgn}, {strength}, TCAV={tcav_mean:.2f}{ci_txt}, {pq}{pres}) — {desc}"

# =========================
# SMARTS helpers (optional rules)
# =========================
# Prefer your vetted v2 SMARTS library
SMARTS_RULES = {}
if SMARTS_V2.exists():
    try:
        SMARTS_RULES = json.load(open(SMARTS_V2, "r"))
    except Exception:
        SMARTS_RULES = {}

def match_toxicophores_with_explanations(smiles: str, label: Optional[str] = None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not SMARTS_RULES: return []
    label_rules = SMARTS_RULES.get(label, []) if label else [r for rules in SMARTS_RULES.values() for r in rules]
    hits = []
    for rule in label_rules:
        patt = Chem.MolFromSmarts(rule["smarts"])
        if patt and mol.HasSubstructMatch(patt):
            hits.append({"name": rule["name"], "explanation": rule.get("explanation","")})
    return hits

def highlight_toxicophores(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: raise ValueError("Invalid SMILES structure.")
    AllChem.Compute2DCoords(mol)
    highlight_atoms = set(); matched = []
    all_rules = [r for rules in SMARTS_RULES.values() for r in rules] if SMARTS_RULES else []
    for rule in all_rules:
        patt = Chem.MolFromSmarts(rule["smarts"])
        if not patt: continue
        matches = mol.GetSubstructMatches(patt)
        if matches:
            matched.append(f"☣️ **{rule['name']}**: {rule.get('explanation','')}")
            for m in matches: highlight_atoms.update(m)
    drawer = rdMolDraw2D.MolDraw2DCairo(500, 300)
    drawer.DrawMolecule(mol, highlightAtoms=list(highlight_atoms),
                        legend="Matched Toxicophores" if matched else "No toxicophores found")
    drawer.FinishDrawing()
    img = Draw.Image.open(io.BytesIO(drawer.GetDrawingText()))
    return matched, img

# =========================
# Narrative + plotting helpers
# =========================
def generate_mechanistic_report(label: str,
                                contrib_df: pd.DataFrame,
                                prob: float,
                                threshold: float,
                                smiles: str,
                                top_k_pos: int = 5,
                                top_k_neg: int = 3) -> str:
    cov = concept_coverage_summary(label)
    cov_note = f"Drivers: {cov.get('drivers',0)}, Counter-evidence: {cov.get('counters',0)}"

    lines = [
        f"### 🔍 {label} — Mechanistic Report\n",
        f"✅ **Prediction confidence**: `{prob:.2f}` (threshold = `{threshold:.2f}`)",
        f"🧭 **Evidence summary**: {cov_note}",
        "_Note: Contribution scores are TCAV-based (not SHAP) and scaled by statistical significance (q)._",
        "",
        _build_one_liner(label, contrib_df, smiles),
        "",
        "📊 **Contributing Concepts (TCAV):**"
    ]

    df = contrib_df.copy()
    drivers = df[df["final_score"] > 0].sort_values(["present","final_score"], ascending=[False, False]).head(int(top_k_pos))
    counters = df[df["final_score"] < 0].sort_values(["present","final_score"], ascending=[False, True]).head(int(top_k_neg))

    if drivers.empty:
        lines.append("- No positive concept drivers passed filters.")
    else:
        for _, row in drivers.iterrows():
            lines.append(_format_concept_bullet(row, label))

    lines.append("")
    lines.append("🚫 **Counter-evidence (TCAV):**")
    if counters.empty:
        lines.append("- No convincing counter-evidence under current filters.")
    else:
        for _, row in counters.iterrows():
            lines.append(_format_concept_bullet(row, label))

    lines.append("\n---")
    return "\n".join(lines)

def generate_toxicity_radar(smiles: str, results: dict):
    import plotly.graph_objects as go
    probs = [results["explanations"].get(lbl, {}).get("prob", 0.0) for lbl in label_cols]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=probs + [probs[0]], theta=label_cols + [label_cols[0]],
        fill='toself', name='Predicted Toxicity',
        line=dict(color='crimson'), marker=dict(symbol='circle'),
        hovertemplate='%{theta}<br>Prob: %{r:.2f}<extra></extra>',
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickfont_size=10)),
        showlegend=False, title=f"Toxicity Radar for: {smiles}",
        margin=dict(l=30, r=30, t=50, b=30), height=500
    )
    return fig

def plot_contrib_bar(contrib_df: pd.DataFrame, top: int = 10):
    import plotly.express as px
    df = contrib_df.copy()
    if df.empty:
        return px.bar(pd.DataFrame({"feature": [], "final_score": []}), x="final_score", y="feature", orientation="h")
    df = df.reindex(df["final_score"].abs().sort_values(ascending=False).index).head(top)
    fig = px.bar(df, x="final_score", y="feature", orientation="h")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
    return fig

def explain_near_misses(smiles: str, top_n: int = 3, top_features: int = 3):
    """Largest negative concept drivers for endpoints that are just below threshold."""
    probs = predict_probs(smiles); rows = []
    for lbl in label_cols:
        p = float(probs[lbl]); thr = float(thresholds.get(lbl, 0.5))
        if p >= thr: continue
        sdf = _tcav_df[_tcav_df["label"] == lbl].copy()
        if sdf.empty: continue
        sdf = sdf.sort_values("tcav_mean", ascending=True)
        neg = sdf.head(int(top_features)).copy()
        neg["feature"] = "CONCEPT_" + neg["concept"].astype(str)
        neg = _apply_tcav_contrib(neg)
        rows.append((lbl, p, thr, neg[["feature","contrib_score"]]))
    rows.sort(key=lambda t: t[1], reverse=True)
    return rows[:int(top_n)]

# =========================
# Public: prediction + explanations for all labels
# =========================
def predict_and_explain_all_labels(smiles: str,
                                   min_tcav: float = SETTINGS["tcav"]["min_tcav"],
                                   max_p: float = SETTINGS["tcav"]["max_p"],
                                   top_k: int = SETTINGS["tcav"]["top_k"]):
    probs = predict_probs(smiles)
    predicted = label_hits_from_probs(probs, thresholds)
    explanations = {}
    for label in predicted:
        contrib_df = _concept_importance_for_label(
            label,
            min_tcav=min_tcav, max_p=max_p, top_k=top_k,
            smiles_for_presence=smiles, drop_absent=False, boost_present=0.5
        )
        explanations[label] = {
            "prob": float(probs[label]),
            "threshold": float(thresholds.get(label, 0.5)),
            "pred_score": float(probs[label]),
            "shap_df": contrib_df,  # app expects this key; holds contrib + final_score
            "coverage": concept_coverage_summary(label, min_tcav=min_tcav, max_p=max_p),
        }
    return {"smiles": smiles, "predicted_labels": predicted, "explanations": explanations}

# =========================
# Diagnostics (useful for app footer)
# =========================
def platform_diagnostics() -> dict:
    return {
        "device": ("cuda" if torch.cuda.is_available() else "cpu"),
        "model_dir": str(MODEL_DIR.resolve()),
        "thresholds": (str(THRESH_PATH.resolve()), _file_sig(THRESH_PATH)),
        "tcav_summary": (str(TCAV_SUMMARY_FILE.resolve()), _file_sig(TCAV_SUMMARY_FILE)),
        "labels": label_cols,
        "settings": get_settings(),
    }

# =========================
# Simple label API
# =========================
def get_labels() -> List[str]:
    return label_cols[:]























# # utils.py — ChemBERTa v2 + TCAV (instance-aware), concept knowledge from JSON

# # --- stdlib ---
# import os, io, json, functools, hashlib, logging, random, datetime as dt
# from logging.handlers import RotatingFileHandler
# from pathlib import Path
# from typing import Dict, List, Tuple, Optional, Callable

# # --- scientific ---
# import numpy as np
# import pandas as pd
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

# # --- viz ---
# import plotly.express as px
# from PIL import Image

# # --- rdkit ---
# from rdkit import Chem
# from rdkit.Chem import AllChem
# from rdkit.Chem.Draw import rdMolDraw2D
# from rdkit import RDLogger
# RDLogger.DisableLog("rdApp.*")

# # --- optional ext lookup ---
# try:
#     import pubchempy as pcp
# except Exception:
#     pcp = None

# # =========================
# # Paths, config, determinism, logging
# # =========================
# ROOT = Path("implementation")
# MODELS_ROOT = ROOT / "models"

# # Model v2
# MODEL_BASE_DIR = MODELS_ROOT / "chemberta_v2"
# META_DIR       = MODEL_BASE_DIR / "metadata"

# # Selected run
# try:
#     _sel = json.load(open(META_DIR / "selected_run.json"))
#     RUN_SUFFIX = _sel.get("suffix", "_v2best")
# except Exception:
#     RUN_SUFFIX = "_v2best"
# WEIGHTS_DIR = MODEL_BASE_DIR / ("v2_best" if RUN_SUFFIX == "_v2best" else RUN_SUFFIX.strip("_"))

# # TCAV v2 locations
# CAV_DIR   = ROOT / "cav_v2"
# STATS_DIR = CAV_DIR / "stats"

# CONFIG_DIR = ROOT / "config"

# SETTINGS: Dict = {
#     "max_len": 256,
#     "use_standardization": True,
#     "enable_pubchem": False,
#     "tcav": {"min_tcav": 0.60, "max_p": 0.05, "top_k": 6},
# }

# cfg_path_env = os.environ.get("TOX21_CONFIG", "")
# SETTINGS_PATH = Path(cfg_path_env) if cfg_path_env else (CONFIG_DIR / "settings.json")
# if SETTINGS_PATH.exists():
#     try:
#         with open(SETTINGS_PATH) as f:
#             user_cfg = json.load(f)
#         def _merge(a, b):
#             for k, v in b.items():
#                 if isinstance(v, dict) and isinstance(a.get(k), dict):
#                     _merge(a[k], v)
#                 else:
#                     a[k] = v
#         _merge(SETTINGS, user_cfg)
#     except Exception:
#         pass

# USE_STANDARDIZATION = bool(SETTINGS.get("use_standardization", True))
# ENABLE_PUBCHEM = bool(SETTINGS.get("enable_pubchem", False))
# MAX_LEN = int(SETTINGS.get("max_len", 256))

# # determinism
# _SEED = 42
# random.seed(_SEED); np.random.seed(_SEED); torch.manual_seed(_SEED)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(_SEED)

# # logging
# LOGS = ROOT / "logs"; LOGS.mkdir(parents=True, exist_ok=True)
# _handler = RotatingFileHandler(LOGS / "app.log", maxBytes=2_000_000, backupCount=3)
# logging.basicConfig(level=logging.INFO, handlers=[_handler], format="%(asctime)s %(levelname)s: %(message)s")
# logger = logging.getLogger("tox21")

# def _file_sig(p: Path) -> str:
#     try:
#         h = hashlib.sha256()
#         with open(p, "rb") as f:
#             for chunk in iter(lambda: f.read(8192), b""):
#                 h.update(chunk)
#         return h.hexdigest()[:12]
#     except Exception:
#         return "missing"

# def _pick_tcav_summary_file() -> Path:
#     candidates = [
#         STATS_DIR / f"tcav_summary{RUN_SUFFIX}_v2concepts_k10.csv",
#         STATS_DIR / f"tcav_summary{RUN_SUFFIX}_v2concepts_k10.json",
#         STATS_DIR / f"tcav_summary{RUN_SUFFIX}_v2concepts.csv",
#         STATS_DIR / f"tcav_summary{RUN_SUFFIX}.csv",
#         STATS_DIR / "tcav_summary_last.csv",
#     ]
#     for p in candidates:
#         if p.exists(): return p
#     return candidates[-1]

# def platform_diagnostics() -> dict:
#     tcav_file = _pick_tcav_summary_file()
#     return {
#         "device": ("cuda" if torch.cuda.is_available() else "cpu"),
#         "model_dir": str(WEIGHTS_DIR.resolve()),
#         "thresholds": (str((META_DIR / "thresholds.json").resolve()), _file_sig(META_DIR / "thresholds.json")),
#         "tcav_summary": (str(tcav_file.resolve()), _file_sig(tcav_file)),
#         "calibration": {
#             "methods": (str((META_DIR / "calibration_methods.json").resolve()), _file_sig(META_DIR / "calibration_methods.json")),
#             "platt":   (str((META_DIR / "platt_params.json").resolve()), _file_sig(META_DIR / "platt_params.json")),
#             "isotonic":(str((META_DIR / "isotonic_params.json").resolve()), _file_sig(META_DIR / "isotonic_params.json")),
#             "temperature": (str((META_DIR / "temperature.npy").resolve()), _file_sig(META_DIR / "temperature.npy")),
#         },
#         "settings": SETTINGS,
#         "seed": _SEED,
#         "run_suffix": RUN_SUFFIX,
#         "timestamp": dt.datetime.utcnow().isoformat() + "Z",
#     }

# def get_settings() -> dict: return SETTINGS
# def set_use_standardization(flag: bool):
#     global USE_STANDARDIZATION; USE_STANDARDIZATION = bool(flag)
# def set_enable_pubchem(flag: bool):
#     global ENABLE_PUBCHEM; ENABLE_PUBCHEM = bool(flag)

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # =========================
# # Model + labels (v2)
# # =========================
# _tok = AutoTokenizer.from_pretrained(str(WEIGHTS_DIR))
# _cfg = AutoConfig.from_pretrained(str(WEIGHTS_DIR))
# _mdl = AutoModelForSequenceClassification.from_pretrained(str(WEIGHTS_DIR), config=_cfg).to(DEVICE).eval()
# if hasattr(_mdl.config, "use_cache"): _mdl.config.use_cache = False

# id2label = _mdl.config.id2label
# label2id = _mdl.config.label2id
# label_cols = [id2label[i] for i in range(len(id2label))]

# # =========================
# # Thresholds (v2)
# # =========================
# THRESH_PATH = META_DIR / "v1TH.json"
# if THRESH_PATH.exists():
#     thresholds: Dict[str, float] = json.load(open(THRESH_PATH))
# else:
#     thresholds = {lbl: 0.5 for lbl in label_cols}
#     logger.warning("thresholds.json not found — defaulting all thresholds to 0.5")

# # =========================
# # Calibration artifacts (v2)
# # =========================
# CAL_METHODS_PATH = META_DIR / "calibration_methods.json"
# PLATT_PATH       = META_DIR / "platt_params.json"
# ISO_PATH         = META_DIR / "isotonic_params.json"
# TEMP_PATH        = META_DIR / "temperature.npy"

# CAL_METHOD: Dict[str, Dict] = json.load(open(CAL_METHODS_PATH)) if CAL_METHODS_PATH.exists() else {}
# PLATT: Dict[str, Dict]      = json.load(open(PLATT_PATH)) if PLATT_PATH.exists() else {}
# ISO: Dict[str, Dict]        = json.load(open(ISO_PATH)) if ISO_PATH.exists() else {}
# TEMP: float                 = float(np.load(TEMP_PATH)[0]) if TEMP_PATH.exists() else 1.0

# def _apply_isotonic_scalar(p: float, X: List[float], Y: List[float]) -> float:
#     if not X or not Y: return float(p)
#     return float(np.interp(float(p), np.asarray(X, float), np.asarray(Y, float)))

# def _calibrate_single(lbl: str, logit: float, prob: float) -> float:
#     method = CAL_METHOD.get(lbl, {}).get("method", "none")
#     if method == "temp":
#         return 1.0 / (1.0 + np.exp(-float(logit) / max(TEMP, 1e-6)))
#     if method == "platt":
#         params = PLATT.get(lbl); 
#         if not params: return float(prob)
#         A, B = float(params["A"]), float(params["B"])
#         return 1.0 / (1.0 + np.exp(-(A * float(logit) + B)))
#     if method == "iso":
#         params = ISO.get(lbl); 
#         if not params: return float(prob)
#         return _apply_isotonic_scalar(float(prob), params.get("X"), params.get("Y"))
#     return float(prob)

# # =========================
# # TCAV summary + FDR q-values (v2)
# # =========================
# TCAV_SUMMARY_FILE = _pick_tcav_summary_file()
# if not TCAV_SUMMARY_FILE.exists():
#     raise FileNotFoundError(
#         f"TCAV summary not found under: {STATS_DIR}\n"
#         f"Expected like: tcav_summary{RUN_SUFFIX}_v2concepts_k10.(csv|json)"
#     )

# if TCAV_SUMMARY_FILE.suffix.lower() == ".json":
#     _tcav_df = pd.DataFrame(json.load(open(TCAV_SUMMARY_FILE)))
# else:
#     _tcav_df = pd.read_csv(TCAV_SUMMARY_FILE)

# if "p_value" not in _tcav_df.columns:
#     if "p_value_ttest" in _tcav_df.columns: _tcav_df["p_value"] = _tcav_df["p_value_ttest"]
#     elif "p_value_binom" in _tcav_df.columns: _tcav_df["p_value"] = _tcav_df["p_value_binom"]
#     else: _tcav_df["p_value"] = 1.0
# if "ci95" not in _tcav_df.columns:
#     _tcav_df["ci95"] = _tcav_df["ci95_t"] if "ci95_t" in _tcav_df.columns else None

# def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
#     p = np.asarray(pvals, dtype=float)
#     if p.size == 0: return p
#     m = p.size; order = np.argsort(p); ranks = np.arange(1, m+1)
#     q_sorted = (p[order] * m / ranks).clip(0, 1)
#     q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
#     q = np.empty_like(p); q[order] = q_sorted
#     return q

# _tcav_df["q_value"] = _bh_fdr(_tcav_df["p_value"].values)

# def _pretty_p(p: float, q: Optional[float] = None) -> str:
#     ptxt = "<1e-12" if p < 1e-12 else (f"{p:.1e}" if p < 1e-6 else f"{p:.3f}")
#     if q is None or not np.isfinite(q): return f"p={ptxt}"
#     qtxt = "<1e-6" if q < 1e-6 else f"{q:.3f}"
#     return f"p={ptxt}, q={qtxt}"

# # =========================
# # Basic helpers
# # =========================
# def validate_smiles(smiles: str) -> Tuple[bool, str]:
#     if not isinstance(smiles, str) or not smiles.strip():
#         return False, "Please enter a non-empty SMILES string."
#     if Chem.MolFromSmiles(smiles) is None:
#         return False, "Invalid SMILES string. Try a simple example like 'CCO' (ethanol)."
#     return True, ""

# def canonical_smiles(smiles: str) -> Optional[str]:
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None: return None
#     try: return Chem.MolToSmiles(mol, canonical=True)
#     except Exception: return smiles

# @functools.lru_cache(maxsize=1024)
# def _forward_logits(smiles: str, max_len: int = 256) -> np.ndarray:
#     enc = _tok(smiles, return_tensors="pt", truncation=True,
#                padding="max_length", max_length=max_len).to(DEVICE)
#     with torch.no_grad():
#         logits = _mdl(**enc).logits.squeeze(0)
#     return logits.detach().cpu().numpy().astype(float)

# def _prep_smiles_for_model(smiles: str) -> str:
#     return (canonical_smiles(smiles) or smiles) if USE_STANDARDIZATION else smiles

# @functools.lru_cache(maxsize=2048)
# def predict_probs(smiles: str, max_len: int = MAX_LEN) -> Dict[str, float]:
#     s = _prep_smiles_for_model(smiles)
#     logits = _forward_logits(s, max_len=max_len)
#     raw_probs = 1.0 / (1.0 + np.exp(-logits))
#     cal_probs = np.zeros_like(raw_probs, dtype=float)
#     for j, lbl in enumerate(label_cols):
#         cal_probs[j] = _calibrate_single(lbl, logits[j], raw_probs[j])
#     return {label_cols[i]: float(cal_probs[i]) for i in range(len(label_cols))}

# def predict_probs_batch(smiles_list: List[str], max_len: int = MAX_LEN) -> List[Dict[str, float]]:
#     smi_proc = [(_prep_smiles_for_model(s) or "") for s in smiles_list]
#     enc = _tok(smi_proc, return_tensors="pt", truncation=True, padding=True, max_length=max_len).to(DEVICE)
#     with torch.no_grad():
#         logits = _mdl(**enc).logits  # [B, L]
#     logits = logits.detach().cpu().numpy()
#     probs = 1.0 / (1.0 + np.exp(-logits))
#     out = []
#     for i in range(probs.shape[0]):
#         cal = np.zeros_like(probs[i])
#         for j, lbl in enumerate(label_cols):
#             cal[j] = _calibrate_single(lbl, logits[i, j], probs[i, j])
#         out.append({label_cols[k]: float(cal[k]) for k in range(len(label_cols))})
#     return out

# def predict_probs_matrix(smiles_list: List[str], max_len: int = MAX_LEN) -> np.ndarray:
#     dicts = predict_probs_batch(smiles_list, max_len=max_len)
#     mat = np.stack([[d[lbl] for lbl in label_cols] for d in dicts], axis=0)
#     return mat

# def label_hits_from_probs(probs: Dict[str, float], thr: Dict[str, float]) -> List[str]:
#     return [lbl for lbl, p in probs.items() if p >= thr.get(lbl, 0.5)]

# @functools.lru_cache(maxsize=256)
# def resolve_compound_name(smiles: str) -> str:
#     if not ENABLE_PUBCHEM or pcp is None:
#         return "Unknown compound"
#     try:
#         comps = pcp.get_compounds(smiles, namespace="smiles")
#         if not comps: return "Unknown compound"
#         c = comps[0]
#         for key in ("iupac_name", "title"):
#             val = getattr(c, key, None)
#             if val: return val
#         if getattr(c, "synonyms", None):
#             return c.synonyms[0]
#         return "Unknown compound"
#     except Exception:
#         return "Unknown compound"

# def _parse_ci95(ci) -> Optional[Tuple[float, float]]:
#     try:
#         if isinstance(ci, str):
#             ci = ci.replace("'", '"'); arr = json.loads(ci)
#             if isinstance(arr, list) and len(arr) == 2:
#                 return float(arr[0]), float(arr[1])
#         elif isinstance(ci, (list, tuple)) and len(ci) == 2:
#             return float(ci[0]), float(ci[1])
#     except Exception:
#         pass
#     return None

# # =========================
# # Concept knowledge (JSON, sanitized)
# # =========================
# CK_PATH = Path("implementation/concept_Knowledge/concept_knowledge_auto_v2.json")
# with open(CK_PATH, "r", encoding="utf-8") as f:
#     _CK_RAW = json.load(f)

# def _sanitize_expl(txt: str) -> str:
#     return txt.split(" — relevant to")[0].strip()

# # dict-of-dicts: {label: {concept: explanation}}
# CONCEPT_KNOWLEDGE: Dict[str, Dict[str, str]] = {}
# for _label, _items in _CK_RAW.items():
#     CONCEPT_KNOWLEDGE[_label] = {it["name"]: _sanitize_expl(it["explanation"]) for it in _items}

# def _kb_phrase(label: str, concept: str) -> str:
#     try:
#         return CONCEPT_KNOWLEDGE[label][concept]
#     except KeyError:
#         return concept

# # =========================
# # TCAV → contribution score + instance awareness
# # =========================
# def _tcav_strength(tcav_mean: float) -> str:
#     d = abs(tcav_mean - 0.5)
#     if d >= 0.40:  return "very strong"
#     if d >= 0.25:  return "strong"
#     if d >= 0.15:  return "moderate"
#     return "weak"

# def _apply_tcav_contrib(df: pd.DataFrame) -> pd.DataFrame:
#     d = df.copy()
#     if "q_value" not in d.columns:
#         d["q_value"] = 1.0
#     d["q_value"] = d["q_value"].clip(0, 1)
#     d["contrib_score"] = (d["tcav_mean"] - 0.5) * (1.0 - d["q_value"])
#     d["direction"] = np.where(d["contrib_score"] >= 0, "↑", "↓")
#     return d

# # SMARTS rules (optional presence detection)
# SMARTS_PATH = META_DIR / "smarts_rules_final.json"
# SMARTS_RULES = json.load(open(SMARTS_PATH)) if SMARTS_PATH.exists() else {}

# # Build Concept -> SMARTS list from SMARTS_RULES by name
# CONCEPT_TO_SMARTS: Dict[str, List[str]] = {}
# if SMARTS_RULES:
#     tmp = {}
#     for _lbl, rules in SMARTS_RULES.items():
#         for r in rules:
#             nm = r.get("name", "")
#             sm = r.get("smarts", "")
#             if nm and sm:
#                 tmp.setdefault(nm, set()).add(sm)
#     CONCEPT_TO_SMARTS = {k: sorted(list(v)) for k, v in tmp.items()}

# def _concept_present_in_smiles(smiles: str, concept: str) -> Tuple[bool, int]:
#     patt_list = CONCEPT_TO_SMARTS.get(concept, [])
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None or not patt_list:
#         return False, 0
#     total = 0
#     for sm in patt_list:
#         p = Chem.MolFromSmarts(sm)
#         if not p: 
#             continue
#         total += len(mol.GetSubstructMatches(p))
#     return (total > 0), total

# def _concept_importance_for_label(label: str,
#                                   min_tcav: float = SETTINGS["tcav"]["min_tcav"],
#                                   max_p: float = SETTINGS["tcav"]["max_p"],
#                                   top_k: int = SETTINGS["tcav"]["top_k"],
#                                   include_negative: bool = True,
#                                   neg_tcav: float = 0.40,
#                                   smiles_for_presence: Optional[str] = None,
#                                   drop_absent: bool = False,
#                                   boost_present: float = 0.5) -> pd.DataFrame:
#     """
#     If smiles_for_presence is given, add instance-aware flags and compute final_score:
#       final_score = contrib_score * (1 + boost_present * present)
#     If drop_absent=True, remove concepts not detected in this molecule.
#     """
#     sdf = _tcav_df[_tcav_df["label"] == label].copy()
#     if sdf.empty:
#         return pd.DataFrame(columns=[
#             "feature","contrib_score","p_value","q_value","tcav_mean",
#             "concept","direction","ci95","present","present_count","final_score"
#         ])

#     pos = sdf[(sdf["tcav_mean"] >= float(min_tcav)) & (sdf["p_value"] <= float(max_p))].copy()
#     neg = pd.DataFrame()
#     if include_negative:
#         neg = sdf[(sdf["tcav_mean"] <= float(neg_tcav)) & (sdf["p_value"] <= float(max_p))].copy()

#     keep = pd.concat([pos, neg], ignore_index=True)
#     if keep.empty:
#         return pd.DataFrame(columns=[
#             "feature","contrib_score","p_value","q_value","tcav_mean",
#             "concept","direction","ci95","present","present_count","final_score"
#         ])

#     keep["feature"] = "CONCEPT_" + keep["concept"].astype(str)
#     keep = _apply_tcav_contrib(keep)

#     pres = []
#     pres_cnt = []
#     if smiles_for_presence:
#         for c in keep["concept"].astype(str).tolist():
#             p, n = _concept_present_in_smiles(smiles_for_presence, c)
#             pres.append(bool(p)); pres_cnt.append(int(n))
#     else:
#         pres = [False]*len(keep); pres_cnt = [0]*len(keep)
#     keep["present"] = pres
#     keep["present_count"] = pres_cnt

#     keep["final_score"] = keep["contrib_score"] * (1.0 + (boost_present if smiles_for_presence else 0.0) * keep["present"].astype(float))

#     if drop_absent and smiles_for_presence:
#         keep = keep[keep["present"]]

#     keep = keep.sort_values(["direction","final_score","tcav_mean"], ascending=[True, False, False])

#     top_pos = keep[keep["direction"]=="↑"].head(top_k)
#     top_neg = keep[keep["direction"]=="↓"].head(2) if include_negative else pd.DataFrame(columns=keep.columns)
#     out = pd.concat([top_pos, top_neg], ignore_index=True)

#     return out.reset_index(drop=True)[
#         ["feature","contrib_score","p_value","q_value","tcav_mean",
#          "concept","direction","ci95","present","present_count","final_score"]
#     ]

# def concept_coverage_summary(label: str,
#                              min_tcav: float = SETTINGS["tcav"]["min_tcav"],
#                              max_p: float = SETTINGS["tcav"]["max_p"],
#                              neg_tcav: float = 0.40) -> Dict[str, int]:
#     df = _concept_importance_for_label(label, min_tcav=min_tcav, max_p=max_p, top_k=999, include_negative=True, neg_tcav=neg_tcav)
#     if df.empty: return {"drivers": 0, "counters": 0}
#     return {"drivers": int((df["contrib_score"] > 0).sum()),
#             "counters": int((df["contrib_score"] < 0).sum())}

# def _format_concept_bullet(row, label: str) -> str:
#     fname = row["feature"]; cname = fname.replace("CONCEPT_", "")
#     sgn = "↑" if row["contrib_score"] >= 0 else "↓"
#     tcav_mean = float(row.get("tcav_mean", 0.5))
#     strength = _tcav_strength(tcav_mean)
#     pval = float(row.get("p_value", 1.0))
#     qval = float(row.get("q_value", np.nan)) if "q_value" in row else None
#     pq = _pretty_p(pval, qval if (qval == qval) else None)
#     desc = _kb_phrase(label, cname)
#     ci = _parse_ci95(row.get("ci95", None))
#     ci_txt = f" [{ci[0]:.2f}–{ci[1]:.2f}]" if ci else ""
#     pres = " • **present**" if bool(row.get("present", False)) else ""
#     cnt  = int(row.get("present_count", 0))
#     pres += f"×{cnt}" if cnt > 1 else ""
#     return f"- **{cname}** ({sgn}, {strength}, TCAV={tcav_mean:.2f}{ci_txt}, {pq}{pres}) — {desc}"

# # =========================
# # SMARTS helpers (optional rules)
# # =========================
# def match_toxicophores_with_explanations(smiles: str, label: Optional[str] = None):
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None or not SMARTS_RULES: return []
#     label_rules = SMARTS_RULES.get(label, []) if label else [r for rules in SMARTS_RULES.values() for r in rules]
#     hits = []
#     for rule in label_rules:
#         patt = Chem.MolFromSmarts(rule["smarts"])
#         if patt and mol.HasSubstructMatch(patt):
#             hits.append({"name": rule["name"], "explanation": rule["explanation"]})
#     return hits

# def highlight_toxicophores(smiles: str):
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None: raise ValueError("Invalid SMILES structure.")
#     AllChem.Compute2DCoords(mol)
#     highlight_atoms = set(); matched = []
#     all_rules = [r for rules in SMARTS_RULES.values() for r in rules] if SMARTS_RULES else []
#     for rule in all_rules:
#         patt = Chem.MolFromSmarts(rule["smarts"])
#         if not patt: continue
#         matches = mol.GetSubstructMatches(patt)
#         if matches:
#             matched.append(f"☣️ **{rule['name']}**: {rule['explanation']}")
#             for m in matches: highlight_atoms.update(m)
#     drawer = rdMolDraw2D.MolDraw2DCairo(500, 300)
#     drawer.DrawMolecule(mol, highlightAtoms=list(highlight_atoms),
#                         legend="Matched Toxicophores" if matched else "No toxicophores found")
#     drawer.FinishDrawing()
#     img = Image.open(io.BytesIO(drawer.GetDrawingText()))
#     return matched, img

# # =========================
# # Narrative helpers
# # =========================
# ENDPOINT_TAIL = {
#     "SR-MMP":   "mitochondrial depolarization consistent with ΔΨm collapse",
#     "SR-ARE":   "electrophilic/oxidative stress consistent with Nrf2/ARE activation",
#     "SR-ATAD5": "replication stress linked to DNA repair/PCNA unloading",
#     "SR-HSE":   "proteotoxic stress consistent with heat-shock response",
#     "SR-p53":   "DNA-damage signalling leading to p53 stabilization",
#     "NR-AR":          "androgen receptor engagement at the LBD",
#     "NR-AR-LBD":      "high-affinity occupation of the AR ligand-binding domain",
#     "NR-AhR":         "planar aromatic binding to AhR and xenobiotic response",
#     "NR-Aromatase":   "CYP19 (aromatase) perturbation of estrogen synthesis",
#     "NR-ER":          "estrogen receptor modulation and downstream transcription",
#     "NR-ER-LBD":      "ligand-binding-domain engagement within ER",
#     "NR-PPAR-gamma":  "PPAR-γ activation affecting metabolic gene programs",
# }

# def _kb_narrative(label: str, df: pd.DataFrame, max_items: int = 3) -> str:
#     if df is None or df.empty:
#         return f"No strong concept evidence passed filters for {label}."
#     pos = df[df["final_score"] > 0].copy()
#     # prefer present concepts first
#     pos = pos.sort_values(["present","final_score"], ascending=[False, False]).head(max_items)
#     neg = df[df["final_score"] < 0].sort_values(["present","final_score"], ascending=[False, True]).head(1)
#     bits = []
#     if not pos.empty:
#         phrases = [_kb_phrase(label, f.replace("CONCEPT_","")) for f in pos["feature"]]
#         bits.append(" ; ".join(phrases))
#     if not neg.empty:
#         phrases_n = [_kb_phrase(label, f.replace("CONCEPT_","")) for f in neg["feature"]]
#         bits.append(f"while features such as {', '.join(phrases_n)} appear inversely associated")
#     if bits:
#         return f"Predicted {label} toxicity is supported by {bits[0]}" + (f", {bits[1]}." if len(bits)>1 else ".")
#     return f"No strong concept evidence passed filters for {label}."

# def _build_one_liner(label: str, contrib_df: pd.DataFrame, smiles: str) -> str:
#     if contrib_df is None or contrib_df.empty:
#         return f"Predicted {label} toxicity lacks clear concept evidence."
#     df = contrib_df.copy()
#     df_pos = df[(df["final_score"] > 0)].copy()
#     df_pos_present = df_pos[df_pos["present"]].sort_values("final_score", ascending=False)
#     if not df_pos_present.empty:
#         pick = df_pos_present.iloc[0]
#     elif not df_pos.empty:
#         pick = df_pos.sort_values("final_score", ascending=False).iloc[0]
#     else:
#         df_neg_present = df[(df["final_score"] < 0) & (df["present"])].sort_values("final_score")
#         pick = (df_neg_present.iloc[0] if not df_neg_present.empty
#                 else df.sort_values("final_score", ascending=False).iloc[0])

#     cname = str(pick["concept"])
#     direction = "driven by" if float(pick["final_score"]) > 0 else "countered by"
#     tail = ENDPOINT_TAIL.get(label, "mechanistic activity")
#     phrase = _kb_phrase(label, cname)
#     pres_note = " (detected in molecule)" if bool(pick.get("present", False)) else ""
#     return f"{direction} **{cname}**{pres_note} — {phrase}; overall consistent with {tail}."

# # =========================
# # Main API used by app.py
# # =========================
# def predict_and_explain_all_labels(smiles: str,
#                                    min_tcav: float = SETTINGS["tcav"]["min_tcav"],
#                                    max_p: float = SETTINGS["tcav"]["max_p"],
#                                    top_k: int = SETTINGS["tcav"]["top_k"]):
#     probs = predict_probs(smiles)
#     predicted = label_hits_from_probs(probs, thresholds)
#     explanations = {}
#     for label in predicted:
#         contrib_df = _concept_importance_for_label(
#             label,
#             min_tcav=min_tcav, max_p=max_p, top_k=top_k,
#             smiles_for_presence=smiles, drop_absent=False, boost_present=0.5
#         )
#         explanations[label] = {
#             "prob": float(probs[label]),
#             "threshold": float(thresholds.get(label, 0.5)),
#             "pred_score": float(probs[label]),
#             "shap_df": contrib_df,  # app expects this key; holds contrib + final_score
#             "coverage": concept_coverage_summary(label, min_tcav=min_tcav, max_p=max_p),
#         }
#     return {"smiles": smiles, "predicted_labels": predicted, "explanations": explanations}

# def generate_toxicity_radar(smiles: str, results: dict):
#     import plotly.graph_objects as go
#     probs = [results["explanations"].get(lbl, {}).get("prob", 0.0) for lbl in label_cols]
#     fig = go.Figure()
#     fig.add_trace(go.Scatterpolar(
#         r=probs + [probs[0]], theta=label_cols + [label_cols[0]],
#         fill='toself', name='Predicted Toxicity',
#         line=dict(color='crimson'), marker=dict(symbol='circle'),
#         hovertemplate='%{theta}<br>Prob: %{r:.2f}<extra></extra>',
#     ))
#     fig.update_layout(
#         polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickfont_size=10)),
#         showlegend=False, title=f"Toxicity Radar for: {smiles}",
#         margin=dict(l=30, r=30, t=50, b=30), height=500
#     )
#     return fig

# def plot_contrib_bar(contrib_df: pd.DataFrame, top: int = 10):
#     df = contrib_df.copy()
#     if df.empty:
#         return px.bar(pd.DataFrame({"feature": [], "final_score": []}), x="final_score", y="feature", orientation="h")
#     df = df.reindex(df["final_score"].abs().sort_values(ascending=False).index).head(top)
#     fig = px.bar(df, x="final_score", y="feature", orientation="h")
#     fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
#     return fig

# def explain_near_misses(smiles: str, top_n: int = 3, top_features: int = 3):
#     probs = predict_probs(smiles); rows = []
#     for lbl in label_cols:
#         p = float(probs[lbl]); thr = float(thresholds.get(lbl, 0.5))
#         if p >= thr: continue
#         sdf = _tcav_df[_tcav_df["label"] == lbl].copy()
#         if sdf.empty: continue
#         sdf = sdf.sort_values("tcav_mean", ascending=True)
#         neg = sdf.head(int(top_features)).copy()
#         neg["feature"] = "CONCEPT_" + neg["concept"].astype(str)
#         neg = _apply_tcav_contrib(neg)
#         # keep compatibility but show contrib score
#         rows.append((lbl, p, thr, neg[["feature","contrib_score"]]))
#     rows.sort(key=lambda t: t[1], reverse=True)
#     return rows[:int(top_n)]

# def generate_mechanistic_report(label: str,
#                                 shap_df: pd.DataFrame,
#                                 prob: float,
#                                 threshold: float,
#                                 smiles: str,
#                                 top_k_pos: int = 5,
#                                 top_k_neg: int = 3) -> str:
#     cov = concept_coverage_summary(label)
#     cov_note = f"Drivers: {cov.get('drivers',0)}, Counter-evidence: {cov.get('counters',0)}"

#     lines = [
#         f"### 🔍 {label} — Mechanistic Report\n",
#         f"✅ **Prediction confidence**: `{prob:.2f}` (threshold = `{threshold:.2f}`)",
#         f"🧭 **Evidence summary**: {cov_note}",
#         "_Note: Contribution scores are TCAV-based (not SHAP) and scaled by statistical significance (q)._",
#         "",
#         _build_one_liner(label, shap_df, smiles),
#         "",
#         "📊 **Contributing Concepts (TCAV):**"
#     ]

#     df = shap_df.copy()
#     drivers = df[df["final_score"] > 0].sort_values(["present","final_score"], ascending=[False, False]).head(int(top_k_pos))
#     counters = df[df["final_score"] < 0].sort_values(["present","final_score"], ascending=[False, True]).head(int(top_k_neg))

#     if drivers.empty:
#         lines.append("- No positive concept drivers passed filters.")
#     else:
#         for _, row in drivers.iterrows():
#             lines.append(_format_concept_bullet(row, label))

#     lines.append("")
#     lines.append("🚫 **Counter-evidence (TCAV):**")
#     if counters.empty:
#         lines.append("- No convincing counter-evidence under current filters.")
#     else:
#         for _, row in counters.iterrows():
#             lines.append(_format_concept_bullet(row, label))

#     lines.append("\n---")
#     return "\n".join(lines)

# def summarize_prediction(result: dict) -> str:
#     smiles = result["smiles"]; predicted = result["predicted_labels"]
#     if not predicted:
#         return f"🔬 The drug (SMILES: `{smiles}`) has **no endpoints above threshold**."
#     ordered = sorted(predicted, key=lambda lb: result["explanations"][lb]["pred_score"], reverse=True)
#     return f"🔬 The drug (SMILES: `{smiles}`) is predicted positive for: **{', '.join(ordered)}**."




















# # import os, io, json, functools, hashlib, logging, random, datetime as dt
# # from logging.handlers import RotatingFileHandler
# # from pathlib import Path
# # from typing import Dict, List, Tuple, Optional

# # import numpy as np
# # import pandas as pd
# # import torch
# # from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

# # # Viz
# # import plotly.graph_objects as go
# # import plotly.express as px
# # from PIL import Image

# # # RDKit (minimal set we still use)
# # from rdkit import Chem
# # from rdkit.Chem import AllChem
# # from rdkit.Chem.Draw import rdMolDraw2D
# # from rdkit import RDLogger
# # RDLogger.DisableLog("rdApp.*")

# # # Optional ext lookup
# # try:
# #     import pubchempy as pcp
# # except Exception:
# #     pcp = None

# # # =========================
# # # Paths, config, determinism, logging
# # # =========================
# # ROOT = Path("implementation")
# # MODELS_ROOT = ROOT / "models"

# # # Model v2
# # MODEL_BASE_DIR = MODELS_ROOT / "chemberta_v2"
# # META_DIR       = MODEL_BASE_DIR / "metadata"

# # # Selected run
# # try:
# #     _sel = json.load(open(META_DIR / "selected_run.json"))
# #     RUN_SUFFIX = _sel.get("suffix", "_v2best")
# # except Exception:
# #     RUN_SUFFIX = "_v2best"
# # WEIGHTS_DIR = MODEL_BASE_DIR / ("v2_best" if RUN_SUFFIX == "_v2best" else RUN_SUFFIX.strip("_"))

# # # TCAV v2 locations
# # CAV_DIR   = ROOT / "cav_v2"
# # STATS_DIR = CAV_DIR / "stats"

# # CONFIG_DIR = ROOT / "config"

# # SETTINGS: Dict = {
# #     "max_len": 256,
# #     "use_standardization": True,
# #     "enable_pubchem": False,
# #     "tcav": {"min_tcav": 0.60, "max_p": 0.05, "top_k": 6},
# # }

# # cfg_path_env = os.environ.get("TOX21_CONFIG", "")
# # SETTINGS_PATH = Path(cfg_path_env) if cfg_path_env else (CONFIG_DIR / "settings.json")
# # if SETTINGS_PATH.exists():
# #     try:
# #         with open(SETTINGS_PATH) as f:
# #             user_cfg = json.load(f)
# #         def _merge(a, b):
# #             for k, v in b.items():
# #                 if isinstance(v, dict) and isinstance(a.get(k), dict):
# #                     _merge(a[k], v)
# #                 else:
# #                     a[k] = v
# #         _merge(SETTINGS, user_cfg)
# #     except Exception:
# #         pass

# # USE_STANDARDIZATION = bool(SETTINGS.get("use_standardization", True))
# # ENABLE_PUBCHEM = bool(SETTINGS.get("enable_pubchem", False))
# # MAX_LEN = int(SETTINGS.get("max_len", 256))

# # # determinism
# # _SEED = 42
# # random.seed(_SEED); np.random.seed(_SEED); torch.manual_seed(_SEED)
# # if torch.cuda.is_available():
# #     torch.cuda.manual_seed_all(_SEED)

# # # logging
# # LOGS = ROOT / "logs"; LOGS.mkdir(parents=True, exist_ok=True)
# # _handler = RotatingFileHandler(LOGS / "app.log", maxBytes=2_000_000, backupCount=3)
# # logging.basicConfig(level=logging.INFO, handlers=[_handler], format="%(asctime)s %(levelname)s: %(message)s")
# # logger = logging.getLogger("tox21")

# # def _file_sig(p: Path) -> str:
# #     try:
# #         h = hashlib.sha256()
# #         with open(p, "rb") as f:
# #             for chunk in iter(lambda: f.read(8192), b""):
# #                 h.update(chunk)
# #         return h.hexdigest()[:12]
# #     except Exception:
# #         return "missing"

# # def _pick_tcav_summary_file() -> Path:
# #     candidates = [
# #         STATS_DIR / f"tcav_summary{RUN_SUFFIX}_v2concepts_k10.csv",
# #         STATS_DIR / f"tcav_summary{RUN_SUFFIX}_v2concepts_k10.json",
# #         STATS_DIR / f"tcav_summary{RUN_SUFFIX}_v2concepts.csv",
# #         STATS_DIR / f"tcav_summary{RUN_SUFFIX}.csv",
# #         STATS_DIR / "tcav_summary_last.csv",
# #     ]
# #     for p in candidates:
# #         if p.exists(): return p
# #     return candidates[-1]

# # def platform_diagnostics() -> dict:
# #     tcav_file = _pick_tcav_summary_file()
# #     return {
# #         "device": ("cuda" if torch.cuda.is_available() else "cpu"),
# #         "model_dir": str(WEIGHTS_DIR.resolve()),
# #         "thresholds": (str((META_DIR / "thresholds.json").resolve()), _file_sig(META_DIR / "thresholds.json")),
# #         "tcav_summary": (str(tcav_file.resolve()), _file_sig(tcav_file)),
# #         "calibration": {
# #             "methods": (str((META_DIR / "calibration_methods.json").resolve()), _file_sig(META_DIR / "calibration_methods.json")),
# #             "platt":   (str((META_DIR / "platt_params.json").resolve()), _file_sig(META_DIR / "platt_params.json")),
# #             "isotonic":(str((META_DIR / "isotonic_params.json").resolve()), _file_sig(META_DIR / "isotonic_params.json")),
# #             "temperature": (str((META_DIR / "temperature.npy").resolve()), _file_sig(META_DIR / "temperature.npy")),
# #         },
# #         "settings": SETTINGS,
# #         "seed": _SEED,
# #         "run_suffix": RUN_SUFFIX,
# #         "timestamp": dt.datetime.utcnow().isoformat() + "Z",
# #     }

# # def get_settings() -> dict: return SETTINGS
# # def set_use_standardization(flag: bool):
# #     global USE_STANDARDIZATION; USE_STANDARDIZATION = bool(flag)
# # def set_enable_pubchem(flag: bool):
# #     global ENABLE_PUBCHEM; ENABLE_PUBCHEM = bool(flag)

# # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # # =========================
# # # Model + labels (v2)
# # # =========================
# # _tok = AutoTokenizer.from_pretrained(str(WEIGHTS_DIR))
# # _cfg = AutoConfig.from_pretrained(str(WEIGHTS_DIR))
# # _mdl = AutoModelForSequenceClassification.from_pretrained(str(WEIGHTS_DIR), config=_cfg).to(DEVICE).eval()
# # if hasattr(_mdl.config, "use_cache"): _mdl.config.use_cache = False

# # id2label = _mdl.config.id2label
# # label2id = _mdl.config.label2id
# # label_cols = [id2label[i] for i in range(len(id2label))]

# # # =========================
# # # Thresholds (v2)
# # # =========================
# # THRESH_PATH = META_DIR / "thresholds.json"
# # if THRESH_PATH.exists():
# #     thresholds: Dict[str, float] = json.load(open(THRESH_PATH))
# # else:
# #     thresholds = {lbl: 0.5 for lbl in label_cols}
# #     logger.warning("thresholds.json not found — defaulting all thresholds to 0.5")

# # # =========================
# # # Calibration artifacts (v2)
# # # =========================
# # CAL_METHODS_PATH = META_DIR / "calibration_methods.json"
# # PLATT_PATH       = META_DIR / "platt_params.json"
# # ISO_PATH         = META_DIR / "isotonic_params.json"
# # TEMP_PATH        = META_DIR / "temperature.npy"

# # CAL_METHOD: Dict[str, Dict] = json.load(open(CAL_METHODS_PATH)) if CAL_METHODS_PATH.exists() else {}
# # PLATT: Dict[str, Dict]      = json.load(open(PLATT_PATH)) if PLATT_PATH.exists() else {}
# # ISO: Dict[str, Dict]        = json.load(open(ISO_PATH)) if ISO_PATH.exists() else {}
# # TEMP: float                 = float(np.load(TEMP_PATH)[0]) if TEMP_PATH.exists() else 1.0

# # def _apply_isotonic_scalar(p: float, X: List[float], Y: List[float]) -> float:
# #     if not X or not Y: return float(p)
# #     return float(np.interp(float(p), np.asarray(X, float), np.asarray(Y, float)))

# # def _calibrate_single(lbl: str, logit: float, prob: float) -> float:
# #     method = CAL_METHOD.get(lbl, {}).get("method", "none")
# #     if method == "temp":
# #         return 1.0 / (1.0 + np.exp(-float(logit) / max(TEMP, 1e-6)))
# #     if method == "platt":
# #         params = PLATT.get(lbl); 
# #         if not params: return float(prob)
# #         A, B = float(params["A"]), float(params["B"])
# #         return 1.0 / (1.0 + np.exp(-(A * float(logit) + B)))
# #     if method == "iso":
# #         params = ISO.get(lbl); 
# #         if not params: return float(prob)
# #         return _apply_isotonic_scalar(float(prob), params.get("X"), params.get("Y"))
# #     return float(prob)

# # # =========================
# # # TCAV summary + FDR q-values (v2)
# # # =========================
# # TCAV_SUMMARY_FILE = _pick_tcav_summary_file()
# # if not TCAV_SUMMARY_FILE.exists():
# #     raise FileNotFoundError(
# #         f"TCAV summary not found under: {STATS_DIR}\n"
# #         f"Expected like: tcav_summary{RUN_SUFFIX}_v2concepts_k10.(csv|json)"
# #     )

# # if TCAV_SUMMARY_FILE.suffix.lower() == ".json":
# #     _tcav_df = pd.DataFrame(json.load(open(TCAV_SUMMARY_FILE)))
# # else:
# #     _tcav_df = pd.read_csv(TCAV_SUMMARY_FILE)

# # if "p_value" not in _tcav_df.columns:
# #     if "p_value_ttest" in _tcav_df.columns: _tcav_df["p_value"] = _tcav_df["p_value_ttest"]
# #     elif "p_value_binom" in _tcav_df.columns: _tcav_df["p_value"] = _tcav_df["p_value_binom"]
# #     else: _tcav_df["p_value"] = 1.0
# # if "ci95" not in _tcav_df.columns:
# #     _tcav_df["ci95"] = _tcav_df["ci95_t"] if "ci95_t" in _tcav_df.columns else None

# # def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
# #     p = np.asarray(pvals, dtype=float)
# #     if p.size == 0: return p
# #     m = p.size; order = np.argsort(p); ranks = np.arange(1, m+1)
# #     q_sorted = (p[order] * m / ranks).clip(0, 1)
# #     q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
# #     q = np.empty_like(p); q[order] = q_sorted
# #     return q

# # _tcav_df["q_value"] = _bh_fdr(_tcav_df["p_value"].values)

# # def _pretty_p(p: float, q: Optional[float] = None) -> str:
# #     ptxt = "<1e-12" if p < 1e-12 else (f"{p:.1e}" if p < 1e-6 else f"{p:.3f}")
# #     if q is None or not np.isfinite(q): return f"p={ptxt}"
# #     qtxt = "<1e-6" if q < 1e-6 else f"{q:.3f}"
# #     return f"p={ptxt}, q={qtxt}"

# # # =========================
# # # Basic helpers
# # # =========================
# # def validate_smiles(smiles: str) -> Tuple[bool, str]:
# #     if not isinstance(smiles, str) or not smiles.strip():
# #         return False, "Please enter a non-empty SMILES string."
# #     if Chem.MolFromSmiles(smiles) is None:
# #         return False, "Invalid SMILES string. Try a simple example like 'CCO' (ethanol)."
# #     return True, ""

# # def canonical_smiles(smiles: str) -> Optional[str]:
# #     mol = Chem.MolFromSmiles(smiles)
# #     if mol is None: return None
# #     try: return Chem.MolToSmiles(mol, canonical=True)
# #     except Exception: return smiles

# # @functools.lru_cache(maxsize=1024)
# # def _forward_logits(smiles: str, max_len: int = 256) -> np.ndarray:
# #     enc = _tok(smiles, return_tensors="pt", truncation=True,
# #                padding="max_length", max_length=max_len).to(DEVICE)
# #     with torch.no_grad():
# #         logits = _mdl(**enc).logits.squeeze(0)
# #     return logits.detach().cpu().numpy().astype(float)

# # def _prep_smiles_for_model(smiles: str) -> str:
# #     return (canonical_smiles(smiles) or smiles) if USE_STANDARDIZATION else smiles

# # @functools.lru_cache(maxsize=2048)
# # def predict_probs(smiles: str, max_len: int = MAX_LEN) -> Dict[str, float]:
# #     s = _prep_smiles_for_model(smiles)
# #     logits = _forward_logits(s, max_len=max_len)
# #     raw_probs = 1.0 / (1.0 + np.exp(-logits))
# #     cal_probs = np.zeros_like(raw_probs, dtype=float)
# #     for j, lbl in enumerate(label_cols):
# #         cal_probs[j] = _calibrate_single(lbl, logits[j], raw_probs[j])
# #     return {label_cols[i]: float(cal_probs[i]) for i in range(len(label_cols))}

# # def predict_probs_batch(smiles_list: List[str], max_len: int = MAX_LEN) -> List[Dict[str, float]]:
# #     smi_proc = [(_prep_smiles_for_model(s) or "") for s in smiles_list]
# #     enc = _tok(smi_proc, return_tensors="pt", truncation=True, padding=True, max_length=max_len).to(DEVICE)
# #     with torch.no_grad():
# #         logits = _mdl(**enc).logits  # [B, L]
# #     logits = logits.detach().cpu().numpy()
# #     probs = 1.0 / (1.0 + np.exp(-logits))
# #     out = []
# #     for i in range(probs.shape[0]):
# #         cal = np.zeros_like(probs[i])
# #         for j, lbl in enumerate(label_cols):
# #             cal[j] = _calibrate_single(lbl, logits[i, j], probs[i, j])
# #         out.append({label_cols[k]: float(cal[k]) for k in range(len(label_cols))})
# #     return out

# # def label_hits_from_probs(probs: Dict[str, float], thr: Dict[str, float]) -> List[str]:
# #     return [lbl for lbl, p in probs.items() if p >= thr.get(lbl, 0.5)]

# # @functools.lru_cache(maxsize=256)
# # def resolve_compound_name(smiles: str) -> str:
# #     if not ENABLE_PUBCHEM or pcp is None:
# #         return "Unknown compound"
# #     try:
# #         comps = pcp.get_compounds(smiles, namespace="smiles")
# #         if not comps: return "Unknown compound"
# #         c = comps[0]
# #         for key in ("iupac_name", "title"):
# #             val = getattr(c, key, None)
# #             if val: return val
# #         if getattr(c, "synonyms", None):
# #             return c.synonyms[0]
# #         return "Unknown compound"
# #     except Exception:
# #         return "Unknown compound"

# # def _parse_ci95(ci) -> Optional[Tuple[float, float]]:
# #     try:
# #         if isinstance(ci, str):
# #             ci = ci.replace("'", '"'); arr = json.loads(ci)
# #             if isinstance(arr, list) and len(arr) == 2:
# #                 return float(arr[0]), float(arr[1])
# #         elif isinstance(ci, (list, tuple)) and len(ci) == 2:
# #             return float(ci[0]), float(ci[1])
# #     except Exception:
# #         pass
# #     return None

# # # =========================
# # # Concept knowledge (for narratives)
# # # =========================
# # CONCEPT_KNOWLEDGE: Dict[str, Dict[str, str]] = {
# #     "_generic": {
# #         "AromaticRing":     "planar π-system enabling hydrophobic/π–π interactions",
# #         "Nitro":            "nitro functionality linked to bioactivation and potential DNA/protein adducts",
# #         "ArylHalide":       "halogenated aryl increasing lipophilicity and metabolic stability",
# #         "Phenol":           "phenolic OH capable of hydrogen bonding and metabolic conjugation",
# #         "TertiaryAmine":    "basic amine promoting cationic binding and membrane permeability",
# #         "MichaelAcceptor":  "α,β-unsaturated system acting as a soft electrophile (Michael acceptor)",
# #         "Quinone":          "redox-active scaffold associated with ROS generation and covalent reactivity",
# #         "CarboxylicAcid":   "acidic handle increasing polarity and H-bonding potential",
# #         "Sulfonamide":      "H-bond rich motif impacting acidity and binding patterns",
# #         "Ester":            "polar carbonyl/alkoxy modulating permeability and metabolism",
# #         "Aniline":          "aromatic amine often linked to metabolic liabilities",
# #         "Pyridine":         "basic heteroaromatic offering HBA capability",
# #     },
# #     "NR-AhR": {
# #         "AromaticRing":    "planar aromatic system consistent with AhR binding preferences",
# #         "Nitro":           "electron-withdrawing group that can stabilize planar conformations",
# #         "Quinone":         "redox cycling potentially triggering xenobiotic response genes",
# #         "ArylHalide":      "halogenation increasing hydrophobicity and receptor affinity",
# #         "Phenol":          "H-bonding potential that may assist binding orientation",
# #         "TertiaryAmine":   "cationic center possibly aiding non-covalent interactions",
# #         "MichaelAcceptor": "electrophilic center that may perturb stress-response pathways",
# #         "CarboxylicAcid":  "acidic group modulating polarity and binding orientation",
# #         "Sulfonamide":     "H-bonding motif that can stabilize binding poses",
# #     },
# #     "NR-AR": {
# #         "AromaticRing":    "hydrophobic/π–π contacts compatible with AR LBD",
# #         "Nitro":           "EWG altering electronic density of the ring system",
# #         "Phenol":          "H-bond donor/acceptor patterns relevant to AR binding",
# #         "ArylHalide":      "lipophilicity increase affecting AR engagement",
# #         "TertiaryAmine":   "cationic interactions with the binding site environment",
# #         "MichaelAcceptor": "electrophilic reactivity potentially perturbing co-regulators",
# #         "CarboxylicAcid":  "polar anchor potentially impacting AR interactions",
# #         "Sulfonamide":     "multi-H-bond motif aiding binding",
# #         "Ester":           "polar carbonyl/alkoxy influencing fit and polarity",
# #         "Aniline":         "aromatic amine influencing electronics and metabolism",
# #         "Pyridine":        "heteroaromatic HBA aiding orientation",
# #     },
# #     "SR-ARE": {
# #         "Quinone":         "ROS generation consistent with Nrf2/ARE pathway activation",
# #         "MichaelAcceptor": "soft electrophile likely to modify Keap1 cysteines",
# #         "Nitro":           "bioactivation potential contributing to oxidative stress",
# #         "Aniline":         "aromatic amine potentially linked to oxidative pathways",
# #     },
# # }

# # def _kb_phrase(label: str, concept: str) -> str:
# #     lbl_map = CONCEPT_KNOWLEDGE.get(label, {})
# #     if concept in lbl_map: return lbl_map[concept]
# #     return CONCEPT_KNOWLEDGE["_generic"].get(concept, concept)

# # # =========================
# # # TCAV → shap-like adapter
# # # =========================
# # def _concept_importance_for_label(label: str,
# #                                   min_tcav: float = SETTINGS["tcav"]["min_tcav"],
# #                                   max_p: float = SETTINGS["tcav"]["max_p"],
# #                                   top_k: int = SETTINGS["tcav"]["top_k"],
# #                                   include_negative: bool = True,
# #                                   neg_tcav: float = 0.40) -> pd.DataFrame:
# #     sdf = _tcav_df[_tcav_df["label"] == label].copy()
# #     if sdf.empty:
# #         return pd.DataFrame(columns=["feature","shap_value","p_value","q_value","tcav_mean","concept","direction","ci95"])
# #     pos = sdf[(sdf["tcav_mean"] >= float(min_tcav)) & (sdf["p_value"] <= float(max_p))].copy()
# #     pos["direction"] = "↑"
# #     neg = pd.DataFrame()
# #     if include_negative:
# #         neg = sdf[(sdf["tcav_mean"] <= float(neg_tcav)) & (sdf["p_value"] <= float(max_p))].copy()
# #         neg["direction"] = "↓"
# #     keep = pd.concat([pos, neg], ignore_index=True)
# #     if keep.empty:
# #         return pd.DataFrame(columns=["feature","shap_value","p_value","q_value","tcav_mean","concept","direction","ci95"])
# #     keep["shap_value"] = keep["tcav_mean"] - 0.5
# #     keep["feature"] = "CONCEPT_" + keep["concept"].astype(str)
# #     keep = keep[["feature","shap_value","p_value","q_value","tcav_mean","concept","direction","ci95"]]
# #     keep = keep.sort_values(["direction","shap_value","tcav_mean"], ascending=[True, False, False])
# #     top_pos = keep[keep["direction"]=="↑"].head(top_k)
# #     top_neg = keep[keep["direction"]=="↓"].head(2) if include_negative else pd.DataFrame(columns=keep.columns)
# #     out = pd.concat([top_pos, top_neg], ignore_index=True)
# #     return out.reset_index(drop=True)

# # def concept_coverage_summary(label: str,
# #                              min_tcav: float = SETTINGS["tcav"]["min_tcav"],
# #                              max_p: float = SETTINGS["tcav"]["max_p"],
# #                              neg_tcav: float = 0.40) -> Dict[str, int]:
# #     df = _concept_importance_for_label(label, min_tcav=min_tcav, max_p=max_p, top_k=999, include_negative=True, neg_tcav=neg_tcav)
# #     if df.empty: return {"drivers": 0, "counters": 0}
# #     return {"drivers": int((df["direction"]=="↑").sum()),
# #             "counters": int((df["direction"]=="↓").sum())}

# # def _tcav_strength(tcav_mean: float) -> str:
# #     d = abs(tcav_mean - 0.5)
# #     if d >= 0.40:  return "very strong"
# #     if d >= 0.25:  return "strong"
# #     if d >= 0.15:  return "moderate"
# #     return "weak"

# # def _format_concept_bullet(row, label: str) -> str:
# #     fname = row["feature"]; cname = fname.replace("CONCEPT_", "")
# #     sgn = "↑" if row["shap_value"] >= 0 else "↓"
# #     tcav_mean = float(row.get("tcav_mean", row["shap_value"] + 0.5))
# #     strength = _tcav_strength(tcav_mean)
# #     pval = float(row.get("p_value", 1.0))
# #     qval = float(row.get("q_value", np.nan)) if "q_value" in row else None
# #     pq = _pretty_p(pval, qval if (qval == qval) else None)
# #     desc = _kb_phrase(label, cname)
# #     ci = _parse_ci95(row.get("ci95", None))
# #     ci_txt = f" [{ci[0]:.2f}–{ci[1]:.2f}]" if ci else ""
# #     return f"- **{cname}** ({sgn}, {strength}, TCAV={tcav_mean:.2f}{ci_txt}, {pq}) — {desc}"

# # # =========================
# # # SMARTS helpers (optional rules)
# # # =========================
# # SMARTS_PATH = META_DIR / "smarts_rules_final.json"
# # SMARTS_RULES = json.load(open(SMARTS_PATH)) if SMARTS_PATH.exists() else {}

# # def match_toxicophores_with_explanations(smiles: str, label: Optional[str] = None):
# #     mol = Chem.MolFromSmiles(smiles)
# #     if mol is None or not SMARTS_RULES: return []
# #     label_rules = SMARTS_RULES.get(label, []) if label else [r for rules in SMARTS_RULES.values() for r in rules]
# #     hits = []
# #     for rule in label_rules:
# #         patt = Chem.MolFromSmarts(rule["smarts"])
# #         if patt and mol.HasSubstructMatch(patt):
# #             hits.append({"name": rule["name"], "explanation": rule["explanation"]})
# #     return hits

# # def highlight_toxicophores(smiles: str):
# #     mol = Chem.MolFromSmiles(smiles)
# #     if mol is None: raise ValueError("Invalid SMILES structure.")
# #     AllChem.Compute2DCoords(mol)
# #     highlight_atoms = set(); matched = []
# #     all_rules = [r for rules in SMARTS_RULES.values() for r in rules] if SMARTS_RULES else []
# #     for rule in all_rules:
# #         patt = Chem.MolFromSmarts(rule["smarts"])
# #         if not patt: continue
# #         matches = mol.GetSubstructMatches(patt)
# #         if matches:
# #             matched.append(f"☣️ **{rule['name']}**: {rule['explanation']}")
# #             for m in matches: highlight_atoms.update(m)
# #     drawer = rdMolDraw2D.MolDraw2DCairo(500, 300)
# #     drawer.DrawMolecule(mol, highlightAtoms=list(highlight_atoms),
# #                         legend="Matched Toxicophores" if matched else "No toxicophores found")
# #     drawer.FinishDrawing()
# #     img = Image.open(io.BytesIO(drawer.GetDrawingText()))
# #     return matched, img

# # # =========================
# # # Main API used by app.py
# # # =========================
# # def predict_and_explain_all_labels(smiles: str,
# #                                    min_tcav: float = SETTINGS["tcav"]["min_tcav"],
# #                                    max_p: float = SETTINGS["tcav"]["max_p"],
# #                                    top_k: int = SETTINGS["tcav"]["top_k"]):
# #     probs = predict_probs(smiles)
# #     predicted = label_hits_from_probs(probs, thresholds)
# #     explanations = {}
# #     for label in predicted:
# #         shap_df = _concept_importance_for_label(label, min_tcav=min_tcav, max_p=max_p, top_k=top_k)
# #         explanations[label] = {
# #             "prob": float(probs[label]),
# #             "threshold": float(thresholds.get(label, 0.5)),
# #             "pred_score": float(probs[label]),
# #             "shap_df": shap_df,
# #             "coverage": concept_coverage_summary(label, min_tcav=min_tcav, max_p=max_p),
# #         }
# #     return {"smiles": smiles, "predicted_labels": predicted, "explanations": explanations}

# # def generate_toxicity_radar(smiles: str, results: dict):
# #     probs = [results["explanations"].get(lbl, {}).get("prob", 0.0) for lbl in label_cols]
# #     fig = go.Figure()
# #     fig.add_trace(go.Scatterpolar(
# #         r=probs + [probs[0]], theta=label_cols + [label_cols[0]],
# #         fill='toself', name='Predicted Toxicity',
# #         line=dict(color='crimson'), marker=dict(symbol='circle'),
# #         hovertemplate='%{theta}<br>Prob: %{r:.2f}<extra></extra>',
# #     ))
# #     fig.update_layout(
# #         polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickfont_size=10)),
# #         showlegend=False, title=f"Toxicity Radar for: {smiles}",
# #         margin=dict(l=30, r=30, t=50, b=30), height=500
# #     )
# #     return fig

# # def plot_shap_bar(shap_df: pd.DataFrame, top: int = 10):
# #     df = shap_df.copy()
# #     df = df.reindex(df["shap_value"].abs().sort_values(ascending=False).index).head(top)
# #     fig = px.bar(df, x="shap_value", y="feature", orientation="h")
# #     fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
# #     return fig

# # def explain_near_misses(smiles: str, top_n: int = 3, top_features: int = 3):
# #     probs = predict_probs(smiles); rows = []
# #     for lbl in label_cols:
# #         p = float(probs[lbl]); thr = float(thresholds.get(lbl, 0.5))
# #         if p >= thr: continue
# #         sdf = _tcav_df[_tcav_df["label"] == lbl].copy()
# #         if sdf.empty: continue
# #         sdf = sdf.sort_values("tcav_mean", ascending=True)
# #         neg = sdf.head(int(top_features)).copy()
# #         neg["feature"] = "CONCEPT_" + neg["concept"].astype(str)
# #         neg["shap_value"] = neg["tcav_mean"] - 0.5
# #         rows.append((lbl, p, thr, neg[["feature","shap_value"]]))
# #     rows.sort(key=lambda t: t[1], reverse=True)
# #     return rows[:int(top_n)]

# # def _kb_narrative(label: str, shap_df: pd.DataFrame, max_items: int = 3) -> str:
# #     if shap_df.empty:
# #         return f"No strong concept evidence passed filters for {label}."
# #     pos = shap_df[shap_df["shap_value"] > 0].sort_values("shap_value", ascending=False).head(max_items)
# #     neg = shap_df[shap_df["shap_value"] < 0].sort_values("shap_value").head(1)
# #     bits = []
# #     if not pos.empty:
# #         phrases = [_kb_phrase(label, f.replace("CONCEPT_","")) for f in pos["feature"]]
# #         bits.append(" ; ".join(phrases))
# #     if not neg.empty:
# #         phrases_n = [_kb_phrase(label, f.replace("CONCEPT_","")) for f in neg["feature"]]
# #         bits.append(f"while features such as {', '.join(phrases_n)} appear inversely associated")
# #     if bits:
# #         return f"Predicted {label} toxicity is supported by {bits[0]}" + (f", {bits[1]}." if len(bits)>1 else ".")
# #     return f"No strong concept evidence passed filters for {label}."

# # def generate_mechanistic_report(label: str,
# #                                 shap_df: pd.DataFrame,
# #                                 prob: float,
# #                                 threshold: float,
# #                                 smiles: str,
# #                                 top_k_pos: int = 5,
# #                                 top_k_neg: int = 3) -> str:
# #     cov = concept_coverage_summary(label)
# #     cov_note = f"Drivers: {cov.get('drivers',0)}, Counter-evidence: {cov.get('counters',0)}"

# #     lines = [
# #         f"### 🔍 {label} — Mechanistic Report\n",
# #         f"✅ **Prediction confidence**: `{prob:.2f}` (threshold = `{threshold:.2f}`)",
# #         f"🧭 **Evidence summary**: {cov_note}",
# #         "",
# #         _kb_narrative(label, shap_df, max_items=min(3, top_k_pos)),
# #         "",
# #         "📊 **Contributing Concepts (TCAV):**"
# #     ]

# #     df = shap_df.copy()
# #     drivers = df[df["shap_value"] > 0].sort_values("shap_value", ascending=False).head(int(top_k_pos))
# #     counters = df[df["shap_value"] < 0].sort_values("shap_value", ascending=True).head(int(top_k_neg))

# #     if drivers.empty:
# #         lines.append("- No positive concept drivers passed filters.")
# #     else:
# #         for _, row in drivers.iterrows():
# #             lines.append(_format_concept_bullet(row, label))

# #     lines.append("")
# #     lines.append("🚫 **Counter-evidence (TCAV):**")
# #     if counters.empty:
# #         lines.append("- No convincing counter-evidence under current filters.")
# #     else:
# #         for _, row in counters.iterrows():
# #             lines.append(_format_concept_bullet(row, label))

# #     lines.append("\n---")
# #     return "\n".join(lines)

# # def summarize_prediction(result: dict) -> str:
# #     smiles = result["smiles"]; predicted = result["predicted_labels"]
# #     if not predicted:
# #         return f"🔬 The drug (SMILES: `{smiles}`) has **no endpoints above threshold**."
# #     ordered = sorted(predicted, key=lambda lb: result["explanations"][lb]["pred_score"], reverse=True)
# #     return f"🔬 The drug (SMILES: `{smiles}`) is predicted positive for: **{', '.join(ordered)}**."
