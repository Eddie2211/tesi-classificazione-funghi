"""
03_shap_analysis.py
------------------------------------------------------------------
Analisi SHAP dei nodi del modello gerarchico LCPN.

Per ogni nodo (classificatore XGBoost) salvato da
01_train_hierarchical_fixed.py, calcola i k-mer piu' influenti
secondo il valore assoluto medio di SHAP.

Output (cartella shap_outputs/):
  - shap_topk_per_node.csv      : top-K k-mer per ogni nodo (con eventuale enzima)
  - shap_restriction_hits.csv   : sottoinsieme in cui un k-mer di alto rango
                                  coincide con un sito di restrizione noto
  - plots/*.png                 : bar plot dei nodi piu' rilevanti
------------------------------------------------------------------
"""
import os, re
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

import kmers, variables, data_utils
from data_utils import LEVELS

# ---------------- CONFIG ----------------
MIN_SAMPLES     = 10
HIER_DIR        = Path("models_hier_fixed10")     # cartella dei nodi
TFIDF_GLOBAL    = HIER_DIR / "tfidf_global.pkl"
OUT_DIR         = Path("shap_outputs")
TOP_K           = 20        # quanti k-mer salvare per nodo
SAMPLE_PER_NODE = 200       # righe campionate per nodo
MAX_PLOTS       = 25        # tetto ai grafici generati
SEED            = 7
rng = np.random.default_rng(SEED)

(OUT_DIR / "plots").mkdir(parents=True, exist_ok=True)

# Siti di restrizione noti
RESTRICTION_SITES = {
    "gaattc": "EcoRI", "aagctt": "HindIII", "ggatcc": "BamHI",
    "gtcgac": "SalI",  "ctgcag": "PstI",    "gagctc": "SacI",
    "tctaga": "XbaI",  "ggtacc": "KpnI",    "cccggg": "SmaI/XmaI",
    "actagt": "SpeI",  "gttaac": "HpaI",    "cagctg": "PvuII",
    "gatatc": "EcoRV", "agatct": "BglII",   "ccatgg": "NcoI",
    "gctagc": "NheI",  "acgcgt": "MluI",    "ttcgaa": "BstBI",
}


def safe_name(v):
    return re.sub(r"[^a-zA-Z0-9_]", "_", str(v))


def plot_node(node_key, kmers_top, vals):
    """Bar plot orizzontale dei k-mer piu' influenti di un nodo."""
    plt.figure(figsize=(7, 5))
    plt.barh(range(len(kmers_top)), vals)
    plt.yticks(range(len(kmers_top)), kmers_top)
    plt.xlabel("|SHAP| medio")
    plt.title(f"Top k-mer  -  {node_key}")
    plt.tight_layout()
    fname = re.sub(r"[^a-zA-Z0-9_]", "_", node_key)[:80]
    plt.savefig(OUT_DIR / "plots" / f"{fname}.png", dpi=130)
    plt.close()


# ---------------- DATI ----------------
print("[1] Caricamento train + pulizia (identica al training)...")
df = pd.read_csv(variables.TRAIN_DATASET_CSV)
df, n_conf, n_dup = data_utils.clean_train(df)
df = kmers.prepare_dataset(df, k=variables.K_MER_SIZE).reset_index(drop=True)
print(f"    Righe: {len(df):,} | conflitti rimossi={n_conf} dup={n_dup}")

print("[2] Caricamento TF-IDF globale (stesso spazio feature del training)...")
tfidf = joblib.load(TFIDF_GLOBAL)
X = tfidf.transform(df["kmers_text"])
feat_names = np.array(tfidf.get_feature_names_out())
n_feat = len(feat_names)
print(f"    Feature (k-mer): {n_feat:,}")

# mappa safe(valore) -> valore reale, per ogni livello (serve a ricostruire il genitore)
safe_lookup = {}
for lvl in LEVELS:
    d = {}
    for v in df[lvl].astype(str).unique():
        d[safe_name(v)] = v
    safe_lookup[lvl] = d


def node_rows(target_level, parent_value):
    """Indici (in df) usati per addestrare il nodo, replicando train_level."""
    if target_level == "phylum":
        sub = np.arange(len(df))
    else:
        pl = LEVELS[LEVELS.index(target_level) - 1]
        sub = np.where(df[pl].astype(str).values == str(parent_value))[0]
    if sub.size == 0:
        return sub
    tl_vals = df[target_level].astype(str).values
    sub = sub[tl_vals[sub] != "Unknown"]
    if sub.size == 0:
        return sub
    vc = pd.Series(tl_vals[sub]).value_counts()
    valid = set(vc[vc >= MIN_SAMPLES].index)
    if not valid:
        return np.array([], dtype=int)
    sub = sub[np.isin(tl_vals[sub], list(valid))]
    return sub


# ---------------- ELENCO NODI ----------------
node_files = []
if (HIER_DIR / "xgb_root.pkl").exists():
    node_files.append(("phylum", None, HIER_DIR / "xgb_root.pkl"))
for lvl in LEVELS[1:]:
    for mp in sorted(HIER_DIR.glob(f"xgb_{lvl}_*.pkl")):
        suf = mp.name[len(f"xgb_{lvl}_"):-4]
        node_files.append((lvl, suf, mp))
print(f"[3] Nodi trovati: {len(node_files)}")

# ---------------- LOOP SHAP ----------------
rows_out = []
plots_done = 0

for tl, suf, mp in node_files:
    if tl == "phylum":
        pv, pl = None, None
    else:
        pl = LEVELS[LEVELS.index(tl) - 1]
        pv = safe_lookup[pl].get(suf, suf)   # ricostruisce il valore reale del genitore

    idx = node_rows(tl, pv)
    if idx.size < 2:
        continue
    if idx.size > SAMPLE_PER_NODE:
        idx = rng.choice(idx, SAMPLE_PER_NODE, replace=False)
    Xd = X[idx].toarray()

    model = joblib.load(mp)
    try:
        le = joblib.load(mp.parent / mp.name.replace("xgb_", "le_", 1))
        n_children = int(len(le.classes_))
    except Exception:
        n_children = None

    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(Xd, check_additivity=False)
    except Exception:
        try:
            explainer = shap.TreeExplainer(model.get_booster())
            sv = explainer.shap_values(Xd, check_additivity=False)
        except Exception as e:
            print(f"    [skip] {mp.name}: {e}")
            continue

    arr = np.abs(np.asarray(sv))
    # asse delle feature = quello di dimensione n_feat
    feat_axes = [a for a, s in enumerate(arr.shape) if s == n_feat]
    if not feat_axes:
        print(f"    [skip] {mp.name}: forma SHAP inattesa {arr.shape}")
        continue
    fa = feat_axes[-1]
    imp = arr.mean(axis=tuple(a for a in range(arr.ndim) if a != fa))  # (n_feat,)

    order = np.argsort(imp)[::-1][:TOP_K]
    node_key = "root" if tl == "phylum" else f"{tl}:{pv}"

    for rank, fi in enumerate(order, 1):
        km = str(feat_names[fi])
        rows_out.append({
            "node_key": node_key,
            "target_level": tl,
            "parent_level": pl,
            "parent_value": pv,
            "n_samples_used": int(len(idx)),
            "n_children": n_children,
            "rank": rank,
            "kmer": km,
            "mean_abs_shap": float(imp[fi]),
            "restriction_enzyme": RESTRICTION_SITES.get(km.lower(), ""),
        })

    #grafico per i nodi con un sito di restrizione nei primi 5
    has_hit = any(RESTRICTION_SITES.get(str(feat_names[fi]).lower()) for fi in order[:5])
    if plots_done < MAX_PLOTS and (has_hit or plots_done < 10):
        top_k = feat_names[order][:15][::-1]
        top_v = imp[order][:15][::-1]
        plot_node(node_key, list(top_k), list(top_v))
        plots_done += 1

    print(f"    ok {node_key}  (n={len(idx)}, classi={n_children})")

# ---------------- SALVATAGGIO ----------------
res = pd.DataFrame(rows_out)
res.to_csv(OUT_DIR / "shap_topk_per_node.csv", index=False)

hits = res[res["restriction_enzyme"] != ""].sort_values(
    ["rank", "mean_abs_shap"], ascending=[True, False]
)
hits.to_csv(OUT_DIR / "shap_restriction_hits.csv", index=False)

print("\n=== FATTO ===")
if len(res):
    print(f"Nodi analizzati: {res['node_key'].nunique()}")
    print(f"Righe totali:    {len(res)}")
    print(f"Hit siti di restrizione (k-mer in top-{TOP_K}): {len(hits)}")
    if len(hits):
        print("  enzimi trovati:", ", ".join(sorted(hits['restriction_enzyme'].unique())))
print(f"Grafici salvati: {plots_done}")
print(f"Output in: {OUT_DIR}/")
