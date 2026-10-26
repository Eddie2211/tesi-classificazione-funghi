"""
02_compare_flat_hier_fixed.py
------------------------------------------------------------------
Confronto EQUO FLAT vs gerarchico sul test_dataset SEPARATO.

  valuta sul test_dataset.csv, NON su un re-split del train
  specie = genere_specie coerente per FLAT, gerarchico e y_true
  confronto alla pari: accuracy overall + coverage + covered-accuracy
  metriche gerarchiche della tesi (hP, hR, hF, TIE a pesi unitari)

richiede che 01_train_hierarchical_fixed.py sia gia stato eseguito
------------------------------------------------------------------
"""
import re, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

import kmers, variables, data_utils
from data_utils import LEVELS

MIN_SAMPLES = 10
TRAIN_CSV = variables.TRAIN_DATASET_CSV
TEST_CSV  = "split_data/test_dataset.csv"  
HIER_DIR  = Path("models_hier_fixed10")
OUT_DIR   = Path("comparison_outputs_fixed10"); OUT_DIR.mkdir(exist_ok=True)

def safe_name(v): return re.sub(r"[^a-zA-Z0-9_]", "_", str(v))

# ---------- DATI ----------
print("[1] Caricamento e pulizia dati...")
train = pd.read_csv(TRAIN_CSV)
train, nc, nd = data_utils.clean_train(train)
test = pd.read_csv(TEST_CSV)
test, no = data_utils.clean_test(test, set(train["sequence"]))
print(f"  Train: {len(train):,} (conflitti={nc}, dup={nd}) | Test: {len(test):,} (overlap tolto={no})")

train = kmers.prepare_dataset(train, k=variables.K_MER_SIZE).reset_index(drop=True)
test  = kmers.prepare_dataset(test,  k=variables.K_MER_SIZE).reset_index(drop=True)

y_true = test["key"].astype(str).values   # genere_specie

# ---------- FLAT ----------
print("[2] Training FLAT (su train pulito, etichetta genere_specie)...")
tfidf_flat = TfidfVectorizer()
Xtr = tfidf_flat.fit_transform(train["kmers_text"])
Xte = tfidf_flat.transform(test["kmers_text"])

counts = train["key"].value_counts()
valid = counts[counts >= MIN_SAMPLES].index
mask = train["key"].isin(valid)
params = variables.XGB_PARAMS_FINAL.copy(); params.pop("early_stopping_rounds", None)
params["n_estimators"] = 200      # FLAT limitato: 200 alberi utilizzati
params["max_bin"] = 128           # accelera hist con impatto minimo sull'accuratezza
params["objective"] = "multi:softprob"; params["num_class"] = int(mask.sum() and train["key"][mask].nunique())
le_flat = LabelEncoder(); ytr = le_flat.fit_transform(train["key"][mask])
flat = XGBClassifier(**params); flat.fit(Xtr[mask.values], ytr, verbose=False)
print(f"  Classi FLAT: {train['key'][mask].nunique()}")
flat_pred = le_flat.inverse_transform(flat.predict_proba(Xte).argmax(axis=1))

# ---------- GERARCHICO ----------
print("[3] Caricamento nodi gerarchici...")
hier_tfidf = joblib.load(HIER_DIR / "tfidf_global.pkl")
Xte_h = hier_tfidf.transform(test["kmers_text"])
nodes = {}
if (HIER_DIR/"xgb_root.pkl").exists():
    nodes[("phylum", None)] = (joblib.load(HIER_DIR/"xgb_root.pkl"), joblib.load(HIER_DIR/"le_root.pkl"))
for level in LEVELS[1:]:
    for mp in HIER_DIR.glob(f"xgb_{level}_*.pkl"):
        suf = mp.name[len(f"xgb_{level}_"):-4]
        lp = HIER_DIR / f"le_{level}_{suf}.pkl"
        if lp.exists(): nodes[(level, suf)] = (joblib.load(mp), joblib.load(lp))
print(f"  Nodi: {len(nodes)}")

print("[4] Inferenza gerarchica top-down...")
n = Xte_h.shape[0]
pred_by_level = {lvl: np.array([None]*n, dtype=object) for lvl in LEVELS}
reached = np.zeros(n, dtype=bool)
groups = {None: np.arange(n, dtype=int)}
for level in LEVELS:
    nxt = {}
    for parent, idxs in groups.items():
        if idxs.size == 0: continue
        key = (level, None) if level == "phylum" else (level, safe_name(parent))
        node = nodes.get(key)
        if node is None: continue
        model, le = node
        probs = model.predict_proba(Xte_h[idxs])
        best = probs.argmax(axis=1)
        labels = le.inverse_transform(best)
        pred_by_level[level][idxs] = labels
        if level == "species":
            reached[idxs] = True
        else:
            for lab in np.unique(labels):
                child = idxs[labels == lab]
                nxt[lab] = np.concatenate([nxt.get(lab, np.array([], dtype=int)), child])
    groups = nxt
    if not groups: break

# ricostruzione genere_specie dalla path
hier_pred = np.array([
    f"{pred_by_level['genus'][i]}_{pred_by_level['species'][i]}" if reached[i] else "__NO_PRED__"
    for i in range(n)
])

# ---------- METRICHE ----------
print("[5] Metriche...")
flat_acc = accuracy_score(y_true, flat_pred)
hier_acc = accuracy_score(y_true, hier_pred)
coverage = reached.mean()
valid_m = reached
hier_cov_acc = accuracy_score(y_true[valid_m], hier_pred[valid_m]) if valid_m.any() else 0.0
flat_on_reached = accuracy_score(y_true[valid_m], flat_pred[valid_m]) if valid_m.any() else 0.0

# metriche gerarchiche hP/hR/hF + TIE (pesi unitari)
def true_path(row):
    return [f"{lvl}:{row[lvl]}" for lvl in LEVELS if row[lvl] != "Unknown"]
def pred_path(i):
    return [f"{lvl}:{pred_by_level[lvl][i]}" for lvl in LEVELS if pred_by_level[lvl][i] is not None]

inter = tot_a = tot_b = 0
tie_sum = 0
for i in range(n):
    b = true_path(test.iloc[i]); a = pred_path(i)
    sa, sb = set(a), set(b)
    inter += len(sa & sb); tot_a += len(sa); tot_b += len(sb)
    # TIE unit: (depth_a - lca) + (depth_b - lca), lca = prefisso comune
    lca = 0
    for pa, pb in zip(a, b):
        if pa == pb: lca += 1
        else: break
    tie_sum += (len(a) - lca) + (len(b) - lca)
hP = inter/tot_a if tot_a else 0.0
hR = inter/tot_b if tot_b else 0.0
hF = 2*hP*hR/(hP+hR) if (hP+hR) else 0.0
tie_mean = tie_sum/n if n else 0.0

report = {
    "flat_overall_accuracy": float(flat_acc),
    "hier_overall_accuracy": float(hier_acc),
    "hier_coverage": float(coverage),
    "hier_covered_accuracy": float(hier_cov_acc),
    "flat_accuracy_on_reached": float(flat_on_reached),
    "hP": float(hP), "hR": float(hR), "hF": float(hF),
    "TIE_mean_unit": float(tie_mean),
    "test_size": int(n),
}
with open(OUT_DIR/"comparison_report.json", "w") as f: json.dump(report, f, indent=2)

print("\n=== RISULTATI (test separato, no leakage) ===")
print(f"FLAT accuracy overall:        {flat_acc*100:.2f}%")
print(f"HIER accuracy overall:        {hier_acc*100:.2f}%")
print(f"HIER coverage:                {coverage*100:.1f}%")
print(f"HIER covered accuracy:        {hier_cov_acc*100:.2f}%")
print(f"FLAT accuracy sui raggiunti:  {flat_on_reached*100:.2f}%")
print(f"hP={hP:.3f}  hR={hR:.3f}  hF={hF:.3f}  TIE medio={tie_mean:.3f}")
