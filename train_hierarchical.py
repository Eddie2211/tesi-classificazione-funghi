import os, re
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

import kmers, variables, data_utils
from data_utils import LEVELS

MIN_SAMPLES = 10
OUT_DIR = "models_hier_fixed10"

print(f"Caricamento train da '{variables.TRAIN_DATASET_CSV}'...")
df = pd.read_csv(variables.TRAIN_DATASET_CSV)
df, n_conf, n_dup = data_utils.clean_train(df)
print(f"  Pulizia: rimossi {n_conf} conflitti + {n_dup} duplicati")
print(f"  Righe: {len(df):,} | Specie (genere_specie): {df['key'].nunique():,}")

print("Generazione K-mers...")
df = kmers.prepare_dataset(df, k=variables.K_MER_SIZE).reset_index(drop=True)

print("TF-IDF globale (fit solo su train)...")
tv = TfidfVectorizer()
X = tv.fit_transform(df["kmers_text"])
os.makedirs(OUT_DIR, exist_ok=True)
joblib.dump(tv, f"{OUT_DIR}/tfidf_global.pkl")
print(f"  Vocabolario: {len(tv.vocabulary_):,} k-mer")


def train_level(df, X, target_level, parent_mask=None):
    if parent_mask is not None:
        idx = np.where(parent_mask)[0]; node_df, node_X = df.iloc[idx], X[idx]
    else:
        node_df, node_X = df, X
    if len(node_df) == 0: return None, None
    m = node_df[target_level] != "Unknown"; idx = np.where(m)[0]
    node_df, node_X = node_df.iloc[idx], node_X[idx]
    if len(node_df) == 0: return None, None
    counts = node_df[target_level].value_counts()
    valid = counts[counts >= MIN_SAMPLES].index
    if len(valid) == 0: return None, None
    fm = node_df[target_level].isin(valid); idx = np.where(fm)[0]
    tdf, tX = node_df.iloc[idx], node_X[idx]
    if len(tdf) == 0 or tdf[target_level].nunique() < 2: return None, None
    y = tdf[target_level]; le = LabelEncoder(); y_enc = le.fit_transform(y)
    params = variables.XGB_PARAMS_FINAL.copy()
    for k, v in variables.LEVEL_PARAMS.get(target_level, {}).items(): params[k] = v
    params.pop("early_stopping_rounds", None)
    nc = len(np.unique(y_enc))
    if nc == 2:
        params["objective"] = "binary:logistic"; params.pop("num_class", None)
    else:
        params["objective"] = "multi:softprob"; params["num_class"] = nc
    model = XGBClassifier(**params); model.fit(tX, y_enc, verbose=False)
    print(f"    -> {target_level}: {len(tdf)} campioni, {y.nunique()} classi")
    return model, le


print("\n--- ROOT (phylum) ---")
m, le = train_level(df, X, "phylum")
if m is not None:
    joblib.dump(m, f"{OUT_DIR}/xgb_root.pkl"); joblib.dump(le, f"{OUT_DIR}/le_root.pkl")

for i in range(1, len(LEVELS)):
    pl, tl = LEVELS[i-1], LEVELS[i]
    print(f"\n--- {tl.upper()} (parent: {pl}) ---")
    for pv in df[pl].unique():
        if pv in ("Unknown", ""): continue
        mask = df[pl] == pv
        if mask.sum() < MIN_SAMPLES: continue
        model, le = train_level(df, X, tl, parent_mask=mask)
        if model is not None:
            safe = re.sub(r"[^a-zA-Z0-9_]", "_", pv)
            joblib.dump(model, f"{OUT_DIR}/xgb_{tl}_{safe}.pkl")
            joblib.dump(le, f"{OUT_DIR}/le_{tl}_{safe}.pkl")

print("\n[OK] Training completato su train.")
