import pandas as pd
import preprocessing

LEVELS = ["phylum", "class", "order", "family", "genus", "species"]


def _base_clean(df):
    df = df.copy()
    df["sequence"] = df["sequence"].astype(str).str.strip()
    df["length"] = df["sequence"].str.len()
    df = preprocessing.length_control(df)
    for col in LEVELS:
        if col not in df.columns:
            df[col] = "Unknown"
        df[col] = df[col].fillna("Unknown").astype(str)
    return df.reset_index(drop=True)


def clean_train(df):
    """Pulisce il TRAIN, costruisce key = genus_epiteto, rimuove conflitti e duplicati."""
    df = _base_clean(df)
    df["key"] = (df["genus"].astype(str) + "_" + df["species"].astype(str)).str.strip()

    nun = df.groupby("sequence")["key"].transform("nunique")
    n_conf = int((nun > 1).sum())
    df = df[nun == 1].copy()

    before = len(df)
    df = df.drop_duplicates(subset="sequence", keep="first").reset_index(drop=True)
    n_dup = before - len(df)
    return df, n_conf, n_dup


def clean_test(df, train_sequences):
    """Pulisce il TEST, key = species (gia genere_specie), rimuove overlap col train e duplicati."""
    df = _base_clean(df)
    df["key"] = df["species"].astype(str).str.strip()

    before = len(df)
    df = df[~df["sequence"].isin(train_sequences)].copy()
    n_overlap = before - len(df)

    df = df.drop_duplicates(subset="sequence", keep="first").reset_index(drop=True)
    return df, n_overlap
