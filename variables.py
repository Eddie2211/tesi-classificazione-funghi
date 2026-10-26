# Percorsi dei principali file che verranno utilizzati
FASTA_FILE = "dataset/CBSITS.fasta"
CLASS_FILE = "dataset/CBSITS.current.classification"

FASTA_FILEITS= "newdataset/2025_12_CBS_ITS/2025_12_CBS_ITS.fasta_reconstructed.fasta"
FASTA_FILE26S= "newdataset/2025_12_CBS_26S/2025_12_CBS_26S.fasta_reconstructed.fasta"

file_crypto= "dataset/cryptococcus.txt"
file_cerevisiae="dataset/scerevisiae.txt"
file_candida= "dataset/Candidaorthopsilosis.txt"
file_gatti="dataset/cryptogatti.txt"
file_pastorianus="dataset/pastorianus.txt"

TRAIN_DATASET_CSV = "split_data/train_dataset.csv"
# Parametri del modello
K_MER_SIZE = 6

# Mappa IUPAC modificata per "azzerare" le ambiguità
# Tutte le lettere ambigue diventano '0'
IUPAC_ZERO_MAP = {
    "A": "A",
    "C": "C",
    "G": "G",
    "T": "T",
    "R": "0",
    "Y": "0",
    "S": "0",
    "W": "0",
    "K": "0",
    "M": "0",
    "B": "0",
    "D": "0",
    "H": "0",
    "V": "0",
    "N": "0"
}

# --- PARAMETRI XGBOOST ---
# Dizionario di configurazione per il classificatore
XGB_PARAMS_CV_A = {
    "n_estimators": 200,       # Numero massimo di alberi
    "learning_rate": 0.1,      # Velocità di apprendimento (più basso = più robusto ma lento)
    "max_depth": 7,             # Profondità massima degli alberi (evita overfitting)
    "subsample": 0.8,           # Frazione di dati usata per ogni albero
    "colsample_bytree": 0.8,    # Frazione di feature (k-mers) usata per albero
    "objective": "multi:softprob", # Output: probabilità per ogni classe
    "n_jobs": -1,               # Usa tutti i core della CPU
    "random_state": 7,          # Riproducibilità
    "eval_metric": "mlogloss",  # Metrica di errore per multiclasse
    "tree_method": "hist"       # Metodo istogramma (molto veloce per grandi dataset)
}
XGB_PARAMS_CV_B = {
    "n_estimators": 180,       # Numero massimo di alberi
    "learning_rate": 0.1,      # Velocità di apprendimento (più basso = più robusto ma lento)
    "max_depth": 6,             # Profondità massima degli alberi (evita overfitting)
    "subsample": 0.8,           # Frazione di dati usata per ogni albero
    "colsample_bytree": 0.8,    # Frazione di feature (k-mers) usata per albero
    "objective": "multi:softprob", # Output: probabilità per ogni classe
    "n_jobs": -1,               # Usa tutti i core della CPU
    "random_state": 7,          # Riproducibilità
    "eval_metric": "mlogloss",  # Metrica di errore per multiclasse
    "tree_method": "hist"       # Metodo istogramma (molto veloce per grandi dataset)
}
XGB_PARAMS_FINAL = {
    "n_estimators": 500,       # Numero massimo di alberi
    "learning_rate": 0.05,      # Velocità di apprendimento (più basso = più robusto ma lento)
    "max_depth": 7,             # Profondità massima degli alberi (evita overfitting)
    "subsample": 0.8,           # Frazione di dati usata per ogni albero
    "colsample_bytree": 0.8,    # Frazione di feature (k-mers) usata per albero
    "objective": "multi:softprob", # Output: probabilità per ogni classe
    "n_jobs": -1,               # Usa tutti i core della CPU
    "random_state": 7,          # Riproducibilità
    "eval_metric": "mlogloss",  # Metrica di errore per multiclasse
    "tree_method": "hist"       # Metodo istogramma (molto veloce per grandi dataset)
}


LEVEL_PARAMS = {
    "phylum": {"max_depth": 4, "learning_rate": 0.1},
    "class":  {"max_depth": 6, "learning_rate": 0.1},
    "order":  {"max_depth": 8, "learning_rate": 0.05},
    "family": {"max_depth": 8, "learning_rate": 0.05},
    "genus":  {"max_depth": 10, "learning_rate": 0.01},
    "species": {"max_depth": 12, "learning_rate": 0.01}
}
