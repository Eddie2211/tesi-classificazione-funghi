from Bio import SeqIO
import pandas as pd
import re

from variables import IUPAC_ZERO_MAP

def length_control(df):
    # aggiungiamo la colonna "length" per una pulizia del dataset successiva in base alla lunghezza della sequenza
    df["length"] = df["sequence"].str.len()
    mask_length=(df["length"] >= 200) & (df["length"] <= 1000)
    df_cleaned = df[mask_length].copy()
    return df_cleaned

def species_control(df, n_species):
    df['species'] = df["species"].fillna("Unknown")
    counts = df["species"].value_counts()
    mask_species = df["species"].isin(counts[counts >= n_species].index)
    df_cleaned = df[mask_species].copy()
    return df_cleaned

def merge_and_load(file1, file2):
    fasta_records=[]
    df_class = pd.read_csv(file2, sep="\t")
    for rec in SeqIO.parse(file1, "fasta"):
        fasta_records.append({
            "id": rec.id,
            "sequence": str(rec.seq).upper()
        })
    df_fasta = pd.DataFrame(fasta_records)
    df_merged = pd.merge(df_class, df_fasta, on="id", how="inner")
    df_merged['species'] = df_merged["species"].fillna("Unknown")
    return df_merged

###################################################################################################################################

def load_data(fasta_path, x):
    rows = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        header = record.description
    
        # taxonomy is after the last "|"
        taxonomy_str = header.split("|")[-1]
    
        tax_dict = {
            "id": record.id + "_" + x,
            "sequence": str(record.seq).upper()
        }
        
        # parse taxonomy
        for item in taxonomy_str.split(";"):
            if "__" in item:
                rank, name = item.split("__", 1)
                tax_dict[rank] = name
    
        rows.append(tax_dict)
        
    df = pd.DataFrame(rows)
    df = df.rename(columns={
        "k": "kingdom",
        "p": "phylum",
        "c": "class",
        "o": "order",
        "f": "family",
        "g": "genus",
        "s": "species"
    })
    return df

def process_fasta_to_df(file_path):
    """
    Legge un file FASTA e lo converte in un DataFrame pandas compatibile
    con la struttura target (id, kingdom, phylum, ..., sequence, length).
    Richiede Biopython.
    """
    
    # Colonne target per garantire la compatibilità
    target_columns = [
        'id', 'kingdom', 'phylum', 'class', 'order', 'family', 
        'genus', 'species', 'strain number/type', 'synonym', 'sequence']
    
    records = []

    stop_words_regex = r'(?:18S|small|large|internal|genes|genomic|rRNA|DNA|sequence|ITS|from|strain|isolate|culture|;|,)'
    
    # Pattern per collezioni note (fallback se non trova "strain ...")
    collections_regex = r'\b(CBS|ATCC|DSM|JCM|NRRL|LEMI|WM)[:\s]+([\w\.]+)'
    
    # Parsing con Biopython
    for record in SeqIO.parse(file_path, "fasta"):
        # Estrazione ID pulito (rimuove eventuali range tipo :18-562)
        clean_id = record.id.split(':')[0]
        
        # Recupero header completo e sequenza
        header = record.description
        full_seq = str(record.seq)
        
        # Parsing dei metadati dall'header
        # Assumiamo formato standard: "ID Genus Species ..."
        parts = header.split()
        genus_name = parts[1] if len(parts) >= 2 else "Unknown"
        species_name = f"{parts[1]} {parts[2]}" if len(parts) >= 3 else "Unknown"
        
        strain = None
        
        # Strategia A: Cerca esplicitamente "strain", "isolate" o "culture"
        # Cattura tutto ciò che segue fino a una stop word o fine riga
        match_explicit = re.search(r'(?:strain|isolate|culture)\s+(?:from\s+)?(.*?)(?=\s+' + stop_words_regex + r'|$)', header, re.IGNORECASE)
        
        if match_explicit:
            candidate = match_explicit.group(1).strip()
            # Pulizia: rimuovi ":" se catturato (es "CBS:123")
            candidate = candidate.replace(':', ' ')
            # Filtro di sicurezza: se è troppo lungo probabilmente ha catturato testo spazzatura
            if len(candidate) < 40:
                strain = candidate
        
        # Strategia B (Fallback o Raffinamento): Cerca collezioni note (es. ATCC 1234)
        # Utile se l'header è tipo "Candida orthopsilosis ATCC 96139..." senza parola "strain"
        if not strain:
            match_collection = re.search(collections_regex, header, re.IGNORECASE)
            if match_collection:
                # Normalizza (es. cbs -> CBS)
                col_name = match_collection.group(1).upper()
                col_id = match_collection.group(2)
                strain = f"{col_name} {col_id}"
        
        records.append({
            'id': clean_id,
            'kingdom': "Fungi",   # Campi vuoti per compatibilità
            'phylum': None,
            'class': None,
            'order': None,
            'family': None,
            'genus': genus_name,
            'species': species_name,
            'strain number/type': strain,
            'synonym': None,
            'sequence': full_seq
        })
        
    # Creazione DataFrame
    df = pd.DataFrame(records)
    
    # Assicura che le colonne siano nell'ordine corretto
    # (Se mancano colonne nel parsing, vengono create come NaN automaticamente)
    for col in target_columns:
        if col not in df.columns:
            df[col] = None
            
    return df[target_columns]

