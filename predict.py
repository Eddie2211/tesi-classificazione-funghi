import joblib
import variables
import preprocessing
import kmers
import numpy as np
import tkinter as tk
import os
from Bio import SeqIO
import pandas as pd
from tqdm import tqdm
from datetime import datetime

def tqdm(iterator, **kwargs): return iterator

class DNAPredictor:
    def __init__(self, model_dir="models", model_suffix="final"):
        """
        Inizializza la classe caricando il modello addestrato e gli strumenti necessari.
        Nota: Assicurarsi che i nomi dei file coincidano con quelli salvati dopo il training.
        """
        self.suffix=model_suffix
        self.model_dir=model_dir

        f_model = f"{model_dir}/xgb_model{model_suffix}.pkl"
        f_tfidf = f"{model_dir}/tfidf{model_suffix}.pkl"
        f_le = f"{model_dir}/label_encoder{model_suffix}.pkl"
        
        # 1. Carichiamo gli artefatti salvati
        try:
            if not os.path.exists(f_model):
                raise FileNotFoundError(f"File non trovato: {f_model}")
                
            print(f"Caricamento modello {model_suffix} in corso...")
            self.model = joblib.load(f_model)
            self.vectorizer = joblib.load(f_tfidf)
            self.encoder = joblib.load(f_le)
            print(f"Modello {model_suffix} pronto.")
        except Execption as e:
            print(f"Errore caricamento modello '{model_suffix}': {e} Hai eseguito train.py?")
            raise

    def predict(self, dna_sequence, top_k=5):
        """
        Prende una stringa raw di DNA e restituisce la specie predetta e la probabilità.
        """
        # 2. Preprocessing: Stessa logica utilizzata durante il training
        kmers_text = kmers.get_clean_kmers_string(dna_sequence, k=variables.K_MER_SIZE)
        
        # Controllo sicurezza: se la sequenza è vuota o troppo corta
        if not kmers_text:
            return "Sequenza non valida o troppo corta", 0.0

        # 3. Vettorizzazione: Trasforma il testo in numeri utilizzando il vocabolario appreso.
        # Utilizzando quindi transform() e non fit_transform()
        X_input = self.vectorizer.transform([kmers_text])

        # 4. Predizione probabilità con predict_proba()
        probabilities = self.model.predict_proba(X_input)[0]

        #5. Ranking: Trova gli indici delle classi con la probabilità più alta
        n_classes= len(probabilities)
        real_k=min(top_k, n_classes)
        top_indices = np.argsort(probabilities)[-real_k:][::-1]
        
        results = []
        for idx in top_indices:
            # Decodifica (Da numero a nome specie)
            species_name = self.encoder.inverse_transform([idx])[0]
            confidence = probabilities[idx]
            results.append((species_name, confidence))

        return results, True

    def predict_batch(self, sequences, top_k=5):
        clean_kmers_list = []
        valid_indices=[]

        for i, seq in enumerate(sequences):
            k_text=kmers.get_clean_kmers_string(seq, k=variables.K_MER_SIZE)
            if k_text:
                clean_kmers_list.append(k_text)
                valid_indices.append(i)
                
        if not clean_kmers_list:
            return []

        print(f"[{self.suffix}] Vettorizzazione di {len(clean_kmers_list)} sequenze...")
        X_input = self.vectorizer.transform(clean_kmers_list)
        
        # Predizione
        print(f"[{self.suffix}] Calcolo probabilità con XGBoost...")
        all_probabilities = self.model.predict_proba(X_input)
        batch_results = []

        print("Estrazione top-5")
        for i in range(len(all_probabilities)):
            probs=all_probabilities[i]
            n_classes = len(probs)
            real_k=min(top_k, n_classes)

            top_indices = np.argsort(probs)[-real_k:][::-1]

            row_res=[]

            for idx in top_indices:
                # Decodifica (Da numero a nome specie)
                species_name = self.encoder.inverse_transform([idx])[0]
                conf = probs[idx]
                row_res.append((species_name, conf))

            batch_results.append(row_res)
            
        return batch_results

    
    def predict_file(self, fasta_file):
        sequences=[]
        ids=[]
        try:
            print(f"Lettura file {fasta_file}...")
            for rec in SeqIO.parse(fasta_file, "fasta"):
                ids.append(rec.id)
                sequences.append(str(rec.seq).upper())
        except Exception as e:
            print(f"Errore durante la lettura del file:  {e}")
            return
        if not sequences:
            print("Il file è vuoto o non è un FASTA valido")
    
        print(f"Trovate {len(sequences)} sequenze. Effettuando la predizione...")

        try:
            batch_predictions= self.predict_batch(sequences, top_k=5)
        except Exception as e:
            print(f"Errore critico durante la predizione: {e}")
            return


        all_results_structured = []
        for i, preds in enumerate(batch_predictions):
            top_species, top_conf = preds[0]
            all_results_structured.append({
                "id": ids[i],
                "preds": preds,
                "top_species": top_species,
                "top_conf": top_conf,
            })

        # Creiamo una versione ordinata per la stampa (Decrescente per confidenza)
        sorted_results = sorted(all_results_structured, key=lambda x: x['top_conf'], reverse=True)
        
        if len(sequences) >100:
            timestamp= datetime.now().strftime("%Y-%m-%d_%H-%M")
            input_dir = os.path.dirname(os.path.abspath(fasta_file))

            

            # Se l'utente sta usando un file interno al container (es. /dataset)
            # forziamo il salvataggio nella cartella montata /data (se esiste),
            # altrimenti l'utente non vedrebbe mai il file.
            if input_dir.startswith("/dataset") and os.path.exists("/data"):
                output_dir = "/data"
            else:
                output_dir = input_dir

            file_csv = os.path.join(output_dir, f"risultati_{self.suffix}_predizione_{timestamp}.csv")
            
            print(f"\nMolte sequenze rilevate ({len(sequences)}).")
            print(f"Salvataggio risultati in '{file_csv}' invece di stampare a schermo tutto...")
            print("Stampiamo solo le prime 100 predizioni...")
            print("\n" + "="*80)
            print(f"{'ID SEQUENZA':<20} | {'TOP 1 PREDICTION':<30} | {'CONFIDENCE':<10}")
            print("="*80)
    
            for item in sorted_results[:100]:
                print(f"{item['id'][:20]:<20} | {item['top_species']:<30} | {item['top_conf']:.2%}")
            
            print("-" * 80)
            
            # --- DETTAGLIO ORDINATO ---
            print("\n--- DETTAGLIO TOP 5 (Prime 100 Migliori) ---")
            for item in sorted_results[:100]:
                print(f"\n> Sequenza ID: {item['id']}")
                for rank, (specie, conf) in enumerate(item['preds'], 1):
                    bar_len = int(conf * 20)
                    bar = "#" * bar_len
                    print(f"   {rank}. {specie:<30} {conf:6.2%} [{bar:<20}]")
            print("="*80 + "\n")

            data_for_df=[]
            for item in sorted_results:
                row_data = {"ID": item['id']}
                for rank, (sp, conf) in enumerate(item['preds'], 1):
                    row_data[f"Top_{rank}_Species"] = sp
                    row_data[f"Top_{rank}_Conf"] = round(conf * 100, 2)
                data_for_df.append(row_data)
            df = pd.DataFrame(data_for_df)
            df.to_csv(file_csv, index=False)
            print("Salvataggio completato.")
        else:
            # Output a video per pochi risultati
            print("\n" + "="*80)
            print(f"{'ID SEQUENZA':<20} | {'TOP 1 PREDICTION':<30} | {'CONFIDENCE':<10}")
            print("="*80)

            # Anche qui uso sorted_results per mostrare prima le migliori
            for item in sorted_results:
                print(f"{item['id'][:20]:<20} | {item['top_species']:<30} | {item['top_conf']:.2%}")

            print("-" * 80)
            print("\n--- DETTAGLIO TOP 5 ---")
            for item in sorted_results:
                print(f"\n> Sequenza ID: {item['id']}")
                for rank, (specie, conf) in enumerate(item['preds'], 1):
                    bar_len = int(conf * 20)
                    bar = "#" * bar_len
                    print(f"   {rank}. {specie:<30} {conf:6.2%} [{bar:<20}]")
            print("="*80 + "\n")
    
def main():
    loaded_models = {}
    current_model = None
    
    # 1. Scansione cartella modelli (Opzionale, per suggerire i nomi)
    print("\n--- DNA CLASSIFIER MULTI-MODEL ---")
    if os.path.exists("models"):
        files = os.listdir("models")
        # Cerca pattern xgb_species_XXXX.pkl
        suffixes = [f.replace("xgb_model", "").replace(".pkl", "") for f in files if f.startswith("xgb_model")]
        print(f"Suffissi modello trovati in cartella: {suffixes}")
    
    while True:
        status = f"Attivo: {current_model.suffix}" if current_model else "Nessuno"
        print(f"\n[STATO MODELLO: {status}]")
        print("1. Carica un nuovo modello")
        print("2. Seleziona modello già caricato")
        print("3. Esegui predizione (File o Sequenza)")
        print("q. Esci")
        
        choice = input("Scegli > ").strip().lower()
        
        if choice == 'q':
            break
            
        elif choice == '1':
            suffix = input("Inserisci il suffisso del modello ( 'A' o 'B'): ").strip()
            if suffix in loaded_models:
                print("Modello già in memoria. Lo seleziono.")
                current_model = loaded_models[suffix]
            else:
                try:
                    new_predictor = DNAPredictor(model_suffix=suffix)
                    loaded_models[suffix] = new_predictor
                    current_model = new_predictor
                except:
                    print("Impossibile caricare il modello.")

        elif choice == '2':
            if not loaded_models:
                print("Nessun modello caricato!")
                continue
            print("Modelli disponibili:", list(loaded_models.keys()))
            sel = input("Quale attivare? ").strip()
            if sel in loaded_models:
                current_model = loaded_models[sel]
            else:
                print("Modello non trovato in memoria.")

        elif choice == '3':
            if not current_model:
                print("❌ ERRORE: Carica prima un modello (Opzione 1)!")
                continue
                
            inp = input(f"\n[{current_model.suffix}] File FASTA o Sequenza: ").strip()
            if not inp: continue
            
            if os.path.isfile(inp):
                current_model.predict_file(inp)
            else:
                # Predizione singola
                preds, ok = current_model.predict(inp.upper())
                if ok:
                    print("-" * 50)
                    for r, (sp, conf) in enumerate(preds, 1):
                        print(f"{r}. {sp:<30} : {conf:.2%}")
                    print("-" * 50)
                else:
                    print("Sequenza non valida.")



# --- ESEMPIO DI UTILIZZO ---
if __name__ == "__main__":
  main()
