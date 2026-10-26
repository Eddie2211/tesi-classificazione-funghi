# Confronto tra Modelli di Classificazione Piatti e Gerarchici
**Candidato:** Eddie Hartard Arata  
**Relatori:** Prof. Osvaldo Gervasi, Dr. Damiano Perri  
**Dipartimento di Matematica e Informatica — Università degli Studi di Perugia**

---

## Descrizione del Progetto
Questo repository contiene la pipeline software, gli script sperimentali e la guida alla riproducibilità relativi alla tesi di laurea in Informatica.

L'obiettivo dello studio è il confronto tra un approccio di classificazione piatto (**FLAT**) e uno gerarchico top-down (**LCPN - Local Classifier Per Node**) applicati alla classificazione tassonomica delle sequenze ITS del DNA fungino lungo 6 livelli gerarchici: Phylum -> Classe -> Ordine -> Famiglia -> Genere -> Specie.

### Aspetti Chiave del Workflow
- **Rappresentazione Alignment-Free:** scomposizione in $k$-mers con $k=6$ e rimozione ambiguità IUPAC.
- **Vettorizzazione:** TF-IDF globale su matrici sparse con vocabolario estratto solo dal training set.
- **Classificatore:** XGBoost con algoritmo *hist* per la gestione nativa della sparsità.
- **Soglia di Confidenza:** vincolo di addestramento a >= 10 campioni (`MIN_SAMPLES = 10`).
- **Explainable AI (XAI):** analisi con SHAP TreeExplainer nodo per nodo e correlazione con siti di restrizione enzimatici noti (es. SpeI, HindIII).

---

## Requisiti e Configurazione Ambiente

1. **Clonare il repository:**
```bash
git clone https://github.com/Eddie2211/tesi-classificazione-funghi.git
cd tesi-classificazione-funghi
```
2. **Creare e attivare l'ambiente virtuale:**
```bash
python -m venv venv
venv\Scripts\activate
```
3. **Installare requirements.txt:**
```bash
pip install -r requirements.txt
```

---

## Struttura del Repository

```text
├── train_hierarchical.py            # Training LCPN gerarchico e salvataggio nodi XGBoost
├── compare_flat_hier.py             # Benchmark FLAT vs LCPN e calcolo metriche (hF, TIE)
├── shap_analysis.py                 # Calcolo valori SHAP e mapping con enzimi di restrizione
├── predict.py                       # Modulo interattivo CLI / batch per file FASTA
├── data_utils.py                    # Routine di deduplica, rimozione leakage e normalizzazione key
├── kmers.py                         # Generazione dei k-mers (k=6) e pulizia caratteri IUPAC
├── preprocessing.py                 # Filtri di qualità sequenze (200-1000 bp) e parser FASTA
├── variables.py                     # Parametri globali, percorsi e configurazioni XGBoost
├── requirements.txt                 # Dipendenze e librerie Python richieste
├── comparison_outputs_fixed10/      # Report JSON con le metriche sperimentali
│   └── comparison_report.json
├── shap_outputs/                    # Risultati XAI, grafici PNG e hit dei siti di restrizione
│   ├── shap_topk_per_node.csv
│   ├── shap_restriction_hits.csv
│   └── plots/
└── README.md                        # Guida alla riproducibilità e documentazione
```

---

## Download di Dataset e Modelli Addestrati

A causa dei limiti di dimensione dei file su GitHub (> 25-100 MB), i dataset completi e i pesi dei modelli serializzati sono ospitati esternamente su cloud storage.

- **Link archivio dataset** `https://drive.google.com/drive/folders/1jB_UCJPqIfEUASs94p9P_bJTh1o7uAZ5?usp=drive_link`

### Istruzioni di posizionamento:
Scarica l'archivio compresso ed estrai le due cartelle direttamente nella cartella principale del progetto:
`split_data/` (contiene `train_dataset.csv` e `test_dataset.csv`)

La root del repository dovrà presentarsi così:
```text
tesi-classificazione-funghi/
├── models_hier_fixed10/
├── split_data/
├── comparison_outputs_fixed10/
├── shap_outputs/
├── train_hierarchical.py
└── ...
```
---

## Guida all'Esecuzione (Workflow Completo)

Gli script sono numerati in ordine sequenziale per garantire la piena riproducibilità degli esperimenti:

### 1. Addestramento del Modello Gerarchico LCPN
Se si desidera ri-addestrare l'intera gerarchia di classificatori partendo dal dataset estratto in `split_data/`:
```bash
python train_hierarchical.py
```
- Esegue la deduplica dei campioni e la normalizzazione delle chiavi tassonomiche tramite data_utils.py.
- Genera i 6-mers ed estrae la matrice TF-IDF globale calcolata esclusivamente sul training set.
- Addestra i classificatori XGBoost (hist) per ciascun nodo che soddisfa il vincolo MIN_SAMPLES = 10.
- Salva i file serializzati .pkl nella cartella models_hier_fixed10/.
## 2. Confronto Modello FLAT vs Gerarchico
Per eseguire il confronto prestazionale tra il modello globale FLAT e l'approccio top-down gerarchico:
```bash
python compare_flat_hier.py
```
- Valuta le predizioni sul test set indipendente split_data/test_dataset.csv, eliminando preventivamente qualsiasi overlap con il train set per scongiurare il data leakage.
- Addestra la baseline FLAT (200 alberi) ed esegue l'inferenza condizionata gerarchica.Calcola le metriche: Coverage, accuratezza su righe coperte vs totale, Precisione/Richiamo/F-Score gerarchici ($hP$, $hR$, $hF$) e la distanza d'errore media sull'albero (Tree Induced Error - TIE).
- Esporta il report completo delle metriche in comparison_outputs_fixed10/comparison_report.json

## 3. Analisi di Spiegabilià con SHAP
Per analizzare il contributo delle feature biologiche e la correlazione con i siti enzimatici:
```bash
python shap_analysis.py
```
Calcola tramite shap.TreeExplainer i valori SHAP per ciascun classificatore locale.Identifica i $k$-mers con il maggior peso decisionale e ne analizza la corrispondenza con siti noti di restrizione.
Salva i file riassuntivi e i grafici a barre in shap_outputs/.
