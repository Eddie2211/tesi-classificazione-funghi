# Confronto tra Modelli di Classificazione Piatti e Gerarchici
**Candidato:** Eddie Hartard Arata  
**Relatori:** Prof. Osvaldo Gervasi, Dr. Damiano Perri  
**Dipartimento di Matematica e Informatica — Università degli Studi di Perugia**

---

##Descrizione del Progetto
Questo repository contiene la pipeline software, gli script sperimentali e la guida alla riproducibilità relativi alla tesi di laurea in Informatica.

L'obiettivo dello studio è il confronto tra un approccio di classificazione piatto (**FLAT**) e uno gerarchico top-down (**LCPN - Local Classifier Per Node**) applicati alla classificazione tassonomica delle sequenze ITS del DNA fungino lungo 6 livelli gerarchici:
$$\text{Phylum} \longrightarrow \text{Classe} \longrightarrow \text{Ordine} \longrightarrow \text{Famiglia} \longrightarrow \text{Genere} \longrightarrow \text{Specie}$$

### Aspetti Chiave del Workflow
- **Rappresentazione Alignment-Free:** scomposizione in $k$-mers con $k=6$ e rimozione ambiguità IUPAC.
- **Vettorizzazione:** TF-IDF globale su matrici sparse con vocabolario estratto solo dal training set.
- **Classificatore:** XGBoost con algoritmo *hist* per la gestione nativa della sparsità.
- **Soglia di Confidenza:** vincolo di addestramento a $\ge 10$ campioni (`MIN_SAMPLES = 10`).
- **Explainable AI (XAI):** analisi con SHAP TreeExplainer nodo per nodo e correlazione con siti di restrizione enzimatici noti (es. SpeI, HindIII).

---

##Struttura del Repository

```text
├── 01_train_hierarchical_fixed.py   # Training LCPN gerarchico e salvataggio nodi XGBoost
├── 02_compare_flat_hier_fixed.py    # Benchmark FLAT vs LCPN e calcolo metriche (hF, TIE)
├── 03_shap_analysis.py              # Calcolo valori SHAP e mapping con enzimi di restrizione
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
