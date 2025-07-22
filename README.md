# Richter'sPredictor AI

Richter'sPredictor AI è un progetto di Machine Learning per la previsione del livello di danno subito da edifici in seguito a un terremoto.  
L’obiettivo è sviluppare un sistema intelligente in grado di classificare ogni edificio in base al grado di danneggiamento, utilizzando le sue caratteristiche strutturali e geografiche.

Il progetto si basa sui dati reali del terremoto in Nepal del 2015 e mira a fornire uno strumento utile per valutare rapidamente l’impatto di un sisma su larga scala.

## Dataset & Preprocessing

<<<<<<< HEAD
Il progetto si basa sui dati della competizione "Nepal Earthquake Damage Assessment", che mette a disposizione informazioni dettagliate su oltre 260.000 edifici colpiti dal terremoto in Nepal nel 2015. I dati includono caratteristiche strutturali, materiali utilizzati, posizione geografica, e altre informazioni legali e urbanistiche. L’obiettivo è prevedere, per ciascun edificio, il livello di danno subito (1 = basso, 2 = medio, 3 = severo).
=======
Il progetto si basa sui dati della competizione " Nepal Earthquake Damage Assessment ", che mette a disposizione informazioni dettagliate su oltre 260.000 edifici colpiti dal terremoto in Nepal nel 2015. I dati includono caratteristiche strutturali, materiali utilizzati, posizione geografica, e altre informazioni legali e urbanistiche. L’obiettivo è prevedere, per ciascun edificio, il livello di danno subito (1 = basso, 2 = medio, 3 = severo).
>>>>>>> 6a8bc239b15b657aa0370371b6f12e10e6c09ae4

Nella fase di preprocessing, i dati sono stati inizialmente puliti e analizzati per verificarne la completezza. Non sono stati rilevati valori mancanti, né nei dati strutturali (`train_values.csv`), né nelle etichette (`train_labels.csv`) o nei dati di test (`test_values.csv`). I dataset sono stati unificati sulla base del campo `building_id`.

Sono stati poi introdotti alcuni miglioramenti tramite **feature engineering**, con l’obiettivo di arricchire l’informazione originale:
- È stata creata una nuova variabile chiamata `superstructure_quality`, che sintetizza la qualità strutturale degli edifici sulla base dei materiali presenti.
- È stato inoltre calcolato un indicatore di rischio geografico (`geo_risk`), ottenuto analizzando la distribuzione dei danni medi per ciascuna area (`geo_level_1_id`), e poi applicato sia al training che al test set.

Per la preparazione finale dei dati, sono stati identificati e analizzati gli outlier tra le variabili numeriche, e successivamente si è applicato uno **scaling robusto** (`RobustScaler`) per limitarne l'impatto. Le variabili categoriche sono state codificate tramite **One-Hot Encoding**, portando a un’espansione di 38 nuove feature.

Il dataset è stato infine suddiviso in due set: uno di training (80%) e uno di validazione (20%), mantenendo la distribuzione originale delle classi. Su questo set è stata applicata una tecnica di **feature selection automatica**, basata su un classificatore Random Forest, che ha permesso di selezionare le 36 feature più rilevanti per il task.

Sebbene sia stata generata anche una versione dei dati bilanciata tramite **SMOTE**, si è deciso consapevolmente di addestrare i modelli sulla versione **sbilanciata**, più rappresentativa della distribuzione reale dei danni in contesto sismico. Questa scelta riflette il desiderio di mantenere fedeltà alle proporzioni naturali del fenomeno, evitando di alterare il segnale informativo originale attraverso un eccessivo oversampling artificiale.

I dati preprocessati sono stati infine salvati in due versioni:
- una sbilanciata (`preprocessed_unbalanced.pkl`) usata per i modelli principali,
- una bilanciata (`preprocessed_data.pkl`) utile per esperimenti secondari e confronti.


## Modelli e Valutazione

Per affrontare il problema di classificazione del livello di danno, sono stati sperimentati e confrontati diversi modelli supervisionati, selezionati per la loro robustezza e capacità di adattarsi a dataset complessi e sbilanciati.

Il training è stato condotto sulla versione **non bilanciata** del dataset, per mantenere una maggiore fedeltà alla distribuzione reale degli edifici danneggiati. L'accuratezza del modello è stata valutata sul validation set, mantenuto separato durante tutto il processo.

### Modelli utilizzati

- **Random Forest**  
  Utilizzato con `class_weight='balanced'` per compensare lo sbilanciamento delle classi. Offre interpretabilità e buone performance su problemi tabellari.

- **XGBoost**  
  Uno dei modelli principali del progetto. È stato sottoposto a tuning automatico con `GridSearchCV` su iperparametri chiave (n_estimators, max_depth, learning_rate, ecc.). La versione ottimizzata viene riutilizzata in più fasi del progetto.

- **CatBoost**  
  Scelto per la sua gestione nativa delle feature categoriche (anche se in questo caso erano già codificate) e per la sua efficienza su dataset di medie dimensioni.

- **LightGBM**  
  Testato con parametri bilanciati e configurato per la classificazione multiclass. Ha mostrato buone performance, pur con maggiore sensibilità alla scelta degli iperparametri.

- **MLPClassifier (Rete Neurale)**  
  È stato testato un classificatore a reti neurali multi-strato per confrontare approcci non basati su alberi, anche se i risultati si sono rivelati meno competitivi rispetto ai modelli precedenti.

- **OrdinalClassifier (LogisticAT)**  
  Un classificatore ordinale è stato sperimentato per valutare se la natura ordinale del problema (danni leggeri, medi, gravi) potesse migliorare le performance. Anche in questo caso, i risultati sono stati informativi ma inferiori a quelli di XGBoost e RandomForest.

- **BaggingClassifier (CVBagging)**  
  È stato implementato un approccio bagging basato su RandomForest con validazione incrociata. Il modello è stato salvato per essere integrato in ensemble successivi.

- **Ensemble Voting (Bagging VotingClassifier)**  
  Il sistema finale più performante si basa su un ensemble soft voting di più modelli (XGBoost, CatBoost, Random Forest, CVBagging), ognuno allenato con diversi seed. La previsione finale viene calcolata per maggioranza su 5 repliche, riducendo l’overfitting e aumentando la stabilità.

### Metriche di valutazione

La valutazione è stata effettuata sul validation set utilizzando:
- **Accuracy**
- **F1 micro** (metrica ufficiale della competizione)
- **F1 macro** (utile per verificare le performance sulle classi minoritarie)

I risultati migliori sono stati ottenuti con XGBoost e con il Voting Ensemble, con un F1-micro score superiore al 0.73 e un miglioramento sostanziale nella recall della classe minoritaria (`damage_grade = 1`).

Ogni modello stampa a console il suo `classification_report` con precision, recall e F1 per ciascuna classe, utile per analisi più approfondite.

## Struttura del progetto

Il progetto è organizzato in modo modulare per separare chiaramente i dati, il codice, i modelli e i risultati. La seguente struttura permette una gestione efficiente del flusso di lavoro e una facile estensione futura.

```text
Richter_sPredictor_AI/
│
├── data/
│   ├── raw/                             # File originali forniti dalla competizione
│   │   ├── train_values.csv
│   │   ├── train_labels.csv
│   │   └── test_values.csv
│   └── processed_unprocessed/           # Dati preprocessati, con e senza bilanciamento
│       ├── preprocessed_data.pkl
│       └── preprocessed_unbalanced.pkl
│
├── models/                              # Modelli addestrati e salvati
│   ├── best_xgb_model.pkl
│   └── cv_bagging_model.pkl
│
├── scripts/                             # Codice suddiviso per funzione
│   ├── preprocessing.py                 # Tutto il flusso di preprocessing dati
│   ├── train.py                         # Addestramento e valutazione di tutti i modelli principali
│   ├── xgb_tuning.py                    # Tuning automatico di XGBoost via GridSearchCV
│   └── esamble.py                       # Ensemble Voting + Bagging finale
│
├── submission/                          # File CSV da caricare sulla piattaforma (opzionale)
│   └── submission.csv
│
├── requirements.txt                     # (opzionale) dipendenze Python usate nel progetto
└── README.md                            # Documentazione del progetto
```
>>>>>>> 6a8bc239b15b657aa0370371b6f12e10e6c09ae4

Ogni cartella ha un ruolo specifico:
- `data/` contiene sia i dati grezzi sia quelli preprocessati.
- `models/` raccoglie i modelli già allenati, pronti per l'inferenza o l'ensemble.
- `scripts/` include tutti gli script Python modulari, divisi per preprocessing, training, tuning e ensemble.
- `submission/` è il punto di raccolta per i file `.csv` da caricare sulla leaderboard.
- `reports/` può contenere output, grafici o file PDF associati al progetto.

Questa struttura garantisce chiarezza e permette di separare nettamente le fasi del flusso di lavoro, semplificando la collaborazione tra più sviluppatori.


## Come eseguire il progetto

L'intero flusso di lavoro si compone di tre fasi principali: preprocessing, addestramento dei modelli e ensemble finale.

Tutti gli script sono eseguibili direttamente da terminale con Python 3.10+ e richiedono le librerie elencate in `requirements.txt`.

### 1 Preprocessing dei dati

Questo script carica i dati grezzi, applica il feature engineering, esegue la selezione delle feature e salva due versioni dei dati preprocessati (bilanciata e non).

```bash
python scripts/preprocessing.py
```

Output:  
- `data/processed_unprocessed/preprocessed_unbalanced.pkl`  
- `data/processed_unprocessed/preprocessed_data.pkl` *(non usato nei modelli finali)*

---

### 2 Addestramento dei modelli

Questo script esegue l’addestramento di tutti i modelli principali, tra cui Random Forest, CatBoost, LightGBM, XGBoost (con tuning automatico se non già eseguito), e salva anche un modello di bagging.

```bash
python scripts/train.py
```

Output:  
- `models/best_xgb_model.pkl`  
- `models/cv_bagging_model.pkl`

Nota: se il file `best_xgb_model.pkl` esiste già, il tuning viene saltato automaticamente.

---

### 3 Ensemble finale

Questo script esegue un ensemble in bagging di 5 repliche di VotingClassifier (Random Forest, CatBoost, XGBoost, CVBagging) con media ponderata delle predizioni.

```bash
python scripts/esamble.py
```

Output:  
A video verranno stampate le metriche finali (accuracy, F1 micro, F1 macro) del modello ensemble.

---

Per ripetere il flusso completo da zero:
1. Cancella la cartella `models/`
2. Rilancia `preprocessing.py`, poi `train.py`, infine `esamble.py`


## Requisiti

Per eseguire il progetto è necessario Python 3.10+ e le seguenti librerie Python:

```txt
pandas
numpy
scikit-learn
xgboost
catboost
lightgbm
imblearn
mord
joblib
```

Puoi installare tutte le dipendenze in un ambiente virtuale con i seguenti comandi:

```bash
# Crea un nuovo ambiente virtuale
python -m venv terremoto_env

# Attiva l'ambiente (Windows)
terremoto_env\Scripts\activate

# Oppure (macOS/Linux)
source terremoto_env/bin/activate

# Installa i pacchetti
pip install -r requirements.txt
```

Se non hai un file `requirements.txt`, puoi crearlo con:

```bash
pip freeze > requirements.txt
```

Nota: il pacchetto `mord` (per la classificazione ordinale) può essere installato con:

```bash
pip install mord
```

## Autori

Il progetto **Richter'sPredictor AI** è stato sviluppato da:

- **Giuseppe Pio Lioi**  
- **Sebastiano Viglialoro**  
- **Siria Sola**

Progetto svolto nell’ambito del corso di **Fondamenti di Intelligenza Artificiale** – UCBM.
