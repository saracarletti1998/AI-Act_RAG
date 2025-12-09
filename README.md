# 🇪🇺 Assessment delle Pipeline RAG sull'AI Act Europeo

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![RAGAS](https://img.shields.io/badge/Evaluation-RAGAS-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

Questo repository contiene il codice sorgente e il dataset sperimentale sviluppati per il progetto denominato **"Assessment delle pipeline RAG con LLM: Analisi quantitativa sull'interpretazione del Regolamento Europeo AI Act"**.

Il progetto propone un approccio metodologico per valutare l'affidabilità degli LLM nell'interpretazione di testi normativi complessi, utilizzando l'**AI Act (Regolamento UE 2024/1689)** come caso di studio.

---

## 🎯 OBIETTIVI

L'utilizzo degli LLM in ambito legale è promettente ma rischioso a causa delle "allucinazioni" generative e delle difficoltà di retrieval. Questo lavoro affronta il problema implementando e valutando una pipeline **Retrieval-Augmented Generation (RAG)** applicata al corpus normativo dell'EU AI Act. 
L'obiettivo è stato condurre un'analisi comparativa rigorosa tra modelli GPT-4o, Claude Sonnet 4.5, Mistral, LLaMA 3 e DeepSeek, misurandone l'affidabilità tramite il framework RAGAS.

Le principali attività svolte includono:
1.  **Costruzione di un Corpus Normativo:** Preprocessing e segmentazione (chunking) del testo integrale dell'AI Act per ottimizzare il recupero semantico.
2.  **Implementazione RAG:** Sviluppo di una pipeline end-to-end che integra un *Vector Store* (FAISS) per il retrieval e diversi LLM per la generazione.
3.  **Benchmarking Comparativo:** Confronto diretto tra modelli **Closed-Source** (GPT-4o, Claude 4.5 Sonnet, Mistral) e modelli **Open-Source** (LLaMA 3, DeepSeek) a parità di condizioni.
4.  **Valutazione Quantitativa:** Utilizzo del framework **RAGAS** per misurare metriche oggettive come *Faithfulness* (aderenza della risposta al contenuto dei chunk),*Context Precision* (quanto della risposta deriva realmente dal contesto), *Answer Relevancy* (pertinenza della risposta alla domanda) e *Context Recall* (quanto i chunk recuperati contengono le informazioni utili).

---

## 📊 PRINCIPALI RISULTATI

L'analisi sperimentale, condotta su un dataset di domande giuridiche e "Golden Answers", ha evidenziato i seguenti trend:

## 🗒️ Tabelle

L'analisi sperimentale, condotta su un dataset di domande giuridiche e "Golden Answers", ha evidenziato i seguenti trend:

| Modello | Tipo | Answer Relevancy | Context Precision | Context Recall | Faithfulness | Analisi Sintetica |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **GPT-4o mini** | Closed | **0.914** | 0.897 | 0.796 | **0.968** | Il modello più solido. Eccelle nel bilanciare precisione nel recupero e fluidità nella generazione, con allucinazioni quasi assenti. |
| **DeepSeek 7B** | Open | 0.912 | 0.903 | 0.769 | 0.923 | Sorprendente performance per un modello open-source 7B, che eguaglia GPT-4o nella rilevanza delle risposte. |
| **Claude Sonnet** | Closed | 0.835 | 0.900 | 0.701 | 0.947 | Estremamente cauto e fedele (*Faithfulness* alta), ma tende a sintetizzare eccessivamente, riducendo la recall delle informazioni. |
| **Mistral Small** | Closed | 0.688 | **0.917** | 0.769 | 0.943 | Molto preciso nel selezionare le fonti (*Context Precision* alta) e fedele al testo, ma penalizzato da risposte spesso troppo vaghe o incomplete. |
| **LLaMA 3 8B** | Open | 0.825 | n.d. | **0.922** | 0.769 | Ottimo nel recuperare i documenti giusti (*Context Recall* alta), ma fatica a rimanere fedele al testo, introducendo inesattezze esterne. |

## 📊 Grafici


<img width="871" height="519" alt="image" src="https://github.com/user-attachments/assets/102f5590-7ea6-4b1f-b364-23e64b472462" />

<img width="585" height="466" alt="image" src="https://github.com/user-attachments/assets/1029e89b-d6ee-4082-bd61-85f5fc26d93a" />


**Conclusione:** Mentre i modelli proprietari offrono ancora le migliori garanzie di sicurezza per l'ambito legale, i modelli open-source (in particolare DeepSeek) mostrano una maturità tale da poter essere impiegati in scenari di assistenza normativa con supervisione umana.

---

## 📂 Struttura del progetto

```text
├── data/                   # Gestione dei Dati
│   ├── raw/                # Testo grezzo dell'AI Act (ai_act_en.txt)
│   ├── eval/               # Dataset di valutazione (domande + gold answers)
│   └── processed/          # Artefatti generati (chunks, database vettoriale)
│
├── src/                    # Codice Sorgente
│   ├── config.py           # Parametri globali (chunk size 512, overlap 64)
│   ├── prepare_corpus.py   # Script di pulizia e segmentazione del testo
│   ├── build_vector_store.py # Creazione dell'indice semantico FAISS
│   ├── rag_pipeline.py     # Logica RAG (Retrieval + Generazione Prompt)
│   ├── llm_*.py            # Classi wrapper per i vari modelli (OpenAI, HuggingFace, ecc.)
│   ├── run_*_experiment.py # Script per eseguire i test sui singoli modelli
│   └── run_ragas_*.py      # Script di valutazione automatica delle metriche
│
├── requirements.txt        # Dipendenze Python necessarie


