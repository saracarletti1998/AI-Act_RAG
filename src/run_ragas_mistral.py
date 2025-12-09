# src/run_ragas_mistral.py

import os

# ⚠️ Importantissimo: settiamo questa variabile PRIMA di importare ragas/git
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

import json

from dotenv import load_dotenv
from datasets import Dataset

from config import PROJECT_ROOT

from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

# File dei risultati generati da run_mistral_experiment.py
RESULTS_FILE = PROJECT_ROOT / "data" / "eval" / "results_mistral_api.jsonl"


def load_results_for_ragas():
    """
    Carica il file results_mistral_api.jsonl e lo converte
    in una lista di dict con i campi che RAGAS si aspetta.

    Dal record Mistral abbiamo:
      - question
      - gold_answer
      - model_answer
      - contexts

    Li mappiamo a:
      - question
      - answer        (risposta del modello)
      - response      (alias richiesto da answer_relevancy)
      - contexts      (lista di stringhe)
      - ground_truth  (risposta gold)
      - reference     (stessa cosa della gold)
    """
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(f"File risultati non trovato: {RESULTS_FILE}")

    examples = []

    with RESULTS_FILE.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)

            question = data["question"]
            model_answer = data["model_answer"]   # 👈 risposta di Mistral salvata dal tuo script
            contexts = data["contexts"]
            gold = data["gold_answer"]

            # Sanity check minimale
            if model_answer is None:
                print(f"[WARN] Riga {idx}: model_answer è None, salto l'esempio.")
                continue

            examples.append(
                {
                    "question": question,
                    "answer": model_answer,
                    "response": model_answer,   # 👈 per soddisfare answer_relevancy nella tua versione di RAGAS
                    "contexts": contexts,
                    "ground_truth": gold,
                    "reference": gold,
                }
            )

    print(f"Caricati {len(examples)} esempi dai risultati Mistral.")
    return examples


def main():
    # 1️⃣ Carichiamo i risultati del modello
    examples = load_results_for_ragas()

    if not examples:
        raise ValueError("Nessun esempio valido caricato da results_mistral_api.jsonl.")

    # 2️⃣ Dataset HuggingFace
    dataset = Dataset.from_list(examples)

    print("[run_ragas_mistral] Colonne del dataset:", dataset.column_names)
    print("[run_ragas_mistral] Prima riga:", dataset[0])

    # 3️⃣ Definiamo l'LLM "giudice" e le embeddings per RAGAS
    judge_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
    )

    judge_embeddings = OpenAIEmbeddings()

    # 4️⃣ Selezioniamo le metriche RAGAS che vogliamo calcolare
    metrics = [
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    ]

    print("Eseguo valutazione RAGAS per Mistral con GPT-4o-mini come LLM giudice...")

    # 5️⃣ Valutazione
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    # result è un EvaluationResult: usiamo to_pandas()
    df = result.to_pandas()

    print("\nPrime righe del DataFrame RAGAS (Mistral):")
    print(df.head())

    # 6️⃣ Medie delle metriche
    mean_scores = df.mean(numeric_only=True)

    print("\n=== RISULTATI RAGAS MISTRAL (media sui casi) ===")
    for metric_name, value in mean_scores.items():
        print(f"{metric_name}: {value:.4f}")

    # 7️⃣ Salviamo le medie in JSON
    out_file = PROJECT_ROOT / "data" / "eval" / "ragas_mistral.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with out_file.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metric_means": {k: float(v) for k, v in mean_scores.items()},
                "num_samples": len(df),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n✅ Risultati RAGAS Mistral salvati in: {out_file}")


if __name__ == "__main__":
    main()
