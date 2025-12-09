# src/run_ragas_openai.py

import os

# ⚠️ Importantissimo: settiamo questa variabile PRIMA di importare ragas/git
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

import json
from pathlib import Path

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

# File dei risultati generati da run_openai_experiment.py
RESULTS_FILE = PROJECT_ROOT / "data" / "eval" / "results_openai_gpt4omini.jsonl"


def load_results_for_ragas():
    """
    Carica il file results_openai_gpt4omini.jsonl e lo converte
    in una lista di dict con i campi che RAGAS si aspetta:
      - question
      - answer        (risposta del modello)
      - contexts      (lista di stringhe)
      - ground_truth  (risposta gold)
    """
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(f"File risultati non trovato: {RESULTS_FILE}")

    examples = []

    with RESULTS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            examples.append(
                {
                    "question": data["question"],
                    "answer": data["model_answer"],
                    "contexts": data["contexts"],
                    "ground_truth": data["gold_answer"],
                }
            )

    print(f"Caricati {len(examples)} esempi dai risultati.")
    return examples


def main():
    # 1️⃣ Carichiamo i risultati del modello
    examples = load_results_for_ragas()

    # 2️⃣ Dataset HuggingFace
    dataset = Dataset.from_list(examples)

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

    print("Eseguo valutazione RAGAS con GPT-4o-mini come LLM giudice...")

    # 5️⃣ Valutazione: qui passiamo esplicitamente llm ed embeddings
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    # result è un EvaluationResult: usiamo to_pandas()
    df = result.to_pandas()

    print("\nPrime righe del DataFrame RAGAS:")
    print(df.head())

    # 6️⃣ Medie delle metriche
    mean_scores = df.mean(numeric_only=True)

    print("\n=== RISULTATI RAGAS (media sui casi) ===")
    for metric_name, value in mean_scores.items():
        print(f"{metric_name}: {value:.4f}")

    # 7️⃣ Salviamo le medie in JSON
    out_file = PROJECT_ROOT / "data" / "eval" / "ragas_openai_gpt4omini.json"
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

    print(f"\n✅ Risultati RAGAS salvati in: {out_file}")


if __name__ == "__main__":
    main()
