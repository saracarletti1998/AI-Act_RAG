# src/run_ragas_deepseek.py

import os

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

RESULTS_FILE = PROJECT_ROOT / "data" / "eval" / "results_deepseek.jsonl"


def load_results_for_ragas():
    """
    Carica il file results_claude_sonnet.jsonl e lo converte
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
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)

            question = data["question"]
            model_answer = data["model_answer"]
            contexts = data["contexts"]
            gold = data["gold_answer"]

            if model_answer is None:
                print(f"[WARN] Riga {idx}: model_answer è None, salto l'esempio.")
                continue

            examples.append(
                {
                    "question": question,
                    "answer": model_answer,
                    "response": model_answer,
                    "contexts": contexts,
                    "ground_truth": gold,
                    "reference": gold,
                }
            )

    print(f"Caricati {len(examples)} esempi dai risultati DeepSeek.")
    return examples


def main():
    examples = load_results_for_ragas()

    if not examples:
        raise ValueError("Nessun esempio valido caricato da results_deepseek.jsonl.")

    dataset = Dataset.from_list(examples)

    print("[run_ragas_deepseek] Colonne del dataset:", dataset.column_names)
    print("[run_ragas_deepseek] Prima riga:", dataset[0])

    judge_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
    )

    judge_embeddings = OpenAIEmbeddings()

    metrics = [
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    ]

    print("Eseguo valutazione RAGAS per DEEPSEEK con GPT-4o-mini come LLM giudice...")

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    df = result.to_pandas()

    print("\nPrime righe del DataFrame RAGAS (DeepSeek):")
    print(df.head())

    mean_scores = df.mean(numeric_only=True)

    print("\n=== RISULTATI RAGAS DEEPSEEK (media sui casi) ===")
    for metric_name, value in mean_scores.items():
        print(f"{metric_name}: {value:.4f}")

    out_file = PROJECT_ROOT / "data" / "eval" / "ragas_deepseek.json"
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

    print(f"\n✅ Risultati RAGAS DEEPSEEK salvati in: {out_file}")


if __name__ == "__main__":
    main()
