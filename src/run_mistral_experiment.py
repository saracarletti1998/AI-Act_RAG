# src/run_mistral_experiment.py

import json
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv

from config import PROJECT_ROOT
from rag_pipeline import answer_question
from llm_mistral_api import MistralLLMClient

load_dotenv()

EVAL_FILE = PROJECT_ROOT / "data" / "eval" / "ai_act_eval.jsonl"
RESULTS_FILE = PROJECT_ROOT / "data" / "eval" / "results_mistral_api.jsonl"


def load_eval_dataset() -> List[Dict]:
    """
    Carica dataset domande-risposte in formato JSONL.
    Se manca 'id', lo aggiunge automaticamente.
    """
    if not EVAL_FILE.exists():
        raise FileNotFoundError(f"File di valutazione non trovato: {EVAL_FILE}")

    examples: List[Dict] = []
    with EVAL_FILE.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            # Se non esiste un ID, lo aggiungiamo automaticamente
            if "id" not in data:
                data["id"] = idx

            examples.append(data)

    print(f"Caricati {len(examples)} esempi di valutazione.")
    return examples


def main():
    # 1) Carica dataset
    eval_examples = load_eval_dataset()

    # 2) Inizializza LLM Mistral
    llm = MistralLLMClient(model_name="mistral-small-latest")

    # 3) Per ogni domanda, esegui la pipeline RAG
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    print(f"Scriverò i risultati in: {RESULTS_FILE}")

    with RESULTS_FILE.open("w", encoding="utf-8") as f_out:
        for ex in eval_examples:
            qid = ex["id"]
            question = ex["question"]
            gold_answer = ex["answer"]

            print(f"\n=== MISTRAL – ESEMPIO {qid} ===")
            print(f"Q: {question}")

            try:
                model_answer, contexts = answer_question(llm, question, top_k=5)
            except Exception as e:
                print(f"Errore durante la generazione per id={qid}: {e}")
                model_answer = f"Errore Mistral: {e}"
                contexts = []

            # Record per RAGAS / analisi
            record = {
                "id": qid,
                "question": question,
                "gold_answer": gold_answer,
                "model_answer": model_answer,
                "contexts": [c["text"] for c in contexts],
            }

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

            print("Mistral answer:")
            print(model_answer[:400], "...")
            print("-" * 60)

    print(f"\n✅ Risultati Mistral salvati in: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
