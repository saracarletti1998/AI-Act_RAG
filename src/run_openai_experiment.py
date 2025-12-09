# src/run_openai_experiment.py

import json
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv

from config import PROJECT_ROOT
from rag_pipeline import answer_question
from llm_openai import OpenAILLMClient

load_dotenv()

EVAL_FILE = PROJECT_ROOT / "data" / "eval" / "ai_act_eval.jsonl"
RESULTS_FILE = PROJECT_ROOT / "data" / "eval" / "results_openai_gpt4omini.jsonl"


def load_eval_dataset() -> List[Dict]:
    """
    Carica il dataset di valutazione (domande + risposta gold).
    Ogni riga del JSONL deve avere almeno:
      - question
      - answer
    """
    if not EVAL_FILE.exists():
        raise FileNotFoundError(f"File di valutazione non trovato: {EVAL_FILE}")

    examples = []
    with EVAL_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            examples.append(data)

    print(f"Caricati {len(examples)} esempi di valutazione.")
    return examples


def main():
    # 1) Carica dataset
    eval_examples = load_eval_dataset()

    # 2) Inizializza LLM OpenAI
    llm = OpenAILLMClient(model_name="gpt-4o-mini")

    # 3) Per ogni domanda, esegui la pipeline RAG
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    with RESULTS_FILE.open("w", encoding="utf-8") as f_out:
        # Generiamo noi un id progressivo
        for idx, ex in enumerate(eval_examples, start=1):
            qid = idx
            question = ex["question"]
            gold_answer = ex["answer"]

            print(f"\n=== ESEMPIO {qid} ===")
            print(f"Q: {question}")

            try:
                model_answer, contexts = answer_question(llm, question, top_k=5)
            except Exception as e:
                print(f"Errore durante la generazione per id={qid}: {e}")
                model_answer = ""
                contexts = []

            # Salviamo un record completo per RAGAS / analisi
            record = {
                "id": qid,
                "question": question,
                "gold_answer": gold_answer,
                "model_answer": model_answer,
                "contexts": [c["text"] for c in contexts],
            }

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

            print("Model answer:")
            print(model_answer[:400], "...")
            print("-" * 60)

    print(f"\n✅ Risultati salvati in: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
