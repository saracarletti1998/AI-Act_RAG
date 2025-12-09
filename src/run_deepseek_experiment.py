# src/run_deepseek_experiment.py

import json
from pathlib import Path
from dotenv import load_dotenv

from config import PROJECT_ROOT
from rag_pipeline import answer_question
from llm_deepseek_hf import DeepSeekHFClient

load_dotenv()

EVAL_FILE = PROJECT_ROOT / "data" / "eval" / "ai_act_eval.jsonl"
RESULTS_FILE = PROJECT_ROOT / "data" / "eval" / "results_deepseek.jsonl"


def load_eval_dataset():
    examples = []
    with EVAL_FILE.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            data = json.loads(line)
            if "id" not in data:
                data["id"] = idx
            examples.append(data)
    print(f"Caricati {len(examples)} esempi.")
    return examples


def main():
    eval_examples = load_eval_dataset()
    llm = DeepSeekHFClient()

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    print(f"Scriverò i risultati in: {RESULTS_FILE}")

    with RESULTS_FILE.open("w", encoding="utf-8") as f_out:
        for ex in eval_examples:
            qid = ex["id"]
            question = ex["question"]
            gold_answer = ex["answer"]

            print(f"\n=== DEEPSEEK – ESEMPIO {qid} ===")

            try:
                model_answer, contexts = answer_question(llm, question)
            except Exception as e:
                model_answer = f"Errore DeepSeek: {e}"
                contexts = []

            record = {
                "id": qid,
                "question": question,
                "gold_answer": gold_answer,
                "model_answer": model_answer,
                "contexts": [c["text"] for c in contexts],
            }

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("✅ Risultati DeepSeek salvati.")


if __name__ == "__main__":
    main()
