# src/rag_pipeline.py

from typing import List, Tuple, Dict

from llm_base import LLMClient
from retriever import retrieve_chunks


def build_rag_prompt(question: str, contexts: List[Dict]) -> str:
    """
    Costruisce il prompt da dare al modello:
    - include i chunk dell'AI Act come CONTEXT
    - include la domanda dell'utente
    """
    context_texts = [c["text"] for c in contexts]
    context_block = "\n\n---\n\n".join(context_texts)

    prompt = f"""You are an assistant specialised in the EU AI Act.
You must answer strictly based on the following excerpts from the Regulation.
If the information is not present, explicitly say that you cannot answer based only on the provided articles.

CONTEXT (excerpts from the AI Act):

{context_block}

---

QUESTION:
{question}

ANSWER (be precise, formal, and refer explicitly to the Regulation when relevant):
"""
    return prompt


def answer_question(
    llm: LLMClient,
    question: str,
    top_k: int = 5,
) -> Tuple[str, List[Dict]]:
    """
    Pipeline RAG:
    1. retrieval dei top_k chunk più rilevanti
    2. costruzione del prompt
    3. chiamata al modello LLM
    4. restituisce (risposta, contesti usati)
    """
    results = retrieve_chunks(question, top_k=top_k)

    # results = lista di (score, chunk_dict); ci servono solo i chunk_dict
    contexts = [chunk for score, chunk in results]

    prompt = build_rag_prompt(question, contexts)

    answer = llm.generate(prompt)

    return answer, contexts


def main():
    # Usiamo OpenAI come primo LLM
    from llm_openai import OpenAILLMClient

    llm = OpenAILLMClient(model_name="gpt-4o-mini")

    question = "What are the main obligations for providers of high-risk AI systems under this Regulation?"
    answer, contexts = answer_question(llm, question, top_k=5)

    print("QUESTION:")
    print(question)
    print("\nANSWER:")

    print(answer)

    print("\n--- CONTEXTS USED ---")
    for i, c in enumerate(contexts):
        print(f"\n[CONTEXT {i}] ID={c['id']}")
        print(c["text"][:400], "...")


if __name__ == "__main__":
    main()
