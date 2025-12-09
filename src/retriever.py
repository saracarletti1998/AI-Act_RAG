# src/retriever.py

import json
from typing import List, Dict, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from config import (
    FAISS_INDEX_FILE,
    CHUNKS_METADATA_FILE,
    EMBEDDING_MODEL_NAME,
)


def load_metadata() -> List[Dict]:
    """
    Carica i chunk (id + text) dal file metadata.
    """
    chunks = []
    with CHUNKS_METADATA_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            chunks.append(data)
    print(f"Caricati {len(chunks)} chunk di metadata.")
    return chunks


def load_faiss_index() -> faiss.IndexFlatIP:
    """
    Carica l'indice FAISS da disco.
    """
    if not FAISS_INDEX_FILE.exists():
        raise FileNotFoundError(f"Indice FAISS non trovato: {FAISS_INDEX_FILE}")
    index = faiss.read_index(str(FAISS_INDEX_FILE))
    print(f"Indice FAISS caricato. Numero vettori: {index.ntotal}")
    return index


def load_embedding_model() -> SentenceTransformer:
    """
    Carica il modello di embeddings (stesso usato per creare l'indice).
    """
    print(f"Carico modello di embeddings per le query: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return model


def retrieve_chunks(
    query: str,
    top_k: int = 5,
) -> List[Tuple[float, Dict]]:
    """
    Data una query testuale, restituisce i top_k chunk più simili.
    Ritorna una lista di tuple (score, chunk_dict).
    """
    # Carichiamo risorse (in un sistema reale le terremo in RAM, qui è ok ricaricarle per test)
    chunks = load_metadata()
    index = load_faiss_index()
    model = load_embedding_model()

    # Embedding della query
    query_embedding = model.encode(query, convert_to_numpy=True)
    query_embedding = np.expand_dims(query_embedding, axis=0)
    faiss.normalize_L2(query_embedding)

    # Ricerca nell'indice
    distances, indices = index.search(query_embedding, top_k)
    distances = distances[0]
    indices = indices[0]

    results: List[Tuple[float, Dict]] = []

    for score, idx in zip(distances, indices):
        if idx == -1:
            continue  # nessun risultato
        # idx è l'indice del vettore; coincide con l'ordine dei chunk
        chunk = chunks[idx]
        results.append((float(score), chunk))

    return results


def main():
    # Esempio di test: cambialo con una domanda più interessante man mano
    query = "What does this regulation establish about AI systems?"
    print(f"Query: {query}")

    results = retrieve_chunks(query, top_k=3)

    print("\n=== RISULTATI ===")
    for score, chunk in results:
        print(f"\nScore: {score:.4f}")
        print(f"ID: {chunk['id']}")
        print("Testo:")
        print(chunk["text"][:500], "...")
        print("-" * 40)


if __name__ == "__main__":
    main()
