# src/build_vector_store.py

import json
from typing import List, Dict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from config import (
    CHUNKS_JSONL,
    VECTOR_STORE_DIR,
    FAISS_INDEX_FILE,
    CHUNKS_METADATA_FILE,
    EMBEDDING_MODEL_NAME,
)


def load_chunks() -> List[Dict]:
    """
    Carica i chunk dal file JSONL generato da prepare_corpus.py.
    Ogni riga deve essere un JSON con almeno: {"id": ..., "text": ...}
    """
    if not CHUNKS_JSONL.exists():
        raise FileNotFoundError(f"File dei chunk non trovato: {CHUNKS_JSONL}")

    chunks = []
    with CHUNKS_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            # Critico: controlliamo che i campi minimi esistano
            if "id" not in data or "text" not in data:
                raise ValueError(f"Chunk malformato: {data}")
            chunks.append(data)

    print(f"Caricati {len(chunks)} chunk da {CHUNKS_JSONL}")
    return chunks


def build_embeddings_model() -> SentenceTransformer:
    """
    Carica il modello di embeddings SentenceTransformers.
    Critico: stesso modello va usato sia per l'indice che per le query.
    """
    print(f"Carico modello di embeddings: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return model


def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Crea un indice FAISS usando inner product (cosine-like similarity).
    Prima normalizziamo i vettori per approssimare la cos similarity.
    """
    # Normalizziamo gli embeddings a norma 1 (per cos similarity)
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    print(f"Creo indice FAISS con dimensione vettori = {dim}")
    index = faiss.IndexFlatIP(dim)  # Inner Product

    index.add(embeddings)
    print(f"Indice FAISS: contiene {index.ntotal} vettori")
    return index


def save_faiss_index(index: faiss.IndexFlatIP):
    """
    Salva l'indice FAISS su disco.
    """
    faiss.write_index(index, str(FAISS_INDEX_FILE))
    print(f"Indice FAISS salvato in: {FAISS_INDEX_FILE}")


def save_metadata(chunks: List[Dict]):
    """
    Salva l'elenco dei chunk (id + text) in un JSONL separato.
    Questo ci serve per mappare gli ID dell'indice al testo.
    """
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    with CHUNKS_METADATA_FILE.open("w", encoding="utf-8") as f:
        for ch in chunks:
            out = {"id": ch["id"], "text": ch["text"]}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Metadata dei chunk salvata in: {CHUNKS_METADATA_FILE}")


def main():
    # 1. Carichiamo i chunk
    chunks = load_chunks()
    texts = [c["text"] for c in chunks]

    if not texts:
        raise ValueError("Nessun testo da indicizzare. Verifica ai_act_chunks.jsonl")

    # 2. Carichiamo il modello di embeddings
    model = build_embeddings_model()

    # 3. Calcoliamo gli embeddings
    print("Calcolo embeddings per tutti i chunk...")
    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    # 4. Creiamo indice FAISS
    index = create_faiss_index(embeddings)

    # 5. Salviamo indice + metadata
    save_faiss_index(index)
    save_metadata(chunks)

    print("✅ Vector store costruito con successo.")


if __name__ == "__main__":
    main()
