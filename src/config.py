#config.py: per centralizzare tutti i percorsi e i parametri globali in un unico posto, per non ripetere i percorsi in tanti file diversi,
# per non rischiare di sbagliare file path
# per cambiare un valore una sola volta per tutto il progetto


# src/config.py

from pathlib import Path

# Radice del progetto
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Cartelle dati
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# File raw dell'AI Act
AI_ACT_RAW_FILE = RAW_DIR / "ai_act_en.txt"

# File dei chunk (generato da prepare_corpus.py)
CHUNKS_JSONL = PROCESSED_DIR / "ai_act_chunks.jsonl"

# Parametri di chunking
CHUNK_MAX_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 64

# ───────── Embeddings & Vector Store ───────── #

# Cartella dove salveremo indice e metadata
VECTOR_STORE_DIR = PROCESSED_DIR / "vector_store"
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# File dell'indice FAISS
FAISS_INDEX_FILE = VECTOR_STORE_DIR / "faiss_index.bin"

# File con la metadata (id → testo)
CHUNKS_METADATA_FILE = VECTOR_STORE_DIR / "chunks_metadata.jsonl"

# Nome del modello di embeddings (SentenceTransformers)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Critico: modello leggero, veloce e decente per testo legale.
