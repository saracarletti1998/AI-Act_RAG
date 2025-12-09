# src/prepare_corpus.py

import json
from typing import List, Dict

import tiktoken  # per stimare i "token" tipo GPT

from config import (
    AI_ACT_RAW_FILE,
    PROCESSED_DIR,
    CHUNKS_JSONL,
    CHUNK_MAX_TOKENS,
    CHUNK_OVERLAP_TOKENS,
)


def load_ai_act_text() -> str:
    """
    Legge il contenuto del file ai_act_en.txt.
    """
    if not AI_ACT_RAW_FILE.exists():
        raise FileNotFoundError(f"File AI Act non trovato: {AI_ACT_RAW_FILE}")
    with AI_ACT_RAW_FILE.open("r", encoding="utf-8") as f:
        return f.read()


def normalize_whitespace(text: str) -> str:
    """
    Cleaning leggerissimo: normalizza newline, tab e spazi doppi.
    Non modifichiamo il contenuto legale.
    """
    text = text.replace("\r\n", "\n")
    text = text.replace("\t", " ")
    # Brutale ma efficace: rimuove spazi doppi ripetuti
    while "  " in text:
        text = text.replace("  ", " ")
    return text


def get_tokenizer():
    """
    Ottiene un tokenizer tiktoken.
    Non ci interessa sia perfetto, ci basta coerente per stimare la lunghezza.
    """
    try:
        enc = tiktoken.encoding_for_model("gpt-4o")
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return enc


def split_into_chunks(text: str) -> List[Dict]:
    """
    Spezza il testo in chunk di max CHUNK_MAX_TOKENS token,
    con overlap CHUNK_OVERLAP_TOKENS.
    """
    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(text)

    chunks: List[Dict] = []
    start = 0
    chunk_id = 0

    while start < len(tokens):
        end = start + CHUNK_MAX_TOKENS
        chunk_tokens = tokens[start:end]
        decoded = tokenizer.decode(chunk_tokens)

        chunks.append({
            "id": f"ai_act_{chunk_id}",
            "text": decoded.strip()
        })

        chunk_id += 1
        # Il prossimo chunk riparte un po' prima per mantenere continuità
        start = end - CHUNK_OVERLAP_TOKENS

    return chunks


def save_chunks(chunks: List[Dict]):
    """
    Salva i chunk in JSONL: una riga = un JSON { "id": ..., "text": ... }
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    with CHUNKS_JSONL.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"✅ {len(chunks)} chunk salvati in {CHUNKS_JSONL}")


def main():
    print(f"Carico il testo da: {AI_ACT_RAW_FILE}")
    text = load_ai_act_text()
    print(f"Lunghezza testo (caratteri): {len(text)}")

    print("Normalizzo whitespace…")
    text = normalize_whitespace(text)

    print("Genero chunk…")
    chunks = split_into_chunks(text)
    print(f"Numero di chunk generati: {len(chunks)}")

    print("Salvo i chunk…")
    save_chunks(chunks)


if __name__ == "__main__":
    main()
