# src/llm_llama_hf.py

import os
from typing import Optional

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()


class LlamaLLMClient:
    """
    Client semplice per usare un modello LLaMA (es. Llama 3 Instruct)
    tramite Hugging Face Inference API.

    L'interfaccia è pensata per essere analoga a MistralLLMClient:
    deve esporre un metodo .generate(prompt: str) -> str
    che la tua rag_pipeline può usare senza modifiche.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        temperature: float = 0.0,
        max_tokens: int = 512,
    ):
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not hf_token:
            raise ValueError(
                "Manca HUGGINGFACEHUB_API_TOKEN nel file .env "
                "o nelle variabili d'ambiente."
            )

        self.client = InferenceClient(model=model_name, token=hf_token)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Genera una risposta data una stringa di prompt.
        Se la tua MistralLLMClient usa un'interfaccia diversa (es. .invoke),
        semplicemente allinea questo metodo allo stesso nome che usi in rag_pipeline.
        """

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        # Usando l'API chat-like di HF
        response = self.client.chat_completion(
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        # La risposta è nel primo choice
        return response.choices[0].message["content"]
