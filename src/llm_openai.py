from dotenv import load_dotenv
load_dotenv()
# src/llm_openai.py

# src/llm_openai.py

import os
from typing import Optional, List

from dotenv import load_dotenv
from openai import OpenAI

from llm_base import LLMClient

# Carica variabili dal .env nella root del progetto
load_dotenv()


class OpenAILLMClient(LLMClient):
    """
    Implementazione di LLMClient per i modelli OpenAI (gpt-4o, gpt-4o-mini, ecc.).
    """

    def __init__(self, model_name: str = "gpt-4o-mini", api_key: Optional[str] = None):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY non trovata. Mettila nel file .env oppure passala al costruttore."
            )

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
        stop: Optional[List[str]] = None,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )
        return response.choices[0].message.content.strip()
