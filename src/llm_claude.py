# src/llm_claude.py

import os
from typing import Optional, List

from dotenv import load_dotenv
import anthropic

from llm_base import LLMClient

load_dotenv()


class ClaudeLLMClient(LLMClient):
    """
    Implementazione di LLMClient per i modelli Anthropic Claude.
    """

    def __init__(self, model_name: str = "claude-3-sonnet-20240229", api_key: Optional[str] = None):
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY non trovata. Aggiungila al file .env oppure passala al costruttore."
            )

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
        stop: Optional[List[str]] = None,
    ) -> str:
        """
        Usa l'API 'messages.create' di Claude 3.
        """
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )

        # Claude restituisce content come lista di blocchi; prendiamo il testo del primo
        # (di solito response.content[0].type == "text")
        if response.content and hasattr(response.content[0], "text"):
            return response.content[0].text.strip()

        # fallback di sicurezza
        return str(response)
