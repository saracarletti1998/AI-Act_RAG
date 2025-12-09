# src/llm_mistral_api.py

import os
from typing import Optional, List

from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()


class MistralLLMClient:
    """
    Client Mistral che espone lo stesso metodo .generate()
    usato nella pipeline RAG (come OpenAILLMClient, Claude, ecc.).
    """

    def __init__(self, model_name: str = "mistral-small-latest"):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError(
                "MISTRAL_API_KEY non trovata. "
                "Aggiungila nel file .env nella root del progetto."
            )
        self.model_name = model_name
        self.client = Mistral(api_key=api_key)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
        stop: Optional[List[str]] = None,
    ) -> str:
        """
        Metodo compatibile con answer_question().
        """
        response = self.client.chat.complete(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )
        # Il contenuto testuale è in choices[0].message.content
        return response.choices[0].message.content
