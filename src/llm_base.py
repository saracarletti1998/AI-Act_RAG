#llm_base.py: per reare un’interfaccia astratta per i vari LLM, in modo che non debba cambiare il codice per i vari LLM

from abc import ABC, abstractmethod
from typing import Optional, List


class LLMClient(ABC):
    """
    Classe astratta: tutti i modelli (OpenAI, Claude, Llama ecc.)
    implementeranno questo metodo.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
        stop: Optional[List[str]] = None,
    ) -> str:
        pass
