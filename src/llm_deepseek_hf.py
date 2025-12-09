import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

class DeepSeekHFClient:
    def __init__(self, model_name="deepseek-ai/DeepSeek-V3", temperature=0.1):
        self.model_name = model_name
        self.temperature = temperature
        token = os.getenv("HF_TOKEN")
        if not token:
            raise RuntimeError("❌ HF_TOKEN non impostato! Aggiungilo al file .env o alle env di PyCharm.")
        self.client = InferenceClient(
            model=model_name,
            token=token
        )

    def generate(self, prompt: str) -> str:
        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=self.temperature
        )
        return response.choices[0].message["content"]


if __name__ == "__main__":
    load_dotenv()  # carica .env
    client = DeepSeekHFClient()
    out = client.generate("Say a very short hello.")
    print("Risposta DeepSeek:", out)
