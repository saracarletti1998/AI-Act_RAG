import os
from dotenv import load_dotenv

load_dotenv()   # carica il file .env dalla root

print("OPENAI_API_KEY =", os.getenv("OPENAI_API_KEY"))
