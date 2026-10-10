import os
from groq import Groq
from src.config import settings

client = Groq(api_key=settings.groq_api_key)
try:
    models = client.models.list()
    print("Available models:")
    for m in models.data:
        print(m.id)
except Exception as e:
    print(f"Error: {e}")
