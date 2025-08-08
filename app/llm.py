import os
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_MODEL = os.getenv("TOGETHER_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")

if not TOGETHER_API_KEY:
    raise RuntimeError("TOGETHER_API_KEY not set in environment")

def ask_llm(question: str, context: str, max_tokens: int = 300, temperature: float = 0.0) -> str:
    """
    Calls Together AI chat completion API with the provided question and context.
    """
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": TOGETHER_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant answering questions based on provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    r = requests.post(url, headers=headers, json=payload)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()
