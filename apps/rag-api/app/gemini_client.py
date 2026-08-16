from openai import OpenAI
from .config import settings

client = OpenAI(base_url=settings.gemini_base_url, api_key=settings.gemini_api_key)

def embed(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=settings.embed_model, input=texts)
    return [d.embedding for d in resp.data]

def chat(messages: list[dict]) -> str:
    resp = client.chat.completions.create(model=settings.llm_model, messages=messages)
    return resp.choices[0].message.content
