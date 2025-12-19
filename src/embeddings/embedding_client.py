# src/explore_kg_llm/embeddings/embedding_client.py
from langchain_openai import OpenAIEmbeddings
from src.config import OPENAI_API_KEY, EMBEDDING_MODEL

def get_embedding_client():
    return OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        model=EMBEDDING_MODEL,
    )
