# src/explore_kg_llm/config.py
import os

from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").strip().lower()

if EMBEDDING_PROVIDER == 'openai':
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
else:
    # sapbert model    
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext')

SAPBERT_DEVICE = os.getenv("SAPBERT_DEVICE", "cpu")
