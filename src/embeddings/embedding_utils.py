# src/explore_kg_llm/embeddings/embedding_utils.py
from langchain_openai import OpenAIEmbeddings
from src.config import OPENAI_API_KEY, EMBEDDING_MODEL


def get_embedding_client():
    return OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        model=EMBEDDING_MODEL,
    )


def print_search_result(results):
    for res in results:
        print("-" * 80)
        print(f"Similarity Score: {res.metadata['score']:.4f}")
        print(f"Document Content: {res.page_content[:200]}...")
        print(f"Metadata: {res.metadata}")
        print("-" * 80)


def extract_entities_from_relationships(rel_results):
    entities = set()

    for doc in rel_results:
        meta = doc.metadata
        if meta.get("llm_subject"):
            entities.add(meta["llm_subject"])
        if meta.get("llm_object"):
            entities.add(meta["llm_object"])

    return list(entities)
