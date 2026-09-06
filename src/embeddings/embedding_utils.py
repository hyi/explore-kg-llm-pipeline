# src/explore_kg_llm/embeddings/embedding_utils.py
from functools import lru_cache

from langchain_core.embeddings import Embeddings

from src.config import EMBEDDING_MODEL, EMBEDDING_PROVIDER

DEFAULT_OPENAI_MODEL = "text-embedding-3-small"
DEFAULT_SAPBERT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
DEFAULT_OPENAI_DIMENSIONS = 1536
DEFAULT_SAPBERT_DIMENSIONS = 768


class SapBERTEmbeddings(Embeddings):
    """LangChain-compatible local SapBERT embedding client."""

    def __init__(
        self,
        model_name: str = DEFAULT_SAPBERT_MODEL,
        device: str = "cpu",
        batch_size: int = 16,
        max_length: int = 256,
        pooling: str | None = None,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.pooling = pooling or (
            "mean" if model_name.endswith("-mean-token") else "cls"
        )
        self._tokenizer = None
        self._model = None

    @property
    def embedding_dimensions(self) -> int:
        if self._model is not None:
            return int(self._model.config.hidden_size)
        return int(DEFAULT_SAPBERT_DIMENSIONS)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        tokenizer, model, torch = self._load_model()
        embeddings = []

        for start in range(0, len(texts), self.batch_size):
            batch = [text or "" for text in texts[start:start + self.batch_size]]
            tokens = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tokens = {key: value.to(self.device) for key, value in tokens.items()}

            with torch.no_grad():
                output = model(**tokens)
                if self.pooling == "mean":
                    vectors = _mean_pool(
                        output.last_hidden_state,
                        tokens["attention_mask"],
                        torch,
                    )
                else:
                    vectors = output.last_hidden_state[:, 0, :]

            embeddings.extend(vectors.cpu().tolist())
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def _load_model(self):
        if self._tokenizer is None or self._model is None:
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer
            except ImportError as exc:
                raise ImportError(
                    "SapBERT embeddings require the 'torch' and 'transformers' "
                    "packages to be installed."
                ) from exc

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
        else:
            import torch

        return self._tokenizer, self._model, torch


def _mean_pool(token_embeddings, attention_mask, torch):
    input_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * input_mask, 1)
    counts = torch.clamp(input_mask.sum(1), min=1e-9)
    return summed / counts


def _embedding_provider(model: str | None = None) -> str:
    return (model or EMBEDDING_PROVIDER).strip().lower()


def get_embedding_property(model: str | None = None) -> str:
    if _embedding_provider(model) == "sapbert":
        return "sapbert_embedding"
    return "embedding"


def get_embedding_index_suffix(model: str | None = None) -> str:
    if _embedding_provider(model) == "sapbert":
        return "_sapbert"
    return ""


def embedding_index_name(base_name: str, model: str | None = None) -> str:
    suffix = get_embedding_index_suffix(model)
    if not suffix:
        return base_name
    if base_name.endswith("_vector_idx"):
        return f"{base_name[:-len('_vector_idx')]}{suffix}_vector_idx"
    if base_name.endswith("_idx"):
        return f"{base_name[:-len('_idx')]}{suffix}_idx"
    return f"{base_name}{suffix}"


def get_embedding_dimensions(embedding_client=None, model: str | None = None) -> int:
    if embedding_client is not None and hasattr(
        embedding_client,
        "embedding_dimensions",
    ):
        return embedding_client.embedding_dimensions
    if _embedding_provider(model) == "sapbert":
        return DEFAULT_SAPBERT_DIMENSIONS
    return DEFAULT_OPENAI_DIMENSIONS


def cypher_escape_identifier(identifier: str) -> str:
    return identifier.replace("`", "``")


@lru_cache(maxsize=1)
def get_embedding_client(model: str | None = None):
    provider = _embedding_provider(model)
    if provider == "sapbert":
        from src.config import SAPBERT_DEVICE
        return SapBERTEmbeddings(
            model_name=_embedding_model(provider),
            device=SAPBERT_DEVICE,
            batch_size=16,
            max_length=256,
        )

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        from src.config import OPENAI_API_KEY
        return OpenAIEmbeddings(
            api_key=OPENAI_API_KEY,
            model=_embedding_model(provider),
        )

    raise ValueError(
        f"Unsupported embedding model '{provider}'. Use 'openai' or 'sapbert'."
    )


def _embedding_model(provider: str) -> str:
    if provider == EMBEDDING_PROVIDER:
        return EMBEDDING_MODEL
    if provider == "sapbert":
        return DEFAULT_SAPBERT_MODEL
    return DEFAULT_OPENAI_MODEL


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
