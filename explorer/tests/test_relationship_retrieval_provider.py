from __future__ import annotations

import pytest

from src.embeddings import embed_relationships


class FakeEmbeddingClient:
    def __init__(self, dimensions: int) -> None:
        self.embedding_dimensions = dimensions

    def embed_query(self, _query: str) -> list[float]:
        return [1.0] * self.embedding_dimensions


class FakeRow:
    def __init__(self, values: dict[str, object]) -> None:
        self.values = values

    def __getitem__(self, key: str):
        return self.values[key]


class FakeSession:
    def __init__(self, captured: dict[str, object], rows=None) -> None:
        self.captured = captured
        self.rows = rows or []

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def run(self, query: str, **kwargs):
        self.captured["query"] = query
        self.captured["kwargs"] = kwargs
        return self.rows


class FakeDriver:
    def __init__(self, captured: dict[str, object], rows=None) -> None:
        self.captured = captured
        self.rows = rows or []

    def session(self) -> FakeSession:
        return FakeSession(self.captured, self.rows)

    def close(self) -> None:
        self.captured["closed"] = True


def test_openai_scan_retrieval_queries_embedding_property(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(embed_relationships, "get_embedding_client", lambda model=None: FakeEmbeddingClient(1536))
    monkeypatch.setattr(embed_relationships.GraphDatabase, "driver", lambda *_args, **_kwargs: FakeDriver(captured))

    embed_relationships._relationship_similarity_search_scan("genes in cancer", k=3, model="openai")

    assert "r.`embedding` IS NOT NULL" in str(captured["query"])
    assert "r.`embedding` AS embedding" in str(captured["query"])
    assert "sapbert_embedding" not in str(captured["query"])
    assert "type(r) IN $relationship_types" not in str(captured["query"])


def test_sapbert_scan_retrieval_queries_sapbert_embedding_property(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(embed_relationships, "get_embedding_client", lambda model=None: FakeEmbeddingClient(768))
    monkeypatch.setattr(embed_relationships.GraphDatabase, "driver", lambda *_args, **_kwargs: FakeDriver(captured))

    embed_relationships._relationship_similarity_search_scan("genes in cancer", k=3, model="sapbert")

    assert "r.`sapbert_embedding` IS NOT NULL" in str(captured["query"])
    assert "r.`sapbert_embedding` AS embedding" in str(captured["query"])


def test_scan_retrieval_records_query_embedding_dimensions(monkeypatch) -> None:
    captured: dict[str, object] = {}
    rows = [
        FakeRow(
            {
                "predicate": "biolink:related_to",
                "metadata": {"id": "rel-1"},
                "text": "semantic text",
                "embedding": [1.0] * 768,
            }
        )
    ]
    monkeypatch.setattr(embed_relationships, "get_embedding_client", lambda model=None: FakeEmbeddingClient(768))
    monkeypatch.setattr(embed_relationships.GraphDatabase, "driver", lambda *_args, **_kwargs: FakeDriver(captured, rows))

    results = embed_relationships._relationship_similarity_search_scan("genes in cancer", k=3, model="sapbert")

    assert results[0].metadata["retrieval_embedding_property"] == "sapbert_embedding"
    assert results[0].metadata["retrieval_expected_dimensions"] == 768
    assert results[0].metadata["retrieval_query_embedding_dimensions"] == 768


def test_vector_retrieval_records_provider_specific_index_and_dimensions(monkeypatch) -> None:
    captured_indexes: list[str] = []
    captured_retrieval_queries: list[str] = []

    class FakeStore:
        def similarity_search_with_score(self, _query: str, k: int):
            return []

    def fake_from_existing_relationship_index(**kwargs):
        captured_indexes.append(kwargs["index_name"])
        captured_retrieval_queries.append(kwargs["retrieval_query"])
        return FakeStore()

    monkeypatch.setattr(
        embed_relationships,
        "_relationship_vector_indexes_by_type",
        lambda: {
            "biolink:related_to": ["biolink_related_to_sapbert_vector_idx"],
            "biolink:custom_predicate": ["biolink_custom_predicate_sapbert_vector_idx"],
        },
    )
    monkeypatch.setattr(embed_relationships, "get_embedding_client", lambda model=None: FakeEmbeddingClient(768))
    monkeypatch.setattr(
        embed_relationships.Neo4jVector,
        "from_existing_relationship_index",
        fake_from_existing_relationship_index,
    )
    embed_relationships._relationship_stores.cache_clear()

    embed_relationships.relationship_similarity_search("genes in cancer", k=3, model="sapbert")

    assert captured_indexes
    assert all("_sapbert_vector_idx" in index_name for index_name in captured_indexes)
    assert "biolink:custom_predicate_sapbert_vector_idx" in captured_indexes
    assert all("elementId(relationship)" in query for query in captured_retrieval_queries)
    assert all("relationship_id:" in query for query in captured_retrieval_queries)
    assert all("subject_labels:" in query for query in captured_retrieval_queries)


def test_openai_index_candidates_exclude_sapbert_indexes() -> None:
    candidates = embed_relationships._relationship_index_candidates(
        "biolink:related_to",
        {
            "biolink:related_to": [
                "biolink_related_to_sapbert_vector_idx",
                "biolink_related_to_vector_idx",
            ]
        },
        model="openai",
    )

    assert "biolink_related_to_vector_idx" in candidates
    assert all("_sapbert" not in index_name for index_name in candidates)


def test_sapbert_index_candidates_exclude_openai_indexes() -> None:
    candidates = embed_relationships._relationship_index_candidates(
        "biolink:related_to",
        {
            "biolink:related_to": [
                "biolink_related_to_vector_idx",
                "biolink_related_to_sapbert_vector_idx",
            ]
        },
        model="sapbert",
    )

    assert "biolink_related_to_sapbert_vector_idx" in candidates
    assert all("_sapbert" in index_name for index_name in candidates)


def test_relationship_store_raises_dimension_mismatch_candidate(monkeypatch) -> None:
    captured_indexes: list[str] = []

    class FakeStore:
        pass

    def fake_from_existing_relationship_index(**kwargs):
        captured_indexes.append(kwargs["index_name"])
        if kwargs["index_name"] == "bad_vector_idx":
            raise ValueError(
                "The provided embedding function and vector index dimensions do not match."
            )
        return FakeStore()

    monkeypatch.setattr(
        embed_relationships,
        "_relationship_vector_indexes_by_type",
        lambda: {"biolink:related_to": ["bad_vector_idx"]},
    )
    monkeypatch.setattr(embed_relationships, "_relationship_index_candidates", lambda *_args, **_kwargs: ["bad_vector_idx", "good_vector_idx"])
    monkeypatch.setattr(embed_relationships, "get_embedding_client", lambda model=None: FakeEmbeddingClient(1536))
    monkeypatch.setattr(
        embed_relationships.Neo4jVector,
        "from_existing_relationship_index",
        fake_from_existing_relationship_index,
    )
    embed_relationships._relationship_stores.cache_clear()

    with pytest.raises(ValueError, match="dimensions do not match"):
        embed_relationships._relationship_stores(model="openai")

    assert captured_indexes == ["bad_vector_idx"]
