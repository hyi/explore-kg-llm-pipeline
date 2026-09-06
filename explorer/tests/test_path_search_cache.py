from __future__ import annotations

import json

import pytest

from explorer.backend.path_search.cache import (
    DEFAULT_CACHE_PATH,
    PathSearchCache,
    _cache_key,
    normalized_query,
)
from explorer.backend.path_search.service import PathSearchService, _embedding_cache_identity
from explorer.backend.semantic_search.ranking import AnchorRankingConfig


def test_path_search_cache_normalizes_query_and_persists(tmp_path) -> None:
    cache = PathSearchCache(tmp_path / "path_search_cache.json")
    paths = [{"id": "path-1", "summary": "cached path"}]

    cache.set("Genes involved in chemoresistance", relationship_k=5, paths=paths)

    assert cache.get("  genes   involved IN chemoresistance  ", relationship_k=5) == paths
    assert cache.get("genes involved in chemoresistance", relationship_k=6) is None


def test_path_search_cache_key_distinguishes_ranking_options(tmp_path) -> None:
    cache = PathSearchCache(tmp_path / "path_search_cache.json")
    paths = [{"id": "path-1", "summary": "cached path"}]
    base_options = {
        "ranking_strategy": "dense_query_aware_v1",
        "anchor_ranking": AnchorRankingConfig(max_per_publication=2).to_cache_dict(),
    }
    changed_options = {
        "ranking_strategy": "dense_query_aware_v1",
        "anchor_ranking": AnchorRankingConfig(max_per_publication=3).to_cache_dict(),
    }

    cache.set("genes involved in chemoresistance", relationship_k=5, paths=paths, options=base_options)

    assert cache.get("genes involved in chemoresistance", relationship_k=5, options=base_options) == paths
    assert cache.get("genes involved in chemoresistance", relationship_k=5, options=changed_options) is None


def test_path_search_cache_stores_readable_key_payload(tmp_path) -> None:
    cache = PathSearchCache(tmp_path / "path_search_cache.json")
    paths = [{"id": "path-1", "summary": "cached path"}]
    options = {
        "embedding_provider": "openai",
        "embedding_model": "text-embedding-3-small",
        "embedding_property": "embedding",
    }

    cache.set("genes involved in chemoresistance", relationship_k=5, paths=paths, options=options)

    payload = json.loads(cache.path.read_text(encoding="utf-8"))
    entry = next(iter(payload["entries"].values()))
    assert "version" not in payload
    assert "version" not in entry["key_payload"]
    assert entry["key_payload"]["options"] == options
    assert entry["paths"] == paths


def test_path_search_cache_ignores_legacy_unstructured_entries(tmp_path) -> None:
    cache_path = tmp_path / "path_search_cache.json"
    key = _cache_key("genes involved in chemoresistance", relationship_k=5)
    cache_path.write_text(
        json.dumps({"entries": {key: [{"id": "path-1"}]}}),
        encoding="utf-8",
    )
    cache = PathSearchCache(cache_path)

    assert cache.get("genes involved in chemoresistance", relationship_k=5) is None


def test_path_search_cache_clear_removes_server_side_cache_file(tmp_path) -> None:
    cache = PathSearchCache(tmp_path / "path_search_cache.json")
    cache.set("genes involved in chemoresistance", relationship_k=5, paths=[{"id": "path-1"}])

    assert cache.path.exists()
    assert cache.clear() is True
    assert cache.path.exists() is False
    assert cache.clear() is False


def test_path_search_cache_options_include_embedding_identity(tmp_path) -> None:
    service = PathSearchService(cache=PathSearchCache(tmp_path / "path_search_cache.json"))

    options = service._cache_options(semantic_fetch_k=10, paths_per_hit=3)

    assert options["embedding_provider"]
    assert options["embedding_model"]
    assert options["embedding_property"] in {"embedding", "sapbert_embedding"}


def test_embedding_cache_identity_prefers_current_dotenv_file(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    (tmp_path / ".env").write_text("EMBEDDING_PROVIDER=sapbert\nEMBEDDING_MODEL=local-sapbert\n", encoding="utf-8")

    assert _embedding_cache_identity() == {
        "embedding_provider": "sapbert",
        "embedding_model": "local-sapbert",
        "embedding_property": "sapbert_embedding",
    }


def test_cache_key_differs_by_embedding_provider(tmp_path) -> None:
    cache = PathSearchCache(tmp_path / "path_search_cache.json")
    openai_options = {
        "embedding_provider": "openai",
        "embedding_model": "text-embedding-3-small",
        "embedding_property": "embedding",
    }
    sapbert_options = {
        "embedding_provider": "sapbert",
        "embedding_model": "local-sapbert",
        "embedding_property": "sapbert_embedding",
    }

    openai_key = cache.key_for("genes involved in chemoresistance", 5, options=openai_options)
    sapbert_key = cache.key_for("genes involved in chemoresistance", 5, options=sapbert_options)

    assert openai_key != sapbert_key


def test_path_search_service_returns_cache_hit_without_live_search(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = PathSearchCache(tmp_path / "path_search_cache.json")
    paths = [{"id": "path-1", "summary": "cached path"}]
    service = PathSearchService(cache=cache)
    cache.set(
        "genes involved in chemoresistance",
        relationship_k=5,
        paths=paths,
        options=service._cache_options(semantic_fetch_k=10, paths_per_hit=3),
    )

    def fail_live_search(*_args, **_kwargs):
        raise AssertionError("live search should not run for cache hits")

    monkeypatch.setattr("explorer.backend.path_search.service.Neo4jGraphAdapter", fail_live_search)

    result = service.search(
        query="genes involved in chemoresistance",
        relationship_k=5,
        semantic_fetch_k=10,
        paths_per_hit=3,
    )

    assert result.cache_hit is True
    assert result.paths == paths


def test_normalized_query_collapses_case_and_whitespace() -> None:
    assert normalized_query("  What ARE   genes involved? ") == "what are genes involved?"


def test_default_cache_path_uses_tmp_runtime_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KG_EXPLORER_PATH_CACHE", raising=False)

    assert DEFAULT_CACHE_PATH.as_posix() == "/tmp/kg_explorer/path_search_cache.json"
    assert PathSearchCache().path == DEFAULT_CACHE_PATH


def test_path_search_cache_ignores_unwritable_cache_path(tmp_path) -> None:
    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("file blocks cache directory creation", encoding="utf-8")
    cache = PathSearchCache(blocked_parent / "path_search_cache.json")

    cache.set("genes involved in chemoresistance", relationship_k=5, paths=[{"id": "path-1"}])

    assert cache.get("genes involved in chemoresistance", relationship_k=5) is None
