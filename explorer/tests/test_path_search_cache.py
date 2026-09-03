from __future__ import annotations

import pytest

from explorer.backend.path_search.cache import (
    DEFAULT_CACHE_PATH,
    PathSearchCache,
    normalized_query,
)
from explorer.backend.path_search.service import PathSearchService


def test_path_search_cache_normalizes_query_and_persists(tmp_path) -> None:
    cache = PathSearchCache(tmp_path / "path_search_cache.json")
    paths = [{"id": "path-1", "summary": "cached path"}]

    cache.set("Genes involved in chemoresistance", relationship_k=5, paths=paths)

    assert cache.get("  genes   involved IN chemoresistance  ", relationship_k=5) == paths
    assert cache.get("genes involved in chemoresistance", relationship_k=6) is None


def test_path_search_service_returns_cache_hit_without_live_search(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = PathSearchCache(tmp_path / "path_search_cache.json")
    paths = [{"id": "path-1", "summary": "cached path"}]
    cache.set("genes involved in chemoresistance", relationship_k=5, paths=paths)

    def fail_live_search(*_args, **_kwargs):
        raise AssertionError("live search should not run for cache hits")

    monkeypatch.setattr("explorer.backend.path_search.service.Neo4jGraphAdapter", fail_live_search)

    result = PathSearchService(cache=cache).search(
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
