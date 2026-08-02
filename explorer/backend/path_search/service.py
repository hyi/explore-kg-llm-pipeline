from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from explorer.backend.graph_adapter.neo4j import Neo4jGraphAdapter
from explorer.backend.path_discovery import PathDiscoveryService
from explorer.backend.path_ranking import PathRankingService
from explorer.backend.path_search.cache import PathSearchCache


@dataclass(frozen=True)
class PathSearchResult:
    paths: list[dict[str, Any]]
    cache_hit: bool


class PathSearchService:
    def __init__(self, cache: PathSearchCache | None = None) -> None:
        self.cache = cache or PathSearchCache()

    def search(
        self,
        query: str,
        relationship_k: int,
        semantic_fetch_k: int,
        paths_per_hit: int,
    ) -> PathSearchResult:
        cached_paths = self.cache.get(query, relationship_k)
        if cached_paths is not None:
            return PathSearchResult(paths=cached_paths, cache_hit=True)

        from explorer.backend.semantic_search import SemanticSearchService

        graph = Neo4jGraphAdapter()
        try:
            semantic_results = SemanticSearchService().search(query, relationship_k=semantic_fetch_k)
            discovery = PathDiscoveryService(graph)
            ranking = PathRankingService()
            paths = discovery.discover_from_semantic_results(
                semantic_results,
                paths_per_hit=paths_per_hit,
                max_paths=relationship_k,
            )
            ranked_paths = ranking.rank(paths)[:relationship_k]
            path_dicts = [path.to_dict() for path in ranked_paths]
        finally:
            graph.close()

        self.cache.set(query, relationship_k, path_dicts)
        return PathSearchResult(paths=path_dicts, cache_hit=False)
