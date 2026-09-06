from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from explorer.backend.graph_adapter.neo4j import Neo4jGraphAdapter
from explorer.backend.path_discovery import PathDiscoveryService
from explorer.backend.path_ranking import PathRankingService
from explorer.backend.path_search.cache import PathSearchCache
from explorer.backend.semantic_search.ranking import (
    ANCHOR_RANKING_STRATEGY,
    AnchorRankingConfig,
)
from src.config import EMBEDDING_MODEL, EMBEDDING_PROVIDER
from src.embeddings.embedding_utils import get_embedding_property


@dataclass(frozen=True)
class PathSearchResult:
    paths: list[dict[str, Any]]
    cache_hit: bool


class PathSearchService:
    def __init__(
        self,
        cache: PathSearchCache | None = None,
        anchor_ranking_config: AnchorRankingConfig | None = None,
    ) -> None:
        self.cache = cache or PathSearchCache()
        self.anchor_ranking_config = anchor_ranking_config or AnchorRankingConfig()

    def search(
        self,
        query: str,
        relationship_k: int,
        semantic_fetch_k: int,
        paths_per_hit: int,
    ) -> PathSearchResult:
        cache_options = self._cache_options(
            semantic_fetch_k=semantic_fetch_k,
            paths_per_hit=paths_per_hit,
        )
        cached_paths = self.cache.get(query, relationship_k, options=cache_options)
        if cached_paths is not None:
            return PathSearchResult(paths=cached_paths, cache_hit=True)

        from explorer.backend.semantic_search import SemanticSearchService

        graph = Neo4jGraphAdapter()
        try:
            semantic_results = SemanticSearchService(
                config=self.anchor_ranking_config,
                metadata_enricher=graph.enrich_relationship_hits,
            ).search(query, relationship_k=semantic_fetch_k)
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

        self.cache.set(query, relationship_k, path_dicts, options=cache_options)
        return PathSearchResult(paths=path_dicts, cache_hit=False)

    def _cache_options(self, semantic_fetch_k: int, paths_per_hit: int) -> dict[str, Any]:
        return {
            "ranking_strategy": ANCHOR_RANKING_STRATEGY,
            "embedding_provider": EMBEDDING_PROVIDER,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_property": get_embedding_property(),
            "semantic_fetch_k": int(semantic_fetch_k),
            "paths_per_hit": int(paths_per_hit),
            "anchor_ranking": self.anchor_ranking_config.to_cache_dict(),
        }
