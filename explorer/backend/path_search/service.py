from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from dotenv import dotenv_values

from explorer.backend.graph_adapter.neo4j import Neo4jGraphAdapter
from explorer.backend.models import Path, SemanticSearchResult
from explorer.backend.path_discovery import PathDiscoveryService
from explorer.backend.path_ranking import PathRankingService
from explorer.backend.path_search.cache import PathSearchCache
from explorer.backend.semantic_search.ranking import (
    ANCHOR_RANKING_STRATEGY,
    AnchorRankingConfig,
)
from explorer.backend.semantic_search.retrieval import RetrievalConfig
from src.embeddings.embedding_utils import (
    DEFAULT_OPENAI_MODEL,
    DEFAULT_SAPBERT_MODEL,
    get_embedding_dimensions,
)


@dataclass(frozen=True)
class PathSearchResult:
    paths: list[dict[str, Any]]
    cache_hit: bool


class PathSearchService:
    def __init__(
        self,
        cache: PathSearchCache | None = None,
        anchor_ranking_config: AnchorRankingConfig | None = None,
        retrieval_config: RetrievalConfig | None = None,
    ) -> None:
        self.cache = cache or PathSearchCache()
        self.anchor_ranking_config = anchor_ranking_config or AnchorRankingConfig()
        self.retrieval_config = retrieval_config or _retrieval_config_from_env()

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
                retrieval_config=self.retrieval_config,
                keyword_retriever=getattr(graph, "keyword_relationship_search", None),
                metadata_enricher=graph.enrich_relationship_hits,
                model=cache_options["embedding_provider"],
            ).search(query, relationship_k=semantic_fetch_k)
            discovery = PathDiscoveryService(graph)
            ranking = PathRankingService()
            path_discovery_limit = max(
                int(relationship_k),
                int(semantic_fetch_k) * max(int(paths_per_hit), 0),
            )
            paths = discovery.discover_from_semantic_results(
                semantic_results,
                paths_per_hit=paths_per_hit,
                max_paths=path_discovery_limit,
            )
            ranked_paths = ranking.rank(paths)[:relationship_k]
            path_dicts = [path.to_dict() for path in ranked_paths]
            diagnostics = _path_search_diagnostics(
                semantic_results=semantic_results,
                discovered_paths=paths,
                displayed_paths=ranked_paths,
            )
        finally:
            graph.close()

        self.cache.set(
            query,
            relationship_k,
            path_dicts,
            options=cache_options,
            metadata={
                "retrieval_diagnostics": semantic_results.diagnostics,
                "path_search_diagnostics": diagnostics,
            },
        )
        return PathSearchResult(paths=path_dicts, cache_hit=False)

    def _cache_options(self, semantic_fetch_k: int, paths_per_hit: int) -> dict[str, Any]:
        return {
            "ranking_strategy": ANCHOR_RANKING_STRATEGY,
            **_embedding_cache_identity(),
            "semantic_fetch_k": int(semantic_fetch_k),
            "paths_per_hit": int(paths_per_hit),
            "path_discovery_strategy": "discover_all_selected_anchors_v1",
            "retrieval_identity_strategy": "relationship_element_id_metadata_v1_keyword_no_semantic_text_v1",
            "retrieval": self.retrieval_config.to_cache_dict(),
            "anchor_ranking": self.anchor_ranking_config.to_cache_dict(),
        }


def _embedding_cache_identity() -> dict[str, str]:
    env_file = dotenv_values(".env")
    provider = str(
        env_file.get("EMBEDDING_PROVIDER")
        or os.getenv("EMBEDDING_PROVIDER")
        or "openai"
    ).strip().lower()
    model = str(
        env_file.get("EMBEDDING_MODEL")
        or os.getenv("EMBEDDING_MODEL")
        or (DEFAULT_SAPBERT_MODEL if provider == "sapbert" else DEFAULT_OPENAI_MODEL)
    ).strip()
    return {
        "embedding_provider": provider,
        "embedding_model": model,
        "embedding_property": "sapbert_embedding" if provider == "sapbert" else "embedding",
        "embedding_dimensions": get_embedding_dimensions(model=provider),
    }


def _retrieval_config_from_env() -> RetrievalConfig:
    env_file = dotenv_values(".env")
    mode = str(
        env_file.get("KG_EXPLORER_RETRIEVAL_MODE")
        or os.getenv("KG_EXPLORER_RETRIEVAL_MODE")
        or "dense"
    ).strip().lower()
    return RetrievalConfig(mode=mode)


def _path_search_diagnostics(
    semantic_results: SemanticSearchResult,
    discovered_paths: list[Path],
    displayed_paths: list[Path],
) -> dict[str, Any]:
    displayed_by_id = {path.id: rank for rank, path in enumerate(displayed_paths)}
    discovered_path_rows = [
        _path_diagnostic(path, displayed_by_id=displayed_by_id)
        for path in discovered_paths
    ]
    paths_by_anchor: dict[str, list[dict[str, Any]]] = {}
    for row in discovered_path_rows:
        anchor_identity = row.get("anchor_relationship_identity")
        if anchor_identity:
            paths_by_anchor.setdefault(str(anchor_identity), []).append(row)

    raw_candidates = [
        _candidate_path_journey(candidate, paths_by_anchor)
        for candidate in semantic_results.diagnostics.get("raw_semantic_candidates", [])
    ]
    reranked_candidates = [
        _candidate_path_journey(candidate, paths_by_anchor)
        for candidate in semantic_results.diagnostics.get("reranked_candidates", [])
    ]
    selected_anchors = [
        _candidate_path_journey(candidate, paths_by_anchor)
        for candidate in semantic_results.diagnostics.get("selected_anchors", [])
    ]

    return {
        "raw_semantic_candidates": raw_candidates,
        "enriched_reranked_candidates": reranked_candidates,
        "diversified_semantic_anchors": selected_anchors,
        "discovered_paths": discovered_path_rows,
        "final_displayed_paths": [
            _path_diagnostic(path, displayed_by_id=displayed_by_id)
            for path in displayed_paths
        ],
    }


def _candidate_path_journey(
    candidate: dict[str, Any],
    paths_by_anchor: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    row = dict(candidate)
    anchor_identity = str(row.get("relationship_identity") or "")
    discovered_paths = paths_by_anchor.get(anchor_identity, [])
    row["number_of_paths_discovered"] = len(discovered_paths)
    row["discovered_paths"] = [
        {
            "path_id": path["path_id"],
            "path_score": path["path_score"],
            "selected_for_display": path["selected_for_display"],
            "final_display_rank": path["final_display_rank"],
        }
        for path in discovered_paths
    ]
    if not row.get("entered_selected_anchor_set"):
        row["path_selection_status"] = "not_selected_as_anchor"
    elif discovered_paths:
        row["path_selection_status"] = "has_displayed_path" if any(
            path["selected_for_display"] for path in discovered_paths
        ) else "paths_discovered_not_displayed"
    else:
        row["path_selection_status"] = "no_paths_discovered"
    return row


def _path_diagnostic(
    path: Path,
    displayed_by_id: dict[str, int],
) -> dict[str, Any]:
    final_rank = displayed_by_id.get(path.id)
    return {
        "path_id": path.id,
        "path_score": path.score,
        "selected_for_display": final_rank is not None,
        "final_display_rank": final_rank,
        "anchor_relationship_identity": path.anchor_metadata.get("relationship_identity"),
        "anchor_score": path.anchor_metadata.get("anchor_score"),
        "semantic_score": path.anchor_metadata.get("semantic_score"),
        "retrieval_score": path.anchor_metadata.get("retrieval_score"),
        "fusion_score": path.anchor_metadata.get("fusion_score"),
        "retrieval_channels": path.anchor_metadata.get("retrieval_channels"),
        "dense_rank": path.anchor_metadata.get("dense_rank"),
        "dense_score": path.anchor_metadata.get("dense_score"),
        "keyword_rank": path.anchor_metadata.get("keyword_rank"),
        "keyword_score": path.anchor_metadata.get("keyword_score"),
        "raw_rank": path.anchor_metadata.get("raw_rank"),
        "seed_subject": path.seed_subject,
        "seed_object": path.seed_object,
        "seed_predicate": path.seed_predicate,
        "summary": path.summary(),
    }
