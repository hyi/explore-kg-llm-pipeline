from __future__ import annotations

from collections.abc import Callable
from typing import Any

from explorer.backend.models import SemanticSearchResult
from explorer.backend.semantic_search.ranking import (
    AnchorRankingConfig,
    candidate_pool_size,
    rerank_and_diversify_relationships,
)
from src.embeddings.embed_relationships import relationship_similarity_search


class SemanticSearchService:
    """Thin wrapper around existing semantic search utilities.

    The explorer's path search only needs relationship hits. Node semantic
    expansion remains available for callers that explicitly request it.
    """

    def __init__(
        self,
        config: AnchorRankingConfig | None = None,
        relationship_retriever: Callable[..., list[Any]] = relationship_similarity_search,
        metadata_enricher: Callable[[list[Any]], list[Any]] | None = None,
    ) -> None:
        self.config = config or AnchorRankingConfig()
        self.relationship_retriever = relationship_retriever
        self.metadata_enricher = metadata_enricher

    def search(
        self,
        query: str,
        relationship_k: int = 10,
        node_k_per_entity: int = 2,
        max_nodes_per_entity: int = 8,
        include_nodes: bool = False,
    ) -> SemanticSearchResult:
        if not include_nodes:
            raw_k = candidate_pool_size(relationship_k, self.config)
            candidates = self.relationship_retriever(query, k=raw_k) if raw_k else []
            if self.metadata_enricher:
                candidates = self.metadata_enricher(candidates)
            return SemanticSearchResult(
                relationships=rerank_and_diversify_relationships(
                    query=query,
                    candidates=candidates,
                    requested_k=relationship_k,
                    config=self.config,
                ),
                nodes={},
            )

        from src.search.semantic_search import run_semantic_search

        evidence_graph = run_semantic_search(
            query=query,
            relationship_k=relationship_k,
            node_k_per_entity=node_k_per_entity,
            max_nodes_per_entity=max_nodes_per_entity,
        )
        return SemanticSearchResult(
            relationships=evidence_graph.relationships,
            nodes=evidence_graph.nodes,
        )
