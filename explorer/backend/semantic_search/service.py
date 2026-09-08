from __future__ import annotations

from collections.abc import Callable
from typing import Any

from explorer.backend.models import SemanticSearchResult
from explorer.backend.semantic_search.ranking import (
    AnchorRankingConfig,
    candidate_pool_size,
    publication_identity,
    relationship_identity,
    rerank_relationships_with_diagnostics,
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
        model: str | None = None,
    ) -> None:
        self.config = config or AnchorRankingConfig()
        self.relationship_retriever = relationship_retriever
        self.metadata_enricher = metadata_enricher
        self.model = model

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
            candidates = self.relationship_retriever(query, k=raw_k, model=self.model) if raw_k else []
            if self.metadata_enricher:
                candidates = self.metadata_enricher(candidates)
            ranking_result = rerank_relationships_with_diagnostics(
                query=query,
                candidates=candidates,
                requested_k=relationship_k,
                config=self.config,
            )
            raw_candidate_diagnostics = sorted(
                ranking_result.diagnostics["ranked_candidates"],
                key=lambda item: item["raw_rank"],
            )
            diagnostics = {
                "requested_relationship_k": relationship_k,
                "raw_candidate_k": raw_k,
                "retrieval_model": self.model or "configured",
                "query_hints": ranking_result.diagnostics["query_hints"],
                "raw_candidates": raw_candidate_diagnostics,
                "raw_semantic_candidates": raw_candidate_diagnostics,
                "reranked_candidates": ranking_result.diagnostics["ranked_candidates"],
                "selected_anchors": ranking_result.diagnostics["selected_anchors"],
            }
            return SemanticSearchResult(
                relationships=ranking_result.relationships,
                nodes={},
                diagnostics=diagnostics,
            )

        from src.search.semantic_search import run_semantic_search

        evidence_graph = run_semantic_search(
            query=query,
            relationship_k=relationship_k,
            node_k_per_entity=node_k_per_entity,
            max_nodes_per_entity=max_nodes_per_entity,
            model=self.model,
        )
        return SemanticSearchResult(
            relationships=evidence_graph.relationships,
            nodes=evidence_graph.nodes,
        )


def _candidate_diagnostics(candidates: list[Any]) -> list[dict[str, Any]]:
    diagnostics = []
    for raw_rank, candidate in enumerate(candidates):
        metadata = dict(getattr(candidate, "metadata", {}) or {})
        diagnostics.append(
            {
                "raw_rank": raw_rank,
                "relationship_identity": relationship_identity(metadata, fallback=f"raw-rank:{raw_rank}"),
                "semantic_score": float(metadata.get("semantic_score", metadata.get("score", 0.0)) or 0.0),
                "predicate": metadata.get("predicate"),
                "subject": metadata.get("llm_subject") or metadata.get("original_subject") or metadata.get("subject"),
                "object": metadata.get("llm_object") or metadata.get("original_object") or metadata.get("object"),
                "publication_id": publication_identity(metadata),
                "retrieval_model": metadata.get("retrieval_model"),
                "retrieval_method": metadata.get("retrieval_method"),
                "retrieval_index_name": metadata.get("retrieval_index_name"),
                "retrieval_embedding_property": metadata.get("retrieval_embedding_property"),
                "retrieval_expected_dimensions": metadata.get("retrieval_expected_dimensions"),
                "retrieval_query_embedding_dimensions": metadata.get("retrieval_query_embedding_dimensions"),
            }
        )
    return diagnostics
