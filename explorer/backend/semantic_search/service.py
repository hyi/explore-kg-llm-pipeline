from __future__ import annotations

from src.search.semantic_search import run_semantic_search
from explorer.backend.models import SemanticSearchResult


class SemanticSearchService:
    """Thin wrapper around the existing src/search semantic workflow."""

    def search(
        self,
        query: str,
        relationship_k: int = 10,
        node_k_per_entity: int = 2,
        max_nodes_per_entity: int = 8,
    ) -> SemanticSearchResult:
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
