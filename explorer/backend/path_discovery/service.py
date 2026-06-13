from __future__ import annotations

from explorer.backend.graph_adapter.neo4j import Neo4jGraphAdapter
from explorer.backend.models import Path, SemanticSearchResult


class PathDiscoveryService:
    def __init__(self, graph: Neo4jGraphAdapter) -> None:
        self.graph = graph

    def discover_from_semantic_results(
        self,
        results: SemanticSearchResult,
        max_hops: int = 3,
        paths_per_hit: int = 3,
    ) -> list[Path]:
        paths = []
        seen = set()
        for hit in results.relationships:
            metadata = dict(getattr(hit, "metadata", {}) or {})
            score = float(metadata.get("score", 0.0) or 0.0)
            evidence_text = getattr(hit, "page_content", None)
            for path in self.graph.candidate_paths_for_semantic_hit(
                metadata=metadata,
                score=score,
                evidence_text=evidence_text,
                max_hops=max_hops,
                limit=paths_per_hit,
            ):
                signature = (
                    tuple(node.element_id for node in path.nodes),
                    tuple(edge.element_id for edge in path.edges),
                )
                if signature in seen:
                    continue
                seen.add(signature)
                paths.append(path)
        return paths

    def expand(self, path: Path, limit: int = 10) -> list[Path]:
        return self.graph.expand_path(path, limit=limit)
