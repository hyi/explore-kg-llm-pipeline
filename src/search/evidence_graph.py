from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class EvidenceNode:
    node_id: str
    label: str
    name: str
    score: float | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceEdge:
    subject: str
    predicate: str
    object: str
    semantic_text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceGraph:
    query: str
    edges: List[EvidenceEdge] = field(default_factory=list)
    nodes: Dict[str, EvidenceNode] = field(default_factory=dict)
    publications: Dict[str, EvidenceNode] = field(default_factory=dict)

    def add_edge(self, edge: EvidenceEdge):
        self.edges.append(edge)

    def add_node(self, node: EvidenceNode):
        self.nodes[node.node_id] = node

    def add_publication(self, pub: EvidenceNode):
        self.publications[pub.node_id] = pub

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "nodes": {k: vars(v) for k, v in self.nodes.items()},
            "edges": [vars(e) for e in self.edges],
            "publications": {k: vars(v) for k, v in self.publications.items()},
        }
