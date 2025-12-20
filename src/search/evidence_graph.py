from dataclasses import dataclass, field


@dataclass
class EvidenceGraph:
    query: str
    relationships: list = field(default_factory=list)
    nodes: dict = field(default_factory=dict)          # keyed by node_id
    publications: dict = field(default_factory=dict)   # keyed by PMID

    def add_relationship(self, doc, score):
        self.relationships.append({
            "text": doc.page_content,
            "score": score,
            "metadata": doc.metadata,
        })

    def add_node(self, doc, score):
        node_id = doc.metadata.get("id") or doc.metadata.get("original_subject")
        self.nodes[node_id] = {
            "text": doc.page_content,
            "score": score,
            "metadata": doc.metadata,
        }

    def add_publication(self, doc, score):
        pmid = doc.metadata.get("pmid") or doc.metadata.get("id")
        self.publications[pmid] = {
            "text": doc.page_content,
            "score": score,
            "metadata": doc.metadata,
        }
