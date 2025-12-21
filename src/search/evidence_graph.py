import pandas as pd
from pathlib import Path


class EvidenceGraph:
    def __init__(self, query: str):
        self.query = query
        self.relationships = []
        self.nodes = {}

    def relationships_df(self) -> pd.DataFrame:
        rows = []
        for r in self.relationships:
            m = r.metadata
            rows.append({
                "score": round(m.get("score"), 4),
                "subject": m.get("llm_subject"),
                "predicate": m.get("predicate"),
                "object": m.get("llm_object"),
                "abstract_title": m.get("abstract_title"),
                "evidence_text": f"{r.page_content[:300]}..."
            })
        return pd.DataFrame(rows).sort_values("score", ascending=False)

    def nodes_df(self) -> pd.DataFrame:
        rows = []
        for node_id, ns in self.nodes.items():
            for n in ns:
                m = n.metadata
                rows.append({
                    "score": round(m["score"], 4),
                    "entity": node_id,
                    "name": m.get("name"),
                    "description": m.get("description", "")
                })
        return pd.DataFrame(rows).sort_values("score", ascending=False)

    def save_tables(self, out_dir='results'):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        self.relationships_df().to_csv(out_dir / "relationships.csv", index=False)
        self.nodes_df().to_csv(out_dir / "nodes.csv", index=False)

    def save_evidence_html(self, out_file="results/evidence_graph.html"):
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)

        with open(out_file, "w") as f:
            f.write(f"<h1>Semantic Search Results</h1>")
            f.write(f"<h3>Query</h3><p>{self.query}</p>")

            f.write("<h2>Relationship Evidence</h2>")
            f.write(self.relationships_df().to_html(index=False))

            f.write("<h2>Expanded Nodes</h2>")
            f.write(self.nodes_df().to_html(index=False))

    def export_as_cypher(self, out_file="results/evidence_graph.cypher"):
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        nodes = set()
        with open(out_file, "w") as f:
            for r in self.relationships:
                m = r.metadata
                subj = m.get("original_subject")
                obj = m.get("original_object")
                if subj:
                    nodes.add((subj, m.get("llm_subject")))
                if obj:
                    nodes.add((obj, m.get("llm_object")))
            for node_id, node_name in sorted(nodes):
                f.write(f"MERGE (e:Entity {{id: '{node_id}'}})"
                        f"SET e.name = '{node_name}';\n")

            for r in self.relationships:
                m = r.metadata
                subj = m.get("original_subject")
                obj = m.get("original_object")
                pred = m.get("predicate", "").replace(":", "_")
                score = float(m.get("score", 0.0))
                title = m.get("abstract_title")

                if not subj or not obj or not pred:
                    continue

                f.write(
                    f"MATCH (s:Entity {{id: '{subj}'}}), (o:Entity {{id: '{obj}'}})\n"
                    f"MERGE (s)-[r:{pred}]->(o)\n"
                    f"SET r.score={score:.4f}, \n"
                    f"    r.abstract_title='{title}';\n"
                )
            f.write("\nMATCH (s:Entity)-[r]->(o:Entity)\nRETURN s, r, o;\n")
