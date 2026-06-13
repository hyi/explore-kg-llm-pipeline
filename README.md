# Exploring Knowledge Graph (KG) + LLM Pipeline for Semantic Search and Prediction

## Source file summary

  - src/config.py centralizes environment loading. Everything that talks to Neo4j or OpenAI imports from here.
  - src/embeddings/embedding_utils.py is the shared utility layer. It builds the OpenAIEmbeddings client and has helpers for printing results and
    extracting entities from relationship hits.
  - src/embeddings/embed_nodes.py creates node vector indexes in Neo4j, embeds selected node labels, and exposes node similarity search.
  - src/embeddings/embed_relationships.py embeds relationship semantic_text, creates relationship vector indexes, and exposes relationship
    similarity search.
  - src/embeddings/embed_publications.py is a specialized embedder for publication abstracts.
  - src/search/semantic_search.py is the main end-to-end search entrypoint. It:
      1. runs relationship similarity search,
      2. extracts entities from those relationship hits,
      3. runs node similarity search for those entities,
      4. writes results.
  - src/search/node_semantic_search.py is a simpler variant that only searches nodes for the query.
  - src/search/evidence_graph.py is the output layer. It turns search results into pandas tables, HTML, CSV, and a Cypher export.
  - src/cypher/create_node_text.cypher and src/cypher/create_semantic_text.cypher prepare the text fields that later get embedded.

## Execution Flow
  1. Load credentials from .env via src/config.py.
  2. Populate textual fields in Neo4j using the Cypher templates.
  3. Run embedding builders in src/embeddings/ to create vectors and Neo4j vector indexes.
  4. Run a search entrypoint in src/search/.
  5. Inspect generated outputs in a results/ directory created by src/search/evidence_graph.py.

## Scripts
  - scripts/dump_embeddings.py exports stored embeddings from Neo4j into JSONL.
  - scripts/restore_embeddings.py restores them back into Neo4j.
  - The snapshot files live in scripts/data/node_embeddings.jsonl and scripts/data/relationship_embeddings.jsonl.