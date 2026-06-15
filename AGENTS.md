# Human-Guided Semantic Graph Exploration

## Purpose

This project is building a toolkit for interactive knowledge graph exploration.

The toolkit combines:

* semantic retrieval
* graph structure
* path discovery
* path ranking
* human-guided exploration

The goal is to help users discover, compare, refine, and save meaningful paths through a knowledge graph.

LitCoin serves as the reference implementation.

The architecture should remain compatible with any Cypher-compatible knowledge graph.

---

## V1 Scope

Implement only:

* Neo4j backend
* semantic node search
* semantic edge search
* path discovery
* path ranking
* exploration sessions
* web application

Favor the simplest implementation that satisfies V1 requirements.

---

## Repository Structure

New functionality should be implemented under:

explorer/

Expected structure:

explorer/

```
backend/

    graph_adapter/
        neo4j/

    semantic_search/

    path_discovery/

    path_ranking/

    exploration_session/

    api/

frontend/

tests/
```

Avoid spreading new V1 functionality throughout unrelated parts of the repository.

Reuse existing code from the repository whenever practical. For example, the src folder already contains cypher and semantic search implementation that explores combining semantic search and graph structure for path exploration.  

---

## Design Principles

This project is NOT:

* a chatbot
* a KGQA system
* an autonomous agent
* an MCP-first application

This project IS:

* a graph exploration tool
* a path-centric discovery tool
* a human-guided exploration system

Human guidance is more important than automated reasoning.

---

## Primary Workflow

The preferred user workflow is:

Search
→ Candidate Paths
→ Human Selection
→ Path Expansion
→ Path Refinement
→ Save Session

Design features around this workflow.

---

## Core Domain Objects

Treat the following as first-class concepts:

* Graph
* Node
* Edge
* Path
* ExplorationSession

Path is the primary object.

Most interactions should revolve around paths rather than individual nodes. But node semantic search can be leveraged for path exploration under human guidance if needed.

---

## Exploration Sessions

ExplorationSession is a core domain object.

Session state should support:

* accepted paths
* rejected paths
* bookmarked paths
* search history
* user notes

Preserving exploration state is more important than user authentication.

---

## Architecture

### graph_adapter

Responsible for graph access.

Initial implementation:

* Neo4j only

Keep Neo4j-specific code isolated in this layer.

---

### semantic_search

Responsible for:

* node embeddings
* edge embeddings
* semantic similarity search

Refer to the `src` folder for previous implementations and reuse them as applicable.
---

### path_discovery

Responsible for:

* shortest paths
* semantic paths
* constrained paths
* path expansion

---

### path_ranking

Responsible for:

* path scoring
* ranking strategies
* path comparison

---

### exploration_session

Responsible for:

* user exploration state
* saved progress
* bookmarks
* notes

---

### web application

The UI should be path-centric.

Avoid node-centric graph browser workflows that quickly become difficult to navigate.

---

## Implementation Rules

Before implementing major features:

* propose the design
* explain tradeoffs
* obtain approval

Keep the implementation simple.

Build the smallest solution that satisfies V1 requirements.

## Testing

* Write all new and modified tests using pytest.
