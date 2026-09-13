# Human-Guided Semantic Graph Exploration

## Purpose

This project provides a toolkit for interactive knowledge graph exploration.

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

## Implemented V1 Baseline

The V1 system is implemented as a Plotly Dash web application backed by Neo4j.

Existing capabilities include:

* semantic relationship and node retrieval
* semantic-hit-driven path discovery
* candidate path ranking and selection
* interactive graph and path expansion
* exploration-session state
* accepted, rejected, and bookmarked paths
* user notes and search history
* path-search caching
* Neo4j access through a graph adapter
* pytest coverage for backend state, ranking, caching, and frontend helpers

Treat these capabilities as the working baseline. Extend the existing
implementation rather than rebuilding the V1 architecture.

---

## Current Repository Structure

New application functionality belongs under `explorer/`:

* `explorer/backend/semantic_search/`: application-level retrieval and anchor ranking
* `explorer/backend/path_discovery/`: paths generated from semantic anchors
* `explorer/backend/path_ranking/`: ranking of discovered paths
* `explorer/backend/path_search/`: search orchestration and caching
* `explorer/backend/graph_adapter/neo4j/`: Neo4j-specific graph access
* `explorer/backend/exploration_session/`: exploration state
* `explorer/frontend/`: Plotly Dash interface
* `explorer/tests/`: application tests

The `src/` directory contains the earlier embedding, Cypher, and semantic-search
utilities. Reuse these where practical, but place new application behavior
under `explorer/`. Avoid duplicating working services or spreading application
logic across unrelated parts of the repository.

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

## Architecture Responsibilities

### graph_adapter

Responsible for graph access.

Keep Neo4j-specific code isolated in this layer.

---

### semantic_search

Responsible for:

* node embeddings
* edge embeddings
* semantic similarity search

Refer to `src/` for existing embedding and vector-search implementations and
reuse them as applicable. Application-level anchor ranking belongs under
`explorer/backend/semantic_search/`.

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

Before implementing a major feature or architectural change:

* propose the design
* explain tradeoffs
* obtain approval

Routine fixes and tests that follow an approved design do not require another
approval pause. Keep implementations simple and preserve the working baseline.

## Testing

* Write all new and modified tests using pytest.

---

## Current Research and Product Goals

The current work investigates how semantic edge retrieval and graph structure
can support human-guided path exploration in the LitCoin knowledge graph.

The work has three related but distinct workstreams:

1. Improve global anchors and query-aware neighborhood exploration using
   semantic, keyword, structural, and diversity signals.
2. Investigate how OpenAI and SapBERT embeddings organize the same LitCoin
   relationships.
3. Prototype bounded cross-graph expansion from LitCoin anchors into ROBOKOP.

Use the following milestone order unless the user approves a change:

1. stabilize query-aware, diversified dense anchors
2. add and evaluate keyword and hybrid retrieval plus ranked local expansion
3. create the analysis notebook and presentation-ready evidence
4. add a minimal, bounded LitCoin-to-ROBOKOP demo path

The notebook may be developed incrementally as each retrieval variant becomes
available, but presentation conclusions must be regenerated from the final
approved implementations. Findings from the offline investigation should
inform later retrieval and interface decisions. They do not, by themselves,
authorize adding embedding projections to the Dash application.

---

## Active Workstream 1: Query-Aware, Diverse Semantic Anchors

### Motivation

Raw edge-vector similarity is a candidate generator, not the final anchor
ranking. In the LitCoin graph, broad document context can dominate the edge
claim. For example, a query about genes involved in cancer chemoresistance can
return several edges from one chemotherapy-neuropathy abstract because the
shared abstract contains words such as cancer, chemotherapy, treatment, and
response.

Improve the initial anchors used for path discovery by combining:

* raw semantic similarity
* query-to-entity-type compatibility
* query-to-predicate compatibility
* lightweight query/context compatibility
* result diversity across publications and seed edges

This is retrieval and ranking functionality, not autonomous scientific
reasoning. Keep the scores and explanations transparent to the user.

### Scope and Location

Implement this feature under `explorer/backend/semantic_search/`. Reuse
`src.embeddings.embed_relationships.relationship_similarity_search` only as
the raw candidate retriever. Do not add the new ranking policy to
`src/search/semantic_search.py`, because the policy belongs to the explorer
application.

Integrate the ranked anchors into the existing flow:

`PathSearchService` -> `SemanticSearchService` -> `PathDiscoveryService`

Do not replace `PathRankingService`: it ranks paths after anchors have already
been selected. Anchor ranking and path ranking are separate concerns.

For the first implementation:

* do not regenerate or change stored embeddings
* do not add an LLM call for query parsing or reranking
* do not introduce an external search service or new database
* do not hard-filter all nonmatching edges; this small KG needs graceful
  fallback when exact matches do not exist

### Required Design

1. Over-fetch raw candidates before reranking. Make the candidate-pool size
   configurable and bounded. A reasonable default is
   `max(requested_k * 5, 25)`, with a safe upper bound such as 100.

2. Parse only explicit, testable query hints. At minimum recognize a gene
   intent from terms such as `gene`, `genes`, `genetic`, `mutation`, and
   `mutations`; recognize resistance/response intent from terms such as
   `resistance`, `resistant`, `chemoresistance`, `sensitivity`, and `response`.
   Keep this logic in a small pure function and make it easy to extend. Do not
   pretend that the heuristic fully understands the query.

3. Preserve the raw vector score as `semantic_score`. Compute a separate
   `anchor_score`; never overwrite the original score without retaining it.
   The ranking must remain interpretable. Record component scores or a compact
   list of ranking reasons in result metadata.

4. Use endpoint node labels when available. For a gene-intent query, strongly
   boost a relationship whose subject or object has a gene-like Neo4j label.
   Use lexical fallback only when endpoint labels are absent. Do not maintain a
   hand-written list of gene symbols.

5. Boost predicate compatibility for resistance/response queries. The initial
   compatible predicate set should include:

   * `biolink:affects_response_to`
   * `biolink:increases_response_to`
   * `biolink:decreases_response_to`
   * `biolink:associated_with_resistance_to`

   Broader predicates such as `affects`, `regulates`, `contributes_to`,
   `causes`, and association predicates may receive a smaller boost, not the
   same boost as explicit response/resistance predicates.

6. Prefer soft boosts and penalties over hard filters. If no edge satisfies a
   requested entity type or predicate family, return the best semantic
   fallbacks and mark them as fallbacks in metadata.

7. Diversify after relevance reranking. Avoid filling the displayed anchors
   with near-duplicate edges from one abstract. Make publication identity use
   the best stable metadata field available, falling back to
   `abstract_title`. Default to at most two selected anchors per publication.
   Also discourage exact repetition of the same subject-object neighborhood.
   A simple greedy selector is sufficient; full MMR is optional.

8. Enrich raw relationship hits with endpoint labels and stable relationship
   and publication identifiers when they are not already returned. Keep
   Neo4j-specific enrichment inside the Neo4j adapter or existing Neo4j
   retrieval layer. Avoid one database query per candidate; enrich candidates
   in one batch where practical.

9. Ensure `PathDiscoveryService` receives anchors in final anchor-rank order.
   It currently stops once `max_paths` is reached, so order affects which
   neighborhoods can become candidate paths.

10. Update cache identity or cache versioning so results produced by the old
    raw-similarity strategy cannot be confused with results from the new
    anchor-ranking strategy. Include every user-visible ranking option in the
    cache key.

### Suggested Interfaces

The first query-aware implementation used query-specific booleans. The active
ranking implementation should instead use generic, schema-oriented query intent
and candidate compatibility. Names may be adjusted to fit the code, but prefer
small testable units such as:

```python
@dataclass(frozen=True)
class QueryIntent:
    requested_subject_categories: tuple[Any, ...] = ()
    requested_object_categories: tuple[Any, ...] = ()
    requested_endpoint_categories: tuple[Any, ...] = ()
    requested_predicate_families: tuple[Any, ...] = ()
    requested_direction: str | None = None
    content_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnchorRankingConfig:
    candidate_multiplier: int = 5
    minimum_candidate_pool: int = 25
    maximum_candidate_pool: int = 100
    max_per_publication: int = 2


def parse_query_intent(query: str) -> QueryIntent: ...


def rerank_and_diversify_relationships(
    query: str,
    candidates: list[Any],
    requested_k: int,
    config: AnchorRankingConfig,
) -> list[Any]: ...
```

Keep scoring constants named and centralized. Document their role as initial
heuristics that require evaluation rather than empirically validated weights.
Do not add generic query-text-overlap as a graph-aware ranking score; dense and
keyword retrieval already measure textual relevance. Matched claim terms may be
kept as diagnostics.

For drug-response predicates, do not treat the normalized Biolink predicate
family alone as a complete structural match. Validate the predicate family
against endpoint categories, original relationship text, endpoint roles, and
claim-level subject/object/statement qualifiers. A gene or variant paired with
a drug/chemical, a drug/chemical paired with a response-bearing phenotype or
process, or explicit qualifier evidence for treatment resistance, sensitivity,
efficacy, or response may produce a complete match. Disease or phenotype
endpoints alone should remain partial or unknown unless claim-level evidence
establishes treatment-response semantics. Drug, treatment, therapy, or
chemotherapy context alone may support partial compatibility but should not
validate complete drug-response compatibility without explicit response
evidence.

Parse query-intent expressions longest-match-first using non-overlapping spans
unless an overlap is explicitly intentional. For example, prefer `gene
products` over `gene` and `associated with resistance to` over its shorter
subphrases. Treat broad `biolink:GenomicEntity` labels as genomic context, not
as exact sequence-variant evidence; exact sequence-variant matching should use
specific categories such as `biolink:SequenceVariant`, `biolink:Allele`, and
`biolink:Haplotype`.

For queries with structural intent, anchor ordering should respect generic
compatibility tiers before retrieval score: validated `complete_match`, then
`partial_match`, then `retrieval_only`, with contextual `biolink:mentions`
available only through labeled fallback passes. Retrieval score ranks
candidates within the same tier. Queries with no structural intent should keep
retrieval ordering except for query-independent relationship quality and
diversity handling.

### Output and UI Compatibility

Maintain compatibility with code that currently reads `metadata["score"]`.
After reranking, it may represent `anchor_score`, provided
`metadata["semantic_score"]` retains the raw similarity. Prefer also exposing:

* `anchor_score`
* `semantic_score`
* `ranking_reasons`
* `is_semantic_fallback`
* `publication_id` when available

Do not expose embedding arrays in API or UI payloads.

If candidate path cards are changed, show the overall anchor/path score and a
short explanation such as `gene endpoint match`, `resistance predicate match`,
or `semantic fallback`. Avoid presenting heuristic scores as scientific
confidence.

### Tests and Acceptance Criteria

Add pytest unit tests that do not require a live Neo4j instance or API keys.
Use small synthetic `Document` objects or equivalent fixtures.

Required tests:

* Raw candidates are over-fetched, then exactly the requested number is
  returned when enough candidates exist.
* A gene-labeled cancer edge outranks a higher raw-similarity non-gene edge for
  `genes involved in chemoresistance in cancer` when the configured boost is
  sufficient.
* An explicit resistance/response predicate outranks an otherwise comparable
  generic predicate.
* Results gracefully fall back to semantic similarity when no structural hint
  matches.
* No more than the configured number of anchors from the same publication are
  selected when alternatives exist.
* Diversity selection is deterministic, including score ties.
* Raw `semantic_score`, final `anchor_score`, and ranking reasons are preserved.
* Empty candidate lists and `requested_k <= 0` are handled explicitly.
* Cache keys distinguish ranking strategy/configuration changes.
* Existing path search, session, and frontend tests continue to pass.

Also add a regression fixture modeled on the observed LitCoin behavior:

* four high-similarity edges from one chemotherapy-induced peripheral
  neuropathy paper
* one PTEN/cancer edge with a slightly lower raw similarity
* one unrelated genetics/opioid-use edge

For a gene/chemoresistance/cancer query, the PTEN/cancer candidate should be
selected ahead of the chemotherapy-neuropathy duplicates and the opioid-use
candidate, while unmatched results may still appear as clearly marked
fallbacks if too few better candidates exist.

Run the complete pytest suite and ruff check before reporting completion. In the handoff,
report the scoring formula and constants, changed cache semantics, test results,
and any metadata limitations discovered in the actual Neo4j records.

---

## Active Workstream 1B: Hybrid Retrieval and Ranked Neighborhood Expansion

### Design Basis

Use the following ideas from Polonuer et al., "Autonomous Knowledge Graph
Exploration with Adaptive Breadth-Depth Retrieval" (ACL 2026) as design
motivation (which can be found in this project folder paper/2026.acl-long.714.pdf), not as an instruction to reproduce ARK:

* retain global retrieval for broad discovery and later re-anchoring
* use bounded one-hop expansion for relational depth
* rank local neighbors against the active query or a user-refined subquery
* filter heterogeneous neighborhoods by node type and predicate when useful
* stop expansion selectively rather than traversing indiscriminately
* evaluate the latency and quality effects of increasing retrieval depth

ARK is autonomous and node-retrieval oriented. This project remains
human-guided and path-centric. Do not add parallel agents, autonomous
trajectories, trajectory voting, or model distillation in this workstream.

### Separate Searchable Text Fields

Do not rely on one undifferentiated text field for every retrieval method.
Preserve or create logically separate representations for:

* `edge_text`: subject, qualifiers, predicate, object, and qualifiers
* `title_text`: publication title
* `context_text`: supporting sentence when available, otherwise abstract text

The exact Neo4j property names may be adjusted after inspecting the data, but
the edge claim must remain distinguishable from publication context. Avoid
copying a complete abstract into API responses or UI state unnecessarily.

### Keyword and Hybrid Global Retrieval

Add BM25 or the nearest practical Neo4j full-text equivalent as a second
global relationship-candidate channel. Primary keyword retrieval should be
edge-claim centered. Publication title and broad context search should be a
separate publication-context retrieval layer or explicit user action, not part
of initial anchor generation except as a labeled fallback.

Build BM25 keyword queries from content-bearing terms separately from parsed
structural intent. Generic endpoint-type expressions such as genes, proteins,
variants, drugs, diseases, and pathways should primarily drive graph-aware
category intent rather than BM25 scoring. Broad relational scaffolding such as
`involved in` and `related to` should be de-emphasized for BM25. Biomedical
content terms and relation-specific concepts, such as `chemoresistance`,
`cancer`, treatment, response, sensitivity, and named entities, should remain
available to keyword retrieval. Keep the dense query text unchanged.

Support independently testable retrieval modes:

* `dense`
* `keyword`
* `hybrid`

Do not add raw cosine and BM25 scores directly. Their scales are unrelated.
Use a rank-based fusion method such as weighted reciprocal rank fusion, with
named, configurable constants. Preserve each candidate's source ranks and raw
scores for diagnosis.

The hybrid candidate flow should be:

1. retrieve bounded dense and keyword candidate pools
2. deduplicate by stable relationship identity
3. fuse channel ranks
4. apply query-to-entity-type and query-to-predicate compatibility
5. diversify by publication and endpoint neighborhood
6. pass ordered anchors to path discovery

If a channel is unavailable, return a clearly identified degraded result
rather than silently labeling a single-channel result as hybrid.

### Human-Guided Neighborhood Retrieval

When a user expands a LitCoin node or path, keep the active query available and
allow an optional user-refined expansion query. Provide bounded controls for:

* adjacent node category
* relationship predicate or predicate family
* incoming, outgoing, or either direction
* number of returned neighbors

Rank the filtered neighborhood against the active or refined query using the
available lexical, semantic, and structural signals. Also provide an explicit
way to reveal additional or unranked neighbors so heuristic ranking does not
hide the graph.

Do not automatically execute an unbounded multi-hop traversal. Preserve human
selection between meaningful expansions. Show unexplored-neighbor counts when
available and make retrieval limits visible rather than presenting them as
complete neighborhoods.

### Re-Anchoring

Global search must remain available after exploration begins. Add or preserve
a workflow that lets the user find alternate anchors and add a new branch
without discarding accepted paths, bookmarks, notes, or the current session.

Treat re-anchoring as recovery from an incomplete or misleading local region,
not as failure of the user. Keep the provenance of the query and retrieval mode
that introduced each anchor.

### Hybrid Retrieval Tests and Evaluation

Add tests that do not require a live Neo4j instance for:

* deterministic keyword tokenization and query handling
* stable rank fusion with ties and missing channels
* relationship deduplication across retrieval channels
* preservation of raw channel scores and ranks
* explicit degraded-mode metadata when one channel fails or is unavailable
* query-conditioned local-neighbor ranking
* node-category, predicate, direction, and limit filtering
* session-preserving re-anchoring

Extend the offline evaluation to compare dense OpenAI, dense SapBERT, keyword,
and both hybrid variants. Do not claim that the ACL paper establishes the best
retriever for LitCoin; its benchmarks are much larger, retrieve nodes rather
than LitCoin paths, and use autonomous agent-generated subqueries.

---

## Active Workstream 2: Comparative Embedding and Retrieval Investigation

### Purpose

Create a reproducible offline investigation of how OpenAI and PubMedBERT
SapBERT relationship embeddings organize the same LitCoin edges, how their
search results differ, and whether keyword and hybrid retrieval improve the
anchors used for graph exploration.

This work supports model comparison, retrieval diagnosis, and presentation of
the investigation. It is not a new primary workflow in the explorer web app.

### Reference note
- PubMedBERT SapBERT model can be found in this project folder `sapbert_model/SapBERT-from-PubMedBERT-fulltext`. 
- EMBEDDING_PROVIDER and EMBEDDING_MODEL environment variables can be set in `.env` file in this project folder to indicate whether sapbert or openai model should be used. This model-switching through environment variables or via a `model` input argument has been implemented in all python code in the `src` folder, but has not been implemented in code in the `explorer` folder.

### Scope and Location

Suggest the following organization, but take it only as a reference and use your best judgement for code organization conforming to best practice:

* `analysis/embedding_comparison.py`: reusable loading, validation, metrics,
  projection, and plotting functions
* `notebooks/embedding_comparison.ipynb`: narrative, configuration, displayed
  results, and interpretation
* `tests/test_embedding_comparison.py`: tests for reusable analysis functions

Keep substantial logic out of notebook cells. The notebook should call tested
functions from the analysis module so results can be reproduced without
manually executing undocumented cell state.

If the repository already has a different established location for notebooks
or analysis code, propose using that convention before creating a parallel
structure.

### Required Data Handling

* Match the two models' vectors using a stable relationship identifier.
* Verify and report whether both models cover the same edge population.
* Fail clearly on duplicate relationship identifiers or embedding-dimension
  inconsistencies within one model.
* Preserve useful metadata, including subject, predicate, object, endpoint
  labels, publication identifier, abstract title, and semantic text.
* Do not place database credentials, API keys, or embedding arrays in committed
  notebook output.
* Prefer a reusable export from Neo4j so the notebook can be rerun without a
  live database when practical.
* Record model names, embedding dimensions, extraction date, projection
  parameters, and random seeds in the output.

### Projection Rules

OpenAI and SapBERT embeddings occupy different vector spaces and may have
different dimensions. Do not treat their raw coordinates as comparable. Do not
concatenate the two vector matrices into one ordinary projection or interpret
the orientation and axis positions of separately fitted projections.

Create separate projections for the same edge set. Use consistent edge IDs,
colors, symbols, labels, and hover fields across panels. Compare neighborhood
membership, clustering, and highlighted query results rather than absolute
coordinates, axis direction, rotation, or visual distance between panels.

Use a fixed random seed. UMAP is appropriate for exploratory local-neighborhood
views; include PCA as a deterministic reference when useful. Treat t-SNE as
optional and do not use it as the only projection.

### Required Analyses

1. **Side-by-side embedding projections**

   * one point per matched relationship edge
   * separate panels for OpenAI and SapBERT
   * consistent coloring by publication, predicate family, or entity type
   * configurable highlighting of top-k results for a selected query
   * informative hover text without embedding arrays

2. **Query-result rank comparison**

   Compare all available retrieval configurations:

   * raw dense OpenAI
   * raw dense SapBERT
   * keyword/BM25
   * hybrid keyword plus OpenAI
   * hybrid keyword plus SapBERT
   * query-aware and diversified final anchors for each applicable mode

   Include at least:

   * `drug resistance in cancer`
   * `genes involved in chemoresistance in cancer`
   * one biomedical entity or synonym-focused query
   * one broader relational query

   Make the query set configurable rather than hard-coding presentation logic
   around only these examples.

3. **Neighborhood agreement**

   Report, as appropriate:

   * top-k retrieval overlap between models
   * per-edge nearest-neighbor Jaccard overlap
   * rank correlation for shared candidates
   * same-publication fraction among nearest neighbors
   * same-predicate or predicate-family fraction among nearest neighbors

4. **Anchor relevance and diversity**

   For each model and query, report:

   * unique publications in top k
   * unique predicates and endpoint neighborhoods
   * requested entity-type matches
   * query-compatible predicate matches
   * raw semantic results versus reranked and diversified anchors
   * optional human judgments such as `relevant`, `adjacent`, or `irrelevant`

5. **Exploration utility**

   Where practical, record whether a useful graph neighborhood or path can be
   reached from each anchor, the number of user-approved expansions required,
   and whether global re-anchoring was needed. An anchor is useful when it
   provides an efficient entry into an informative region, even if it is not a
   complete answer by itself.

### Outputs

Produce presentation-ready interactive HTML figures when practical. Also make
it straightforward to export static figures for slides without requiring the
Dash application. Tables used in the notebook should be exportable as CSV.

Include a concise conclusions section suitable for adapting into group-meeting
slides. Separate direct observations, interpretations, and future hypotheses.
Every numerical or visual claim must be reproducible from notebook inputs.

Every figure must state the model, projection method, important parameters,
and what can and cannot be inferred from the view. Do not describe a 2D
projection as proof that one embedding model is globally superior.

### Tests and Acceptance Criteria

Tests must not require Neo4j, OpenAI access, or downloading SapBERT. Use small
synthetic vectors and metadata fixtures.

Required tests:

* edge matching is stable regardless of input row order
* missing and duplicate relationship IDs are reported clearly
* mismatched model coverage is reported rather than silently discarded
* nearest-neighbor and top-k-overlap metrics return known values on fixtures
* rank-fusion and retrieval-comparison tables retain the retrieval-mode label
* same-publication fractions are calculated correctly
* projection and sampling are reproducible with a fixed seed
* query-result highlighting uses stable relationship IDs
* raw and reranked result sets remain distinguishable
* embedding arrays are excluded from exported hover and table payloads

Run the complete pytest suite before reporting completion. In the handoff,
list generated artifacts, data assumptions, dependency changes, reproducibility
settings, and limitations of the visual comparison.

---

## Active Workstream 3: LitCoin-to-ROBOKOP Cross-Graph Expansion

### Goal

Prototype a human-guided cross-graph workflow in which a user:

1. finds and selects useful anchors in the LitCoin graph
2. explores LitCoin paths and neighborhoods
3. chooses a normalized LitCoin entity as a bridge
4. expands that entity into the much larger ROBOKOP knowledge graph
5. optionally continues bounded one-hop exploration in either graph

The purpose is to demonstrate how a small, literature-focused KG can provide
interpretable anchors into a large integrated biomedical KG. Do not ingest or
replicate the complete ROBOKOP graph into the LitCoin Neo4j database.

### Integration Reference and Current Constraints

RoboMCP is available at `https://github.com/cbizon/RoboMCP`, including the
`robokop-mcp`, `nodenormalizer-mcp`, `name-resolver-mcp`, and `biolink-mcp`
servers.

At the inspected version, `robokop-mcp` provides:

* `get_node(curie)`
* `get_edges(curie, category, predicate, limit, offset, count_only)`
* `get_edge_summary(curie)`
* `get_edges_between(curie1, curie2)`

These operations support known-CURIE lookup and adjacency expansion; they do
not provide global semantic or keyword search over all ROBOKOP nodes. Use
LitCoin retrieval to establish the initial anchor and use ROBOKOP after a
bridge CURIE is known.

The inspected MCP implementation returns human-formatted strings. Do not parse
those strings in production application code. Before implementation, perform a
short integration spike to determine whether the installed MCP client exposes
structured content not visible in the current source. If not, propose one of:

* a typed adapter to the same ROBOKOP HTTP API
* a small upstream-compatible RoboMCP enhancement that returns structured data

Keep either option behind the explorer's own provider interface so transport
details do not leak into session, path, ranking, or frontend code. Obtain
approval for the selected option before implementing it.

Do not use the current `get_edges_between` implementation for arbitrary
high-degree nodes without review: it may request up to 10,000 edges from one
endpoint and filter client-side.

### Identifier Normalization and Bridge Resolution

LitCoin and ROBOKOP should have already been normalized to use shared Biolink categories and predicates and the same canonical CURIE for an entity, but do a quick verification before external expansion.

### Provider Boundary

Add a small external graph-provider abstraction rather than teaching the
existing Neo4j adapter to call remote services. The exact interface should be
proposed after the integration spike, but it should support operations such as:

* node lookup by normalized CURIE
* edge-summary retrieval
* bounded incident-edge retrieval with category, predicate, direction, limit,
  and pagination controls where supported
* optional direct-edge lookup between two known CURIEs

Convert remote results into explorer domain objects only after retaining:

* source graph (`litcoin` or `robokop`)
* stable or reproducible edge identity
* original and normalized endpoint CURIEs
* Biolink categories and predicates
* edge direction
* qualifiers
* primary knowledge source
* publications and supporting sentences when available

Never merge two edges merely because their subject, predicate, and object look
similar. Preserve source-specific evidence and provenance.

### Bounded Cross-Graph Interaction

The first demo should support only user-initiated, one-hop ROBOKOP expansion.
For a selected bridge entity:

1. fetch and display an edge summary first
2. let the user choose or confirm category and predicate filters
3. fetch a small page of incident edges
4. rank the returned page against the active or refined query
5. let the user select which remote edges to add to the visible exploration

Because ROBOKOP has no stored embeddings available through RoboMCP, begin with
keyword relevance, query-intent compatibility, and explicit type/predicate
filters. On-demand embedding reranking of a small fetched candidate set may be
proposed later, but it must be cached, bounded, labeled, and evaluated against
the simpler lexical baseline.

Do not imply that a limited page is the complete ROBOKOP neighborhood. Show
counts, limits, pagination state, loading state, source badges, and remote
errors clearly. A ROBOKOP timeout or outage must not corrupt the LitCoin
session or remove locally explored paths.

### Demo Scope

For the demo, optimize for one reliable, explainable
cross-graph story rather than broad feature coverage. Prepare at least one
scripted example that demonstrates:

* semantic, keyword, or hybrid discovery of a LitCoin anchor
* query-aware local exploration
* CURIE normalization and bridge confirmation
* bounded ROBOKOP expansion with visible provenance
* user selection of a remote edge or branch

Keep a local fixture or recorded, non-sensitive response for deterministic UI
testing and presentation fallback. Clearly label fixture data in development
and never present it as a live response. Do not silently fall back from live to
recorded data during the demo.

### Cross-Graph Tests and Acceptance Criteria

Tests must not depend on live ROBOKOP or Translator services. Use committed,
small response fixtures that preserve realistic response structure and
provenance.

Required tests:

* one normalized identifier maps a LitCoin node to the intended ROBOKOP lookup
* zero and ambiguous mappings require explicit handling
* display-name equality alone cannot create a bridge
* gene/protein and drug/chemical conflation settings affect cache identity
* category, predicate, limit, offset, and direction controls are propagated
* remote results retain source graph, identifiers, qualifiers, publications,
  and primary knowledge source
* duplicate-looking LitCoin and ROBOKOP edges retain separate provenance
* high-degree expansion is bounded and paginated
* timeout, malformed response, and unavailable-provider states are visible and
  leave the LitCoin session intact
* fixture mode and live mode cannot be confused in the UI

Run a separate opt-in integration check against the configured live endpoint.
Do not make live network access part of the default pytest suite.

---

## Group Presentation and Demo Readiness

The near-term milestone is a group presentation that uses evidence from the
analysis notebook and a brief live demonstration of the explorer. The goal is
to communicate practical design insights that may improve an existing path
finder application or motivate future grant proposals.

Prioritize a coherent evidence-to-demo story over feature count. The notebook
is the source of quantitative and visual claims; the app demonstrates the
human-guided workflow.

Before presentation handoff:

* identify one primary and one backup query
* preflight the expected LitCoin anchors and local expansions
* preflight the bounded ROBOKOP example if Workstream 3 is approved and ready
* provide a fixture-backed test route for development and a clearly separate
  live-demo route
* document startup commands, required environment variables, expected latency,
  and recovery steps
* ensure every visible edge has source and evidence provenance
* avoid claims of model superiority based only on 2D projections or two example
  queries
* prepare a short table of implemented findings, observed limitations, and
  grant-worthy future directions

Do not add slide-generation or presentation-authoring dependencies to the
explorer unless explicitly requested. Produce reusable figures, tables, and
concise conclusions that can be transferred into slides.

---

## Deferred Product Decision: Embedding Visualization in the Dash App

The primary application remains a path-centric, human-guided exploration tool.
Do not add UMAP, t-SNE, PCA, or a general embedding-space browser to the Dash
application during the active workstreams.

After the offline investigation is reviewed, a coding agent may propose a
small optional diagnostic feature only if it supports a concrete exploration
decision that ranked anchors and paths do not already support. A possible
future feature is an `Embedding neighborhood` or `Why this anchor?` panel that
shows nearby edges, model-specific rankings, publication concentration, and
ranking reasons for a selected anchor.

Any such integration is a new product decision. Present its user value,
interaction design, performance implications, and maintenance cost, and obtain
approval before implementation.

---
