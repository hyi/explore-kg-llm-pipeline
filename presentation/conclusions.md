# Presentation Conclusions

## Direct Observations

- The compared exports contain `2003` matched LitCoin relationship embeddings across OpenAI and SapBERT with same-population coverage `True`.
- OpenAI and SapBERT nearest-neighbor sets are related but not identical: mean top-10 Jaccard is `0.542`.
- For `genes involved in chemoresistance in cancer`, top-15 query results share `7` relationships with Jaccard `0.304`.
- The current anchor-diversity run reports `20` unique publications, `17` unique predicates, and `6` query-compatible predicate matches.
- Among final-anchor result sets, `hybrid` / `sapbert` has the highest publication diversity in this artifact set with `9` unique publications.

## Interpretations

- The models organize the same LitCoin edges differently enough that retrieval diagnostics should compare ranked neighborhoods rather than relying on one projection view.
- Query-aware reranking is useful for separating structural compatibility from raw text similarity, especially for gene and treatment-response queries.
- Hybrid/keyword retrieval gives a practical recovery path when dense retrieval over-concentrates on one publication context.
- Bounded ROBOKOP expansion is best presented as human-guided follow-up from a confirmed bridge CURIE, not as global ROBOKOP search.

## Limitations And Future Directions

- The evidence is based on the exported relationship population and configured queries, so claims should stay within that scope.
- 2D projections can show local patterns and outliers, but cannot establish global model superiority.
- Live ROBOKOP availability and latency remain external risks; fixture mode is deterministic but must be clearly labeled.
- Grant-worthy next steps: evaluate retrieval quality with human labels, add explicit bridge-selection UX for ambiguous IDs, and compare whether external expansion improves time-to-useful-path.
