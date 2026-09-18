# Demo Runbook

## Preflight

1. Regenerate analysis artifacts if source data changed:

```bash
uv run python scripts/generate_embedding_comparison_artifacts.py --projection-method pca
uv run python scripts/build_presentation_package.py
```

2. Run default validation:

```bash
env PYTHONDONTWRITEBYTECODE=1 uv run pytest
env PYTHONDONTWRITEBYTECODE=1 uv run ruff check
```

3. Optional live ROBOKOP check:

```bash
env PYTHONDONTWRITEBYTECODE=1 RUN_LIVE_ROBOKOP=1 uv run pytest explorer/tests/test_robokop_live_integration.py
```

## Environment

- Required for LitCoin search: `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, and the embedding provider settings already used by the app.
- Deterministic ROBOKOP development route: `ROBOKOP_MODE=fixture`.
- Live ROBOKOP route: `ROBOKOP_MODE=live`, optional `ROBOKOP_URL`, optional `ROBOKOP_TIMEOUT_SECONDS`.
- The app defaults to `HOST=0.0.0.0` and `PORT=8050`.

## Start The App

Fixture-backed development/demo fallback:

```bash
env ROBOKOP_MODE=fixture PORT=8050 uv run python -m explorer.frontend.dash_app
```

Live demo route:

```bash
env ROBOKOP_MODE=live PORT=8050 uv run python -m explorer.frontend.dash_app
```

## Primary Story

1. Search: `genes involved in chemoresistance in cancer`.
2. Select a candidate path whose anchor reasons include gene endpoint or drug-response compatibility.
3. Accept or bookmark the path to show that session state is preserved.
4. Select a normalized LitCoin entity with a visible CURIE, preferably `NCBIGene:5728` / PTEN if present.
5. Use `Fetch ROBOKOP summary`; confirm the UI shows mode `fixture` or `live` explicitly.
6. Choose a category/predicate filter, fetch a bounded ROBOKOP page, and point out limit/offset pagination.
7. Add one remote edge to the graph and identify its ROBOKOP source badge and primary knowledge source.

## Backup Story

Use `drug resistance in cancer` if the primary query does not surface a clean bridge quickly. The fallback discussion is still valid: LitCoin retrieves an interpretable local anchor, then ROBOKOP expansion starts only after a normalized CURIE is confirmed.

## Expected Latency

- Cached LitCoin path searches should usually respond faster than first-run searches.
- ROBOKOP fixture mode should be effectively local.
- ROBOKOP live mode depends on the remote service; keep the requested page size at 10 or lower for the presentation.

## Recovery Steps

- If LitCoin search is slow or unavailable, use the copied HTML figures and CSV summaries in `presentation/artifacts`.
- If live ROBOKOP fails, restart with `ROBOKOP_MODE=fixture`; do not present fixture data as live data.
- If a selected node has no bridge CURIE, choose another path node with an explicit normalized identifier. Display-name equality is intentionally rejected.
