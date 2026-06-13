from __future__ import annotations

from explorer.backend.models import Path


class PathRankingService:
    """V1 ranking uses the semantic search similarity score directly."""

    def rank(self, paths: list[Path]) -> list[Path]:
        return sorted(paths, key=lambda path: path.score, reverse=True)
