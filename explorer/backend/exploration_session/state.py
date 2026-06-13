from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from explorer.backend.models import Path


@dataclass
class ExplorationSession:
    id: str = field(default_factory=lambda: str(uuid4()))
    accepted_paths: dict[str, Path] = field(default_factory=dict)
    rejected_paths: dict[str, Path] = field(default_factory=dict)
    bookmarked_paths: dict[str, Path] = field(default_factory=dict)
    search_history: list[dict[str, Any]] = field(default_factory=list)
    user_notes: dict[str, str] = field(default_factory=dict)

    def add_search(self, query: str, result_count: int) -> None:
        self.search_history.append(
            {
                "query": query,
                "result_count": result_count,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

    def accept(self, path: Path) -> None:
        self.accepted_paths[path.id] = path
        self.rejected_paths.pop(path.id, None)

    def reject(self, path: Path) -> None:
        self.rejected_paths[path.id] = path
        self.accepted_paths.pop(path.id, None)
        self.bookmarked_paths.pop(path.id, None)

    def bookmark(self, path: Path) -> None:
        self.bookmarked_paths[path.id] = path

    def save_note(self, path_id: str, note: str) -> None:
        if note.strip():
            self.user_notes[path_id] = note.strip()
        else:
            self.user_notes.pop(path_id, None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "accepted_paths": [path.to_dict() for path in self.accepted_paths.values()],
            "rejected_paths": [path.to_dict() for path in self.rejected_paths.values()],
            "bookmarked_paths": [path.to_dict() for path in self.bookmarked_paths.values()],
            "search_history": self.search_history,
            "user_notes": self.user_notes,
        }
