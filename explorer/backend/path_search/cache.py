from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

DEFAULT_CACHE_PATH = Path("/tmp/kg_explorer/path_search_cache.json")


class PathSearchCache:
    def __init__(self, path: Path | None = None) -> None:
        configured_path = os.getenv("KG_EXPLORER_PATH_CACHE")
        self.path = Path(configured_path) if configured_path else path or DEFAULT_CACHE_PATH

    def key_for(
        self,
        query: str,
        relationship_k: int,
        options: dict[str, Any] | None = None,
    ) -> str:
        return _cache_key(query, relationship_k, options=options)

    def get(
        self,
        query: str,
        relationship_k: int,
        options: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]] | None:
        try:
            cache = self._read()
        except OSError:
            return None
        key_payload = _cache_key_payload(query, relationship_k, options=options)
        entry = cache.get(_cache_key_from_payload(key_payload))
        if not isinstance(entry, dict):
            return None
        if entry.get("key_payload") != key_payload:
            return None
        paths = entry.get("paths")
        return paths if isinstance(paths, list) else None

    def set(
        self,
        query: str,
        relationship_k: int,
        paths: list[dict[str, Any]],
        options: dict[str, Any] | None = None,
    ) -> None:
        try:
            cache = self._read()
            key_payload = _cache_key_payload(query, relationship_k, options=options)
            cache[_cache_key_from_payload(key_payload)] = {
                "key_payload": key_payload,
                "paths": paths,
            }
            self._write(cache)
        except OSError:
            # Cache persistence is an optimization only; never fail search because
            # the container filesystem or mounted volume is not writable.
            return

    def clear(self) -> bool:
        try:
            self.path.unlink()
            return True
        except FileNotFoundError:
            return False

    def _read(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}

        try:
            with self.path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return {}

        entries = payload.get("entries")
        return entries if isinstance(entries, dict) else {}

    def _write(self, entries: dict[str, dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"entries": entries}
        with NamedTemporaryFile("w", encoding="utf-8", dir=self.path.parent, delete=False) as handle:
            json.dump(payload, handle)
            handle.write("\n")
            temp_path = Path(handle.name)
        temp_path.replace(self.path)


def normalized_query(query: str) -> str:
    return " ".join(query.casefold().split())


def _cache_key(
    query: str,
    relationship_k: int,
    options: dict[str, Any] | None = None,
) -> str:
    return _cache_key_from_payload(_cache_key_payload(query, relationship_k, options=options))


def _cache_key_payload(
    query: str,
    relationship_k: int,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "query": normalized_query(query),
        "relationship_k": int(relationship_k),
        "options": options or {},
    }


def _cache_key_from_payload(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
