from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

CURIE_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*:[^\s:][^\s]*$")


@dataclass(frozen=True)
class BridgeResolutionConfig:
    conflate_gene_protein: bool = True
    conflate_drug_chemical: bool = True

    def to_cache_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BridgeResolution:
    status: str
    curie: str | None = None
    candidates: tuple[str, ...] = ()
    reason: str = ""
    config: BridgeResolutionConfig = BridgeResolutionConfig()

    @property
    def is_resolved(self) -> bool:
        return self.status == "resolved" and bool(self.curie)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "curie": self.curie,
            "candidates": list(self.candidates),
            "reason": self.reason,
            "config": self.config.to_cache_dict(),
        }


def resolve_litcoin_bridge(
    node: Mapping[str, Any] | None,
    config: BridgeResolutionConfig | None = None,
) -> BridgeResolution:
    """Resolve a selected LitCoin node to a normalized CURIE for ROBOKOP lookup.

    Display labels are intentionally ignored. A bridge requires an explicit
    identifier-like field, otherwise equal names across graphs could create
    false joins.
    """
    config = config or BridgeResolutionConfig()
    if not node:
        return BridgeResolution(
            status="zero_mapping",
            reason="No selected LitCoin node is available.",
            config=config,
        )

    primary = _first_curie(node.get("curie"), node.get("normalized_curie"))
    if primary:
        return BridgeResolution(
            status="resolved",
            curie=primary,
            candidates=(primary,),
            reason="Resolved from explicit normalized CURIE.",
            config=config,
        )

    properties = node.get("properties") if isinstance(node.get("properties"), Mapping) else {}
    primary = _first_curie(properties.get("curie"), properties.get("normalized_curie"), properties.get("id"))
    if primary:
        return BridgeResolution(
            status="resolved",
            curie=primary,
            candidates=(primary,),
            reason="Resolved from node properties.",
            config=config,
        )

    candidate_fields = (
        properties.get("equivalent_identifiers"),
        properties.get("equivalent_ids"),
        properties.get("same_as"),
        properties.get("xref"),
    )
    candidates = _unique_curies_from_values(candidate_fields)
    if len(candidates) == 1:
        return BridgeResolution(
            status="resolved",
            curie=candidates[0],
            candidates=tuple(candidates),
            reason="Resolved from a single equivalent identifier.",
            config=config,
        )
    if len(candidates) > 1:
        return BridgeResolution(
            status="ambiguous_mapping",
            candidates=tuple(candidates),
            reason="Multiple normalized identifiers are available; choose one explicitly before remote expansion.",
            config=config,
        )

    return BridgeResolution(
        status="zero_mapping",
        reason="No normalized CURIE is available; display-name equality is not sufficient for bridging.",
        config=config,
    )


def looks_like_curie(value: Any) -> bool:
    return isinstance(value, str) and bool(CURIE_PATTERN.match(value.strip()))


def _first_curie(*values: Any) -> str | None:
    for value in values:
        if looks_like_curie(value):
            return str(value).strip()
    return None


def _unique_curies_from_values(values: tuple[Any, ...]) -> list[str]:
    seen: set[str] = set()
    curies: list[str] = []
    for value in values:
        for candidate in _flatten(value):
            if not looks_like_curie(candidate):
                continue
            curie = str(candidate).strip()
            if curie not in seen:
                seen.add(curie)
                curies.append(curie)
    return curies


def _flatten(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        if "identifier" in value:
            return [value["identifier"]]
        if "id" in value:
            return [value["id"]]
        return []
    if isinstance(value, (list, tuple, set)):
        flattened: list[Any] = []
        for item in value:
            flattened.extend(_flatten(item))
        return flattened
    return [value]
