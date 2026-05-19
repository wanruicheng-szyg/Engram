"""
Engram Miner — HTTP body → Synapse helpers

Converts raw JSON request bodies into typed IngestSynapse / QuerySynapse
objects so the mapping logic is testable independently of the HTTP server.
"""

from __future__ import annotations

from typing import Any

from engram.protocol import IngestSynapse, QuerySynapse


def _parse_int(value: Any) -> int | None:
    """Return int(value) or None — never coerces 0 to None unlike `value or None`."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def ingest_synapse_from_body(body: dict[str, Any]) -> IngestSynapse:
    """Map an /IngestSynapse JSON body to a typed IngestSynapse."""
    return IngestSynapse(
        text                  = body.get("text") or None,
        raw_embedding         = body.get("raw_embedding") or None,
        metadata              = body.get("metadata") or {},
        model_version         = body.get("model_version") or "v1",
        namespace             = body.get("namespace") or None,
        namespace_hotkey      = body.get("namespace_hotkey") or None,
        namespace_sig         = body.get("namespace_sig") or None,
        namespace_timestamp_ms= _parse_int(body.get("namespace_timestamp_ms")),
        namespace_key         = body.get("namespace_key") or None,
    )


def query_synapse_from_body(body: dict[str, Any]) -> QuerySynapse:
    """Map a /QuerySynapse JSON body to a typed QuerySynapse."""
    return QuerySynapse(
        query_text            = body.get("query_text") or None,
        query_vector          = body.get("query_vector") or None,
        top_k                 = int(body.get("top_k") or 10),
        namespace             = body.get("namespace") or None,
        namespace_hotkey      = body.get("namespace_hotkey") or None,
        namespace_sig         = body.get("namespace_sig") or None,
        namespace_timestamp_ms= _parse_int(body.get("namespace_timestamp_ms")),
        namespace_key         = body.get("namespace_key") or None,
    )
