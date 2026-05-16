"""
Tests that namespace signature fields (namespace_hotkey, namespace_sig,
namespace_timestamp_ms) are correctly wired through the miner HTTP handler
boundary into IngestSynapse and QuerySynapse, and that the sig-based auth
path in IngestHandler and QueryHandler is exercised end-to-end.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from engram.protocol import IngestSynapse, QuerySynapse
from engram.miner.ingest import IngestHandler
from engram.miner.query import QueryHandler
from engram.miner.store import FAISSStore, VectorRecord


_NS = "team_ns"
_HK = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
_SIG = "aabbccddeeff" * 8   # 96-char hex placeholder
_TS = 1_700_000_000_000


# ── Synapse construction from body dict (the HTTP handler boundary) ───────────

def test_ingest_synapse_carries_sig_fields():
    """Simulates what handle_ingest does: construct IngestSynapse from body."""
    body = {
        "text": "hello",
        "metadata": {},
        "namespace": _NS,
        "namespace_hotkey": _HK,
        "namespace_sig": _SIG,
        "namespace_timestamp_ms": _TS,
        "namespace_key": None,
    }
    syn = IngestSynapse(
        text                   = body.get("text"),
        raw_embedding          = body.get("raw_embedding"),
        metadata               = body.get("metadata") or {},
        namespace              = body.get("namespace") or None,
        namespace_hotkey       = body.get("namespace_hotkey") or None,
        namespace_sig          = body.get("namespace_sig") or None,
        namespace_timestamp_ms = body.get("namespace_timestamp_ms") or None,
        namespace_key          = body.get("namespace_key") or None,
    )
    assert syn.namespace_hotkey == _HK
    assert syn.namespace_sig == _SIG
    assert syn.namespace_timestamp_ms == _TS
    assert syn.namespace_key is None


def test_query_synapse_carries_sig_fields():
    """Simulates what handle_query does: construct QuerySynapse from body."""
    body = {
        "query_text": "find stuff",
        "top_k": 5,
        "namespace": _NS,
        "namespace_hotkey": _HK,
        "namespace_sig": _SIG,
        "namespace_timestamp_ms": _TS,
        "namespace_key": None,
    }
    syn = QuerySynapse(
        query_text             = body.get("query_text"),
        query_vector           = body.get("query_vector"),
        top_k                  = int(body.get("top_k", 10)),
        namespace              = body.get("namespace") or None,
        namespace_hotkey       = body.get("namespace_hotkey") or None,
        namespace_sig          = body.get("namespace_sig") or None,
        namespace_timestamp_ms = body.get("namespace_timestamp_ms") or None,
        namespace_key          = body.get("namespace_key") or None,
    )
    assert syn.namespace_hotkey == _HK
    assert syn.namespace_sig == _SIG
    assert syn.namespace_timestamp_ms == _TS
    assert syn.namespace_key is None


def test_ingest_synapse_missing_sig_fields_are_none():
    """Body without sig fields must not inject unexpected values."""
    body = {"text": "hello", "namespace": _NS, "namespace_key": "legacykey1234567"}
    syn = IngestSynapse(
        text                   = body.get("text"),
        namespace              = body.get("namespace") or None,
        namespace_hotkey       = body.get("namespace_hotkey") or None,
        namespace_sig          = body.get("namespace_sig") or None,
        namespace_timestamp_ms = body.get("namespace_timestamp_ms") or None,
        namespace_key          = body.get("namespace_key") or None,
    )
    assert syn.namespace_hotkey is None
    assert syn.namespace_sig is None
    assert syn.namespace_timestamp_ms is None
    assert syn.namespace_key == "legacykey1234567"


# ── Sig-based auth path through IngestHandler / QueryHandler ─────────────────

@pytest.fixture
def mock_ns_registry():
    reg = MagicMock()
    reg.verify_sig.return_value = True
    reg.exists.return_value = True
    reg.owner_hotkey.return_value = _HK
    return reg


@pytest.fixture
def ingest_handler(mock_ns_registry):
    store = FAISSStore(dim=4)
    embedder = MagicMock()
    embedder.embed.return_value = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    return IngestHandler(
        store=store,
        embedder=embedder,
        subtensor=None,
        netuid=1,
        namespace_registry=mock_ns_registry,
    )


@pytest.fixture
def query_handler(mock_ns_registry):
    store = FAISSStore(dim=4)
    store.upsert(VectorRecord(
        cid="test_cid",
        embedding=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        namespace=_NS,
    ))
    embedder = MagicMock()
    embedder.embed.return_value = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    return QueryHandler(store=store, embedder=embedder, namespace_registry=mock_ns_registry)


def test_ingest_handler_accepts_sig_auth(ingest_handler, mock_ns_registry):
    syn = IngestSynapse(
        text                   = "hello world",
        namespace              = _NS,
        namespace_hotkey       = _HK,
        namespace_sig          = _SIG,
        namespace_timestamp_ms = _TS,
    )
    result = ingest_handler.handle(syn)
    assert result.error is None
    mock_ns_registry.verify_sig.assert_called_once_with(_NS, _HK, _SIG, _TS)


def test_ingest_handler_rejects_bad_sig(ingest_handler, mock_ns_registry):
    mock_ns_registry.verify_sig.return_value = False
    syn = IngestSynapse(
        text                   = "hello",
        namespace              = _NS,
        namespace_hotkey       = _HK,
        namespace_sig          = "badsig",
        namespace_timestamp_ms = _TS,
    )
    result = ingest_handler.handle(syn)
    assert result.error is not None
    assert "signature" in result.error.lower() or "invalid" in result.error.lower()


def test_query_handler_accepts_sig_auth(query_handler, mock_ns_registry):
    syn = QuerySynapse(
        query_vector           = [0.1, 0.2, 0.3, 0.4],
        top_k                  = 5,
        namespace              = _NS,
        namespace_hotkey       = _HK,
        namespace_sig          = _SIG,
        namespace_timestamp_ms = _TS,
    )
    result = query_handler.handle(syn)
    assert result.error is None
    mock_ns_registry.verify_sig.assert_called_once_with(_NS, _HK, _SIG, _TS)


def test_query_handler_rejects_bad_sig(query_handler, mock_ns_registry):
    mock_ns_registry.verify_sig.return_value = False
    syn = QuerySynapse(
        query_vector           = [0.1, 0.2, 0.3, 0.4],
        top_k                  = 5,
        namespace              = _NS,
        namespace_hotkey       = _HK,
        namespace_sig          = "badsig",
        namespace_timestamp_ms = _TS,
    )
    result = query_handler.handle(syn)
    assert result.error is not None
