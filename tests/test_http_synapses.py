"""Regression tests for HTTP body → Synapse mapping helpers."""

from engram.miner.http_synapses import ingest_synapse_from_body, query_synapse_from_body


# ── IngestSynapse ─────────────────────────────────────────────────────────────

def test_ingest_preserves_sig_fields():
    body = {
        "text": "hello world",
        "namespace": "my_ns",
        "namespace_hotkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
        "namespace_sig": "0xdeadbeef1234",
        "namespace_timestamp_ms": 1700000000000,
    }
    s = ingest_synapse_from_body(body)
    assert s.namespace == "my_ns"
    assert s.namespace_hotkey == "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    assert s.namespace_sig == "0xdeadbeef1234"
    assert s.namespace_timestamp_ms == 1700000000000
    assert s.namespace_key is None


def test_ingest_legacy_namespace_key_fallback():
    body = {"text": "hello", "namespace": "legacy_ns", "namespace_key": "secretkey12345678"}
    s = ingest_synapse_from_body(body)
    assert s.namespace == "legacy_ns"
    assert s.namespace_key == "secretkey12345678"
    assert s.namespace_sig is None
    assert s.namespace_hotkey is None


def test_ingest_empty_namespace_becomes_none():
    s = ingest_synapse_from_body({"text": "hello", "namespace": ""})
    assert s.namespace is None


def test_ingest_timestamp_zero_not_dropped():
    # `value or None` bug: 0 or None == None — must NOT drop a zero timestamp
    s = ingest_synapse_from_body({"text": "x", "namespace_timestamp_ms": 0})
    assert s.namespace_timestamp_ms == 0


def test_ingest_no_namespace_fields():
    s = ingest_synapse_from_body({"text": "public content"})
    assert s.namespace is None
    assert s.namespace_hotkey is None
    assert s.namespace_sig is None
    assert s.namespace_timestamp_ms is None
    assert s.namespace_key is None


def test_ingest_raw_embedding_preserved():
    vec = [0.1, 0.2, 0.3]
    s = ingest_synapse_from_body({"raw_embedding": vec, "metadata": {"src": "test"}})
    assert s.raw_embedding == vec
    assert s.metadata == {"src": "test"}
    assert s.text is None


def test_ingest_model_version_default():
    s = ingest_synapse_from_body({"text": "hi"})
    assert s.model_version == "v1"


def test_ingest_model_version_custom():
    s = ingest_synapse_from_body({"text": "hi", "model_version": "v2"})
    assert s.model_version == "v2"


# ── QuerySynapse ──────────────────────────────────────────────────────────────

def test_query_preserves_sig_fields():
    body = {
        "query_text": "find memories",
        "top_k": 5,
        "namespace": "q_ns",
        "namespace_hotkey": "5ABC",
        "namespace_sig": "0xsig123",
        "namespace_timestamp_ms": 9_999_999,
    }
    s = query_synapse_from_body(body)
    assert s.namespace == "q_ns"
    assert s.namespace_hotkey == "5ABC"
    assert s.namespace_sig == "0xsig123"
    assert s.namespace_timestamp_ms == 9_999_999
    assert s.namespace_key is None


def test_query_legacy_namespace_key_fallback():
    body = {"query_text": "find", "namespace": "old_ns", "namespace_key": "legacykey12345678"}
    s = query_synapse_from_body(body)
    assert s.namespace_key == "legacykey12345678"
    assert s.namespace_sig is None


def test_query_empty_namespace_becomes_none():
    s = query_synapse_from_body({"query_text": "q", "namespace": ""})
    assert s.namespace is None


def test_query_timestamp_zero_not_dropped():
    s = query_synapse_from_body({"query_text": "q", "namespace_timestamp_ms": 0})
    assert s.namespace_timestamp_ms == 0


def test_query_top_k_default():
    s = query_synapse_from_body({"query_text": "q"})
    assert s.top_k == 10


def test_query_vector_preserved():
    vec = [0.5, 0.5]
    s = query_synapse_from_body({"query_vector": vec, "top_k": 3})
    assert s.query_vector == vec
    assert s.top_k == 3
    assert s.query_text is None
