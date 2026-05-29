"""
Tests for engram/validator/consensus.py — ConsensusEngine.

Covers:
  - ScoreVector serialization round-trip
  - Canonical JSON is deterministic (key order independent)
  - cosine_distance: identical, orthogonal, zero vectors
  - create_vector stores and exposes own vector
  - ingest_peer_vector: accepted, self-rejected, stale-rejected
  - has_quorum: min=0 always passes, min=2 needs 2 peers
  - Disagreeing peer not counted toward quorum
  - aggregate: single validator, two validators, missing UIDs
  - Aggregate gives median (odd and even validator counts)
  - Persistence: reload from disk
  - agreement_report structure
  - peer_count
  - Quorum not met when own vector is for a different block
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from engram.validator.consensus import ConsensusEngine, ScoreVector, cosine_distance


# ── Helpers ───────────────────────────────────────────────────────────────────

def _engine(tmp_path: Path, min_validators: int = 0, hotkey: str = "5ValidatorA") -> ConsensusEngine:
    return ConsensusEngine(
        validator_hotkey=hotkey,
        db_path=tmp_path / "consensus.db",
        min_validators=min_validators,
        score_tolerance=0.15,
        vector_ttl_secs=600,
    )


def _scores(base: float = 1.0, n: int = 5) -> dict[int, float]:
    return {uid: base * (1.0 - uid * 0.05) for uid in range(n)}


def _peer_vec(
    hotkey: str = "5PeerB",
    block: int = 100,
    scores: dict[int, float] | None = None,
    ts: int | None = None,
) -> ScoreVector:
    return ScoreVector(
        validator_hotkey=hotkey,
        block=block,
        scores=scores or _scores(0.95),
        timestamp_ms=ts if ts is not None else int(time.time() * 1000),
        signature="",
    )


# ── ScoreVector ───────────────────────────────────────────────────────────────

def test_score_vector_round_trip():
    sv = ScoreVector(
        validator_hotkey="5ABC",
        block=42,
        scores={0: 0.9, 1: 0.7},
        timestamp_ms=1_700_000_000_000,
        signature="0xdeadbeef",
    )
    d = sv.to_dict()
    sv2 = ScoreVector.from_dict(d)
    assert sv2.validator_hotkey == sv.validator_hotkey
    assert sv2.block == sv.block
    assert sv2.scores == sv.scores
    assert sv2.timestamp_ms == sv.timestamp_ms
    assert sv2.signature == sv.signature


def test_canonical_json_is_deterministic():
    sv = ScoreVector("5A", 1, {3: 0.1, 1: 0.5, 2: 0.3}, 999, "")
    j1 = sv.canonical_json()
    j2 = sv.canonical_json()
    assert j1 == j2
    # Keys in scores must be sorted
    parsed = json.loads(j1)
    keys = list(parsed["scores"].keys())
    assert keys == sorted(keys)


def test_canonical_json_key_order_independent():
    a = ScoreVector("5A", 1, {3: 0.1, 1: 0.5, 2: 0.3}, 999, "")
    b = ScoreVector("5A", 1, {1: 0.5, 2: 0.3, 3: 0.1}, 999, "")
    assert a.canonical_json() == b.canonical_json()


# ── cosine_distance ───────────────────────────────────────────────────────────

def test_cosine_distance_identical():
    s = {0: 0.8, 1: 0.6, 2: 0.4}
    assert cosine_distance(s, s) == pytest.approx(0.0, abs=1e-9)


def test_cosine_distance_orthogonal():
    a = {0: 1.0, 1: 0.0}
    b = {0: 0.0, 1: 1.0}
    assert cosine_distance(a, b) == pytest.approx(1.0, abs=1e-9)


def test_cosine_distance_zero_vector():
    a = {0: 0.0, 1: 0.0}
    b = {0: 0.5, 1: 0.5}
    assert cosine_distance(a, b) == 1.0


def test_cosine_distance_missing_uid_treated_as_zero():
    a = {0: 1.0, 1: 1.0}
    b = {0: 1.0}          # uid 1 missing → treated as 0.0
    # dist = 1 - (1*1+1*0)/(sqrt(2)*1) = 1 - 1/sqrt(2) ≈ 0.293
    assert cosine_distance(a, b) == pytest.approx(1.0 - 1.0 / (2 ** 0.5), abs=1e-6)


def test_cosine_distance_empty_dicts():
    assert cosine_distance({}, {}) == 0.0


def test_cosine_distance_symmetric():
    a = {0: 0.9, 1: 0.3}
    b = {0: 0.7, 1: 0.8}
    assert cosine_distance(a, b) == pytest.approx(cosine_distance(b, a), abs=1e-12)


# ── create_vector ─────────────────────────────────────────────────────────────

def test_create_vector_stores_own_vector(tmp_path):
    eng = _engine(tmp_path)
    vec = eng.create_vector(_scores(), block=100)
    assert eng._own_vector is not None
    assert eng._own_vector.block == 100
    assert vec.validator_hotkey == "5ValidatorA"


def test_create_vector_unsigned_by_default(tmp_path):
    eng = _engine(tmp_path)
    vec = eng.create_vector(_scores(), block=100)
    assert vec.signature == ""


def test_create_vector_signed_with_keypair(tmp_path):
    eng = _engine(tmp_path)
    keypair = MagicMock()
    keypair.sign.return_value = bytes(64)  # 64 zero bytes
    vec = eng.create_vector(_scores(), block=100, keypair=keypair)
    assert vec.signature.startswith("0x")
    keypair.sign.assert_called_once()


def test_create_vector_persists_to_db(tmp_path):
    eng = _engine(tmp_path)
    eng.create_vector({0: 0.8, 1: 0.6}, block=50)
    # Reload engine from same DB
    eng2 = ConsensusEngine(
        validator_hotkey="5ValidatorA",
        db_path=tmp_path / "consensus.db",
    )
    # Own vector is re-created, but DB contains the stored entry
    rows = eng2._db.execute(
        "SELECT block FROM score_vectors WHERE validator_hotkey='5ValidatorA'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == 50


# ── ingest_peer_vector ────────────────────────────────────────────────────────

def test_ingest_peer_vector_accepted(tmp_path):
    eng = _engine(tmp_path)
    peer = _peer_vec()
    assert eng.ingest_peer_vector(peer) is True


def test_ingest_self_vector_rejected(tmp_path):
    eng = _engine(tmp_path)
    self_vec = _peer_vec(hotkey="5ValidatorA")
    assert eng.ingest_peer_vector(self_vec) is False


def test_ingest_stale_vector_rejected(tmp_path):
    eng = _engine(tmp_path, min_validators=1)
    stale_ts = int((time.time() - 700) * 1000)  # 700s ago > TTL 600s
    old_vec = _peer_vec(ts=stale_ts)
    assert eng.ingest_peer_vector(old_vec, check_ttl=True) is False


def test_ingest_stale_vector_accepted_without_check(tmp_path):
    eng = _engine(tmp_path)
    stale_ts = int((time.time() - 700) * 1000)
    old_vec = _peer_vec(ts=stale_ts)
    assert eng.ingest_peer_vector(old_vec, check_ttl=False) is True


def test_ingest_duplicate_overrides(tmp_path):
    eng = _engine(tmp_path)
    peer = _peer_vec(hotkey="5PeerB", block=100, scores={0: 0.5})
    eng.ingest_peer_vector(peer)
    peer2 = _peer_vec(hotkey="5PeerB", block=100, scores={0: 0.9})
    eng.ingest_peer_vector(peer2)
    # Only one row per (hotkey, block)
    rows = eng._db.execute(
        "SELECT COUNT(*) FROM score_vectors WHERE validator_hotkey='5PeerB' AND block=100"
    ).fetchone()[0]
    assert rows == 1


# ── has_quorum ────────────────────────────────────────────────────────────────

def test_has_quorum_min_zero_always_true(tmp_path):
    eng = _engine(tmp_path, min_validators=0)
    # No own vector, no peers — still True because min=0
    assert eng.has_quorum(block=100) is True


def test_has_quorum_no_own_vector(tmp_path):
    eng = _engine(tmp_path, min_validators=2)
    assert eng.has_quorum(block=100) is False


def test_has_quorum_wrong_block(tmp_path):
    eng = _engine(tmp_path, min_validators=1)
    eng.create_vector(_scores(), block=99)
    assert eng.has_quorum(block=100) is False  # own vector is for block 99


def test_has_quorum_met_with_agreeing_peers(tmp_path):
    eng = _engine(tmp_path, min_validators=2)
    eng.create_vector(_scores(1.0), block=100)
    # Two peers with nearly identical scores
    eng.ingest_peer_vector(_peer_vec("5PeerB", 100, _scores(0.98)))
    eng.ingest_peer_vector(_peer_vec("5PeerC", 100, _scores(0.97)))
    assert eng.has_quorum(block=100) is True


def test_has_quorum_not_met_insufficient_peers(tmp_path):
    eng = _engine(tmp_path, min_validators=2)
    eng.create_vector(_scores(1.0), block=100)
    eng.ingest_peer_vector(_peer_vec("5PeerB", 100, _scores(0.98)))
    # Only 1 peer, need 2
    assert eng.has_quorum(block=100) is False


def test_has_quorum_disagreeing_peer_not_counted(tmp_path):
    eng = _engine(tmp_path, min_validators=1)
    eng.create_vector({0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}, block=100)
    # Completely different score distribution (cosine dist ≈ 1.0)
    eng.ingest_peer_vector(_peer_vec("5PeerB", 100, {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 1.0}))
    assert eng.has_quorum(block=100) is False


# ── aggregate ─────────────────────────────────────────────────────────────────

def test_aggregate_single_validator_returns_own_scores(tmp_path):
    eng = _engine(tmp_path)
    own = {0: 0.8, 1: 0.6}
    eng.create_vector(own, block=42)
    result = eng.aggregate(block=42)
    assert result == pytest.approx(own)


def test_aggregate_two_validators_median(tmp_path):
    eng = _engine(tmp_path)
    eng.create_vector({0: 1.0, 1: 0.5}, block=10)
    eng.ingest_peer_vector(_peer_vec("5PeerB", 10, {0: 0.8, 1: 0.7}))
    result = eng.aggregate(block=10)
    # median([1.0, 0.8]) = 0.9, median([0.5, 0.7]) = 0.6
    assert result[0] == pytest.approx(0.9, abs=1e-6)
    assert result[1] == pytest.approx(0.6, abs=1e-6)


def test_aggregate_three_validators_median(tmp_path):
    eng = _engine(tmp_path)
    eng.create_vector({0: 1.0}, block=5)
    eng.ingest_peer_vector(_peer_vec("5B", 5, {0: 0.6}))
    eng.ingest_peer_vector(_peer_vec("5C", 5, {0: 0.8}))
    result = eng.aggregate(block=5)
    # median([1.0, 0.6, 0.8]) = 0.8
    assert result[0] == pytest.approx(0.8, abs=1e-6)


def test_aggregate_missing_uid_filled_with_zero(tmp_path):
    eng = _engine(tmp_path)
    eng.create_vector({0: 1.0, 1: 0.5}, block=7)
    # Peer only has uid 0; uid 1 missing → 0.0
    eng.ingest_peer_vector(_peer_vec("5B", 7, {0: 0.8}))
    result = eng.aggregate(block=7)
    assert 1 in result
    # median([0.5, 0.0]) = 0.25
    assert result[1] == pytest.approx(0.25, abs=1e-6)


def test_aggregate_no_own_vector_returns_empty(tmp_path):
    eng = _engine(tmp_path)
    assert eng.aggregate(block=99) == {}


def test_aggregate_wrong_block_returns_empty(tmp_path):
    eng = _engine(tmp_path)
    eng.create_vector({0: 1.0}, block=1)
    assert eng.aggregate(block=99) == {}


# ── peer_count ────────────────────────────────────────────────────────────────

def test_peer_count_zero_initially(tmp_path):
    eng = _engine(tmp_path)
    assert eng.peer_count(block=100) == 0


def test_peer_count_increments(tmp_path):
    eng = _engine(tmp_path)
    eng.ingest_peer_vector(_peer_vec("5B", 100))
    eng.ingest_peer_vector(_peer_vec("5C", 100))
    assert eng.peer_count(block=100) == 2


def test_peer_count_different_block_not_counted(tmp_path):
    eng = _engine(tmp_path)
    eng.ingest_peer_vector(_peer_vec("5B", block=99))
    assert eng.peer_count(block=100) == 0


# ── agreement_report ─────────────────────────────────────────────────────────

def test_agreement_report_structure(tmp_path):
    eng = _engine(tmp_path)
    eng.create_vector({0: 1.0, 1: 0.8}, block=100)
    eng.ingest_peer_vector(_peer_vec("5B", 100, {0: 0.95, 1: 0.75}))
    report = eng.agreement_report(block=100)
    assert len(report) == 1
    entry = report[0]
    assert "validator" in entry
    assert "cosine_distance" in entry
    assert "agrees" in entry
    assert isinstance(entry["cosine_distance"], float)


def test_agreement_report_empty_without_own_vector(tmp_path):
    eng = _engine(tmp_path)
    assert eng.agreement_report(block=100) == []


# ── Persistence ───────────────────────────────────────────────────────────────

def test_peer_vectors_survive_restart(tmp_path):
    db = tmp_path / "consensus.db"
    eng = ConsensusEngine(
        validator_hotkey="5ValidatorA",
        db_path=db,
        vector_ttl_secs=600,
    )
    eng.create_vector({0: 1.0}, block=200)
    eng.ingest_peer_vector(_peer_vec("5PeerB", 200, {0: 0.95}))

    # Reload
    eng2 = ConsensusEngine(
        validator_hotkey="5ValidatorA",
        db_path=db,
        vector_ttl_secs=600,
    )
    # Peer vector should still be in DB
    peers = eng2._peer_vectors(block=200)
    assert len(peers) == 1
    assert peers[0].validator_hotkey == "5PeerB"
