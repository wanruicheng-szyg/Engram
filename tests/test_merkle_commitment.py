"""
Merkle memory commitment — Python-layer tests.

Exercises the engram_core Python bindings for:
  build_commitment, generate_inclusion_proof, verify_inclusion,
  and the InclusionProof JSON round-trip.

These tests are skipped if the Rust wheel is not installed.
"""

from __future__ import annotations

import json
import pytest

try:
    import engram_core as ec
    _RUST = True
    _MERKLE = hasattr(ec, "build_commitment")
except ImportError:
    _RUST = False
    _MERKLE = False

pytestmark = pytest.mark.skipif(
    not _MERKLE,
    reason="engram_core Merkle API not in installed wheel (rebuild with maturin)",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fake_pair(n: int) -> tuple[str, str]:
    cid      = f"v1::{n:064x}"
    emb_hash = f"{(n + 100):064x}"
    return cid, emb_hash


# ── MemoryCommitment ──────────────────────────────────────────────────────────

def test_empty_corpus_has_zero_root() -> None:
    c = ec.build_commitment([], [])
    assert c.root_hex == "0" * 64
    assert c.count == 0


def test_single_memory_produces_nonzero_root() -> None:
    cid, emb = _fake_pair(1)
    c = ec.build_commitment([cid], [emb])
    assert c.root_hex != "0" * 64
    assert len(c.root_hex) == 64
    assert c.count == 1


def test_root_is_deterministic_regardless_of_ingest_order() -> None:
    pairs = [_fake_pair(i) for i in range(5)]
    cids  = [p[0] for p in pairs]
    embs  = [p[1] for p in pairs]

    c_forward  = ec.build_commitment(cids, embs)
    c_reversed = ec.build_commitment(list(reversed(cids)), list(reversed(embs)))
    assert c_forward.root_hex == c_reversed.root_hex


def test_adding_memory_changes_root() -> None:
    c1, e1 = _fake_pair(1)
    c2, e2 = _fake_pair(2)
    a = ec.build_commitment([c1], [e1])
    b = ec.build_commitment([c1, c2], [e1, e2])
    assert a.root_hex != b.root_hex


def test_corrupted_embedding_changes_root() -> None:
    cid, emb = _fake_pair(1)
    a = ec.build_commitment([cid], [emb])
    b = ec.build_commitment([cid], ["ff" * 32])
    assert a.root_hex != b.root_hex


# ── Inclusion proofs ──────────────────────────────────────────────────────────

def test_inclusion_proof_verifies() -> None:
    cid, emb = _fake_pair(1)
    commitment = ec.build_commitment([cid], [emb])
    proof = ec.generate_inclusion_proof(commitment, cid, emb)
    assert ec.verify_inclusion(commitment.root_hex, cid, emb, proof)


def test_inclusion_proof_for_all_corpus_sizes() -> None:
    for n in range(1, 18):
        pairs = [_fake_pair(i) for i in range(n)]
        cids  = [p[0] for p in pairs]
        embs  = [p[1] for p in pairs]
        commitment = ec.build_commitment(cids, embs)
        for cid, emb in pairs:
            proof = ec.generate_inclusion_proof(commitment, cid, emb)
            assert ec.verify_inclusion(commitment.root_hex, cid, emb, proof), (
                f"proof failed for n={n} cid={cid}"
            )


def test_wrong_cid_fails_verification() -> None:
    c1, e1 = _fake_pair(1)
    c2, e2 = _fake_pair(2)
    commitment = ec.build_commitment([c1, c2], [e1, e2])
    proof = ec.generate_inclusion_proof(commitment, c1, e1)
    assert not ec.verify_inclusion(commitment.root_hex, c2, e2, proof)


def test_wrong_embedding_fails_verification() -> None:
    cid, emb = _fake_pair(1)
    commitment = ec.build_commitment([cid], [emb])
    proof = ec.generate_inclusion_proof(commitment, cid, emb)
    assert not ec.verify_inclusion(commitment.root_hex, cid, "ff" * 32, proof)


def test_non_member_raises() -> None:
    c1, e1 = _fake_pair(1)
    c2, e2 = _fake_pair(2)
    commitment = ec.build_commitment([c1], [e1])
    with pytest.raises((ValueError, Exception)):
        ec.generate_inclusion_proof(commitment, c2, e2)


def test_proof_against_wrong_root_fails() -> None:
    c1, e1 = _fake_pair(1)
    c2, e2 = _fake_pair(2)
    ca = ec.build_commitment([c1], [e1])
    cb = ec.build_commitment([c2], [e2])
    proof = ec.generate_inclusion_proof(ca, c1, e1)
    assert not ec.verify_inclusion(cb.root_hex, c1, e1, proof)


# ── InclusionProof JSON round-trip ────────────────────────────────────────────

def test_proof_json_round_trip() -> None:
    c1, e1 = _fake_pair(1)
    c2, e2 = _fake_pair(2)
    commitment = ec.build_commitment([c1, c2], [e1, e2])
    proof = ec.generate_inclusion_proof(commitment, c1, e1)

    # Serialize
    proof_json = proof.to_json()
    assert isinstance(proof_json, str)
    parsed = json.loads(proof_json)
    assert "leaf_hex" in parsed
    assert "steps" in parsed

    # Deserialize and verify
    proof2 = ec.MemoryInclusionProof.from_json(proof_json)
    assert ec.verify_inclusion(commitment.root_hex, c1, e1, proof2)


def test_tampered_proof_json_fails() -> None:
    cid, emb = _fake_pair(7)
    commitment = ec.build_commitment([_fake_pair(i)[0] for i in range(8)],
                                     [_fake_pair(i)[1] for i in range(8)])
    proof = ec.generate_inclusion_proof(commitment, cid, emb)
    data = json.loads(proof.to_json())
    # Flip first byte of the first sibling
    if data["steps"]:
        old_sib = data["steps"][0]["sibling"]
        flipped  = f"{int(old_sib[:2], 16) ^ 0xff:02x}" + old_sib[2:]
        data["steps"][0]["sibling"] = flipped
        bad_proof = ec.MemoryInclusionProof.from_json(json.dumps(data))
        assert not ec.verify_inclusion(commitment.root_hex, cid, emb, bad_proof)


def test_proof_depth_matches_expected() -> None:
    """Proof depth is ceil(log2(n)) — validate the InclusionProof.depth field."""
    import math
    for n in [1, 2, 4, 8, 16]:
        pairs = [_fake_pair(i) for i in range(n)]
        cids  = [p[0] for p in pairs]
        embs  = [p[1] for p in pairs]
        commitment = ec.build_commitment(cids, embs)
        proof = ec.generate_inclusion_proof(commitment, cids[0], embs[0])
        expected_depth = 0 if n == 1 else math.ceil(math.log2(n))
        assert proof.depth == expected_depth, f"n={n}: depth={proof.depth}, expected={expected_depth}"
