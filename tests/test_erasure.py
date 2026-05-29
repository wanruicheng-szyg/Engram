"""
Tests for engram/storage/erasure.py — (k, n) erasure coding.

Covers:
  - Correct number and shape of shards
  - Round-trip from all n shards
  - Round-trip from any k-subset (all C(n,k) combinations for small n)
  - Round-trip with parity-only shards (worst case for condition number)
  - All-zero and ones embeddings
  - Padding: dimensions not evenly divisible by k
  - Full 1536-dim embedding (default EMBEDDING_DIM)
  - Insufficient shards raises ValueError
  - Out-of-range shard indices raise ValueError
  - Invalid constructor parameters raise ValueError
  - MDS property (any k×k submatrix invertible)
  - shard_cid / parse_shard_cid helpers
  - is_shard_cid helper
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from engram.storage.erasure import (
    ErasureCoder,
    _chunk_size,
    is_shard_cid,
    parse_shard_cid,
    shard_cid,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def coder35() -> ErasureCoder:
    return ErasureCoder(k=3, n=5)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


def _rand(dim: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)  # unit vector (typical embedding)


# ── Shard count and shapes ────────────────────────────────────────────────────

def test_encode_returns_n_shards(coder35, rng):
    shards = coder35.encode(_rand(15, rng))
    assert len(shards) == 5


def test_shard_shapes_consistent(coder35, rng):
    dim = 15
    shards = coder35.encode(_rand(dim, rng))
    expected_chunk = _chunk_size(dim, 3)
    for s in shards:
        assert s.shape == (expected_chunk,)


def test_shard_dtype_is_float32(coder35, rng):
    shards = coder35.encode(_rand(9, rng))
    for s in shards:
        assert s.dtype == np.float32


def test_generator_matrix_shape(coder35):
    assert coder35._G.shape == (5, 3)


def test_shard_size_helper():
    c = ErasureCoder(k=3, n=5)
    assert c.shard_size(9) == 3     # 9/3 = 3 exactly
    assert c.shard_size(10) == 4    # ceil(10/3) = 4


# ── Round-trip: all n shards ──────────────────────────────────────────────────

def test_roundtrip_all_shards(coder35, rng):
    emb = _rand(15, rng)
    shards = coder35.encode(emb)
    recovered = coder35.decode({i: shards[i] for i in range(5)}, orig_dim=15)
    np.testing.assert_allclose(recovered, emb, atol=1e-5)


def test_roundtrip_preserves_orig_dim(coder35, rng):
    emb = _rand(15, rng)
    shards = coder35.encode(emb)
    recovered = coder35.decode({i: shards[i] for i in range(5)}, orig_dim=15)
    assert recovered.shape == (15,)


# ── Round-trip: every C(n, k) subset ─────────────────────────────────────────

@pytest.mark.parametrize("indices", list(combinations(range(5), 3)))
def test_roundtrip_all_k_subsets(indices, rng):
    coder = ErasureCoder(k=3, n=5)
    emb = _rand(15, rng)
    shards = coder.encode(emb)
    subset = {i: shards[i] for i in indices}
    recovered = coder.decode(subset, orig_dim=15)
    np.testing.assert_allclose(recovered, emb, atol=1e-4)


# ── Round-trip: parity shards only ───────────────────────────────────────────

def test_roundtrip_parity_shards_only(coder35, rng):
    """Reconstruct using only the n-k parity shards — worst-case condition."""
    emb = _rand(15, rng)
    shards = coder35.encode(emb)
    # Parity shards are indices 3, 4 + one more (need k=3 total)
    subset = {2: shards[2], 3: shards[3], 4: shards[4]}
    recovered = coder35.decode(subset, orig_dim=15)
    np.testing.assert_allclose(recovered, emb, atol=1e-4)


# ── Special embeddings ────────────────────────────────────────────────────────

def test_all_zero_embedding(coder35):
    emb = np.zeros(12, dtype=np.float32)
    shards = coder35.encode(emb)
    # All shards should also be (near) zero
    for s in shards:
        np.testing.assert_allclose(s, 0.0, atol=1e-9)
    recovered = coder35.decode({i: shards[i] for i in range(3)}, orig_dim=12)
    np.testing.assert_allclose(recovered, emb, atol=1e-9)


def test_all_ones_embedding(coder35):
    emb = np.ones(12, dtype=np.float32)
    shards = coder35.encode(emb)
    recovered = coder35.decode({i: shards[i] for i in range(3)}, orig_dim=12)
    np.testing.assert_allclose(recovered, emb, atol=1e-5)


def test_unit_basis_vector(coder35):
    dim = 12
    emb = np.zeros(dim, dtype=np.float32)
    emb[0] = 1.0
    shards = coder35.encode(emb)
    recovered = coder35.decode({i: shards[i] for i in range(3)}, orig_dim=dim)
    np.testing.assert_allclose(recovered, emb, atol=1e-5)


# ── Padding: dim not divisible by k ──────────────────────────────────────────

def test_padding_roundtrip_non_divisible(rng):
    coder = ErasureCoder(k=3, n=5)
    dim = 10  # ceil(10/3) = 4, padded to 12
    emb = _rand(dim, rng)
    shards = coder.encode(emb)
    recovered = coder.decode({i: shards[i] for i in range(3)}, orig_dim=dim)
    assert recovered.shape == (dim,)
    np.testing.assert_allclose(recovered, emb, atol=1e-4)


def test_padding_roundtrip_dim_1(rng):
    """Extreme case: 1-element embedding."""
    coder = ErasureCoder(k=2, n=3)
    emb = np.array([3.14159], dtype=np.float32)
    shards = coder.encode(emb)
    recovered = coder.decode({0: shards[0], 1: shards[1]}, orig_dim=1)
    np.testing.assert_allclose(recovered, emb, atol=1e-5)


# ── Full 1536-dim embedding ───────────────────────────────────────────────────

def test_roundtrip_full_embedding_dim(rng):
    """EMBEDDING_DIM=1536, k=3, n=5 — default configuration."""
    from engram.config import EMBEDDING_DIM
    coder = ErasureCoder(k=3, n=5)
    emb = _rand(EMBEDDING_DIM, rng)
    shards = coder.encode(emb)
    assert len(shards) == 5
    assert all(s.shape == (512,) for s in shards)   # 1536/3 = 512

    # Reconstruct from three different subsets
    for combo in [(0, 1, 2), (0, 3, 4), (1, 2, 4)]:
        subset = {i: shards[i] for i in combo}
        recovered = coder.decode(subset, orig_dim=EMBEDDING_DIM)
        np.testing.assert_allclose(recovered, emb, atol=1e-4)


# ── Error cases ───────────────────────────────────────────────────────────────

def test_insufficient_shards_raises(coder35, rng):
    emb = _rand(15, rng)
    shards = coder35.encode(emb)
    with pytest.raises(ValueError, match="at least 3"):
        coder35.decode({0: shards[0], 1: shards[1]})  # only 2


def test_out_of_range_shard_index_raises(coder35, rng):
    emb = _rand(15, rng)
    shards = coder35.encode(emb)
    with pytest.raises(ValueError, match="out of range"):
        coder35.decode({0: shards[0], 1: shards[1], 99: shards[2]})


def test_invalid_k_less_than_2():
    with pytest.raises(ValueError, match="k must be"):
        ErasureCoder(k=1, n=3)


def test_invalid_n_less_than_k():
    with pytest.raises(ValueError, match="n must be >= k"):
        ErasureCoder(k=5, n=3)


def test_invalid_n_too_large():
    with pytest.raises(ValueError, match="<= 255"):
        ErasureCoder(k=3, n=256)


# ── MDS property ─────────────────────────────────────────────────────────────

def test_mds_property_k3_n5():
    """All C(5,3)=10 submatrices must be non-singular."""
    assert ErasureCoder(k=3, n=5).is_mds()


def test_mds_property_k2_n4():
    assert ErasureCoder(k=2, n=4).is_mds()


def test_mds_property_k4_n6():
    assert ErasureCoder(k=4, n=6).is_mds()


# ── k == n (degenerate: zero parity shards) ───────────────────────────────────

def test_k_equals_n_roundtrip(rng):
    coder = ErasureCoder(k=4, n=4)
    emb = _rand(8, rng)
    shards = coder.encode(emb)
    assert len(shards) == 4
    recovered = coder.decode({i: shards[i] for i in range(4)}, orig_dim=8)
    np.testing.assert_allclose(recovered, emb, atol=1e-5)


# ── decode without orig_dim ───────────────────────────────────────────────────

def test_decode_without_orig_dim_returns_padded(rng):
    """Without orig_dim, caller gets the zero-padded result."""
    coder = ErasureCoder(k=3, n=5)
    dim = 10  # not divisible by 3; padded to 12
    emb = _rand(dim, rng)
    shards = coder.encode(emb)
    recovered = coder.decode({i: shards[i] for i in range(3)})
    assert recovered.shape == (12,)
    np.testing.assert_allclose(recovered[:dim], emb, atol=1e-4)


# ── More than k shards provided ───────────────────────────────────────────────

def test_decode_uses_first_k_when_more_provided(coder35, rng):
    """Providing all 5 shards is allowed; decoder uses the first k."""
    emb = _rand(15, rng)
    shards = coder35.encode(emb)
    recovered = coder35.decode({i: shards[i] for i in range(5)}, orig_dim=15)
    np.testing.assert_allclose(recovered, emb, atol=1e-5)


# ── Shard CID helpers ─────────────────────────────────────────────────────────

def test_shard_cid_format():
    cid = shard_cid("v1::abc123", 2, 3, 5)
    assert cid == "v1::abc123::ec:2/3/5"


def test_parse_shard_cid_valid():
    cid = shard_cid("v1::abc123", 1, 3, 5)
    result = parse_shard_cid(cid)
    assert result == ("v1::abc123", 1, 3, 5)


def test_parse_shard_cid_non_shard_returns_none():
    assert parse_shard_cid("v1::abc123") is None


def test_parse_shard_cid_malformed_returns_none():
    assert parse_shard_cid("v1::abc::ec:bad/data") is None
    assert parse_shard_cid("v1::abc::ec:1/2") is None    # only 2 parts
    assert parse_shard_cid("v1::abc::ec:1/2/3/4") is None  # 4 parts


def test_parse_shard_cid_out_of_range_index():
    # shard 5 out of range for n=5 (valid: 0..4)
    assert parse_shard_cid("v1::abc::ec:5/3/5") is None


def test_is_shard_cid_true():
    assert is_shard_cid("v1::abc::ec:0/3/5") is True


def test_is_shard_cid_false():
    assert is_shard_cid("v1::abc123") is False


def test_shard_cid_roundtrip_all_shards():
    parent = "v1::" + "f" * 64
    for i in range(5):
        cid = shard_cid(parent, i, 3, 5)
        parsed = parse_shard_cid(cid)
        assert parsed is not None
        assert parsed == (parent, i, 3, 5)
