"""
(k, n) erasure coding for float32 embedding vectors.

Encodes an embedding into n shards using a Vandermonde generator matrix;
any k of the n shards reconstruct the original. Shard i is:

    S_i = G[i, :] @ D

where D is the embedding split into k equal-length row-vectors and G is the
n×k Vandermonde matrix with evaluation points 1..n. Any k×k Vandermonde
submatrix with distinct integer evaluation points is invertible, guaranteeing
reconstruction from any k shards.

Default k=3, n=5: 1.67× storage overhead, tolerates 2 simultaneous miner losses.
Compare to 3× full replication: tolerates 2 losses at 3× overhead.

Shard CIDs extend the parent CID: "{parent}::ec:{index}/{k}/{n}"
"""

from __future__ import annotations

import numpy as np

from engram.config import ERASURE_K, ERASURE_N


class ErasureCoder:
    """
    Systematic-free (k, n) erasure coder for fixed-dimension float32 vectors.

    Thread-safe: the generator matrix is built once at construction and is
    read-only afterwards.
    """

    def __init__(self, k: int = ERASURE_K, n: int = ERASURE_N) -> None:
        if k < 2:
            raise ValueError(f"k must be >= 2, got {k}")
        if n < k:
            raise ValueError(f"n must be >= k, got n={n} k={k}")
        if n > 255:
            raise ValueError(f"n must be <= 255, got {n}")
        self.k = k
        self.n = n
        # G: n×k Vandermonde with evaluation points 1..n (increasing=True means
        # columns go [x^0, x^1, ..., x^(k-1)] — lower condition number than
        # the default decreasing form).
        self._G: np.ndarray = np.vander(
            np.arange(1, n + 1, dtype=np.float64), N=k, increasing=True
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def encode(self, embedding: np.ndarray) -> list[np.ndarray]:
        """
        Split embedding into n shards.

        Returns n float32 arrays each of shape (ceil(dim/k),).
        The caller should store ``(shard_index, orig_dim)`` alongside each shard
        so decode() can strip zero-padding after reconstruction.
        """
        orig_dim = embedding.shape[0]
        chunk = _chunk_size(orig_dim, self.k)

        padded = np.zeros(chunk * self.k, dtype=np.float64)
        padded[:orig_dim] = embedding.astype(np.float64)

        D = padded.reshape(self.k, chunk)       # k × chunk
        S = self._G @ D                         # n × chunk

        return [S[i].astype(np.float32) for i in range(self.n)]

    def decode(
        self,
        shards: dict[int, np.ndarray],
        orig_dim: int | None = None,
    ) -> np.ndarray:
        """
        Reconstruct the original embedding from any k (or more) shards.

        Args:
            shards:   Dict mapping shard index (0..n-1) to its float32 array.
                      Must contain at least k entries.
            orig_dim: Original embedding dimension before padding. When None,
                      the full (possibly zero-padded) vector is returned.

        Returns:
            float32 array of shape (orig_dim,) or (k*chunk,) if orig_dim is None.

        Raises:
            ValueError: if fewer than k shards are provided, or any shard index
                        is out of range [0, n).
        """
        if len(shards) < self.k:
            raise ValueError(
                f"Need at least {self.k} shards to reconstruct, got {len(shards)}"
            )
        bad = [i for i in shards if not (0 <= i < self.n)]
        if bad:
            raise ValueError(f"Shard indices out of range [0, {self.n}): {bad}")

        indices = sorted(shards.keys())[: self.k]   # take any k, sorted
        G_k = self._G[indices, :]                   # k × k
        S_k = np.stack(
            [shards[i].astype(np.float64) for i in indices]
        )                                           # k × chunk

        # G_k @ D = S_k  →  D = G_k^{-1} @ S_k
        D = np.linalg.solve(G_k, S_k)              # k × chunk

        flat = D.reshape(-1).astype(np.float32)
        if orig_dim is not None:
            return flat[:orig_dim]
        return flat

    # ── Helpers ───────────────────────────────────────────────────────────────

    def shard_size(self, orig_dim: int) -> int:
        """Number of float32 elements in each shard for a given embedding dim."""
        return _chunk_size(orig_dim, self.k)

    def is_mds(self) -> bool:
        """True when any k×k submatrix of the generator is invertible (MDS property)."""
        from itertools import combinations
        for indices in combinations(range(self.n), self.k):
            sub = self._G[list(indices), :]
            if abs(np.linalg.det(sub)) < 1e-10:
                return False
        return True


# ── Shard CID helpers ─────────────────────────────────────────────────────────

def shard_cid(parent_cid: str, shard_index: int, k: int, n: int) -> str:
    """Build the CID for shard ``shard_index`` of an erasure-coded object."""
    return f"{parent_cid}::ec:{shard_index}/{k}/{n}"


def parse_shard_cid(cid: str) -> tuple[str, int, int, int] | None:
    """
    Parse a shard CID back to (parent_cid, shard_index, k, n).
    Returns None for non-shard CIDs.
    """
    marker = "::ec:"
    if marker not in cid:
        return None
    parent, rest = cid.rsplit(marker, 1)
    parts = rest.split("/")
    if len(parts) != 3:
        return None
    try:
        i, k, n = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None
    if not (0 <= i < n and k >= 2 and n >= k):
        return None
    return parent, i, k, n


def is_shard_cid(cid: str) -> bool:
    """Return True if the CID refers to a shard rather than a full embedding."""
    return "::ec:" in cid


# ── Internal ──────────────────────────────────────────────────────────────────

def _chunk_size(dim: int, k: int) -> int:
    """Return ceil(dim / k) — the number of floats per shard."""
    return (dim + k - 1) // k
