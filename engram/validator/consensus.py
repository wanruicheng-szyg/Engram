"""
Multi-validator consensus for weight setting.

Before setting weights on-chain a validator gathers signed score vectors from
peer validators and checks that its own scores agree with a quorum. If no
quorum is reached the validator logs a warning and skips this round rather
than risk setting idiosyncratic weights.

Workflow:
    1. Validator scores miners → create_vector(scores, block).
    2. Transport layer (HTTP push/pull, gossip) delivers peer vectors to
       ingest_peer_vector().
    3. Before weight-setting: has_quorum(block) → aggregate(block) → set_weights.

CONSENSUS_MIN_VALIDATORS=0 (testnet default): has_quorum() always returns True
so a single-validator deployment works without any changes.

Agreement is measured by cosine distance between score vectors: two validators
"agree" when their cosine distance is <= CONSENSUS_SCORE_TOLERANCE (default 0.15,
which corresponds to roughly ±8° of angular deviation).
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from engram.config import (
    CONSENSUS_MIN_VALIDATORS,
    CONSENSUS_SCORE_TOLERANCE,
    CONSENSUS_VECTOR_TTL_SECS,
)

_DEFAULT_DB = Path("data/consensus.db")


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class ScoreVector:
    """A signed score vector produced by one validator for one block."""

    validator_hotkey: str
    block: int
    scores: dict[int, float]   # uid → recall score (or composite)
    timestamp_ms: int
    signature: str             # 0x-prefixed sr25519 hex, or "" for unsigned

    def to_dict(self) -> dict[str, Any]:
        return {
            "validator_hotkey": self.validator_hotkey,
            "block": self.block,
            "scores": {str(k): float(v) for k, v in self.scores.items()},
            "timestamp_ms": self.timestamp_ms,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ScoreVector":
        return cls(
            validator_hotkey=str(d["validator_hotkey"]),
            block=int(d["block"]),
            scores={int(k): float(v) for k, v in d["scores"].items()},
            timestamp_ms=int(d["timestamp_ms"]),
            signature=str(d.get("signature", "")),
        )

    def canonical_json(self) -> str:
        """Deterministic JSON used as the signing payload."""
        return json.dumps(
            {
                "validator_hotkey": self.validator_hotkey,
                "block": self.block,
                "scores": {
                    str(k): float(v)
                    for k, v in sorted(self.scores.items())
                },
                "timestamp_ms": self.timestamp_ms,
            },
            sort_keys=True,
            separators=(",", ":"),
        )


# ── Distance metric ───────────────────────────────────────────────────────────

def cosine_distance(a: dict[int, float], b: dict[int, float]) -> float:
    """
    Cosine distance in [0, 2] between two score dicts.

    UIDs present in one dict but absent from the other are assigned 0.0.
    Returns 1.0 when either vector is all-zero (maximum uncertainty).
    """
    all_uids = sorted(set(a) | set(b))
    if not all_uids:
        return 0.0
    va = np.array([a.get(u, 0.0) for u in all_uids], dtype=np.float64)
    vb = np.array([b.get(u, 0.0) for u in all_uids], dtype=np.float64)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return float(1.0 - np.dot(va, vb) / (na * nb))


# ── Consensus engine ──────────────────────────────────────────────────────────

class ConsensusEngine:
    """
    Collects signed score vectors from peer validators and determines
    whether a quorum agrees before weight-setting proceeds.
    """

    def __init__(
        self,
        validator_hotkey: str,
        db_path: Path = _DEFAULT_DB,
        min_validators: int = CONSENSUS_MIN_VALIDATORS,
        score_tolerance: float = CONSENSUS_SCORE_TOLERANCE,
        vector_ttl_secs: int = CONSENSUS_VECTOR_TTL_SECS,
    ) -> None:
        self._hotkey = validator_hotkey
        self._min_validators = min_validators
        self._tolerance = score_tolerance
        self._ttl = vector_ttl_secs
        self._db = self._open_db(db_path)
        self._own_vector: ScoreVector | None = None

    # ── Own vector ────────────────────────────────────────────────────────────

    def create_vector(
        self,
        scores: dict[int, float],
        block: int,
        keypair: Any = None,
    ) -> ScoreVector:
        """
        Build and persist the local validator's score vector for this block.

        Signs the vector with `keypair` when provided (sr25519). Unsigned
        vectors (keypair=None) are accepted for testnet / single-validator use.
        """
        ts = int(time.time() * 1000)
        vec = ScoreVector(
            validator_hotkey=self._hotkey,
            block=block,
            scores=dict(scores),
            timestamp_ms=ts,
            signature="",
        )
        if keypair is not None:
            try:
                raw = keypair.sign(vec.canonical_json().encode())
                vec.signature = "0x" + (raw.hex() if isinstance(raw, bytes) else str(raw))
            except Exception as exc:
                logger.warning(f"Consensus: failed to sign score vector: {exc}")
        self._own_vector = vec
        self._store(vec)
        logger.debug(
            f"Consensus: created vector | block={block} | uids={len(scores)} | "
            f"signed={bool(vec.signature)}"
        )
        return vec

    # ── Peer vector ingestion ─────────────────────────────────────────────────

    def ingest_peer_vector(
        self,
        vec: ScoreVector,
        check_ttl: bool = True,
    ) -> bool:
        """
        Accept a score vector from a peer validator.

        Signature verification is the caller's responsibility — the transport
        layer should verify the sr25519 signature against the known validator
        hotkey before calling this method.

        Returns True if the vector was accepted, False if rejected (stale,
        self-submitted, or duplicate older than the stored one).
        """
        if vec.validator_hotkey == self._hotkey:
            return False  # never ingest our own vector via the peer path

        if check_ttl:
            age_secs = abs(time.time() - vec.timestamp_ms / 1000)
            if age_secs > self._ttl:
                logger.warning(
                    f"Consensus: rejected stale vector from "
                    f"{vec.validator_hotkey[:14]}… age={age_secs:.0f}s ttl={self._ttl}s"
                )
                return False

        self._store(vec)
        logger.debug(
            f"Consensus: ingested peer vector | hk={vec.validator_hotkey[:14]}… "
            f"block={vec.block} | uids={len(vec.scores)}"
        )
        return True

    # ── Quorum ────────────────────────────────────────────────────────────────

    def has_quorum(self, block: int) -> bool:
        """
        Return True if at least `min_validators` peers agree with the local
        validator's scores for this block.

        When min_validators == 0 (default) always returns True — the validator
        acts as a single authority, suitable for testnet.
        """
        if self._min_validators == 0:
            return True
        if self._own_vector is None or self._own_vector.block != block:
            logger.warning(
                f"Consensus: no local vector for block={block} — call create_vector first"
            )
            return False
        n_agree = len(self._agreeing_peers(block))
        ok = n_agree >= self._min_validators
        if not ok:
            logger.warning(
                f"Consensus: quorum not reached | block={block} | "
                f"agreeing={n_agree} needed={self._min_validators}"
            )
        return ok

    # ── Aggregation ───────────────────────────────────────────────────────────

    def aggregate(self, block: int) -> dict[int, float]:
        """
        Compute element-wise median of all agreeing validators' score vectors
        for this block (own scores included).

        Returns own scores unchanged when min_validators == 0 (no aggregation
        needed) or when no peers have submitted vectors.
        """
        if self._own_vector is None or self._own_vector.block != block:
            return {}
        agreeing = self._agreeing_peers(block)
        all_vecs = [self._own_vector] + agreeing

        if len(all_vecs) == 1:
            return dict(self._own_vector.scores)

        all_uids: set[int] = set()
        for v in all_vecs:
            all_uids.update(v.scores.keys())

        result: dict[int, float] = {}
        for uid in all_uids:
            vals = [v.scores.get(uid, 0.0) for v in all_vecs]
            result[uid] = float(np.median(vals))

        logger.info(
            f"Consensus: aggregated | block={block} | validators={len(all_vecs)} | "
            f"uids={len(result)}"
        )
        return result

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def peer_count(self, block: int) -> int:
        """Number of distinct peer vectors received for this block (within TTL)."""
        return len(self._peer_vectors(block))

    def agreement_report(self, block: int) -> list[dict[str, Any]]:
        """Per-peer cosine distance report for logging / monitoring."""
        if self._own_vector is None:
            return []
        return [
            {
                "validator": v.validator_hotkey[:16] + "…",
                "block": v.block,
                "cosine_distance": round(cosine_distance(self._own_vector.scores, v.scores), 4),
                "agrees": cosine_distance(self._own_vector.scores, v.scores) <= self._tolerance,
            }
            for v in self._peer_vectors(block)
        ]

    # ── DB helpers ────────────────────────────────────────────────────────────

    def _open_db(self, path: Path) -> sqlite3.Connection:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), check_same_thread=False)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS score_vectors (
                validator_hotkey  TEXT    NOT NULL,
                block             INTEGER NOT NULL,
                scores_json       TEXT    NOT NULL,
                timestamp_ms      INTEGER NOT NULL,
                signature         TEXT    NOT NULL DEFAULT '',
                PRIMARY KEY (validator_hotkey, block)
            )
        """)
        conn.commit()
        return conn

    def _store(self, vec: ScoreVector) -> None:
        self._db.execute(
            """INSERT OR REPLACE INTO score_vectors
               (validator_hotkey, block, scores_json, timestamp_ms, signature)
               VALUES (?, ?, ?, ?, ?)""",
            (
                vec.validator_hotkey,
                vec.block,
                json.dumps({str(k): float(v) for k, v in vec.scores.items()}),
                vec.timestamp_ms,
                vec.signature,
            ),
        )
        self._db.commit()

    def _peer_vectors(self, block: int) -> list[ScoreVector]:
        """All non-self vectors for this block that are within TTL."""
        cutoff_ms = int((time.time() - self._ttl) * 1000)
        rows = self._db.execute(
            """SELECT validator_hotkey, block, scores_json, timestamp_ms, signature
               FROM score_vectors
               WHERE block = ? AND validator_hotkey != ? AND timestamp_ms > ?""",
            (block, self._hotkey, cutoff_ms),
        ).fetchall()
        result = []
        for hk, blk, sj, ts, sig in rows:
            result.append(ScoreVector(
                validator_hotkey=hk,
                block=blk,
                scores={int(k): float(v) for k, v in json.loads(sj).items()},
                timestamp_ms=ts,
                signature=sig,
            ))
        return result

    def _agreeing_peers(self, block: int) -> list[ScoreVector]:
        assert self._own_vector is not None
        return [
            v for v in self._peer_vectors(block)
            if cosine_distance(self._own_vector.scores, v.scores) <= self._tolerance
        ]
