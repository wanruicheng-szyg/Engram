"""
Cloud mining session lifecycle.

A CloudMiningSession represents one user's active mining node:
  - The phone (controller_hotkey) owns the session.
  - A managed cloud node is allocated and runs miner.py under that hotkey.
  - The session expires when the prepaid compute time runs out.
  - Stats are polled from the node and cached here for the mobile dashboard.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger


class SessionStatus(str, Enum):
    PROVISIONING = "provisioning"
    ACTIVE       = "active"
    STOPPING     = "stopping"
    STOPPED      = "stopped"
    FAILED       = "failed"


@dataclass
class CloudMiningSession:
    session_id:        str
    controller_hotkey: str          # phone's sr25519 pubkey (SS58 or hex)
    status:            SessionStatus
    created_at:        float
    expires_at:        float        # Unix timestamp — compute credits run out here
    node_endpoint:     str | None = None   # https://<ip>:<port> of the mining node
    node_hotkey:       str | None = None   # node's registered Bittensor hotkey
    payment_tx:        str | None = None   # x402 payment proof (on-chain tx hash)
    network:           str = "base"        # chain used for payment
    amount_paid_usd:   float = 0.0
    stats:             dict[str, Any] = field(default_factory=dict)
    error:             str | None = None

    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    def remaining_seconds(self) -> float:
        return max(0.0, self.expires_at - time.time())

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        d["remaining_seconds"] = self.remaining_seconds()
        return d


class SessionRegistry:
    """
    Thread-safe, file-persisted registry of all cloud mining sessions.

    Capacity rules
    ──────────────
    Each Akash deployment gets its own provider IP, so IP-collisions on the
    Bittensor network are not a concern for cloud miners.  The real constraint
    is how many pre-registered node hotkeys the gateway operator has available.

    We enforce two limits:
      MAX_PER_HOTKEY   – concurrent sessions a single phone key may own (default 3).
                         One per tier makes operational sense; more is fine.
      MAX_TOTAL_ACTIVE – total active sessions across all users the gateway will
                         manage at once (default 50). Raise when you add more
                         pre-registered hotkeys to the node pool.
    """

    MAX_PER_HOTKEY   = 3
    MAX_TOTAL_ACTIVE = 50

    def __init__(self, path: Path | None = None) -> None:
        self._path   = path or Path("data/cloud_sessions.json")
        self._lock   = threading.Lock()
        self._sessions: dict[str, CloudMiningSession] = {}
        self._load()

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def create(
        self,
        controller_hotkey: str,
        duration_hours: float,
        payment_tx: str,
        network: str,
        amount_paid_usd: float,
    ) -> CloudMiningSession:
        with self._lock:
            active_for_hotkey = sum(
                1 for s in self._sessions.values()
                if s.controller_hotkey == controller_hotkey
                and s.status in (SessionStatus.PROVISIONING, SessionStatus.ACTIVE)
            )
            if active_for_hotkey >= self.MAX_PER_HOTKEY:
                raise ValueError(
                    f"Limit reached: {self.MAX_PER_HOTKEY} concurrent sessions per device. "
                    "Stop an existing session before starting a new one."
                )
            total_active = sum(
                1 for s in self._sessions.values()
                if s.status in (SessionStatus.PROVISIONING, SessionStatus.ACTIVE)
            )
            if total_active >= self.MAX_TOTAL_ACTIVE:
                raise ValueError("Gateway at capacity — please try again shortly.")

        session_id = str(uuid.uuid4())
        now = time.time()
        session = CloudMiningSession(
            session_id        = session_id,
            controller_hotkey = controller_hotkey,
            status            = SessionStatus.PROVISIONING,
            created_at        = now,
            expires_at        = now + duration_hours * 3600,
            payment_tx        = payment_tx,
            network           = network,
            amount_paid_usd   = amount_paid_usd,
        )
        with self._lock:
            self._sessions[session_id] = session
            self._flush()
        logger.info(f"Session created | id={session_id[:8]} hotkey={controller_hotkey[:12]}… ({active_for_hotkey+1}/{self.MAX_PER_HOTKEY} for this device)")
        return session

    def get(self, session_id: str) -> CloudMiningSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def list_for_hotkey(self, hotkey: str) -> list[CloudMiningSession]:
        with self._lock:
            return [s for s in self._sessions.values() if s.controller_hotkey == hotkey]

    def update(self, session: CloudMiningSession) -> None:
        with self._lock:
            self._sessions[session.session_id] = session
            self._flush()

    def mark_active(self, session_id: str, node_endpoint: str, node_hotkey: str) -> None:
        with self._lock:
            s = self._sessions.get(session_id)
            if s:
                s.status        = SessionStatus.ACTIVE
                s.node_endpoint = node_endpoint
                s.node_hotkey   = node_hotkey
                self._flush()

    def mark_failed(self, session_id: str, error: str) -> None:
        with self._lock:
            s = self._sessions.get(session_id)
            if s:
                s.status = SessionStatus.FAILED
                s.error  = error
                self._flush()

    def mark_stopped(self, session_id: str) -> None:
        with self._lock:
            s = self._sessions.get(session_id)
            if s:
                s.status = SessionStatus.STOPPED
                self._flush()

    def update_stats(self, session_id: str, stats: dict) -> None:
        with self._lock:
            s = self._sessions.get(session_id)
            if s:
                s.stats = stats

    # ── Housekeeping ──────────────────────────────────────────────────────────

    def expire_stale(self) -> list[str]:
        """Return session IDs that just expired; caller should stop their nodes."""
        expired = []
        with self._lock:
            for s in self._sessions.values():
                if s.status == SessionStatus.ACTIVE and s.is_expired():
                    s.status = SessionStatus.STOPPING
                    expired.append(s.session_id)
            if expired:
                self._flush()
        return expired

    # ── Persistence ───────────────────────────────────────────────────────────

    def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {sid: s.to_dict() for sid, s in self._sessions.items()}
        self._path.write_text(json.dumps(data, indent=2))

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            for sid, d in data.items():
                d["status"] = SessionStatus(d["status"])
                d.pop("remaining_seconds", None)
                self._sessions[sid] = CloudMiningSession(**d)
        except Exception as exc:
            logger.warning(f"Session registry load failed: {exc}")
