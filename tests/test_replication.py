"""Tests for replication manager."""

import pytest
from pathlib import Path
from engram.storage.dht import DHTRouter, Peer
from engram.storage.replication import ReplicationManager, ReplicationStatus
from engram.config import REPLICATION_FACTOR


def make_peer(uid: int) -> Peer:
    return Peer(uid=uid, hotkey=f"hotkey_{uid:04d}")


def make_manager(n_peers: int = 10) -> ReplicationManager:
    local = make_peer(0)
    router = DHTRouter(local_peer=local)
    for i in range(1, n_peers + 1):
        router.add_peer(make_peer(i))
    return ReplicationManager(router=router, db_path=Path(":memory:"))


TEST_CID = "v1::abc123def456abc123def456abc123def456abc123def456abc123def456abc1"


def test_register_creates_record():
    mgr = make_manager()
    record = mgr.register(TEST_CID)
    assert record.cid == TEST_CID
    assert len(record.assigned_uids) == REPLICATION_FACTOR


def test_initial_status_lost():
    mgr = make_manager()
    record = mgr.register(TEST_CID)
    assert record.status == ReplicationStatus.LOST


def test_confirm_updates_status():
    mgr = make_manager()
    record = mgr.register(TEST_CID)
    uid = record.assigned_uids[0]
    mgr.confirm(TEST_CID, uid)
    assert record.replica_count == 1
    assert record.status == ReplicationStatus.CRITICAL


def test_fully_replicated_is_healthy():
    mgr = make_manager()
    record = mgr.register(TEST_CID)
    for uid in record.assigned_uids[:REPLICATION_FACTOR]:
        mgr.confirm(TEST_CID, uid)
    assert record.status == ReplicationStatus.HEALTHY
    assert not record.needs_replication


def test_unconfirm_reduces_count():
    mgr = make_manager()
    record = mgr.register(TEST_CID)
    uid = record.assigned_uids[0]
    mgr.confirm(TEST_CID, uid)
    mgr.unconfirm(TEST_CID, uid)
    assert record.replica_count == 0


def test_miner_offline_returns_affected_cids():
    mgr = make_manager()
    record = mgr.register(TEST_CID)
    uid = record.assigned_uids[0]
    mgr.confirm(TEST_CID, uid)
    affected = mgr.handle_miner_offline(uid)
    assert TEST_CID in affected


def test_health_summary_keys():
    mgr = make_manager()
    mgr.register(TEST_CID)
    summary = mgr.health_summary()
    assert "healthy" in summary
    assert "degraded" in summary
    assert "lost" in summary


def test_total_cids():
    mgr = make_manager()
    mgr.register(TEST_CID)
    mgr.register("v1::" + "b" * 64)
    assert mgr.total_cids() == 2


def test_get_repair_targets():
    mgr = make_manager(10)
    record = mgr.register(TEST_CID)
    # Confirm only one replica
    mgr.confirm(TEST_CID, record.assigned_uids[0])
    targets = mgr.get_repair_targets(TEST_CID)
    assert len(targets) > 0


# ── Status path tests ────────────────────────────────────────────────────────

def test_lost_status_zero_replicas():
    mgr = make_manager()
    record = mgr.register(TEST_CID)
    assert record.status == ReplicationStatus.LOST


def test_critical_status_one_replica():
    mgr = make_manager()
    record = mgr.register(TEST_CID)
    mgr.confirm(TEST_CID, record.assigned_uids[0])
    assert record.status == ReplicationStatus.CRITICAL


def test_degraded_status_partial_replicas():
    mgr = make_manager()
    record = mgr.register(TEST_CID)
    # Confirm 2 out of REPLICATION_FACTOR (need >1 and <REPLICATION_FACTOR)
    assert REPLICATION_FACTOR >= 3, "test requires REPLICATION_FACTOR >= 3"
    mgr.confirm(TEST_CID, record.assigned_uids[0])
    mgr.confirm(TEST_CID, record.assigned_uids[1])
    assert record.status == ReplicationStatus.DEGRADED


def test_repair_queue_priority_order():
    """LOST tasks must sort before CRITICAL, which sort before DEGRADED."""
    mgr = make_manager(20)
    cid_lost     = "v1::" + "a" * 64
    cid_critical = "v1::" + "b" * 64
    cid_degraded = "v1::" + "c" * 64

    for cid in (cid_lost, cid_critical, cid_degraded):
        rec = mgr.register(cid)

    rec_crit = mgr.get_record(cid_critical)
    mgr.confirm(cid_critical, rec_crit.assigned_uids[0])

    rec_deg = mgr.get_record(cid_degraded)
    mgr.confirm(cid_degraded, rec_deg.assigned_uids[0])
    mgr.confirm(cid_degraded, rec_deg.assigned_uids[1])

    queue = mgr.prioritized_repair_queue()
    statuses = [t.status for t in queue]
    lost_idx     = statuses.index(ReplicationStatus.LOST)
    critical_idx = statuses.index(ReplicationStatus.CRITICAL)
    degraded_idx = statuses.index(ReplicationStatus.DEGRADED)
    assert lost_idx < critical_idx < degraded_idx


def test_repair_queue_no_duplicates():
    """Each CID appears at most once in the repair queue even after multi-miner failure."""
    mgr = make_manager(10)
    record = mgr.register(TEST_CID)
    # Confirm two replicas then take both miners offline simultaneously
    uid_a, uid_b = record.assigned_uids[0], record.assigned_uids[1]
    mgr.confirm(TEST_CID, uid_a)
    mgr.confirm(TEST_CID, uid_b)
    mgr.handle_miners_offline([uid_a, uid_b])

    queue = mgr.prioritized_repair_queue()
    cids_in_queue = [t.cid for t in queue]
    assert cids_in_queue.count(TEST_CID) == 1


def test_handle_miners_offline_deduplicates():
    """handle_miners_offline processes multi-miner failure atomically — no duplicate tasks."""
    mgr = make_manager(10)
    record = mgr.register(TEST_CID)
    uid_a, uid_b = record.assigned_uids[0], record.assigned_uids[1]
    mgr.confirm(TEST_CID, uid_a)
    mgr.confirm(TEST_CID, uid_b)

    tasks = mgr.handle_miners_offline([uid_a, uid_b])
    cids = [t.cid for t in tasks]
    assert cids.count(TEST_CID) == 1


def test_healthy_cid_not_in_repair_queue():
    mgr = make_manager(10)
    record = mgr.register(TEST_CID)
    for uid in record.assigned_uids[:REPLICATION_FACTOR]:
        mgr.confirm(TEST_CID, uid)
    assert record.status == ReplicationStatus.HEALTHY
    queue = mgr.prioritized_repair_queue()
    assert all(t.cid != TEST_CID for t in queue)
