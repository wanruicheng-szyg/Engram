"""
Akash SDL (Stack Definition Language) manifest generator for Engram miners.

Each cloud mining session gets its own SDL — a YAML spec describing:
  - The Docker image to run (engram miner)
  - CPU / memory / storage resources
  - Exposed port (miner HTTP)
  - Environment variables (hotkey, bittensor config)

Akash providers read this and bid on the deployment.

Tiers:
  "lite"     — 1 vCPU, 2 GB RAM  — light mining, cheaper
  "standard" — 2 vCPU, 4 GB RAM  — recommended
  "pro"      — 4 vCPU, 8 GB RAM  — high-throughput embedding

The miner image is built from the Engram repo and pushed to GHCR.
"""

from __future__ import annotations

import os
from typing import Any

MINER_IMAGE = os.getenv(
    "ENGRAM_MINER_IMAGE",
    "ghcr.io/dipraise1/engram:latest",
)

TIERS: dict[str, dict[str, Any]] = {
    "lite":     {"cpu": 1000, "memory": "2Gi", "storage": "10Gi", "price_akt_per_hour": 0.05},
    "standard": {"cpu": 2000, "memory": "4Gi", "storage": "20Gi", "price_akt_per_hour": 0.10},
    "pro":      {"cpu": 4000, "memory": "8Gi", "storage": "40Gi", "price_akt_per_hour": 0.18},
}


def build_sdl(
    session_id: str,
    controller_hotkey: str,
    netuid: int = 450,
    tier: str = "standard",
    miner_port: int = 8091,
    subtensor_endpoint: str = "wss://test.finney.opentensor.ai:443",
) -> str:
    """
    Generate an Akash SDL YAML for one Engram mining node.

    The node runs miner.py and accepts only commands signed by controller_hotkey.
    Session ID is injected as an env var so the node can self-identify.
    """
    t = TIERS.get(tier, TIERS["standard"])
    cpu_m  = t["cpu"]    # millicores (1000 = 1 vCPU)
    memory = t["memory"]
    storage = t["storage"]

    return f"""---
version: "2.0"

services:
  miner:
    image: {MINER_IMAGE}
    env:
      - SESSION_ID={session_id}
      - CONTROLLER_HOTKEY={controller_hotkey}
      - NETUID={netuid}
      - SUBTENSOR_ENDPOINT={subtensor_endpoint}
      - MINER_PORT={miner_port}
      - QDRANT_HOST=localhost
      - QDRANT_PORT=6333
    expose:
      - port: {miner_port}
        as: 80
        to:
          - global: true
      - port: 6333
        as: 6333
        to:
          - service: miner   # Qdrant only internal

  qdrant:
    image: qdrant/qdrant:latest
    expose:
      - port: 6333
        as: 6333
        to:
          - service: miner

profiles:
  compute:
    miner:
      resources:
        cpu:
          units: {cpu_m}m
        memory:
          size: {memory}
        storage:
          - size: {storage}
    qdrant:
      resources:
        cpu:
          units: 500m
        memory:
          size: 512Mi
        storage:
          - size: 5Gi

  placement:
    anywhere:
      pricing:
        miner:
          denom: uakt
          amount: 100
        qdrant:
          denom: uakt
          amount: 50

deployment:
  miner:
    anywhere:
      profile: miner
      count: 1
  qdrant:
    anywhere:
      profile: qdrant
      count: 1
"""


def tier_info() -> list[dict]:
    """Return pricing and specs for each tier — used by the mobile app."""
    return [
        {
            "tier":             name,
            "cpu_vcpu":         t["cpu"] // 1000,
            "memory_gb":        int(t["memory"].replace("Gi", "")),
            "storage_gb":       int(t["storage"].replace("Gi", "")),
            "price_akt_per_hour": t["price_akt_per_hour"],
        }
        for name, t in TIERS.items()
    ]
