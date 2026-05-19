# Engram Miner — Docker image for Akash Network deployment
#
# engram-core (Rust) is excluded: it's an optional accelerator with Python
# fallbacks in every import site (try/except). The Rust build would require
# a full Rust toolchain install inside Docker, greatly increasing build time
# and image size for an optional dependency.
FROM python:3.11-slim

WORKDIR /app

# Runtime libs: libgomp1 for faiss-cpu, libgmp-dev for bittensor crypto
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential pkg-config libssl-dev git \
    libgomp1 libgmp-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python deps first (cached layer — only rebuilds when pyproject.toml changes)
COPY pyproject.toml README.md ./
COPY engram/ engram/
RUN pip install --no-cache-dir ".[node]"

COPY neurons/miner.py neurons/miner.py
RUN mkdir -p data

ENV MINER_PORT=8091
ENV QDRANT_HOST=localhost
ENV QDRANT_PORT=6333
ENV NETUID=450
ENV SUBTENSOR_ENDPOINT=wss://test.finney.opentensor.ai:443

EXPOSE 8091

CMD ["sh", "-c", "python neurons/miner.py --port ${MINER_PORT} --netuid ${NETUID} --subtensor.chain_endpoint ${SUBTENSOR_ENDPOINT}"]
