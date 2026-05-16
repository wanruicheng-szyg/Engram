// engram-core/src/lib.rs
//
// PyO3 bindings — CID, storage proofs, and Merkle memory commitment.
//
// CID:
//   cid = engram_core.generate_cid([0.1, 0.2, 0.3], {}, "v1")
//   ok  = engram_core.verify_cid(cid, [0.1, 0.2, 0.3], {}, "v1")
//
// Storage proofs (validator ↔ miner):
//   challenge = engram_core.generate_challenge("v1::abc...", 30, validator_hotkey_hex)
//   response  = engram_core.generate_response(challenge, [0.1, 0.2, 0.3])
//   ok        = engram_core.verify_response(challenge, response, [0.1, 0.2, 0.3])
//
// Batch proofs:
//   batch    = engram_core.generate_batch_challenge(["v1::aaa", "v1::bbb"], 30, validator_hotkey_hex)
//   response = engram_core.generate_batch_response(batch, [[0.1, 0.2], [0.3, 0.4]])
//   results  = engram_core.verify_batch_response(batch, response, [[0.1, 0.2], [0.3, 0.4]])
//
// Merkle memory commitment (full-corpus integrity):
//   commitment = engram_core.build_commitment(cids, embedding_hashes)
//   # commitment.root_hex  — 64-char hex fingerprint of the whole memory set
//   proof = engram_core.generate_inclusion_proof(commitment, cid, embedding_hash)
//   ok    = engram_core.verify_inclusion(commitment.root_hex, cid, embedding_hash, proof)

use pyo3::prelude::*;
use std::collections::BTreeMap;

mod cid;
mod merkle;
mod proof;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Parse a 64-char hex string into a 32-byte array.
fn parse_hotkey_hex(hex_str: &str) -> PyResult<[u8; 32]> {
    let bytes = hex::decode(hex_str).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "validator_hotkey must be a 64-char hex string (SR25519 pubkey): {e}"
        ))
    })?;
    bytes.try_into().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err(
            "validator_hotkey must decode to exactly 32 bytes (64 hex chars)",
        )
    })
}

// ── CID bindings ──────────────────────────────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (embedding, metadata=None, model_version="v1"))]
fn generate_cid(
    embedding: Vec<f32>,
    metadata: Option<std::collections::HashMap<String, String>>,
    model_version: &str,
) -> PyResult<String> {
    let meta: BTreeMap<String, String> = metadata
        .unwrap_or_default()
        .into_iter()
        .collect();
    Ok(cid::generate_cid(&embedding, &meta, model_version))
}

#[pyfunction]
#[pyo3(signature = (cid_str, embedding, metadata=None, model_version="v1"))]
fn verify_cid(
    cid_str: &str,
    embedding: Vec<f32>,
    metadata: Option<std::collections::HashMap<String, String>>,
    model_version: &str,
) -> PyResult<bool> {
    let meta: BTreeMap<String, String> = metadata
        .unwrap_or_default()
        .into_iter()
        .collect();
    Ok(cid::verify_cid(cid_str, &embedding, &meta, model_version))
}

#[pyfunction]
fn parse_cid(cid_str: &str) -> PyResult<(String, String)> {
    cid::parse_cid(cid_str)
        .map(|(v, d)| (v.to_string(), d.to_string()))
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
}

// ── Single-CID Challenge / Proof bindings ─────────────────────────────────────

/// Python-visible Challenge object
#[pyclass]
#[derive(Clone)]
struct Challenge {
    inner: proof::Challenge,
}

#[pymethods]
impl Challenge {
    #[getter]
    fn cid(&self) -> &str { &self.inner.cid }
    #[getter]
    fn nonce_hex(&self) -> String { hex::encode(self.inner.nonce) }
    #[getter]
    fn issued_at(&self) -> u64 { self.inner.issued_at }
    #[getter]
    fn expires_at(&self) -> u64 { self.inner.expires_at }
    #[getter]
    fn validator_hotkey_hex(&self) -> String { hex::encode(self.inner.validator_hotkey) }
}

/// Python-visible ProofResponse object
#[pyclass]
#[derive(Clone)]
struct ProofResponse {
    inner: proof::ProofResponse,
}

#[pymethods]
impl ProofResponse {
    #[getter]
    fn cid(&self) -> &str { &self.inner.cid }
    #[getter]
    fn nonce_hex(&self) -> &str { &self.inner.nonce_hex }
    #[getter]
    fn embedding_hash(&self) -> &str { &self.inner.embedding_hash }
    #[getter]
    fn proof(&self) -> &str { &self.inner.proof }
}

/// Generate a challenge for a single CID.
///
/// Args:
///     cid_str:             CID to challenge
///     timeout_secs:        validity window in seconds (default 30)
///     validator_hotkey_hex: validator's SR25519 public key as a 64-char hex string
#[pyfunction]
#[pyo3(signature = (cid_str, timeout_secs=30, validator_hotkey_hex="0000000000000000000000000000000000000000000000000000000000000000"))]
fn generate_challenge(
    cid_str: &str,
    timeout_secs: u64,
    validator_hotkey_hex: &str,
) -> PyResult<Challenge> {
    let hotkey = parse_hotkey_hex(validator_hotkey_hex)?;
    Ok(Challenge {
        inner: proof::generate_challenge(cid_str, timeout_secs, hotkey),
    })
}

#[pyfunction]
fn generate_response(challenge: &Challenge, embedding: Vec<f32>) -> ProofResponse {
    ProofResponse {
        inner: proof::generate_response(&challenge.inner, &embedding),
    }
}

#[pyfunction]
fn verify_response(
    challenge: &Challenge,
    response: &ProofResponse,
    embedding: Vec<f32>,
) -> bool {
    proof::verify_response(&challenge.inner, &response.inner, &embedding)
}

// ── Batch Challenge / Proof bindings ─────────────────────────────────────────

/// Python-visible BatchChallenge: one nonce covering N CIDs.
#[pyclass]
#[derive(Clone)]
struct BatchChallenge {
    inner: proof::BatchChallenge,
}

#[pymethods]
impl BatchChallenge {
    #[getter]
    fn cids(&self) -> Vec<String> { self.inner.cids.clone() }
    #[getter]
    fn nonce_hex(&self) -> String { hex::encode(self.inner.nonce) }
    #[getter]
    fn issued_at(&self) -> u64 { self.inner.issued_at }
    #[getter]
    fn expires_at(&self) -> u64 { self.inner.expires_at }
    #[getter]
    fn validator_hotkey_hex(&self) -> String { hex::encode(self.inner.validator_hotkey) }
}

/// Python-visible per-entry proof within a batch response.
#[pyclass]
#[derive(Clone)]
struct BatchProofEntry {
    inner: proof::BatchProofEntry,
}

#[pymethods]
impl BatchProofEntry {
    #[getter]
    fn cid(&self) -> &str { &self.inner.cid }
    #[getter]
    fn embedding_hash(&self) -> &str { &self.inner.embedding_hash }
    #[getter]
    fn proof(&self) -> &str { &self.inner.proof }
}

/// Python-visible BatchProofResponse.
#[pyclass]
#[derive(Clone)]
struct BatchProofResponse {
    inner: proof::BatchProofResponse,
}

#[pymethods]
impl BatchProofResponse {
    #[getter]
    fn nonce_hex(&self) -> &str { &self.inner.nonce_hex }
    #[getter]
    fn entries(&self) -> Vec<BatchProofEntry> {
        self.inner.entries.iter().map(|e| BatchProofEntry { inner: e.clone() }).collect()
    }
}

/// Generate a batch challenge covering multiple CIDs in one round trip.
///
/// Args:
///     cids:                list of CID strings to challenge
///     timeout_secs:        validity window in seconds (default 30)
///     validator_hotkey_hex: validator's SR25519 public key as a 64-char hex string
#[pyfunction]
#[pyo3(signature = (cids, timeout_secs=30, validator_hotkey_hex="0000000000000000000000000000000000000000000000000000000000000000"))]
fn generate_batch_challenge(
    cids: Vec<String>,
    timeout_secs: u64,
    validator_hotkey_hex: &str,
) -> PyResult<BatchChallenge> {
    let hotkey = parse_hotkey_hex(validator_hotkey_hex)?;
    let cid_refs: Vec<&str> = cids.iter().map(String::as_str).collect();
    Ok(BatchChallenge {
        inner: proof::generate_batch_challenge(&cid_refs, timeout_secs, hotkey),
    })
}

/// Miner side: respond to a batch challenge.
#[pyfunction]
fn generate_batch_response(
    batch: &BatchChallenge,
    embeddings: Vec<Vec<f32>>,
) -> BatchProofResponse {
    BatchProofResponse {
        inner: proof::generate_batch_response(&batch.inner, &embeddings),
    }
}

/// Validator side: verify a miner's batch response.
///
/// Returns a list[bool] — one result per CID in the original batch order.
/// Expired challenges or nonce mismatches return all-False.
#[pyfunction]
fn verify_batch_response(
    batch: &BatchChallenge,
    response: &BatchProofResponse,
    embeddings: Vec<Vec<f32>>,
) -> Vec<bool> {
    proof::verify_batch_response(&batch.inner, &response.inner, &embeddings)
}

// ── Merkle Memory Commitment bindings ─────────────────────────────────────────

/// Python-visible Merkle commitment: fingerprint of an AI's full memory corpus.
#[pyclass]
#[derive(Clone)]
struct MemoryCommitment {
    inner: merkle::MerkleCommitment,
}

#[pymethods]
impl MemoryCommitment {
    /// 64-char hex fingerprint of the entire memory corpus.
    /// Store this on-chain or in the agent's context to detect tampering.
    #[getter]
    fn root_hex(&self) -> String { hex::encode(self.inner.root) }

    /// Number of distinct memories in this commitment.
    #[getter]
    fn count(&self) -> usize { self.inner.leaves.len() }
}

/// Python-visible inclusion proof for one memory.
#[pyclass]
#[derive(Clone)]
struct MemoryInclusionProof {
    inner: merkle::InclusionProof,
}

#[pymethods]
impl MemoryInclusionProof {
    /// Hex-encoded leaf hash for the proved memory.
    #[getter]
    fn leaf_hex(&self) -> String { hex::encode(self.inner.leaf_hash) }

    /// Number of sibling hashes in the proof (= tree depth = ceil(log2(N))).
    #[getter]
    fn depth(&self) -> usize { self.inner.steps.len() }

    /// Serialise proof to JSON for wire transmission.
    fn to_json(&self) -> String {
        let steps: Vec<serde_json::Value> = self.inner.steps.iter().map(|s| {
            serde_json::json!({
                "sibling": hex::encode(s.sibling),
                "side": match s.side {
                    merkle::Side::Left  => "left",
                    merkle::Side::Right => "right",
                },
            })
        }).collect();
        serde_json::json!({
            "leaf_hex": hex::encode(self.inner.leaf_hash),
            "steps": steps,
        }).to_string()
    }

    /// Deserialise a proof from JSON (produced by to_json).
    #[staticmethod]
    fn from_json(json_str: &str) -> PyResult<MemoryInclusionProof> {
        let v: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("JSON parse error: {e}")))?;

        let leaf_hex = v["leaf_hex"].as_str().unwrap_or("");
        let leaf_bytes = hex::decode(leaf_hex)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("leaf_hex decode: {e}")))?;
        let mut leaf_hash = [0u8; 32];
        if leaf_bytes.len() != 32 {
            return Err(pyo3::exceptions::PyValueError::new_err("leaf_hex must be 64 hex chars"));
        }
        leaf_hash.copy_from_slice(&leaf_bytes);

        let steps_json = v["steps"].as_array()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("steps must be an array"))?;

        let mut steps = Vec::new();
        for step in steps_json {
            let sib_hex = step["sibling"].as_str().unwrap_or("");
            let sib_bytes = hex::decode(sib_hex)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("sibling decode: {e}")))?;
            let mut sibling = [0u8; 32];
            if sib_bytes.len() != 32 {
                return Err(pyo3::exceptions::PyValueError::new_err("sibling must be 64 hex chars"));
            }
            sibling.copy_from_slice(&sib_bytes);
            let side = match step["side"].as_str().unwrap_or("right") {
                "left"  => merkle::Side::Left,
                _       => merkle::Side::Right,
            };
            steps.push(merkle::ProofStep { sibling, side });
        }

        Ok(MemoryInclusionProof {
            inner: merkle::InclusionProof { leaf_hash, steps },
        })
    }
}

/// Build a Merkle commitment over a miner's full memory corpus.
///
/// Args:
///     cids:             list of CID strings (one per stored memory)
///     embedding_hashes: corresponding SHA-256 hashes of embedding bytes
///
/// Returns a MemoryCommitment whose root_hex is a 64-char hex fingerprint
/// of the entire corpus. Same set of memories → same root, regardless of order.
#[pyfunction]
fn build_commitment(
    cids: Vec<String>,
    embedding_hashes: Vec<String>,
) -> PyResult<MemoryCommitment> {
    if cids.len() != embedding_hashes.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "cids and embedding_hashes must have the same length"
        ));
    }
    let cid_refs: Vec<&str> = cids.iter().map(String::as_str).collect();
    let emb_refs: Vec<&str> = embedding_hashes.iter().map(String::as_str).collect();
    Ok(MemoryCommitment {
        inner: merkle::build_commitment(&cid_refs, &emb_refs),
    })
}

/// Generate a Merkle inclusion proof for one memory.
///
/// Args:
///     commitment:      MemoryCommitment built from the full corpus
///     cid:             CID of the memory to prove
///     embedding_hash:  SHA-256 hex of that memory's embedding bytes
///
/// Returns a MemoryInclusionProof, or raises ValueError if the memory is
/// not in this commitment (i.e., the miner doesn't hold it).
#[pyfunction]
fn generate_inclusion_proof(
    commitment: &MemoryCommitment,
    cid: &str,
    embedding_hash: &str,
) -> PyResult<MemoryInclusionProof> {
    merkle::generate_inclusion_proof(&commitment.inner, cid, embedding_hash)
        .map(|inner| MemoryInclusionProof { inner })
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
            format!("CID {cid} is not in this commitment — miner does not hold this memory")
        ))
}

/// Verify a Merkle inclusion proof against a known root.
///
/// Args:
///     root_hex:        64-char hex root (from on-chain or MemoryCommitment.root_hex)
///     cid:             CID being proved
///     embedding_hash:  SHA-256 hex of the embedding
///     proof:           MemoryInclusionProof from generate_inclusion_proof
///
/// Returns True only if the memory is genuinely in the committed corpus.
#[pyfunction]
fn verify_inclusion(
    root_hex: &str,
    cid: &str,
    embedding_hash: &str,
    proof: &MemoryInclusionProof,
) -> bool {
    merkle::verify_inclusion_hex(root_hex, cid, embedding_hash, &proof.inner)
}

// ── Module ────────────────────────────────────────────────────────────────────

#[pymodule]
fn engram_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // CID
    m.add_function(wrap_pyfunction!(generate_cid, m)?)?;
    m.add_function(wrap_pyfunction!(verify_cid, m)?)?;
    m.add_function(wrap_pyfunction!(parse_cid, m)?)?;
    // Single-CID proofs
    m.add_class::<Challenge>()?;
    m.add_class::<ProofResponse>()?;
    m.add_function(wrap_pyfunction!(generate_challenge, m)?)?;
    m.add_function(wrap_pyfunction!(generate_response, m)?)?;
    m.add_function(wrap_pyfunction!(verify_response, m)?)?;
    // Batch proofs
    m.add_class::<BatchChallenge>()?;
    m.add_class::<BatchProofEntry>()?;
    m.add_class::<BatchProofResponse>()?;
    m.add_function(wrap_pyfunction!(generate_batch_challenge, m)?)?;
    m.add_function(wrap_pyfunction!(generate_batch_response, m)?)?;
    m.add_function(wrap_pyfunction!(verify_batch_response, m)?)?;
    // Merkle memory commitment
    m.add_class::<MemoryCommitment>()?;
    m.add_class::<MemoryInclusionProof>()?;
    m.add_function(wrap_pyfunction!(build_commitment, m)?)?;
    m.add_function(wrap_pyfunction!(generate_inclusion_proof, m)?)?;
    m.add_function(wrap_pyfunction!(verify_inclusion, m)?)?;
    Ok(())
}
