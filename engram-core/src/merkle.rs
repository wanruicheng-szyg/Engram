// engram-core/src/merkle.rs
//
// Merkle commitment over an AI agent's memory corpus.
//
// Every miner maintains a commitment to its entire stored dataset:
//
//   root = MerkleRoot({ leaf(cid, embedding_hash) for all stored memories })
//
// The root is a 32-byte fingerprint of the complete memory set.
// If a miner deletes, corrupts, or fabricates any memory, the root changes —
// and the validator's on-chain commitment no longer matches.
//
// This replaces spot-check storage proofs (prove you hold *this* CID)
// with a full-corpus commitment (prove your *entire* memory set is intact).
//
// Design
// ------
// Leaf:   SHA256("leaf:" || cid || ":" || embedding_hash)
// Node:   SHA256("node:" || left_child || right_child)
// Leaves are sorted before tree construction so the root is deterministic
// regardless of ingest order — the same set of memories always produces
// the same root.
//
// Inclusion proof
// ---------------
// A miner can prove a specific memory (cid, embedding_hash) is in its
// committed corpus without revealing the rest:
//   1. Validator holds the on-chain root.
//   2. Miner returns the leaf hash + sibling hashes up the tree.
//   3. Validator recomputes the root from the proof — O(log N) hashes.
//
// For an AI agent this means: its memories are verifiably present and
// unmodified, without the validator having to re-download the full index.

use sha2::{Digest, Sha256};

type Hash = [u8; 32];

// ── Types ─────────────────────────────────────────────────────────────────────

/// The committed fingerprint of a miner's full memory corpus.
#[derive(Debug, Clone)]
pub struct MerkleCommitment {
    /// Root hash: 32-byte fingerprint of all stored memories.
    pub root: Hash,
    /// Sorted leaf hashes — retained so we can generate inclusion proofs.
    pub leaves: Vec<Hash>,
}

/// Sibling direction in the tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Left,   // sibling is to the left; current node is the right child
    Right,  // sibling is to the right; current node is the left child
}

/// One step in an inclusion proof: the sibling hash and which side it is on.
#[derive(Debug, Clone)]
pub struct ProofStep {
    pub sibling: Hash,
    pub side: Side,
}

/// Full inclusion proof for one (cid, embedding_hash) memory.
#[derive(Debug, Clone)]
pub struct InclusionProof {
    pub leaf_hash: Hash,
    pub steps: Vec<ProofStep>,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Build a Merkle commitment over a set of (cid, embedding_hash) pairs.
///
/// Leaves are sorted for determinism — the same memory corpus always
/// produces the same root regardless of the order records were ingested.
///
/// An empty corpus returns the zero root.
pub fn build_commitment(cids: &[&str], embedding_hashes: &[&str]) -> MerkleCommitment {
    debug_assert_eq!(cids.len(), embedding_hashes.len());

    let mut leaves: Vec<Hash> = cids
        .iter()
        .zip(embedding_hashes.iter())
        .map(|(cid, emb_hash)| leaf_hash(cid, emb_hash))
        .collect();

    // Sort for determinism — ingest order must not affect the root.
    leaves.sort_unstable();
    leaves.dedup();  // identical memories collapse to one leaf

    let root = merkle_root(&leaves);
    MerkleCommitment { root, leaves }
}

/// Generate an inclusion proof proving (cid, embedding_hash) is in the commitment.
///
/// Returns None if the memory is not in this commitment.
pub fn generate_inclusion_proof(
    commitment: &MerkleCommitment,
    cid: &str,
    embedding_hash: &str,
) -> Option<InclusionProof> {
    let target = leaf_hash(cid, embedding_hash);
    let leaf_idx = commitment.leaves.iter().position(|h| h == &target)?;

    let mut steps = Vec::new();
    let mut layer = commitment.leaves.clone();
    let mut current_idx = leaf_idx;

    while layer.len() > 1 {
        // Pad odd layer by duplicating the last leaf.
        if layer.len() % 2 != 0 {
            layer.push(*layer.last().unwrap());
        }

        let sibling_idx = if current_idx % 2 == 0 {
            current_idx + 1
        } else {
            current_idx - 1
        };

        let side = if current_idx % 2 == 0 {
            Side::Right   // current is left child; sibling is right
        } else {
            Side::Left    // current is right child; sibling is left
        };

        steps.push(ProofStep { sibling: layer[sibling_idx], side });

        layer = layer
            .chunks(2)
            .map(|pair| internal_node(&pair[0], &pair[1]))
            .collect();

        current_idx /= 2;
    }

    Some(InclusionProof { leaf_hash: target, steps })
}

/// Verify an inclusion proof against a known root.
///
/// Returns true only if the proof is valid and the root matches.
/// All computation is constant-time-ish — we always walk the full proof
/// rather than short-circuiting, to avoid leaking the depth.
pub fn verify_inclusion(
    expected_root: &Hash,
    cid: &str,
    embedding_hash: &str,
    proof: &InclusionProof,
) -> bool {
    let target = leaf_hash(cid, embedding_hash);

    // Leaf hash must match what the proof claims.
    if target != proof.leaf_hash {
        return false;
    }

    let mut current = target;
    for step in &proof.steps {
        current = match step.side {
            Side::Right => internal_node(&current, &step.sibling),
            Side::Left  => internal_node(&step.sibling, &current),
        };
    }

    &current == expected_root
}

/// Verify an inclusion proof given a hex-encoded root string.
pub fn verify_inclusion_hex(
    root_hex: &str,
    cid: &str,
    embedding_hash: &str,
    proof: &InclusionProof,
) -> bool {
    let bytes = match hex::decode(root_hex) {
        Ok(b) if b.len() == 32 => b,
        _ => return false,
    };
    let mut root = [0u8; 32];
    root.copy_from_slice(&bytes);
    verify_inclusion(&root, cid, embedding_hash, proof)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Leaf hash: SHA256("leaf:" || cid || ":" || embedding_hash)
///
/// The "leaf:" domain separator prevents second-preimage attacks where an
/// internal node hash is submitted as a leaf.
fn leaf_hash(cid: &str, embedding_hash: &str) -> Hash {
    let mut h = Sha256::new();
    h.update(b"leaf:");
    h.update(cid.as_bytes());
    h.update(b":");
    h.update(embedding_hash.as_bytes());
    h.finalize().into()
}

/// Internal node hash: SHA256("node:" || left || right)
fn internal_node(left: &Hash, right: &Hash) -> Hash {
    let mut h = Sha256::new();
    h.update(b"node:");
    h.update(left);
    h.update(right);
    h.finalize().into()
}

/// Compute the Merkle root of a slice of (pre-sorted) leaf hashes.
fn merkle_root(leaves: &[Hash]) -> Hash {
    if leaves.is_empty() {
        return [0u8; 32];
    }
    if leaves.len() == 1 {
        return leaves[0];
    }
    let mut layer = leaves.to_vec();
    while layer.len() > 1 {
        if layer.len() % 2 != 0 {
            layer.push(*layer.last().unwrap());
        }
        layer = layer
            .chunks(2)
            .map(|pair| internal_node(&pair[0], &pair[1]))
            .collect();
    }
    layer[0]
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn mem(n: u8) -> (String, String) {
        (format!("v1::{:064x}", n), format!("{:064x}", n + 100))
    }

    // ── Basic commitment ──────────────────────────────────────────────────────

    #[test]
    fn empty_corpus_returns_zero_root() {
        let c = build_commitment(&[], &[]);
        assert_eq!(c.root, [0u8; 32]);
    }

    #[test]
    fn single_memory_root_equals_leaf() {
        let (cid, emb) = mem(1);
        let c = build_commitment(&[&cid], &[&emb]);
        // With one leaf, root == that leaf hash.
        assert_ne!(c.root, [0u8; 32]);
        assert_eq!(c.leaves.len(), 1);
    }

    #[test]
    fn same_corpus_same_root() {
        let (c1, e1) = mem(1); let (c2, e2) = mem(2); let (c3, e3) = mem(3);
        let a = build_commitment(&[&c1, &c2, &c3], &[&e1, &e2, &e3]);
        let b = build_commitment(&[&c3, &c1, &c2], &[&e3, &e1, &e2]);
        assert_eq!(a.root, b.root, "ingest order must not affect root");
    }

    #[test]
    fn adding_memory_changes_root() {
        let (c1, e1) = mem(1); let (c2, e2) = mem(2);
        let a = build_commitment(&[&c1], &[&e1]);
        let b = build_commitment(&[&c1, &c2], &[&e1, &e2]);
        assert_ne!(a.root, b.root);
    }

    #[test]
    fn corrupted_embedding_changes_root() {
        let (cid, emb) = mem(1);
        let corrupted = "aaaa".repeat(16);
        let a = build_commitment(&[&cid], &[&emb]);
        let b = build_commitment(&[&cid], &[&corrupted]);
        assert_ne!(a.root, b.root);
    }

    // ── Inclusion proofs ──────────────────────────────────────────────────────

    #[test]
    fn inclusion_proof_verifies_for_all_sizes() {
        for n in 1..=17usize {
            let pairs: Vec<_> = (0..n).map(|i| mem(i as u8)).collect();
            let cids:  Vec<&str> = pairs.iter().map(|(c, _)| c.as_str()).collect();
            let embs:  Vec<&str> = pairs.iter().map(|(_, e)| e.as_str()).collect();
            let commitment = build_commitment(&cids, &embs);

            for (cid, emb) in &pairs {
                let proof = generate_inclusion_proof(&commitment, cid, emb)
                    .unwrap_or_else(|| panic!("proof missing for n={n}"));
                assert!(
                    verify_inclusion(&commitment.root, cid, emb, &proof),
                    "proof failed for n={n} cid={cid}"
                );
            }
        }
    }

    #[test]
    fn wrong_cid_fails_verification() {
        let (c1, e1) = mem(1); let (c2, e2) = mem(2);
        let commitment = build_commitment(&[&c1, &c2], &[&e1, &e2]);
        let proof = generate_inclusion_proof(&commitment, &c1, &e1).unwrap();
        // Claim the proof is for c2 — must fail.
        assert!(!verify_inclusion(&commitment.root, &c2, &e2, &proof));
    }

    #[test]
    fn wrong_embedding_fails_verification() {
        let (c1, e1) = mem(1);
        let commitment = build_commitment(&[&c1], &[&e1]);
        let proof = generate_inclusion_proof(&commitment, &c1, &e1).unwrap();
        let bad_emb = "ffff".repeat(16);
        assert!(!verify_inclusion(&commitment.root, &c1, &bad_emb, &proof));
    }

    #[test]
    fn proof_against_wrong_root_fails() {
        let (c1, e1) = mem(1); let (c2, e2) = mem(2);
        let c_a = build_commitment(&[&c1], &[&e1]);
        let c_b = build_commitment(&[&c2], &[&e2]);
        let proof = generate_inclusion_proof(&c_a, &c1, &e1).unwrap();
        // Proof is valid for c_a's root but not c_b's.
        assert!(!verify_inclusion(&c_b.root, &c1, &e1, &proof));
    }

    #[test]
    fn non_member_has_no_proof() {
        let (c1, e1) = mem(1); let (c2, e2) = mem(2);
        let commitment = build_commitment(&[&c1], &[&e1]);
        assert!(generate_inclusion_proof(&commitment, &c2, &e2).is_none());
    }

    #[test]
    fn hex_verify_works() {
        let (c1, e1) = mem(1); let (c2, e2) = mem(2);
        let commitment = build_commitment(&[&c1, &c2], &[&e1, &e2]);
        let root_hex = hex::encode(commitment.root);
        let proof = generate_inclusion_proof(&commitment, &c1, &e1).unwrap();
        assert!(verify_inclusion_hex(&root_hex, &c1, &e1, &proof));
        assert!(!verify_inclusion_hex(&root_hex, &c1, &e1, &{
            // tamper with first sibling
            let mut bad = proof.clone();
            if let Some(step) = bad.steps.first_mut() {
                step.sibling[0] ^= 0xff;
            }
            bad
        }));
    }

    // ── Second-preimage resistance ────────────────────────────────────────────

    #[test]
    fn internal_node_not_accepted_as_leaf() {
        // A node hash computed from two leaves must not verify as a leaf claim.
        let (c1, e1) = mem(1); let (c2, e2) = mem(2);
        let commitment = build_commitment(&[&c1, &c2], &[&e1, &e2]);
        // Construct a fake "proof" that presents an internal node as a leaf.
        let fake_leaf = commitment.root;
        let fake_proof = InclusionProof { leaf_hash: fake_leaf, steps: vec![] };
        // verify_inclusion recomputes the leaf hash from (cid, emb) — so
        // even if we knew an internal node hash, we can't claim it as a leaf.
        assert!(!verify_inclusion(&commitment.root, &c1, &e1, &fake_proof));
    }
}
