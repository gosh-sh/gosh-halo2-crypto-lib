//! One-hop L7 referenced-block circuit primitive for Acki Nacki multi-thread
//! inclusion proofs.
//!
//! Given a current block `B_cur` (the prover holds its `block_id` and the eight
//! leaves of its block-id 8-leaf SHA-256 Merkle tree), this primitive proves:
//!
//!   1. `block_id(B_cur) == sha256_pair(...)`  — the 8-leaf SHA-256 tree binds
//!      `leaves[0..8]` to the public `block_id_cur`. Leaf 7 is the L7
//!      "referenced blocks" Poseidon root (`compute_referenced_blocks_root`).
//!
//!   2. A `next_block_id` chosen by the prover sits at some index `idx` in the
//!      Poseidon dense-Merkle tree under `leaves[7]`, with the leaf hashed via
//!      `compute_referenced_block_leaf_hash(idx, next_block_id)`. Both tag
//!      variants (parent at `idx==0`, ref otherwise) are supported.
//!
//!   3. `next_block_id` is exposed as a public output so that the next hop (or
//!      the thread-0 anchor proof) can pick it up.
//!
//! Off-chain native semantics mirrored from `acki-nacki/node/libs/history-proof`:
//!   * `compute_referenced_block_leaf_hash(index, block_id)`
//!   * `compute_referenced_blocks_root([parent, refs..])`
//!   * 8-leaf block-id Merkle = `sha256_pair` over `[L0..L7]` (depth 3).
//!
//! See `dexdo-halo2-kit/MULTITHREAD_CIRCUIT_SPEC.md` §1, §4 for the full design
//! and how this primitive is reused inside MultiHopProof.

pub mod circuit;
pub mod multi_hop;
pub mod native;

/// 37-byte tag prepended before `block_id` when computing the L7 leaf hash for
/// the parent slot (`index == 0`). Mirrors
/// `acki-nacki/node/libs/history-proof/src/lib.rs::REFERENCED_PARENT_BLOCK_TAG`.
pub const REFERENCED_PARENT_BLOCK_TAG: &[u8] = b"acki-nacki:referenced-block:parent:v1";

/// 34-byte tag prepended before `block_id` when computing the L7 leaf hash for
/// the non-parent ref slots (`index >= 1`). Mirrors
/// `acki-nacki/node/libs/history-proof/src/lib.rs::REFERENCED_REF_BLOCK_TAG`.
pub const REFERENCED_REF_BLOCK_TAG: &[u8] = b"acki-nacki:referenced-block:ref:v1";

/// Locked upper bound on `proof_block_refs.len()` for a single block.
///
/// Per `MULTITHREAD_CIRCUIT_SPEC.md` §10.1 and §4.2, the L7 Poseidon dense
/// Merkle tree is padded to width 256, i.e. depth 8.
pub const MAX_PROOF_BLOCK_REFS: usize = 256;

/// Depth of the per-block L7 Poseidon dense-Merkle tree (`log2(MAX_PROOF_BLOCK_REFS)`).
pub const L7_TREE_DEPTH: usize = 8;

/// Number of leaves in the 8-leaf SHA-256 block-id Merkle tree.
pub const BLOCK_ID_LEAF_COUNT: usize = 8;

/// Depth of the 8-leaf block-id Merkle tree (`log2(BLOCK_ID_LEAF_COUNT)`).
pub const BLOCK_ID_TREE_DEPTH: usize = 3;

/// Position (0-indexed) of the L7 leaf in the block-id 8-leaf Merkle tree.
pub const L7_LEAF_INDEX: usize = 7;

/// Default maximum number of L7 hops per MultiHopProof snark. With
/// `1 DexFinalProof + 4 MultiHopProof` per claim and `L_MAX = 20` L7 hops
/// total, each MultiHopProof carries `MAX_HOPS = 5` hops.
///
/// Note: this is a default. The MultiHopChip is parameterized by `H` so tests
/// can run smaller circuits (e.g. `H = 2`) without committing to MAX_HOPS at
/// compile time.
pub const MAX_HOPS: usize = 5;

/// Domain-tag byte string used to bind the voucher's `sk_u` into the per-hop
/// salt. Differentiates this salt from any other Poseidon image of `sk_u`
/// produced elsewhere (e.g. spend nullifier, deposit commitment).
///
/// Length-checked at runtime in `domain_tag_hop_salt_fr()` to keep this `const`
/// stable under refactoring.
pub const DOMAIN_TAG_HOP_SALT_BYTES: &[u8] = b"acki-nacki:voucher-hop-salt:v1";
