//! Dense balanced Merkle tree proof verification.
//!
//! Off-circuit preprocessing + in-circuit algebraic linking for
//! `hash_bytes_flat`-based Poseidon Merkle trees (BN254 Fr).

use halo2_base::{
    gates::{GateInstructions, RangeInstructions},
    halo2_proofs::halo2curves::{
        bn256::Fr,
        ff::PrimeField,
    },
    poseidon::hasher::PoseidonHasher,
    AssignedValue, Context,
};
use pse_poseidon::Poseidon;

const T: usize = 3;
const RATE: usize = 2;
const R_F: usize = 8;
const R_P: usize = 57;

/// Maximum number of dense balanced trees in a verification chain.
/// Both gosh-dark-dex and layer-hashes-update circuits use this constant
/// so that one verification key works for any chain length in [1..MAX_CHAIN_LEN].
pub const MAX_CHAIN_LEN: usize = 11;

// ---------------------------------------------------------------------------
// Native (off-circuit) helpers
// ---------------------------------------------------------------------------

/// Native (off-circuit) Poseidon hash of BN254 Fr field elements.
pub fn poseidon_hash_native(inputs: &[Fr]) -> Fr {
    let mut poseidon = Poseidon::<Fr, T, RATE>::new(R_F, R_P);
    poseidon.update(inputs);
    poseidon.squeeze()
}

/// Conditional swap using gate.select: if swap=1, return (b, a); else (a, b).
pub fn cond_swap(
    ctx: &mut Context<Fr>,
    gate: &impl GateInstructions<Fr>,
    a: AssignedValue<Fr>,
    b: AssignedValue<Fr>,
    swap: AssignedValue<Fr>,
) -> (AssignedValue<Fr>, AssignedValue<Fr>) {
    let new_a = gate.select(ctx, b, a, swap);
    let new_b = gate.select(ctx, a, b, swap);
    (new_a, new_b)
}

/// Convert a 32-byte LE array to Fr.
pub fn bytes_to_fr(bytes: &[u8; 32]) -> Fr {
    Fr::from_raw([
        u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
        u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
        u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
        u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
    ])
}

/// Convert Fr to 32-byte LE array.
pub fn fr_to_bytes(fr: Fr) -> [u8; 32] {
    fr.to_repr()
}

// ---------------------------------------------------------------------------
// Proof types
// ---------------------------------------------------------------------------

/// Preprocessed data for one level of a dense balanced tree proof.
/// Contains the sibling hash and the precomputed 31-byte chunks used by `hash_bytes_flat`.
#[derive(Clone, Debug)]
pub struct DenseProofLevel {
    /// The sibling hash at this level (32 bytes)
    pub sibling: [u8; 32],
    /// Direction bit: false = current node is left child, true = current node is right child
    pub direction_bit: bool,
    /// chunk0 = Fr from left[0..31] zero-padded to 32 bytes LE
    pub chunk0: Fr,
    /// chunk1 = Fr from left[31]||right[0..30] zero-padded to 32 bytes LE (boundary-crossing chunk)
    pub chunk1: Fr,
    /// chunk2 = Fr from right[30..32] zero-padded to 32 bytes LE
    pub chunk2: Fr,
    /// left[31] as a single byte (used for algebraic linking)
    pub left_hi: u8,
}

/// A fully preprocessed dense balanced tree proof ready for circuit verification.
#[derive(Clone, Debug)]
pub struct DenseTreeProof {
    /// The leaf hash (32 bytes)
    pub leaf: [u8; 32],
    /// Preprocessed data for each tree level, from leaf to root.
    pub levels: Vec<DenseProofLevel>,
}

// ---------------------------------------------------------------------------
// Preprocessing
// ---------------------------------------------------------------------------

/// Convert 64 bytes into 3 Fr elements using 31-byte chunking (matching `hash_bytes_flat`).
///
/// Given `data[0..64]`:
///   chunk0 = Fr::from_le_bytes(data[0..31] ++ [0])
///   chunk1 = Fr::from_le_bytes(data[31..62] ++ [0])
///   chunk2 = Fr::from_le_bytes(data[62..64] ++ [0; 30])
fn chunk_64_bytes_to_fr(data: &[u8; 64]) -> (Fr, Fr, Fr) {
    let mut buf0 = [0u8; 32];
    buf0[..31].copy_from_slice(&data[0..31]);
    let chunk0 = bytes_to_fr(&buf0);

    let mut buf1 = [0u8; 32];
    buf1[..31].copy_from_slice(&data[31..62]);
    let chunk1 = bytes_to_fr(&buf1);

    let mut buf2 = [0u8; 32];
    buf2[..2].copy_from_slice(&data[62..64]);
    let chunk2 = bytes_to_fr(&buf2);

    (chunk0, chunk1, chunk2)
}

/// Preprocess a dense balanced tree proof for circuit verification.
///
/// Takes the raw leaf hash, sibling hashes (bottom-up), and the leaf position.
/// Returns a `DenseTreeProof` with all chunks precomputed for each level.
///
/// The direction at each level is determined by the tree index:
///   - Start with `idx = width - 1 + pos` (where `width = 2^depth`)
///   - If `idx` is odd, current node is left child (direction_bit = false)
///   - If `idx` is even, current node is right child (direction_bit = true)
///   - Move up: `idx = (idx - 1) / 2`
pub fn preprocess_dense_proof(
    leaf: [u8; 32],
    siblings: &[[u8; 32]],
    pos: usize,
) -> DenseTreeProof {
    let depth = siblings.len();
    let width = 1usize << depth;
    let mut idx = width - 1 + pos;
    let mut cur_bytes = leaf;
    let mut levels = Vec::with_capacity(depth);

    for j in 0..depth {
        // Determine direction: odd index = left child, even index = right child
        let is_right = idx % 2 == 0;
        let (left_bytes, right_bytes) = if !is_right {
            (cur_bytes, siblings[j])
        } else {
            (siblings[j], cur_bytes)
        };

        // Concatenate left || right (64 bytes)
        let mut concat = [0u8; 64];
        concat[..32].copy_from_slice(&left_bytes);
        concat[32..].copy_from_slice(&right_bytes);

        // Chunk into 3 Fr elements
        let (chunk0, chunk1, chunk2) = chunk_64_bytes_to_fr(&concat);
        let left_hi = left_bytes[31];

        levels.push(DenseProofLevel {
            sibling: siblings[j],
            direction_bit: is_right,
            chunk0,
            chunk1,
            chunk2,
            left_hi,
        });

        // Compute native hash for next level
        let hash_output = poseidon_hash_native(&[chunk0, chunk1, chunk2]);
        cur_bytes = fr_to_bytes(hash_output);

        // Move to parent
        idx = (idx - 1) / 2;
    }

    DenseTreeProof { leaf, levels }
}

/// Preprocess a dense balanced tree proof, padding to `max_depth` levels.
///
/// The first `siblings.len()` levels are identical to `preprocess_dense_proof`.
/// The remaining `max_depth - siblings.len()` levels are dummy: `sibling=[0;32]`,
/// `direction_bit=false`, chunks computed from `cur_bytes || [0;32]` (identity
/// pair — in-circuit `gate.select` will keep `cur` unchanged for these levels).
///
/// Panics if `siblings.len() > max_depth`.
pub fn preprocess_dense_proof_padded(
    leaf: [u8; 32],
    siblings: &[[u8; 32]],
    pos: usize,
    max_depth: usize,
) -> DenseTreeProof {
    assert!(
        siblings.len() <= max_depth,
        "siblings.len()={} exceeds max_depth={}",
        siblings.len(),
        max_depth
    );

    let real_depth = siblings.len();
    let width = 1usize << real_depth;
    let mut idx = width - 1 + pos;
    let mut cur_bytes = leaf;
    let mut levels = Vec::with_capacity(max_depth);

    // Real levels (identical to preprocess_dense_proof)
    for j in 0..real_depth {
        let is_right = idx % 2 == 0;
        let (left_bytes, right_bytes) = if !is_right {
            (cur_bytes, siblings[j])
        } else {
            (siblings[j], cur_bytes)
        };

        let mut concat = [0u8; 64];
        concat[..32].copy_from_slice(&left_bytes);
        concat[32..].copy_from_slice(&right_bytes);
        let (chunk0, chunk1, chunk2) = chunk_64_bytes_to_fr(&concat);
        let left_hi = left_bytes[31];

        levels.push(DenseProofLevel {
            sibling: siblings[j],
            direction_bit: is_right,
            chunk0,
            chunk1,
            chunk2,
            left_hi,
        });

        let hash_output = poseidon_hash_native(&[chunk0, chunk1, chunk2]);
        cur_bytes = fr_to_bytes(hash_output);
        idx = (idx - 1) / 2;
    }

    // Dummy levels: cur_bytes is the real root, sibling = [0;32], direction = false (left child).
    // Chunks are computed from cur_bytes || [0;32] so witnesses are consistent.
    // In-circuit, gate.select keeps `cur` unchanged for inactive levels, so
    // cur_bytes must NOT advance — all dummy levels use the same cur_bytes.
    let dummy_cur_bytes = cur_bytes;
    for _ in real_depth..max_depth {
        let dummy_sibling = [0u8; 32];
        let mut concat = [0u8; 64];
        concat[..32].copy_from_slice(&dummy_cur_bytes);
        concat[32..].copy_from_slice(&dummy_sibling);
        let (chunk0, chunk1, chunk2) = chunk_64_bytes_to_fr(&concat);
        let left_hi = dummy_cur_bytes[31];

        levels.push(DenseProofLevel {
            sibling: dummy_sibling,
            direction_bit: false,
            chunk0,
            chunk1,
            chunk2,
            left_hi,
        });
    }

    DenseTreeProof { leaf, levels }
}

/// Compute the root hash natively from a preprocessed proof.
/// Useful for testing that preprocessing matches the tree's own root computation.
pub fn compute_root_native(proof: &DenseTreeProof) -> Fr {
    let mut cur = Fr::zero();
    for level in &proof.levels {
        cur = poseidon_hash_native(&[level.chunk0, level.chunk1, level.chunk2]);
    }
    cur
}

// ---------------------------------------------------------------------------
// In-circuit verification
// ---------------------------------------------------------------------------

/// Verify a dense balanced tree Merkle proof in-circuit.
///
/// Given a preprocessed `DenseTreeProof`, this function:
/// 1. Loads the leaf as an Fr witness
/// 2. At each level, performs conditional swap, loads chunk witnesses,
///    enforces algebraic linking constraints, range checks, and hashes.
/// 3. Returns the computed root as an `AssignedValue<Fr>`.
///
/// ## Chunking layout for `left[32] || right[32]` (64 bytes):
///
/// `hash_bytes_flat` splits at 31-byte boundaries:
///   - chunk0 = Fr(left[0..31])              — 31 bytes (248 bits)
///   - chunk1 = Fr(left[31] || right[0..30]) — 31 bytes (248 bits)
///   - chunk2 = Fr(right[30..32])            — 2 bytes  (16 bits)
///
/// ## Algebraic linking constraints:
///   - left_fr  = chunk0 + left_hi * 2^248       (left_hi = left[31], 1 byte)
///   - right_fr = right_low + chunk2 * 2^240      (right_low = right[0..30], 30 bytes)
///   - chunk1   = left_hi + 256 * right_low
///
/// ## Range checks (canonical decomposition):
///   - chunk0 < 2^248, right_low < 2^240
///   - left_hi < 2^8, chunk2 < 2^16
///
/// The `leaf_fr` parameter is the leaf hash as an assigned Fr value.
/// The caller is responsible for loading and constraining it (e.g., linking
/// it to a SHA-256 output). Use `bytes_to_fr(&leaf_bytes)` for the witness value.
pub fn dense_merkle_root_circuit(
    ctx: &mut Context<Fr>,
    range: &impl RangeInstructions<Fr>,
    hasher: &PoseidonHasher<Fr, T, RATE>,
    proof: &DenseTreeProof,
    leaf_fr: AssignedValue<Fr>,
) -> AssignedValue<Fr> {
    let gate = range.gate();

    // 2^248 = 256^31: splits left[0..31] from left[31]
    let pow_248 = ctx.load_constant(Fr::from_raw([0u64, 0u64, 0u64, 1u64 << 56]));
    // 2^240 = 256^30: splits right[0..30] from right[30..32]
    let pow_240 = ctx.load_constant(Fr::from_raw([0u64, 0u64, 0u64, 1u64 << 48]));
    let two56 = ctx.load_constant(Fr::from(256u64));

    let mut cur = leaf_fr;

    for level in &proof.levels {
        // Load sibling as witness
        let sibling_fr = ctx.load_witness(bytes_to_fr(&level.sibling));

        // Load direction bit
        let bit_val = if level.direction_bit { Fr::one() } else { Fr::zero() };
        let bit = ctx.load_witness(bit_val);
        gate.assert_bit(ctx, bit);

        // Conditional swap: if bit=1 (right child), swap so sibling is on left
        let (left, right) = cond_swap(ctx, gate, cur, sibling_fr, bit);

        // Load chunk witnesses
        let c0 = ctx.load_witness(level.chunk0);
        let c1 = ctx.load_witness(level.chunk1);
        let c2 = ctx.load_witness(level.chunk2);
        let left_hi = ctx.load_witness(Fr::from(level.left_hi as u64));

        // Constraint 1: chunk0 + left_hi * 2^248 == left_fr
        let lhs = gate.mul_add(ctx, left_hi, pow_248, c0);
        ctx.constrain_equal(&lhs, &left);

        // Constraint 2: right_low = right_fr - chunk2 * 2^240
        let c2_shifted = gate.mul(ctx, c2, pow_240);
        let right_low = gate.sub(ctx, right, c2_shifted);

        // Constraint 3: chunk1 == left_hi + 256 * right_low
        let rhs = gate.mul_add(ctx, right_low, two56, left_hi);
        ctx.constrain_equal(&rhs, &c1);

        // Range checks for canonical decomposition
        range.range_check(ctx, c0, 248);       // 31 bytes
        range.range_check(ctx, right_low, 240); // 30 bytes
        range.range_check(ctx, left_hi, 8);     // 1 byte
        range.range_check(ctx, c2, 16);         // 2 bytes

        // Hash the 3 chunks to get the next level's value
        cur = hasher.hash_fix_len_array(ctx, gate, &[c0, c1, c2]);
    }

    cur
}

/// Verify a dense balanced tree Merkle proof in-circuit with fixed iteration count.
///
/// Like `dense_merkle_root_circuit`, but always iterates over all `proof.levels`
/// and uses `gate.select` to mask out inactive levels (where `j >= num_active_levels`).
/// This ensures the circuit shape is identical regardless of the real tree depth,
/// so one verification key works for any depth up to `proof.levels.len()`.
///
/// `num_active_levels` must be in `[0, proof.levels.len()]` (caller must range-check).
pub fn dense_merkle_root_circuit_padded(
    ctx: &mut Context<Fr>,
    range: &impl RangeInstructions<Fr>,
    hasher: &PoseidonHasher<Fr, T, RATE>,
    proof: &DenseTreeProof,
    leaf_fr: AssignedValue<Fr>,
    num_active_levels: AssignedValue<Fr>,
) -> AssignedValue<Fr> {
    let gate = range.gate();

    let pow_248 = ctx.load_constant(Fr::from_raw([0u64, 0u64, 0u64, 1u64 << 56]));
    let pow_240 = ctx.load_constant(Fr::from_raw([0u64, 0u64, 0u64, 1u64 << 48]));
    let two56 = ctx.load_constant(Fr::from(256u64));

    let mut cur = leaf_fr;

    for (j, level) in proof.levels.iter().enumerate() {
        // active = (j < num_active_levels)
        let j_const = ctx.load_constant(Fr::from(j as u64));
        let active = range.is_less_than(ctx, j_const, num_active_levels, 4);

        // Load sibling as witness
        let sibling_fr = ctx.load_witness(bytes_to_fr(&level.sibling));

        // Load direction bit
        let bit_val = if level.direction_bit { Fr::one() } else { Fr::zero() };
        let bit = ctx.load_witness(bit_val);
        gate.assert_bit(ctx, bit);

        // Conditional swap: if bit=1 (right child), swap so sibling is on left
        let (left, right) = cond_swap(ctx, gate, cur, sibling_fr, bit);

        // Load chunk witnesses
        let c0 = ctx.load_witness(level.chunk0);
        let c1 = ctx.load_witness(level.chunk1);
        let c2 = ctx.load_witness(level.chunk2);
        let left_hi = ctx.load_witness(Fr::from(level.left_hi as u64));

        // Constraint 1: chunk0 + left_hi * 2^248 == left_fr
        let lhs = gate.mul_add(ctx, left_hi, pow_248, c0);
        ctx.constrain_equal(&lhs, &left);

        // Constraint 2: right_low = right_fr - chunk2 * 2^240
        let c2_shifted = gate.mul(ctx, c2, pow_240);
        let right_low = gate.sub(ctx, right, c2_shifted);

        // Constraint 3: chunk1 == left_hi + 256 * right_low
        let rhs = gate.mul_add(ctx, right_low, two56, left_hi);
        ctx.constrain_equal(&rhs, &c1);

        // Range checks for canonical decomposition
        range.range_check(ctx, c0, 248);
        range.range_check(ctx, right_low, 240);
        range.range_check(ctx, left_hi, 8);
        range.range_check(ctx, c2, 16);

        // Hash the 3 chunks to get the computed value for this level
        let computed = hasher.hash_fix_len_array(ctx, gate, &[c0, c1, c2]);

        // If active, use computed; otherwise keep cur unchanged
        cur = gate.select(ctx, computed, cur, active);
    }

    cur
}

// ---------------------------------------------------------------------------
// Chain support
// ---------------------------------------------------------------------------

/// One link in a chain of dense balanced tree Merkle proofs.
///
/// Used to prove a sequence of Merkle memberships where each step's leaf
/// is the previous step's computed root (or an independently supplied value).
#[derive(Clone, Debug)]
pub struct DenseChainLink {
    /// Whether this link is active (contains a real proof).
    pub active: bool,
    /// Sibling hashes for the Merkle proof (bottom-up), one per tree level.
    pub siblings: Vec<[u8; 32]>,
    /// Leaf position in the tree.
    pub position: usize,
    /// Leaf bytes for this link.
    pub leaf_native: [u8; 32],
}

impl DenseChainLink {
    /// Create an inactive (padding) link with the given leaf bytes and depth.
    pub fn inactive(leaf_bytes: [u8; 32], depth: usize) -> Self {
        Self {
            active: false,
            siblings: vec![[0u8; 32]; depth],
            position: 0,
            leaf_native: leaf_bytes,
        }
    }
}

/// Verify a chain of dense balanced tree proofs in-circuit.
///
/// Starting from `initial_leaf_fr`, each active link proves that the current
/// root is a leaf of the next tree. The first `num_active_steps` links are
/// active; the rest are masked out via `gate.select`.
///
/// The caller MUST constrain `num_active_steps` to `[1, chain.len()]` before
/// calling this function (e.g., via range checks).
///
/// `chain` must have exactly `MAX_CHAIN_LEN` entries. The circuit shape is
/// fixed regardless of how many steps are actually active, so one verification
/// key works for any chain length in `[1, MAX_CHAIN_LEN]`.
///
/// Returns the final root (last active tree's root).
pub fn verify_chain_of_dense_proofs(
    ctx: &mut Context<Fr>,
    range: &impl RangeInstructions<Fr>,
    hasher: &PoseidonHasher<Fr, T, RATE>,
    initial_leaf_fr: AssignedValue<Fr>,
    chain: &[DenseChainLink],
    num_active_steps: AssignedValue<Fr>,
) -> AssignedValue<Fr> {
    assert_eq!(chain.len(), MAX_CHAIN_LEN);
    let gate = range.gate();

    let mut current = initial_leaf_fr;

    for (j, link) in chain.iter().enumerate() {
        // active = (j < num_active_steps)
        let j_const = ctx.load_constant(Fr::from(j as u64));
        let active = range.is_less_than(ctx, j_const, num_active_steps, 4);

        // Preprocess the proof using leaf_native bytes (off-circuit).
        let dense_proof = preprocess_dense_proof(
            link.leaf_native,
            &link.siblings,
            link.position,
        );

        // Verify the proof in-circuit with current as the leaf.
        let computed_root = dense_merkle_root_circuit(
            ctx, range, hasher, &dense_proof, current,
        );

        // If active, use computed_root; otherwise keep current.
        current = gate.select(ctx, computed_root, current, active);
    }

    current
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_64_bytes() {
        let mut data = [0u8; 64];
        for i in 0..64 {
            data[i] = i as u8;
        }

        let (c0, c1, c2) = chunk_64_bytes_to_fr(&data);

        // Verify chunk0: bytes 0..31 in LE
        let mut expected0 = [0u8; 32];
        expected0[..31].copy_from_slice(&data[0..31]);
        assert_eq!(c0, bytes_to_fr(&expected0));

        // Verify chunk1: bytes 31..62 in LE
        let mut expected1 = [0u8; 32];
        expected1[..31].copy_from_slice(&data[31..62]);
        assert_eq!(c1, bytes_to_fr(&expected1));

        // Verify chunk2: bytes 62..64 in LE
        let mut expected2 = [0u8; 32];
        expected2[..2].copy_from_slice(&data[62..64]);
        assert_eq!(c2, bytes_to_fr(&expected2));
    }

    #[test]
    fn test_dense_chain_link_inactive() {
        let leaf = [0xABu8; 32];
        let link = DenseChainLink::inactive(leaf, 8);
        assert!(!link.active);
        assert_eq!(link.siblings.len(), 8);
        assert_eq!(link.leaf_native, leaf);
        assert_eq!(link.position, 0);
    }

    #[test]
    fn test_preprocess_padded_matches_unpadded() {
        // When max_depth == real_depth, padded should produce identical levels.
        let leaf = [0x42u8; 32];
        let siblings: Vec<[u8; 32]> = (0..3)
            .map(|i| {
                let mut s = [0u8; 32];
                s[0] = (i + 1) as u8;
                s
            })
            .collect();

        let unpadded = preprocess_dense_proof(leaf, &siblings, 2);
        let padded = preprocess_dense_proof_padded(leaf, &siblings, 2, 3);

        assert_eq!(unpadded.levels.len(), 3);
        assert_eq!(padded.levels.len(), 3);
        for (u, p) in unpadded.levels.iter().zip(padded.levels.iter()) {
            assert_eq!(u.sibling, p.sibling);
            assert_eq!(u.direction_bit, p.direction_bit);
            assert_eq!(u.chunk0, p.chunk0);
            assert_eq!(u.chunk1, p.chunk1);
            assert_eq!(u.chunk2, p.chunk2);
            assert_eq!(u.left_hi, p.left_hi);
        }
    }

    #[test]
    fn test_preprocess_padded_extra_levels() {
        let leaf = [0x11u8; 32];
        let siblings: Vec<[u8; 32]> = vec![[0xAAu8; 32], [0xBBu8; 32]];

        let padded = preprocess_dense_proof_padded(leaf, &siblings, 0, 5);
        assert_eq!(padded.levels.len(), 5);

        // First 2 levels should match unpadded
        let unpadded = preprocess_dense_proof(leaf, &siblings, 0);
        for i in 0..2 {
            assert_eq!(padded.levels[i].sibling, unpadded.levels[i].sibling);
            assert_eq!(padded.levels[i].direction_bit, unpadded.levels[i].direction_bit);
            assert_eq!(padded.levels[i].chunk0, unpadded.levels[i].chunk0);
        }

        // Dummy levels: sibling=[0;32], direction=false, all identical chunks
        for j in 2..5 {
            assert_eq!(padded.levels[j].sibling, [0u8; 32]);
            assert!(!padded.levels[j].direction_bit);
        }
        // All dummy levels must have identical chunks (same cur_bytes, no advancement)
        for j in 3..5 {
            assert_eq!(padded.levels[j].chunk0, padded.levels[2].chunk0);
            assert_eq!(padded.levels[j].chunk1, padded.levels[2].chunk1);
            assert_eq!(padded.levels[j].chunk2, padded.levels[2].chunk2);
            assert_eq!(padded.levels[j].left_hi, padded.levels[2].left_hi);
        }
    }

    #[test]
    fn test_preprocess_padded_zero_depth() {
        let leaf = [0xFFu8; 32];
        let padded = preprocess_dense_proof_padded(leaf, &[], 0, 4);
        assert_eq!(padded.levels.len(), 4);
        assert_eq!(padded.leaf, leaf);
        // All levels are dummy
        for level in &padded.levels {
            assert_eq!(level.sibling, [0u8; 32]);
            assert!(!level.direction_bit);
        }
    }
}
