//! Native (off-circuit) helpers that mirror the on-chain semantics from
//! `acki-nacki/node/libs/history-proof/src/lib.rs`.
//!
//! These are the single source of truth that the circuit must agree with:
//!
//!   * [`poseidon_hash_bytes_flat_native`] — Poseidon over a flat byte string,
//!     chunked to 31-byte LE field elements. Matches
//!     `PoseidonSponge::hash_bytes_flat` (which `PoseidonHasher::digest` calls).
//!   * [`sha256_pair_native`] — single SHA-256 over `left || right` (64 bytes).
//!   * [`compute_referenced_block_leaf_hash_native`] — `tag || block_id`
//!     digested via Poseidon. `tag` switches on `index == 0`.
//!   * [`compute_referenced_blocks_root_native`] — Poseidon dense Merkle root
//!     over the tagged ref leaves, with `next_power_of_two` zero-padding.
//!   * [`compute_block_id_native`] — 8-leaf SHA-256 Merkle root over `[L0..L7]`.
//!   * [`block_id_path_siblings`] — 3-deep sibling list for a chosen leaf.
//!
//! Refer to `MULTITHREAD_CIRCUIT_SPEC.md` §1.1 (block-id tree), §1.3 (L7), §4.

use gosh_dense_balanced_tree::{bytes_to_fr, fr_to_bytes, poseidon_hash_native};
use halo2_base::halo2_proofs::halo2curves::bn256::Fr;
use sha2::{Digest, Sha256};

use crate::{
    BLOCK_ID_LEAF_COUNT, BLOCK_ID_TREE_DEPTH, L7_LEAF_INDEX, REFERENCED_PARENT_BLOCK_TAG,
    REFERENCED_REF_BLOCK_TAG,
};

// ---------------------------------------------------------------------------
// Poseidon over flat bytes — mirrors `PoseidonSponge::hash_bytes_flat`
// ---------------------------------------------------------------------------

/// Chunk a flat byte string into 31-byte LE Fr elements (last chunk
/// zero-extended to 32 bytes), then run `poseidon_hash_native` over them.
///
/// Mirrors `PoseidonSponge::hash_bytes_flat` as called by
/// `PoseidonHasher::digest` in `acki-nacki/node/libs/history-proof`. The 31-byte
/// chunking matches the `chunk_64_bytes_to_fr` convention used in
/// `gosh-dense-balanced-tree::dense_merkle_root_circuit`.
///
/// Returns the squeezed Poseidon field element as 32 LE bytes.
pub fn poseidon_hash_bytes_flat_native(bytes: &[u8]) -> [u8; 32] {
    let mut inputs: Vec<Fr> = Vec::with_capacity(bytes.len().div_ceil(31));
    let mut i = 0;
    while i < bytes.len() {
        let end = (i + 31).min(bytes.len());
        let mut buf = [0u8; 32];
        buf[..end - i].copy_from_slice(&bytes[i..end]);
        inputs.push(bytes_to_fr(&buf));
        i = end;
    }
    // Empty input: hash the empty sponge state (Poseidon::squeeze on empty inputs).
    // The reference `hash_bytes_flat` is not invoked with empty input for any
    // of the call sites we mirror here, so we still produce a consistent value.
    let out = poseidon_hash_native(&inputs);
    fr_to_bytes(out)
}

// ---------------------------------------------------------------------------
// SHA-256 pair — the 8-leaf block-id Merkle combiner
// ---------------------------------------------------------------------------

/// `SHA-256(left || right)` — the combine function for the 8-leaf block-id
/// Merkle tree. Inputs and output are exactly 32 bytes (the canonical SHA-256
/// big-endian digest).
pub fn sha256_pair_native(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(left);
    hasher.update(right);
    let out = hasher.finalize();
    let mut result = [0u8; 32];
    result.copy_from_slice(&out);
    result
}

// ---------------------------------------------------------------------------
// L7 referenced-block leaf and root
// ---------------------------------------------------------------------------

/// Hash one leaf of the L7 referenced-blocks tree.
///
/// `index == 0` uses [`REFERENCED_PARENT_BLOCK_TAG`] (37 bytes), all higher
/// indices use [`REFERENCED_REF_BLOCK_TAG`] (34 bytes). The preimage is
/// `tag || block_id` (block_id is 32 bytes). Mirrors
/// `history_proof::compute_referenced_block_leaf_hash`.
pub fn compute_referenced_block_leaf_hash_native(index: usize, block_id: &[u8; 32]) -> [u8; 32] {
    let tag: &[u8] =
        if index == 0 { REFERENCED_PARENT_BLOCK_TAG } else { REFERENCED_REF_BLOCK_TAG };
    let mut buf = Vec::with_capacity(tag.len() + 32);
    buf.extend_from_slice(tag);
    buf.extend_from_slice(block_id);
    poseidon_hash_bytes_flat_native(&buf)
}

/// Compute the Poseidon dense-Merkle root of `proof_block_refs` exactly as
/// the on-chain helper does it.
///
/// `refs[0]` is the parent block id (tagged with the parent tag),
/// `refs[1..]` are CommonSection refs (tagged with the ref tag).
///
/// Tree shape: leaves are padded to `next_power_of_two(refs.len())` with
/// `[0u8; 32]`, then combined via Poseidon over the 31-byte chunked
/// concatenation `left || right`.
///
/// Returns `[0u8; 32]` for an empty input (matches the on-chain early-return).
pub fn compute_referenced_blocks_root_native(refs: &[[u8; 32]]) -> [u8; 32] {
    if refs.is_empty() {
        return [0u8; 32];
    }
    let width = refs.len().next_power_of_two();
    let mut level: Vec<[u8; 32]> = Vec::with_capacity(width);
    for (i, block_id) in refs.iter().enumerate() {
        level.push(compute_referenced_block_leaf_hash_native(i, block_id));
    }
    while level.len() < width {
        level.push([0u8; 32]);
    }
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len() / 2);
        for pair in level.chunks(2) {
            let mut buf = [0u8; 64];
            buf[..32].copy_from_slice(&pair[0]);
            buf[32..].copy_from_slice(&pair[1]);
            next.push(poseidon_hash_bytes_flat_native(&buf));
        }
        level = next;
    }
    level[0]
}

/// Compute the L7 sibling path for `ref_index` in the Poseidon dense-Merkle
/// tree formed from `refs` (with `next_power_of_two` zero-padding).
///
/// Returns `(siblings_from_leaf_to_root, real_depth)`. `siblings` has
/// `real_depth` entries; the caller pads to a fixed max depth before feeding
/// the circuit's variable-depth dense_merkle primitive.
///
/// Panics if `ref_index >= refs.len()` or `refs` is empty.
pub fn compute_referenced_blocks_path_native(
    refs: &[[u8; 32]],
    ref_index: usize,
) -> (Vec<[u8; 32]>, usize) {
    assert!(!refs.is_empty(), "refs must not be empty");
    assert!(ref_index < refs.len(), "ref_index out of range");

    let real_depth = (refs.len().next_power_of_two()).trailing_zeros() as usize;
    let width = 1usize << real_depth;
    let mut level: Vec<[u8; 32]> = Vec::with_capacity(width);
    for (i, block_id) in refs.iter().enumerate() {
        level.push(compute_referenced_block_leaf_hash_native(i, block_id));
    }
    while level.len() < width {
        level.push([0u8; 32]);
    }

    let mut siblings = Vec::with_capacity(real_depth);
    let mut idx = ref_index;
    while level.len() > 1 {
        let sibling_idx = idx ^ 1;
        siblings.push(level[sibling_idx]);
        let mut next = Vec::with_capacity(level.len() / 2);
        for pair in level.chunks(2) {
            let mut buf = [0u8; 64];
            buf[..32].copy_from_slice(&pair[0]);
            buf[32..].copy_from_slice(&pair[1]);
            next.push(poseidon_hash_bytes_flat_native(&buf));
        }
        level = next;
        idx /= 2;
    }
    (siblings, real_depth)
}

// ---------------------------------------------------------------------------
// 8-leaf SHA-256 block-id Merkle tree
// ---------------------------------------------------------------------------

/// Compute the block_id from the 8 outer Merkle leaves `[L0..L7]` using
/// SHA-256 as the combine function (`sha256_pair`). Tree depth is 3.
pub fn compute_block_id_native(leaves: &[[u8; 32]; BLOCK_ID_LEAF_COUNT]) -> [u8; 32] {
    let mut level: Vec<[u8; 32]> = leaves.to_vec();
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len() / 2);
        for pair in level.chunks(2) {
            next.push(sha256_pair_native(&pair[0], &pair[1]));
        }
        level = next;
    }
    level[0]
}

/// Return the 3 sibling hashes (bottom-up) for the SHA-256 path that opens
/// `leaves[leaf_index]` in the 8-leaf block-id Merkle tree.
pub fn block_id_path_siblings(
    leaves: &[[u8; 32]; BLOCK_ID_LEAF_COUNT],
    leaf_index: usize,
) -> [[u8; 32]; BLOCK_ID_TREE_DEPTH] {
    assert!(leaf_index < BLOCK_ID_LEAF_COUNT, "leaf_index out of range");
    let mut level: Vec<[u8; 32]> = leaves.to_vec();
    let mut siblings = [[0u8; 32]; BLOCK_ID_TREE_DEPTH];
    let mut idx = leaf_index;
    for s in siblings.iter_mut() {
        let sibling_idx = idx ^ 1;
        *s = level[sibling_idx];
        let mut next = Vec::with_capacity(level.len() / 2);
        for pair in level.chunks(2) {
            next.push(sha256_pair_native(&pair[0], &pair[1]));
        }
        level = next;
        idx /= 2;
    }
    siblings
}

/// Convenience: extract the L7 sibling path for the 8-leaf tree.
pub fn block_id_l7_siblings(
    leaves: &[[u8; 32]; BLOCK_ID_LEAF_COUNT],
) -> [[u8; 32]; BLOCK_ID_TREE_DEPTH] {
    block_id_path_siblings(leaves, L7_LEAF_INDEX)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference vector for `compute_referenced_block_leaf_hash_native(0, ..)`
    /// produced by mirroring the algorithm in pure Rust here. Acts as a smoke
    /// test that the parent tag branch is used at `index == 0`.
    #[test]
    fn parent_and_ref_tags_produce_different_leaf_hashes() {
        let block_id = [0x11u8; 32];
        let parent = compute_referenced_block_leaf_hash_native(0, &block_id);
        let ref_leaf = compute_referenced_block_leaf_hash_native(1, &block_id);
        assert_ne!(parent, ref_leaf, "parent and ref tag must diverge at index boundary");
    }

    #[test]
    fn ref_tag_is_used_for_all_nonzero_indices() {
        let block_id = [0x22u8; 32];
        let r1 = compute_referenced_block_leaf_hash_native(1, &block_id);
        let r2 = compute_referenced_block_leaf_hash_native(2, &block_id);
        let r5 = compute_referenced_block_leaf_hash_native(5, &block_id);
        assert_eq!(r1, r2, "index 1..N must all hash with the same ref tag");
        assert_eq!(r2, r5, "index 1..N must all hash with the same ref tag");
    }

    #[test]
    fn referenced_blocks_root_empty_is_zero() {
        assert_eq!(compute_referenced_blocks_root_native(&[]), [0u8; 32]);
    }

    #[test]
    fn referenced_blocks_root_single_element_is_just_the_leaf() {
        let block_id = [0x33u8; 32];
        let root = compute_referenced_blocks_root_native(&[block_id]);
        let leaf = compute_referenced_block_leaf_hash_native(0, &block_id);
        assert_eq!(root, leaf, "single-leaf tree has root == leaf (width=1)");
    }

    #[test]
    fn referenced_blocks_root_two_elements_matches_manual_combine() {
        let parent = [0x44u8; 32];
        let r1 = [0x55u8; 32];
        let root = compute_referenced_blocks_root_native(&[parent, r1]);

        let left = compute_referenced_block_leaf_hash_native(0, &parent);
        let right = compute_referenced_block_leaf_hash_native(1, &r1);
        let mut buf = [0u8; 64];
        buf[..32].copy_from_slice(&left);
        buf[32..].copy_from_slice(&right);
        let expected = poseidon_hash_bytes_flat_native(&buf);
        assert_eq!(root, expected);
    }

    #[test]
    fn referenced_blocks_root_three_elements_pads_to_four_with_zero() {
        let parent = [0xA1u8; 32];
        let r1 = [0xA2u8; 32];
        let r2 = [0xA3u8; 32];
        let root = compute_referenced_blocks_root_native(&[parent, r1, r2]);

        // Build width=4 tree manually.
        let l0 = compute_referenced_block_leaf_hash_native(0, &parent);
        let l1 = compute_referenced_block_leaf_hash_native(1, &r1);
        let l2 = compute_referenced_block_leaf_hash_native(2, &r2);
        let l3 = [0u8; 32];

        let combine = |a: &[u8; 32], b: &[u8; 32]| -> [u8; 32] {
            let mut buf = [0u8; 64];
            buf[..32].copy_from_slice(a);
            buf[32..].copy_from_slice(b);
            poseidon_hash_bytes_flat_native(&buf)
        };

        let l01 = combine(&l0, &l1);
        let l23 = combine(&l2, &l3);
        let expected = combine(&l01, &l23);
        assert_eq!(root, expected);
    }

    #[test]
    fn referenced_blocks_path_recomputes_root_for_every_index() {
        let refs: Vec<[u8; 32]> =
            (0..5u8).map(|i| [i.wrapping_add(0x70); 32]).collect();
        let root = compute_referenced_blocks_root_native(&refs);

        for index in 0..refs.len() {
            let (siblings, depth) = compute_referenced_blocks_path_native(&refs, index);
            // refs.len() == 5 → next_power_of_two = 8 → depth = 3
            assert_eq!(depth, 3);
            assert_eq!(siblings.len(), depth);

            // Replay the path manually.
            let mut cur = compute_referenced_block_leaf_hash_native(index, &refs[index]);
            let mut idx = index;
            for sib in &siblings {
                let mut buf = [0u8; 64];
                let (left, right) =
                    if idx % 2 == 0 { (&cur, sib) } else { (sib, &cur) };
                buf[..32].copy_from_slice(left);
                buf[32..].copy_from_slice(right);
                cur = poseidon_hash_bytes_flat_native(&buf);
                idx /= 2;
            }
            assert_eq!(cur, root, "path replay failed at index {index}");
        }
    }

    #[test]
    fn block_id_native_matches_manual_sha_combine() {
        let leaves: [[u8; 32]; 8] = std::array::from_fn(|i| [(i as u8).wrapping_mul(17); 32]);
        let got = compute_block_id_native(&leaves);

        let pair = |a: &[u8; 32], b: &[u8; 32]| sha256_pair_native(a, b);
        let l01 = pair(&leaves[0], &leaves[1]);
        let l23 = pair(&leaves[2], &leaves[3]);
        let l45 = pair(&leaves[4], &leaves[5]);
        let l67 = pair(&leaves[6], &leaves[7]);
        let l0123 = pair(&l01, &l23);
        let l4567 = pair(&l45, &l67);
        let expected = pair(&l0123, &l4567);

        assert_eq!(got, expected);
    }

    #[test]
    fn block_id_path_recomputes_root_for_every_leaf_index() {
        let leaves: [[u8; 32]; 8] = std::array::from_fn(|i| [0xC0u8 ^ (i as u8); 32]);
        let root = compute_block_id_native(&leaves);

        for leaf_index in 0..BLOCK_ID_LEAF_COUNT {
            let siblings = block_id_path_siblings(&leaves, leaf_index);
            let mut cur = leaves[leaf_index];
            let mut idx = leaf_index;
            for sib in &siblings {
                cur = if idx % 2 == 0 {
                    sha256_pair_native(&cur, sib)
                } else {
                    sha256_pair_native(sib, &cur)
                };
                idx /= 2;
            }
            assert_eq!(cur, root, "block_id path replay failed at leaf {leaf_index}");
        }
    }

    #[test]
    fn block_id_l7_siblings_matches_explicit_path() {
        let leaves: [[u8; 32]; 8] = std::array::from_fn(|i| [(0xD0 + i as u8); 32]);
        let l7_only = block_id_l7_siblings(&leaves);
        let general = block_id_path_siblings(&leaves, L7_LEAF_INDEX);
        assert_eq!(l7_only, general);
    }

    /// Parity vectors emitted by
    /// `acki-nacki/node/libs/history-proof/tests/golden_vectors.rs` on the
    /// `poseidon_dex` branch (canonical multi-thread design). These lock our
    /// `pse_poseidon` parameters and 31-byte chunking convention against the
    /// on-chain `tvm_vm::executor::zk_stuff::bn254::poseidon::PoseidonSponge`.
    ///
    /// Regenerate with:
    ///   cargo test -p history-proof --test golden_vectors -- --nocapture
    /// (from `acki-nacki/`).
    #[test]
    fn parity_with_acki_nacki_history_proof_vectors() {
        // Vector A: compute_referenced_block_leaf_hash(0, [0x11; 32])
        let a = compute_referenced_block_leaf_hash_native(0, &[0x11u8; 32]);
        assert_eq!(
            hex::encode(a),
            "eab0366c0d4c74d1928e39cfcd79872f4c89325929cf62b3c89df86cbb58c01e",
            "parent-tag leaf hash diverged from acki-nacki history-proof"
        );

        // Vector B: compute_referenced_block_leaf_hash(1, [0x22; 32])
        let b = compute_referenced_block_leaf_hash_native(1, &[0x22u8; 32]);
        assert_eq!(
            hex::encode(b),
            "c3440b45d47c45e5c58cfe3e667adc8e8694d802025d542ad5554ae69447be23",
            "ref-tag leaf hash diverged from acki-nacki history-proof"
        );

        // Vector C: compute_referenced_blocks_root([[0x44; 32], [0x55; 32]])
        let c = compute_referenced_blocks_root_native(&[[0x44u8; 32], [0x55u8; 32]]);
        assert_eq!(
            hex::encode(c),
            "be62a0d9c3dfb394bfeb64b240cb022c478ee41c7eb929e14444c00cf07d8c25",
            "L7 root (width=2) diverged from acki-nacki history-proof"
        );

        // Vector D: compute_referenced_blocks_root([[0xA1;32],[0xA2;32],[0xA3;32]])
        // Verifies that next_power_of_two zero-padding (width=4) agrees.
        let d = compute_referenced_blocks_root_native(&[
            [0xA1u8; 32],
            [0xA2u8; 32],
            [0xA3u8; 32],
        ]);
        assert_eq!(
            hex::encode(d),
            "d7c110973862e4517ddb7c6044e42c570e7a43b1767cf123f1f3ece5a105000f",
            "L7 root (width=4 with zero-padding) diverged from acki-nacki history-proof"
        );

        // Vector H: compute_block_leaf_hash(block_id, envelope, ext_out_root)
        // = Poseidon over a 96-byte flat buffer. Confirms our 31-byte chunking
        // matches PoseidonHasher::digest for the layer-1 batch-tree leaf.
        let mut buf = [0u8; 96];
        buf[..32].copy_from_slice(&[0x01u8; 32]);
        buf[32..64].copy_from_slice(&[0x02u8; 32]);
        buf[64..96].copy_from_slice(&[0x03u8; 32]);
        let h = poseidon_hash_bytes_flat_native(&buf);
        assert_eq!(
            hex::encode(h),
            "ef72b6347d7882a00125c65a51fc7e3c1f414dea235fc6c7454401d4749fe518",
            "poseidon_hash_bytes_flat_native over 96 bytes diverged from \
             compute_block_leaf_hash — chunking or Poseidon params do not match"
        );
    }

    #[test]
    fn poseidon_hash_bytes_flat_chunking_matches_31_byte_boundary() {
        // Two preimages that differ only beyond the first 31 bytes must hash
        // differently — proves the second chunk is absorbed.
        let mut a = [0u8; 35];
        let mut b = [0u8; 35];
        a[34] = 1;
        b[34] = 2;
        assert_ne!(
            poseidon_hash_bytes_flat_native(&a),
            poseidon_hash_bytes_flat_native(&b)
        );
    }
}
