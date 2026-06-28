//! MockProver integration tests for `HopChip::verify_hop`.
//!
//! Test design: build a synthetic block whose 8 outer SHA-256 leaves include
//! a real L7 Poseidon root (over a small ref list), then prove that the
//! resulting block_id correctly "hops" to a chosen ref. Both happy paths
//! (varying depths and tag boundaries) and tampering cases are exercised.

use std::convert::TryInto;

use gosh_dense_balanced_tree::{preprocess_dense_proof_padded, DenseTreeProof};
use gosh_referenced_block_hop::{
    circuit::{HopChip, HopOutput},
    native::{
        compute_block_id_native, compute_referenced_block_leaf_hash_native,
        compute_referenced_blocks_path_native, compute_referenced_blocks_root_native,
    },
    L7_TREE_DEPTH,
};
use gosh_sha256_chip::Sha256Chip;
use halo2_base::{
    gates::{circuit::builder::BaseCircuitBuilder, RangeInstructions},
    halo2_proofs::{dev::MockProver, halo2curves::bn256::Fr},
    poseidon::hasher::{spec::OptimizedPoseidonSpec, PoseidonHasher as Halo2PoseidonHasher},
    AssignedValue, Context,
};

const T: usize = 3;
const RATE: usize = 2;
const K: usize = 16;

// ---------------------------------------------------------------------------
// Synthetic block construction
// ---------------------------------------------------------------------------

/// One "block" of synthetic test data: its 8 outer SHA-256 leaves (with the
/// L7 slot set to the Poseidon root over `refs`), the resulting block_id,
/// and the refs vector that was committed.
struct SyntheticBlock {
    leaves: [[u8; 32]; 8],
    cur_block_id: [u8; 32],
    refs: Vec<[u8; 32]>,
}

/// Build a `SyntheticBlock` with deterministic leaves[0..7] and the supplied
/// refs list slotted into leaves[7] via Poseidon.
fn make_block(refs: Vec<[u8; 32]>, seed: u8) -> SyntheticBlock {
    // Arbitrary distinct content for leaves[0..7].
    let mut leaves: [[u8; 32]; 8] = std::array::from_fn(|i| {
        let mut a = [0u8; 32];
        for j in 0..32 {
            a[j] = seed.wrapping_add(i as u8).wrapping_mul(j as u8 | 1);
        }
        a
    });
    leaves[7] = compute_referenced_blocks_root_native(&refs);
    let cur_block_id = compute_block_id_native(&leaves);
    SyntheticBlock { leaves, cur_block_id, refs }
}

// ---------------------------------------------------------------------------
// Witness building & circuit driver
// ---------------------------------------------------------------------------

struct HopWitness {
    leaves: [[u8; 32]; 8],
    next_block_id: [u8; 32],
    is_parent: bool,
    l7_tree_proof: DenseTreeProof,
    num_active_levels: usize,
    /// Expected `cur_block_id` to compare against the circuit output.
    expected_cur_block_id: [u8; 32],
}

/// Build the canonical (honest) `HopWitness` for the given block and ref index.
fn build_witness(block: &SyntheticBlock, ref_index: usize) -> HopWitness {
    let leaf_hash = compute_referenced_block_leaf_hash_native(ref_index, &block.refs[ref_index]);
    let (siblings, real_depth) =
        compute_referenced_blocks_path_native(&block.refs, ref_index);
    let l7_tree_proof =
        preprocess_dense_proof_padded(leaf_hash, &siblings, ref_index, L7_TREE_DEPTH);

    HopWitness {
        leaves: block.leaves,
        next_block_id: block.refs[ref_index],
        is_parent: ref_index == 0,
        l7_tree_proof,
        num_active_levels: real_depth,
        expected_cur_block_id: block.cur_block_id,
    }
}

/// Build a `BaseCircuitBuilder` populated with one `HopChip::verify_hop` call
/// on `w`, configure params, return `(builder, public_instances)`.
fn build_hop_circuit(w: &HopWitness) -> (BaseCircuitBuilder<Fr>, Vec<Vec<Fr>>) {
    let mut builder = BaseCircuitBuilder::new(false)
        .use_k(K)
        .use_lookup_bits(K - 1)
        .use_instance_columns(1);

    let range = builder.range_chip();
    let sha256 = Sha256Chip::new(&range);

    let mut public_block_id: Vec<AssignedValue<Fr>> = Vec::with_capacity(32);
    let mut public_next_block_id: Vec<AssignedValue<Fr>> = Vec::with_capacity(32);

    {
        let ctx: &mut Context<Fr> = builder.main(0);

        // Initialize the Poseidon hasher.
        let spec = OptimizedPoseidonSpec::<Fr, T, RATE>::new::<8, 57, 0>();
        let mut poseidon = Halo2PoseidonHasher::<Fr, T, RATE>::new(spec);
        poseidon.initialize_consts(ctx, range.gate());

        let hop = HopChip::new(&range, &sha256, &poseidon);

        // Assign leaves[0..8] as 8 × 32 byte witnesses.
        let leaves_assigned: [[AssignedValue<Fr>; 32]; 8] = std::array::from_fn(|i| {
            let row: [AssignedValue<Fr>; 32] =
                w.leaves[i].iter().map(|b| ctx.load_witness(Fr::from(*b as u64)))
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();
            row
        });

        // Assign next_block_id as 32 byte witnesses.
        let next_block_id_assigned: [AssignedValue<Fr>; 32] = w
            .next_block_id
            .iter()
            .map(|b| ctx.load_witness(Fr::from(*b as u64)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let is_parent_bit =
            ctx.load_witness(if w.is_parent { Fr::one() } else { Fr::zero() });
        let num_active_levels =
            ctx.load_witness(Fr::from(w.num_active_levels as u64));

        let HopOutput { cur_block_id } = hop.verify_hop(
            ctx,
            &leaves_assigned,
            &next_block_id_assigned,
            is_parent_bit,
            &w.l7_tree_proof,
            num_active_levels,
        );

        public_block_id.extend_from_slice(&cur_block_id);
        public_next_block_id.extend_from_slice(&next_block_id_assigned);
    }

    // Expose cur_block_id (32 bytes) and next_block_id (32 bytes) as instances.
    for v in public_block_id.iter().chain(public_next_block_id.iter()) {
        builder.assigned_instances[0].push(*v);
    }
    builder.calculate_params(Some(20));

    // Expected public input values: expected_cur_block_id (BE), then next_block_id.
    let mut instance: Vec<Fr> = Vec::with_capacity(64);
    instance.extend(w.expected_cur_block_id.iter().map(|b| Fr::from(*b as u64)));
    instance.extend(w.next_block_id.iter().map(|b| Fr::from(*b as u64)));
    (builder, vec![instance])
}

fn run_mock_prover_ok(w: &HopWitness) {
    let (builder, instances) = build_hop_circuit(w);
    let prover = MockProver::<Fr>::run(K as u32, &builder, instances).unwrap();
    prover.assert_satisfied();
}

fn run_mock_prover_should_fail(w: &HopWitness) {
    let (builder, instances) = build_hop_circuit(w);
    let result = MockProver::<Fr>::run(K as u32, &builder, instances);
    match result {
        Err(_) => { /* synthesis error counts as a fail */ }
        Ok(prover) => {
            assert!(
                prover.verify().is_err(),
                "expected MockProver verification to fail, but it succeeded"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Happy-path tests
// ---------------------------------------------------------------------------

#[test]
fn hop_three_refs_to_parent_at_index_0() {
    let refs = vec![[0x10u8; 32], [0x11u8; 32], [0x12u8; 32]];
    let block = make_block(refs, 0xA0);
    let w = build_witness(&block, 0);
    assert!(w.is_parent);
    assert_eq!(w.num_active_levels, 2); // 3 refs → next_pow2 = 4 → depth 2
    run_mock_prover_ok(&w);
}

#[test]
fn hop_three_refs_to_ref_at_index_2() {
    let refs = vec![[0x20u8; 32], [0x21u8; 32], [0x22u8; 32]];
    let block = make_block(refs, 0xB0);
    let w = build_witness(&block, 2);
    assert!(!w.is_parent);
    assert_eq!(w.num_active_levels, 2);
    run_mock_prover_ok(&w);
}

#[test]
fn hop_single_ref_depth_zero() {
    // With a single ref the L7 tree is degenerate: root == leaf, so the
    // dense-tree path has zero Merkle combining steps. The chip must accept
    // num_active_levels=0 as a valid value.
    let refs = vec![[0x30u8; 32]];
    let block = make_block(refs, 0xC0);
    let w = build_witness(&block, 0);
    assert!(w.is_parent);
    assert_eq!(w.num_active_levels, 0);
    run_mock_prover_ok(&w);
}

#[test]
fn hop_two_refs_to_ref_at_index_1() {
    let refs = vec![[0x40u8; 32], [0x41u8; 32]];
    let block = make_block(refs, 0xD0);
    let w = build_witness(&block, 1);
    assert!(!w.is_parent);
    assert_eq!(w.num_active_levels, 1);
    run_mock_prover_ok(&w);
}

#[test]
fn hop_max_depth_256_refs() {
    // Exercise the full L7 depth (depth = 8, width = 256).
    let refs: Vec<[u8; 32]> = (0..256u32)
        .map(|i| {
            let mut a = [0u8; 32];
            a[0] = (i & 0xFF) as u8;
            a[1] = ((i >> 8) & 0xFF) as u8;
            a
        })
        .collect();
    let block = make_block(refs, 0xE0);
    // Hop to a non-parent position deep in the tree.
    let w = build_witness(&block, 173);
    assert!(!w.is_parent);
    assert_eq!(w.num_active_levels, L7_TREE_DEPTH);
    run_mock_prover_ok(&w);
}

// ---------------------------------------------------------------------------
// Negative (tampering) tests
// ---------------------------------------------------------------------------

#[test]
fn hop_negative_tampered_next_block_id() {
    let refs = vec![[0x50u8; 32], [0x51u8; 32], [0x52u8; 32]];
    let block = make_block(refs, 0xF0);
    let mut w = build_witness(&block, 1);
    // Flip one byte of next_block_id. The dense-tree leaf hash will no longer
    // match the actual ref leaf and the root verification will diverge.
    w.next_block_id[5] ^= 0x01;
    run_mock_prover_should_fail(&w);
}

#[test]
fn hop_negative_tampered_l7_leaf_in_block() {
    let refs = vec![[0x60u8; 32], [0x61u8; 32]];
    let mut block = make_block(refs, 0xF1);
    let mut w = build_witness(&block, 1);
    // Tamper leaves[7]: this breaks bytes_to_fr(leaves[7]) == L7 root equation.
    block.leaves[7][0] ^= 0xFF;
    w.leaves[7][0] ^= 0xFF;
    // Update expected cur_block_id so the public instances reflect the tamper
    // (otherwise the test fails for the trivial reason of a public-input
    // mismatch rather than the constraint we want to exercise).
    w.expected_cur_block_id = compute_block_id_native(&block.leaves);
    run_mock_prover_should_fail(&w);
}

#[test]
fn hop_negative_lie_parent_at_non_zero_position() {
    // Honest hop at index 2 (ref tag). Claim is_parent=1 (parent tag) → leaf
    // hash diverges → dense-tree root will not match leaves[7].
    let refs = vec![[0x70u8; 32], [0x71u8; 32], [0x72u8; 32]];
    let block = make_block(refs, 0xF2);
    let mut w = build_witness(&block, 2);
    w.is_parent = true;
    run_mock_prover_should_fail(&w);
}

#[test]
fn hop_negative_wrong_num_active_levels() {
    // Real depth is 2 (refs.len()=3 → next_pow2=4). Send depth=3 instead.
    let refs = vec![[0x80u8; 32], [0x81u8; 32], [0x82u8; 32]];
    let block = make_block(refs, 0xF3);
    let mut w = build_witness(&block, 0);
    w.num_active_levels = 3;
    run_mock_prover_should_fail(&w);
}

#[test]
fn hop_negative_out_of_range_num_active_levels() {
    let refs: Vec<[u8; 32]> = (0..8u32)
        .map(|i| {
            let mut a = [0u8; 32];
            a[0] = i as u8;
            a
        })
        .collect();
    let block = make_block(refs, 0xF5);
    let mut w = build_witness(&block, 0);
    // 9 > 8 (max L7 depth) — must be rejected by depth_minus_one range check.
    w.num_active_levels = 9;
    run_mock_prover_should_fail(&w);
}

// ---------------------------------------------------------------------------
// Sanity: public-input mismatch is caught
// ---------------------------------------------------------------------------

#[test]
fn hop_negative_wrong_public_block_id() {
    let refs = vec![[0xA0u8; 32], [0xA1u8; 32]];
    let block = make_block(refs, 0xF6);
    let mut w = build_witness(&block, 0);
    // Lie about cur_block_id in the public instance.
    w.expected_cur_block_id[10] ^= 0x42;
    run_mock_prover_should_fail(&w);
}
