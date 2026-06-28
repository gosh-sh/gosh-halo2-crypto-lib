//! MockProver integration tests for `MultiHopChip::verify_multi_hop`.
//!
//! Test strategy: build a chain of synthetic blocks where each block's L7
//! refs list contains the previous block's id, so `verify_hop` cleanly walks
//! from `B_0` to `B_K`. Adjacent hops are linked via byte-by-byte equality of
//! `hop_i.cur_block_id == hop_{i-1}.next_block_id`.
//!
//! Tests use `H = 2` for speed (each hop costs ~7 in-circuit SHA-256 calls,
//! so larger H makes CI slow). One slower test exercises `H = 3` to confirm
//! the chain logic scales.

use std::convert::TryInto;

use gosh_dense_balanced_tree::{
    bytes_to_fr, fr_to_bytes, preprocess_dense_proof_padded, DenseTreeProof,
};
use gosh_referenced_block_hop::{
    multi_hop::{
        compute_salt_commitment_native, compute_salt_native, compute_salted_block_id_native,
        HopWitnessAssigned, MultiHopChip, MultiHopOutput,
    },
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
const K: usize = 17; // Larger circuit than single-hop (5 hops worth of SHA-256).

// ---------------------------------------------------------------------------
// Synthetic chain construction
// ---------------------------------------------------------------------------

/// A synthetic block in the chain: its 8 outer SHA-256 leaves, its block_id,
/// and the refs list whose Poseidon root sits at leaves[7].
#[derive(Clone)]
struct ChainBlock {
    leaves: [[u8; 32]; 8],
    block_id_be: [u8; 32],
    refs: Vec<[u8; 32]>,
}

/// Build a chain of `n_blocks` blocks where:
///   block[0] is the deepest (oldest).
///   block[k] (k > 0) has block[k-1].block_id as parent (refs[0]) and one
///   arbitrary ref at refs[1].
///
/// Returns blocks in chain order (oldest → newest). Hopping forward means
/// starting from `block[n-1]` (the event block) and walking to `block[0]`.
fn build_chain(n_blocks: usize, seed: u8) -> Vec<ChainBlock> {
    assert!(n_blocks >= 1);
    let mut chain: Vec<ChainBlock> = Vec::with_capacity(n_blocks);

    for k in 0..n_blocks {
        // For the oldest block, parent is arbitrary. For others, parent = prev block's id.
        let parent_id = if k == 0 {
            let mut p = [0u8; 32];
            for j in 0..32 {
                p[j] = seed.wrapping_add(0x55).wrapping_mul(j as u8 | 1);
            }
            p
        } else {
            chain[k - 1].block_id_be
        };

        // One extra ref for non-trivial L7 width.
        let extra_ref = {
            let mut r = [0u8; 32];
            for j in 0..32 {
                r[j] = seed.wrapping_add(k as u8 + 0x33).wrapping_mul((j as u8) | 1);
            }
            r
        };
        let refs = vec![parent_id, extra_ref];
        let l7_root_le = compute_referenced_blocks_root_native(&refs); // returns LE bytes

        // leaves[0..7]: arbitrary distinct content. leaves[7]: LE Poseidon root.
        let mut leaves: [[u8; 32]; 8] = std::array::from_fn(|i| {
            let mut a = [0u8; 32];
            for j in 0..32 {
                a[j] = seed.wrapping_add(k as u8).wrapping_add(i as u8).wrapping_mul(j as u8 | 1);
            }
            a
        });
        leaves[7] = l7_root_le;

        let block_id_be = compute_block_id_native(&leaves);
        chain.push(ChainBlock { leaves, block_id_be, refs });
    }
    chain
}

/// Build the per-hop witness for "hop from `block` to its parent (`refs[0]`)".
struct PerHopWitness {
    leaves: [[u8; 32]; 8],
    next_block_id_be: [u8; 32],
    is_parent: bool,
    l7_tree_proof: DenseTreeProof,
    num_active_levels: usize,
}

fn hop_to_parent(block: &ChainBlock) -> PerHopWitness {
    let ref_index = 0; // parent slot
    let leaf_hash = compute_referenced_block_leaf_hash_native(ref_index, &block.refs[ref_index]);
    let (siblings, real_depth) =
        compute_referenced_blocks_path_native(&block.refs, ref_index);
    let l7_tree_proof =
        preprocess_dense_proof_padded(leaf_hash, &siblings, ref_index, L7_TREE_DEPTH);
    PerHopWitness {
        leaves: block.leaves,
        next_block_id_be: block.refs[ref_index],
        is_parent: ref_index == 0,
        l7_tree_proof,
        num_active_levels: real_depth,
    }
}

// ---------------------------------------------------------------------------
// Circuit driver
// ---------------------------------------------------------------------------

/// Top-level test witness for a MultiHopProof.
struct MultiHopTestWitness {
    sk_u: Fr,
    hops: Vec<PerHopWitness>,
    num_active_hops: usize,
    /// Expected `hop_0.cur_block_id` (32 BE bytes — the head, oldest block in chain order).
    expected_head_block_id_be: [u8; 32],
    /// Expected tail block_id (32 BE bytes).
    expected_tail_block_id_be: [u8; 32],
}

/// Build a canonical (honest) MultiHopTestWitness over a chain of `h_total`
/// blocks, configured with `num_active_hops` active and `padding_hops` dummy
/// hops appended (also valid hops from a separate chain, but with continuity
/// gated by active[]).
fn build_witness(num_active_hops: usize, padding_hops: usize, sk_u: Fr) -> MultiHopTestWitness {
    let h = num_active_hops + padding_hops;
    assert!(h >= 1);

    // Active chain: newest block at index num_active_hops-1, oldest at 0.
    // (Hopping order = walking from newest to oldest = index num_active_hops-1 down to 0.)
    // For the multi-hop chip, hop[0] is the FIRST hop = the newest block walking to its parent.
    let active_chain = build_chain(num_active_hops + 1, 0xA0);

    let mut hops: Vec<PerHopWitness> = Vec::with_capacity(h);
    // hop[0] = walk from block[num_active_hops] (newest) → block[num_active_hops - 1] (its parent)
    // hop[1] = walk from block[num_active_hops - 1] → block[num_active_hops - 2]
    // ...
    // hop[num_active_hops - 1] = walk from block[1] → block[0] (oldest).
    for i in 0..num_active_hops {
        let block_idx = num_active_hops - i;
        hops.push(hop_to_parent(&active_chain[block_idx]));
    }

    // Padding hops: a *separate* synthetic chain that won't connect to the
    // active one. Each padding hop is a perfectly valid HopChip witness — the
    // chain continuity is gated by active[] and skipped here.
    if padding_hops > 0 {
        let padding_chain = build_chain(padding_hops + 1, 0xF0);
        for i in 0..padding_hops {
            let block_idx = padding_hops - i;
            hops.push(hop_to_parent(&padding_chain[block_idx]));
        }
    }

    let (expected_head_block_id_be, expected_tail_block_id_be) = if num_active_hops == 0 {
        // Degenerate: no active hops. tail == head == hop[0].cur_block_id (the prover
        // could choose anything; we use the first padding hop's block).
        let padding_chain = build_chain(padding_hops + 1, 0xF0);
        let id = padding_chain[padding_hops].block_id_be;
        (id, id)
    } else {
        let head = active_chain[num_active_hops].block_id_be;
        let tail = active_chain[0].block_id_be;
        (head, tail)
    };

    MultiHopTestWitness {
        sk_u,
        hops,
        num_active_hops,
        expected_head_block_id_be,
        expected_tail_block_id_be,
    }
}

/// Helper to reverse BE → LE (or vice versa).
fn reverse(bytes: &[u8; 32]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = bytes[31 - i];
    }
    out
}

/// Build a BaseCircuitBuilder populated with one MultiHopChip::verify_multi_hop call.
fn build_multi_hop_circuit(
    w: &MultiHopTestWitness,
) -> (BaseCircuitBuilder<Fr>, Vec<Vec<Fr>>) {
    let mut builder = BaseCircuitBuilder::new(false)
        .use_k(K)
        .use_lookup_bits(K - 1)
        .use_instance_columns(1);

    let range = builder.range_chip();
    let sha256 = Sha256Chip::new(&range);

    let public_salt_commitment: AssignedValue<Fr>;
    let public_head_salted: AssignedValue<Fr>;
    let public_tail_salted: AssignedValue<Fr>;

    {
        let ctx: &mut Context<Fr> = builder.main(0);

        let spec = OptimizedPoseidonSpec::<Fr, T, RATE>::new::<8, 57, 0>();
        let mut poseidon = Halo2PoseidonHasher::<Fr, T, RATE>::new(spec);
        poseidon.initialize_consts(ctx, range.gate());

        let multi_hop = MultiHopChip::new(&range, &sha256, &poseidon);

        // Assign sk_u as a witness.
        let sk_u_assigned = ctx.load_witness(w.sk_u);

        // Assign per-hop witnesses. We allocate all 8-byte arrays first into
        // owned storage so we can produce stable references for HopWitnessAssigned.
        // Storage uses Vec to keep elements alive for the duration of the function.
        let mut leaves_storage: Vec<[[AssignedValue<Fr>; 32]; 8]> =
            Vec::with_capacity(w.hops.len());
        let mut next_block_id_storage: Vec<[AssignedValue<Fr>; 32]> =
            Vec::with_capacity(w.hops.len());
        let mut is_parent_storage: Vec<AssignedValue<Fr>> = Vec::with_capacity(w.hops.len());
        let mut num_levels_storage: Vec<AssignedValue<Fr>> = Vec::with_capacity(w.hops.len());

        for hop in w.hops.iter() {
            let leaves_assigned: [[AssignedValue<Fr>; 32]; 8] = std::array::from_fn(|i| {
                let row: [AssignedValue<Fr>; 32] = hop.leaves[i]
                    .iter()
                    .map(|b| ctx.load_witness(Fr::from(*b as u64)))
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();
                row
            });
            let next_assigned: [AssignedValue<Fr>; 32] = hop
                .next_block_id_be
                .iter()
                .map(|b| ctx.load_witness(Fr::from(*b as u64)))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            let is_parent = ctx.load_witness(if hop.is_parent { Fr::one() } else { Fr::zero() });
            let num_levels = ctx.load_witness(Fr::from(hop.num_active_levels as u64));

            leaves_storage.push(leaves_assigned);
            next_block_id_storage.push(next_assigned);
            is_parent_storage.push(is_parent);
            num_levels_storage.push(num_levels);
        }

        // Build HopWitnessAssigned slice borrowing from storage.
        let hops_assigned: Vec<HopWitnessAssigned> = (0..w.hops.len())
            .map(|i| HopWitnessAssigned {
                leaves: &leaves_storage[i],
                next_block_id: &next_block_id_storage[i],
                is_parent: is_parent_storage[i],
                l7_tree_proof: &w.hops[i].l7_tree_proof,
                num_active_levels: num_levels_storage[i],
            })
            .collect();

        let num_active_assigned = ctx.load_witness(Fr::from(w.num_active_hops as u64));

        let MultiHopOutput {
            salt_commitment,
            head_salted_block_id,
            tail_salted_block_id,
        } = multi_hop.verify_multi_hop(ctx, sk_u_assigned, &hops_assigned, num_active_assigned);

        public_salt_commitment = salt_commitment;
        public_head_salted = head_salted_block_id;
        public_tail_salted = tail_salted_block_id;
    }

    builder.assigned_instances[0].push(public_salt_commitment);
    builder.assigned_instances[0].push(public_head_salted);
    builder.assigned_instances[0].push(public_tail_salted);
    builder.calculate_params(Some(20));

    // Expected public inputs.
    let salt = compute_salt_native(w.sk_u);
    let salt_commitment = compute_salt_commitment_native(salt);
    // Reverse BE → LE for the salted-id Poseidon (matches circuit's pack_bytes_le_to_fr
    // on cur_block_id, which is composed of BE bytes that the circuit packs in
    // index-0-first order; in our test the bytes match positionally with `compute_block_id_native`
    // output. The circuit packs index 0 as LSB; we mirror by packing the BE byte string
    // directly without reversal — since the digest_bytes output IS in the same byte order
    // as compute_block_id_native, byte[i] in the witness equals byte[i] of block_id_be,
    // which gets packed with 256^i. So fr = Σ block_id_be[i] * 256^i.)
    // → That is exactly `bytes_to_fr(&block_id_be)` because bytes_to_fr also does LE = byte[i] * 256^i.
    let head_salted = compute_salted_block_id_native(salt, &w.expected_head_block_id_be);
    let tail_salted = compute_salted_block_id_native(salt, &w.expected_tail_block_id_be);

    let instance = vec![salt_commitment, head_salted, tail_salted];
    let _ = reverse; // silence unused-import warning in case we don't reverse
    let _ = bytes_to_fr; // silence unused-import warning
    let _ = fr_to_bytes; // silence
    (builder, vec![instance])
}

fn run_mock_prover_ok(w: &MultiHopTestWitness) {
    let (builder, instances) = build_multi_hop_circuit(w);
    let prover = MockProver::<Fr>::run(K as u32, &builder, instances).unwrap();
    prover.assert_satisfied();
}

fn run_mock_prover_should_fail(w: &MultiHopTestWitness) {
    let (builder, instances) = build_multi_hop_circuit(w);
    let result = MockProver::<Fr>::run(K as u32, &builder, instances);
    match result {
        Err(_) => { /* synthesis error counts */ }
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
fn multi_hop_h2_both_active() {
    // H = 2, both hops on the active chain.
    let w = build_witness(/*num_active_hops*/ 2, /*padding*/ 0, Fr::from(0x1234u64));
    run_mock_prover_ok(&w);
}

#[test]
fn multi_hop_h2_one_active_one_padding() {
    // H = 2, only first hop carries chain continuity. Second is disconnected.
    let w = build_witness(1, 1, Fr::from(0xBEEFu64));
    run_mock_prover_ok(&w);
}

#[test]
fn multi_hop_h2_zero_active_pure_passthrough() {
    // H = 2, num_active = 0. tail == head == hop_0.cur_block_id.
    let w = build_witness(0, 2, Fr::from(0xCAFEu64));
    run_mock_prover_ok(&w);
}

#[test]
fn multi_hop_h3_all_active() {
    // Larger H to exercise the chain logic across 3 hops.
    let w = build_witness(3, 0, Fr::from(0xDEADu64));
    run_mock_prover_ok(&w);
}

// ---------------------------------------------------------------------------
// Negative tests
// ---------------------------------------------------------------------------

#[test]
fn multi_hop_breaks_when_chain_link_is_lied_about() {
    // H = 2, both active. We construct two unrelated chains and pretend they
    // connect by swapping in a wrong hop[1]. Chain continuity at i=1 must fail.
    let chain_a = build_chain(2, 0xA0);
    let chain_b = build_chain(2, 0xB0);

    let hop0 = hop_to_parent(&chain_a[1]); // chain_a's newest → its parent
    let hop1 = hop_to_parent(&chain_b[1]); // chain_b's newest → its parent (does NOT match hop0.next)

    let w = MultiHopTestWitness {
        sk_u: Fr::from(0x11u64),
        hops: vec![hop0, hop1],
        num_active_hops: 2,
        // Whatever we put here as expected, the circuit's chain-continuity
        // constraint at i=1 will fail because hop1.cur_block_id !=
        // hop0.next_block_id (refs[0]).
        expected_head_block_id_be: chain_a[1].block_id_be,
        expected_tail_block_id_be: chain_b[0].block_id_be,
    };
    run_mock_prover_should_fail(&w);
}

#[test]
fn multi_hop_breaks_when_num_active_out_of_range() {
    // H = 2, claim num_active = 3 (too large).
    let mut w = build_witness(2, 0, Fr::from(0x22u64));
    w.num_active_hops = 3;
    run_mock_prover_should_fail(&w);
}

#[test]
fn multi_hop_breaks_when_wrong_sk_u_in_public_input() {
    // Build a valid circuit with sk_u = X, but supply public salt_commitment
    // computed from a different sk_u = Y.
    let mut w = build_witness(2, 0, Fr::from(0x55u64));
    let (builder, _instances) = build_multi_hop_circuit(&w);

    // Override the public instance with a wrong salt_commitment.
    let wrong_salt = compute_salt_native(Fr::from(0x66u64));
    let wrong_comm = compute_salt_commitment_native(wrong_salt);
    let salt_real = compute_salt_native(w.sk_u);
    let head_real =
        compute_salted_block_id_native(salt_real, &w.expected_head_block_id_be);
    let tail_real =
        compute_salted_block_id_native(salt_real, &w.expected_tail_block_id_be);
    let instances = vec![vec![wrong_comm, head_real, tail_real]];

    let result = MockProver::<Fr>::run(K as u32, &builder, instances);
    match result {
        Err(_) => {}
        Ok(prover) => assert!(prover.verify().is_err()),
    }
    let _ = &mut w; // silence
}

#[test]
fn multi_hop_breaks_when_tail_public_input_tampered() {
    let w = build_witness(2, 0, Fr::from(0x77u64));
    let (builder, _instances) = build_multi_hop_circuit(&w);

    let salt = compute_salt_native(w.sk_u);
    let comm = compute_salt_commitment_native(salt);
    let head = compute_salted_block_id_native(salt, &w.expected_head_block_id_be);
    // Tamper: use head as tail.
    let instances = vec![vec![comm, head, head]];

    let result = MockProver::<Fr>::run(K as u32, &builder, instances);
    match result {
        Err(_) => {}
        Ok(prover) => assert!(prover.verify().is_err()),
    }
}
