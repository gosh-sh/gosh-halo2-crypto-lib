//! `HopChip` — one-hop L7 referenced-block circuit primitive.
//!
//! Given the 8 outer SHA-256 Merkle leaves of a block `B_cur`, a candidate
//! `next_block_id`, and an L7 Poseidon dense-Merkle path, this chip enforces:
//!
//!   (1) `cur_block_id == sha256_8leaf_root(leaves)` — reconstruct the
//!       block-id from `[L0..L7]` using 7 `sha256_pair` calls.
//!   (2) `leaf_fr = Poseidon(tag || next_block_id)`, where `tag` is the
//!       parent tag when `is_parent == 1` and the ref tag otherwise.
//!   (3) `dense_merkle_root_padded(leaf_fr, l7_path, num_active_levels)`
//!       equals `bytes_to_fr(leaves[7])` — i.e. the supplied leaf opens to
//!       the L7 Poseidon root that lives at slot 7 of the block-id tree.
//!
//! The caller is responsible for exposing `cur_block_id` and `next_block_id`
//! as public outputs (so a follow-on hop or the thread-0 anchor proof can pick
//! up `next_block_id` and prove inclusion of a further block).
//!
//! ## Why `is_parent` instead of `ref_index`
//!
//! Only the *tag boundary* (`index == 0` vs `index >= 1`) is observable: the
//! direction bits inside the dense-tree open are private witnesses already
//! constrained to be valid bits. A prover could "lie" about the position
//! within the non-zero region (e.g. open position 5 while claiming index 3) —
//! both produce the same ref tag, the leaf hash is identical for the same
//! `next_block_id`, and the dense tree still verifies. This is harmless
//! because the only public binding is `next_block_id == refs[some non-zero
//! position]`. The tag-confusion attack (claim parent at non-zero position)
//! IS prevented because the parent tag at any non-zero position produces a
//! leaf hash different from the actual ref-tagged leaf, so the dense-tree
//! verification fails. Hence `is_parent` (1 bit) is sufficient.
//!
//! See `MULTITHREAD_CIRCUIT_SPEC.md` §4 for the full design.

use gosh_dense_balanced_tree::{
    bytes_to_fr, dense_merkle_root_circuit_padded, DenseTreeProof,
};
use gosh_sha256_chip::Sha256Chip;
use halo2_base::{
    gates::{GateInstructions, RangeChip, RangeInstructions},
    halo2_proofs::halo2curves::bn256::Fr,
    poseidon::hasher::PoseidonHasher,
    AssignedValue, Context, QuantumCell,
};

use crate::{
    BLOCK_ID_LEAF_COUNT, L7_LEAF_INDEX, REFERENCED_PARENT_BLOCK_TAG, REFERENCED_REF_BLOCK_TAG,
};

/// Poseidon parameters used everywhere in this crate. Must match the on-chain
/// `tvm_vm::executor::zk_stuff::bn254::poseidon::PoseidonSponge` (`T=3, RATE=2,
/// R_F=8, R_P=57`). The parity test in `native.rs` locks these values down
/// against the acki-nacki `history-proof` crate.
const T: usize = 3;
const RATE: usize = 2;

/// Output of one L7 hop verification.
pub struct HopOutput {
    /// 32 big-endian bytes of `block_id(B_cur)` reconstructed from the 8 outer
    /// SHA-256 leaves. Caller typically exposes this as a public input.
    pub cur_block_id: [AssignedValue<Fr>; 32],
}

/// In-circuit L7 hop primitive. Borrows its dependencies; safe to instantiate
/// once per hop or reuse across many hops inside the same circuit.
pub struct HopChip<'a> {
    range: &'a RangeChip<Fr>,
    sha256: &'a Sha256Chip<'a, Fr>,
    poseidon: &'a PoseidonHasher<Fr, T, RATE>,
}

impl<'a> HopChip<'a> {
    /// `poseidon` must already be initialized via `initialize_consts(ctx, gate)`
    /// before any call to [`verify_hop`].
    pub fn new(
        range: &'a RangeChip<Fr>,
        sha256: &'a Sha256Chip<'a, Fr>,
        poseidon: &'a PoseidonHasher<Fr, T, RATE>,
    ) -> Self {
        Self { range, sha256, poseidon }
    }

    /// Enforce one L7 hop. See the module-level doc for the constraint set.
    ///
    /// `leaves`: 8 × 32 assigned bytes, one row per outer Merkle leaf. The
    /// caller is responsible for binding these to whatever input feed they
    /// come from (e.g. a private witness in a standalone test, or copy
    /// constraints from a previous-hop output in a multi-hop circuit).
    ///
    /// `next_block_id`: 32 assigned bytes. Each byte is range-checked to 8
    /// bits inside this call.
    ///
    /// `is_parent`: 1 iff the referenced block sits at position 0 in the L7
    /// tree (parent slot), else 0. Bit-validity is asserted here.
    ///
    /// `l7_tree_proof`: preprocessed L7 dense-Merkle path padded to depth 8
    /// (use `gosh_dense_balanced_tree::preprocess_dense_proof_padded(.., 8)`).
    ///
    /// `num_active_levels`: real depth of the L7 tree in `[0, 8]`. Depth 0
    /// is the legitimate single-ref case: with one ref, the L7 root equals the
    /// (Poseidon-hashed) parent leaf directly — no Merkle combining steps.
    /// Range-checked here as `num_active_levels ∈ [0, 8]` via a 4-bit lookup
    /// plus a `< 9` assertion.
    pub fn verify_hop(
        &self,
        ctx: &mut Context<Fr>,
        leaves: &[[AssignedValue<Fr>; 32]; BLOCK_ID_LEAF_COUNT],
        next_block_id: &[AssignedValue<Fr>; 32],
        is_parent: AssignedValue<Fr>,
        l7_tree_proof: &DenseTreeProof,
        num_active_levels: AssignedValue<Fr>,
    ) -> HopOutput {
        let gate = self.range.gate();

        // -- (a) bit/range validity ------------------------------------------------
        gate.assert_bit(ctx, is_parent);
        // num_active_levels ∈ [0, 8]: 4-bit lookup gives [0, 15], plus the
        // `< 9` assertion narrows it to [0, 8].
        self.range.range_check(ctx, num_active_levels, 4);
        let nine = ctx.load_constant(Fr::from(9u64));
        let in_range = self.range.is_less_than(ctx, num_active_levels, nine, 4);
        let one = ctx.load_constant(Fr::one());
        ctx.constrain_equal(&in_range, &one);

        // next_block_id bytes — range to 8 bits. (Leaves bytes are checked
        // implicitly when sha256.digest_bytes consumes them.)
        for b in next_block_id.iter() {
            self.range.range_check(ctx, *b, 8);
        }

        // -- (b) cur_block_id via 8-leaf SHA-256 tree ------------------------------
        let cur_block_id = self.compute_block_id_from_leaves(ctx, leaves);

        // -- (c) L7 leaf Fr = Poseidon(tag || next_block_id) -----------------------
        let leaf_fr = self.compute_l7_leaf_fr(ctx, next_block_id, is_parent);

        // -- (d) Dense-tree open; equate with bytes_to_fr(leaves[7]) ---------------
        let l7_computed = dense_merkle_root_circuit_padded(
            ctx,
            self.range,
            self.poseidon,
            l7_tree_proof,
            leaf_fr,
            num_active_levels,
        );
        let l7_expected = pack_bytes_le_to_fr(ctx, gate, &leaves[L7_LEAF_INDEX]);
        ctx.constrain_equal(&l7_computed, &l7_expected);

        HopOutput { cur_block_id }
    }

    /// 8-leaf SHA-256 Merkle root: 4 + 2 + 1 = 7 `sha256_pair` calls.
    fn compute_block_id_from_leaves(
        &self,
        ctx: &mut Context<Fr>,
        leaves: &[[AssignedValue<Fr>; 32]; BLOCK_ID_LEAF_COUNT],
    ) -> [AssignedValue<Fr>; 32] {
        let mut level: Vec<Vec<AssignedValue<Fr>>> =
            leaves.iter().map(|l| l.to_vec()).collect();
        while level.len() > 1 {
            let mut next = Vec::with_capacity(level.len() / 2);
            for pair in level.chunks(2) {
                let mut input = Vec::with_capacity(64);
                input.extend_from_slice(&pair[0]);
                input.extend_from_slice(&pair[1]);
                next.push(self.sha256.digest_bytes(ctx, &input));
            }
            level = next;
        }
        let root_vec = level.into_iter().next().expect("non-empty levels");
        root_vec.try_into().expect("sha256 root is 32 bytes")
    }

    /// Compute `Poseidon(31-byte chunks of (tag || next_block_id))` where
    /// `tag = parent_tag if is_parent else ref_tag`.
    ///
    /// `parent_tag` is 37 bytes → preimage = 69 bytes → 3 chunks of sizes
    /// 31, 31, 7.  `ref_tag` is 34 bytes → preimage = 66 bytes → 3 chunks of
    /// sizes 31, 31, 4. Both produce 3 chunks, so the Poseidon shape is
    /// identical and one `gate.select` per chunk picks the right Fr value.
    ///
    /// chunk0 is pure tag bytes → two constants, selected.
    /// chunk1 has a constant tag tail and a 25-byte (parent) / 28-byte (ref)
    ///   prefix of `next_block_id` → linear combinations, selected.
    /// chunk2 has only `next_block_id` bytes (7 / 4 of them) → linear
    ///   combinations, selected.
    fn compute_l7_leaf_fr(
        &self,
        ctx: &mut Context<Fr>,
        next_block_id: &[AssignedValue<Fr>; 32],
        is_parent: AssignedValue<Fr>,
    ) -> AssignedValue<Fr> {
        let gate = self.range.gate();

        debug_assert_eq!(REFERENCED_PARENT_BLOCK_TAG.len(), 37);
        debug_assert_eq!(REFERENCED_REF_BLOCK_TAG.len(), 34);

        let two56 = Fr::from(256u64);

        // ---- chunk0: bytes [0..31] of preimage = tag[0..31] (constants only) ----
        let mut c0p_bytes = [0u8; 32];
        c0p_bytes[..31].copy_from_slice(&REFERENCED_PARENT_BLOCK_TAG[..31]);
        let c0_parent = ctx.load_constant(bytes_to_fr(&c0p_bytes));

        let mut c0r_bytes = [0u8; 32];
        c0r_bytes[..31].copy_from_slice(&REFERENCED_REF_BLOCK_TAG[..31]);
        let c0_ref = ctx.load_constant(bytes_to_fr(&c0r_bytes));

        let c0 = gate.select(ctx, c0_parent, c0_ref, is_parent);

        // ---- chunk1: bytes [31..62] of preimage --------------------------------
        // Parent (tag len 37): tag bytes [31..37] (6 const bytes) at positions
        //   0..6, then next_block_id[0..25] at positions 6..31.
        let parent_const_part_c1 = {
            let mut acc = Fr::zero();
            let mut pow = Fr::one();
            for i in 0..6 {
                acc += Fr::from(REFERENCED_PARENT_BLOCK_TAG[31 + i] as u64) * pow;
                pow *= two56;
            }
            acc
        };
        // pow at this point is 256^6 — the coefficient for next_block_id[0].
        let c1_parent = lc_const_plus_witness_bytes(
            ctx,
            gate,
            parent_const_part_c1,
            Fr::from_raw([1u64 << 48, 0, 0, 0]), // 256^6 = 2^48
            &next_block_id[0..25],
        );

        // Ref (tag len 34): tag bytes [31..34] (3 const bytes) at positions
        //   0..3, then next_block_id[0..28] at positions 3..31.
        let ref_const_part_c1 = {
            let mut acc = Fr::zero();
            let mut pow = Fr::one();
            for i in 0..3 {
                acc += Fr::from(REFERENCED_REF_BLOCK_TAG[31 + i] as u64) * pow;
                pow *= two56;
            }
            acc
        };
        // pow at this point is 256^3 — the coefficient for next_block_id[0].
        let c1_ref = lc_const_plus_witness_bytes(
            ctx,
            gate,
            ref_const_part_c1,
            Fr::from(1u64 << 24), // 256^3 = 2^24
            &next_block_id[0..28],
        );

        let c1 = gate.select(ctx, c1_parent, c1_ref, is_parent);

        // ---- chunk2: bytes [62..preimage_end] zero-padded ----------------------
        // Parent: next_block_id[25..32] (7 witness bytes at positions 0..7).
        let c2_parent = lc_const_plus_witness_bytes(
            ctx,
            gate,
            Fr::zero(),
            Fr::one(),
            &next_block_id[25..32],
        );
        // Ref: next_block_id[28..32] (4 witness bytes at positions 0..4).
        let c2_ref = lc_const_plus_witness_bytes(
            ctx,
            gate,
            Fr::zero(),
            Fr::one(),
            &next_block_id[28..32],
        );

        let c2 = gate.select(ctx, c2_parent, c2_ref, is_parent);

        // ---- Poseidon over [c0, c1, c2] ----------------------------------------
        self.poseidon.hash_fix_len_array(ctx, gate, &[c0, c1, c2])
    }
}

/// Linear combination `const_part + Σ_{j} witnesses[j] * (start_coeff * 256^j)`
/// using a single `gate.inner_product` call.
///
/// Used to assemble the variable parts of the L7-leaf Poseidon chunks where
/// some bytes are constant tag bytes and the rest are witness `next_block_id`
/// bytes.
fn lc_const_plus_witness_bytes(
    ctx: &mut Context<Fr>,
    gate: &impl GateInstructions<Fr>,
    const_part: Fr,
    start_coeff: Fr,
    witnesses: &[AssignedValue<Fr>],
) -> AssignedValue<Fr> {
    let two56 = Fr::from(256u64);
    let mut a_vec: Vec<QuantumCell<Fr>> = Vec::with_capacity(witnesses.len() + 1);
    let mut b_vec: Vec<QuantumCell<Fr>> = Vec::with_capacity(witnesses.len() + 1);

    // Constant term: 1 * const_part.
    a_vec.push(QuantumCell::Constant(Fr::one()));
    b_vec.push(QuantumCell::Constant(const_part));

    let mut coeff = start_coeff;
    for w in witnesses.iter() {
        a_vec.push(QuantumCell::Existing(*w));
        b_vec.push(QuantumCell::Constant(coeff));
        coeff *= two56;
    }
    gate.inner_product(ctx, a_vec, b_vec)
}

/// Pack 32 LE bytes into a single `Fr` field element via the constraint
/// `Σ_i bytes[i] * 256^i == Fr` (mod p). Used to equate `leaves[7]` (a SHA-256
/// output, exposed as 32 BE bytes) with the Fr value computed by the dense
/// tree opening.
///
/// Note: the byte ordering passed in must be little-endian. Since the existing
/// `gosh_dense_balanced_tree::bytes_to_fr` and `fr_to_bytes` use LE, and
/// on-chain `fr_to_bytes` returns the canonical LE encoding, the caller must
/// reverse `leaves[7]` (which is BE per SHA-256 convention) before feeding it
/// here — OR equivalently, the `leaves[7]` bytes can be supplied in LE form by
/// the witness builder if the prover prefers to reverse off-chip.
fn pack_bytes_le_to_fr(
    ctx: &mut Context<Fr>,
    gate: &impl GateInstructions<Fr>,
    bytes: &[AssignedValue<Fr>; 32],
) -> AssignedValue<Fr> {
    let two56 = Fr::from(256u64);
    let mut a_vec: Vec<QuantumCell<Fr>> = Vec::with_capacity(32);
    let mut b_vec: Vec<QuantumCell<Fr>> = Vec::with_capacity(32);
    let mut coeff = Fr::one();
    for b in bytes.iter() {
        a_vec.push(QuantumCell::Existing(*b));
        b_vec.push(QuantumCell::Constant(coeff));
        coeff *= two56;
    }
    gate.inner_product(ctx, a_vec, b_vec)
}
