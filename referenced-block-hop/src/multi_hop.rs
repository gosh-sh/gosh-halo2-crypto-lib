//! `MultiHopChip` — chain `H` L7 hops into a single proof and expose salted
//! endpoint commitments.
//!
//! ## Role in the multi-thread design
//!
//! One claim consists of `1 DexFinalProof + N_BUNDLE * MultiHopProof` snarks
//! (default `N_BUNDLE = 4`). The MultiHopProofs walk the L7 referenced-block
//! chain from the dex-thread event block toward an anchor in
//! `GlobalHistoricalData`. Each MultiHopProof handles `H` hops; the orchestrator
//! contract (`RootPN.sol`) chains adjacent MultiHopProofs by checking
//! `head_salted_block_id` of proof `k+1` equals `tail_salted_block_id` of proof
//! `k`, and all snarks share the same `salt_commitment`.
//!
//! Block ids in the chain are **never exposed in cleartext** — only their salted
//! Poseidon images. The salt is derived from the voucher secret `sk_u`, so an
//! attacker who steals one MultiHopProof cannot link it to a specific voucher
//! event without also breaking the salt commitment scheme.
//!
//! ## Variable active hop count
//!
//! `num_active_hops ∈ [0, H]` is a private witness. `active[i] = (i < num_active_hops)`
//! is derived as a 0/1 mask. The chain continuity constraint
//! `hop_i.cur_block_id == hop_{i-1}.next_block_id` is enforced **only when
//! `active[i] == 1`** (byte-by-byte). The `tail_block_id` exposed is the
//! `next_block_id` of the last active hop (or the head when no hop is active).
//!
//! `num_active_hops == 0` is a valid degenerate case: it expresses a no-op
//! MultiHopProof where head == tail. The orchestrator uses these to pad the
//! tail of the bundle when the real chain is shorter than `N_BUNDLE * H`.
//!
//! Inactive hops still run the full HopChip constraints (8-leaf SHA-256 tree,
//! L7 dense-tree opening). The prover supplies arbitrary valid synthetic
//! witness data for them. The chain continuity is the **only** constraint
//! gated by `active[i]`.
//!
//! ## Public outputs (3 Fr)
//!
//! 1. `salt_commitment = Poseidon([salt])`
//! 2. `head_salted_block_id = Poseidon([salt, fr(hop_0.cur_block_id)])`
//! 3. `tail_salted_block_id = Poseidon([salt, fr(tail_block_id)])`
//!
//! where `salt = Poseidon([DOMAIN_TAG_HOP_SALT_FR, sk_u])` and
//! `fr(bytes)` is `pack_bytes_le_to_fr` applied to the byte string in LE order
//! (caller reverses BE SHA-256 outputs to LE before packing — same convention
//! as `leaves[7]` in `HopChip`).
//!
//! See `MULTITHREAD_CIRCUIT_SPEC.md` §6 for the bundle composition.

use gosh_dense_balanced_tree::{bytes_to_fr, poseidon_hash_native, DenseTreeProof};
use gosh_sha256_chip::Sha256Chip;
use halo2_base::{
    gates::{GateInstructions, RangeChip, RangeInstructions},
    halo2_proofs::halo2curves::bn256::Fr,
    poseidon::hasher::PoseidonHasher,
    AssignedValue, Context, QuantumCell,
};

use crate::{
    circuit::{HopChip, HopOutput},
    BLOCK_ID_LEAF_COUNT, DOMAIN_TAG_HOP_SALT_BYTES,
};

/// Poseidon parameters — must match the on-chain
/// `tvm_vm::executor::zk_stuff::bn254::poseidon::PoseidonSponge`.
const T: usize = 3;
const RATE: usize = 2;

// ---------------------------------------------------------------------------
// Native helpers
// ---------------------------------------------------------------------------

/// Derive `DOMAIN_TAG_HOP_SALT_FR` from the ASCII tag bytes by zero-padding to
/// 32 LE-bytes and interpreting as Fr. The tag is shorter than 31 bytes so this
/// is collision-free with respect to other tags of the same encoding family.
pub fn domain_tag_hop_salt_fr() -> Fr {
    assert!(
        DOMAIN_TAG_HOP_SALT_BYTES.len() <= 31,
        "DOMAIN_TAG_HOP_SALT_BYTES must fit in 31 LE bytes (one Fr) to keep \
         the in-circuit constant a single field element"
    );
    let mut buf = [0u8; 32];
    buf[..DOMAIN_TAG_HOP_SALT_BYTES.len()].copy_from_slice(DOMAIN_TAG_HOP_SALT_BYTES);
    bytes_to_fr(&buf)
}

/// Native: `salt = Poseidon([DOMAIN_TAG_HOP_SALT_FR, sk_u])`.
pub fn compute_salt_native(sk_u: Fr) -> Fr {
    poseidon_hash_native(&[domain_tag_hop_salt_fr(), sk_u])
}

/// Native: `salt_commitment = Poseidon([salt])`.
pub fn compute_salt_commitment_native(salt: Fr) -> Fr {
    poseidon_hash_native(&[salt])
}

/// Native: `salted_block_id = Poseidon([salt, bytes_to_fr(block_id_le)])`.
///
/// `block_id_le` is the **little-endian** byte representation of the block id.
/// SHA-256 outputs are big-endian by convention, so the caller must reverse
/// before invoking this if working with raw SHA-256 outputs. The reason is
/// consistency with the existing `leaves[7]` convention in `HopChip` (where
/// the L7 root is supplied LE because it comes from `fr_to_bytes`).
pub fn compute_salted_block_id_native(salt: Fr, block_id_le: &[u8; 32]) -> Fr {
    poseidon_hash_native(&[salt, bytes_to_fr(block_id_le)])
}

// ---------------------------------------------------------------------------
// Circuit
// ---------------------------------------------------------------------------

/// One hop's full set of private witnesses, assigned to in-circuit cells.
///
/// Constructed by the caller and passed to [`MultiHopChip::verify_multi_hop`].
pub struct HopWitnessAssigned<'a> {
    /// 8 × 32 SHA-256 outer Merkle leaves of this hop's `B_cur`. `leaves[7]`
    /// must be supplied in **LE** (matches `HopChip` convention).
    pub leaves: &'a [[AssignedValue<Fr>; 32]; BLOCK_ID_LEAF_COUNT],
    /// 32 BE bytes of the `next_block_id` referenced by this hop.
    pub next_block_id: &'a [AssignedValue<Fr>; 32],
    /// 1 iff the referenced block is the parent (`index == 0`) in the L7 tree.
    pub is_parent: AssignedValue<Fr>,
    /// Dense-Merkle proof padded to depth 8 via
    /// `gosh_dense_balanced_tree::preprocess_dense_proof_padded`.
    pub l7_tree_proof: &'a DenseTreeProof,
    /// Real depth of the L7 tree in `[0, 8]`.
    pub num_active_levels: AssignedValue<Fr>,
}

/// Output of [`MultiHopChip::verify_multi_hop`].
pub struct MultiHopOutput {
    /// `Poseidon([salt])`. Bound across all snarks of one bundle.
    pub salt_commitment: AssignedValue<Fr>,
    /// `Poseidon([salt, fr(hop_0.cur_block_id_le)])`.
    pub head_salted_block_id: AssignedValue<Fr>,
    /// `Poseidon([salt, fr(tail_block_id_le)])` where `tail_block_id` is
    /// `hop_{num_active_hops - 1}.next_block_id` (or `head_block_id` when
    /// `num_active_hops == 0`).
    pub tail_salted_block_id: AssignedValue<Fr>,
}

/// In-circuit primitive that chains `H` hops and exposes the salted endpoint
/// commitments. Reuses `HopChip` for each hop; the chain continuity and salt
/// logic live here.
pub struct MultiHopChip<'a> {
    hop: HopChip<'a>,
    range: &'a RangeChip<Fr>,
    poseidon: &'a PoseidonHasher<Fr, T, RATE>,
}

impl<'a> MultiHopChip<'a> {
    /// `poseidon` must already be initialized via `initialize_consts(ctx, gate)`.
    pub fn new(
        range: &'a RangeChip<Fr>,
        sha256: &'a Sha256Chip<'a, Fr>,
        poseidon: &'a PoseidonHasher<Fr, T, RATE>,
    ) -> Self {
        Self { hop: HopChip::new(range, sha256, poseidon), range, poseidon }
    }

    /// Enforce `H` chained hops with a variable active count.
    ///
    /// `sk_u` is the voucher secret (1 Fr witness).
    /// `hops` has length exactly `H`; the chip enforces continuity for the
    /// first `num_active_hops` of them.
    /// `num_active_hops` is a witness in `[0, H]`; range-checked here.
    ///
    /// Returns the three public salted commitments. The caller is responsible
    /// for exposing them as instance values.
    pub fn verify_multi_hop(
        &self,
        ctx: &mut Context<Fr>,
        sk_u: AssignedValue<Fr>,
        hops: &[HopWitnessAssigned<'_>],
        num_active_hops: AssignedValue<Fr>,
    ) -> MultiHopOutput {
        let h = hops.len();
        assert!(h >= 1, "MultiHopChip requires at least one hop slot");
        let gate = self.range.gate();

        // -- (a) num_active_hops range-check ∈ [0, H] ----------------------------
        // 4-bit lookup covers [0, 15]; assert < (H+1) to narrow.
        // Restrict H to ≤ 15 so the 4-bit witness suffices. Real prod targets
        // H = MAX_HOPS = 5, well within range.
        assert!(h <= 15, "MultiHopChip current implementation supports H ≤ 15");
        self.range.range_check(ctx, num_active_hops, 4);
        let h_plus_one = ctx.load_constant(Fr::from((h + 1) as u64));
        let in_range = self.range.is_less_than(ctx, num_active_hops, h_plus_one, 4);
        let one = ctx.load_constant(Fr::one());
        ctx.constrain_equal(&in_range, &one);

        // -- (b) Run each HopChip and collect cur_block_id outputs ---------------
        let mut hop_outputs: Vec<HopOutput> = Vec::with_capacity(h);
        for w in hops.iter() {
            let out = self.hop.verify_hop(
                ctx,
                w.leaves,
                w.next_block_id,
                w.is_parent,
                w.l7_tree_proof,
                w.num_active_levels,
            );
            hop_outputs.push(out);
        }

        // -- (c) Derive active[i] = (i < num_active_hops) -----------------------
        // active[i] is computed in-circuit as `is_less_than(constant_i, num_active_hops)`.
        // All values fit in 4 bits.
        let mut active: Vec<AssignedValue<Fr>> = Vec::with_capacity(h);
        for i in 0..h {
            let i_const = ctx.load_constant(Fr::from(i as u64));
            let a = self.range.is_less_than(ctx, i_const, num_active_hops, 4);
            active.push(a);
        }

        // -- (d) Chain continuity (conditional, byte-by-byte) -------------------
        //   For i in 1..H:
        //     active[i] * (hop_i.cur_block_id[b] - hop_{i-1}.next_block_id[b]) == 0
        //   for each byte b ∈ 0..32.
        //
        // Implemented as `gate.mul` + `ctx.constrain_equal(result, zero)`.
        let zero = ctx.load_constant(Fr::zero());
        for i in 1..h {
            for b in 0..32 {
                let cur_b = hop_outputs[i].cur_block_id[b];
                let prev_next_b = hops[i - 1].next_block_id[b];
                // diff = cur_b - prev_next_b; both are byte witnesses.
                let diff = gate.sub(
                    ctx,
                    QuantumCell::Existing(cur_b),
                    QuantumCell::Existing(prev_next_b),
                );
                let masked = gate.mul(ctx, active[i], diff);
                ctx.constrain_equal(&masked, &zero);
            }
        }

        // -- (e) Compute salt and salt_commitment -------------------------------
        let domain_tag = ctx.load_constant(domain_tag_hop_salt_fr());
        let salt = self.poseidon.hash_fix_len_array(ctx, gate, &[domain_tag, sk_u]);
        let salt_commitment = self.poseidon.hash_fix_len_array(ctx, gate, &[salt]);

        // -- (f) head_salted_block_id from hop_0.cur_block_id -------------------
        // cur_block_id is 32 BE bytes (SHA-256 output). Pack as LE for
        // consistency with `leaves[7]` and `bytes_to_fr` conventions.
        // (BE/LE choice is consistent across head and tail, so the on-chain
        // verifier sees one canonical encoding regardless of which end it is.)
        let head_fr = pack_bytes_le_to_fr(ctx, gate, &hop_outputs[0].cur_block_id);
        let head_salted_block_id = self.poseidon.hash_fix_len_array(ctx, gate, &[salt, head_fr]);

        // -- (g) tail_block_id via running-update selector ----------------------
        //   tail = head_block_id
        //   for i in 0..H:
        //     tail = active[i] ? hop_i.next_block_id : tail   (byte-by-byte)
        // After this loop, `tail` is the next_block_id of the LAST active hop
        // (or hop_0.cur_block_id when no hop is active).
        let mut tail: [AssignedValue<Fr>; 32] = hop_outputs[0].cur_block_id;
        for i in 0..h {
            let mut next_tail = [tail[0]; 32]; // placeholder, overwritten below
            for b in 0..32 {
                next_tail[b] = gate.select(
                    ctx,
                    hops[i].next_block_id[b],
                    tail[b],
                    active[i],
                );
            }
            tail = next_tail;
        }
        let tail_fr = pack_bytes_le_to_fr(ctx, gate, &tail);
        let tail_salted_block_id = self.poseidon.hash_fix_len_array(ctx, gate, &[salt, tail_fr]);

        MultiHopOutput { salt_commitment, head_salted_block_id, tail_salted_block_id }
    }
}

/// Pack 32 LE bytes into a single Fr: `Σ_i bytes[i] * 256^i`. Mirrors the
/// (private) `pack_bytes_le_to_fr` in `circuit.rs`. Duplicated here to avoid
/// changing the public API of `circuit.rs`.
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

// ---------------------------------------------------------------------------
// Tests for native helpers
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn domain_tag_constant_is_stable() {
        // Sanity: derivation is deterministic and the byte tag fits in 31 LE
        // bytes (so the entire tag lives in one Fr without crossing the
        // 31-byte chunking boundary of poseidon_hash_bytes_flat).
        let a = domain_tag_hop_salt_fr();
        let b = domain_tag_hop_salt_fr();
        assert_eq!(a, b);
        assert_eq!(DOMAIN_TAG_HOP_SALT_BYTES, b"acki-nacki:voucher-hop-salt:v1");
        assert!(DOMAIN_TAG_HOP_SALT_BYTES.len() <= 31);
    }

    #[test]
    fn salt_chain_is_deterministic_and_distinct_per_sk_u() {
        let s1 = compute_salt_native(Fr::from(42u64));
        let s2 = compute_salt_native(Fr::from(42u64));
        let s3 = compute_salt_native(Fr::from(43u64));
        assert_eq!(s1, s2);
        assert_ne!(s1, s3, "different sk_u must yield different salt");

        let c1 = compute_salt_commitment_native(s1);
        let c2 = compute_salt_commitment_native(s1);
        let c3 = compute_salt_commitment_native(s3);
        assert_eq!(c1, c2);
        assert_ne!(c1, c3, "different salt must yield different salt_commitment");
    }

    #[test]
    fn salted_block_id_depends_on_both_salt_and_block_id() {
        let salt_a = compute_salt_native(Fr::from(1u64));
        let salt_b = compute_salt_native(Fr::from(2u64));
        let id_x = [0xAAu8; 32];
        let id_y = [0xBBu8; 32];

        let s_ax = compute_salted_block_id_native(salt_a, &id_x);
        let s_ay = compute_salted_block_id_native(salt_a, &id_y);
        let s_bx = compute_salted_block_id_native(salt_b, &id_x);

        assert_ne!(s_ax, s_ay, "same salt + different block_id must differ");
        assert_ne!(s_ax, s_bx, "same block_id + different salt must differ");
    }

    #[test]
    fn salt_commitment_does_not_leak_sk_u_directly() {
        // Smoke test: salt_commitment != sk_u, != salt. (Not a security proof —
        // just guards against an accidental no-op Poseidon path.)
        let sk_u = Fr::from(0x1234_5678u64);
        let salt = compute_salt_native(sk_u);
        let comm = compute_salt_commitment_native(salt);
        assert_ne!(comm, sk_u);
        assert_ne!(comm, salt);
    }
}
