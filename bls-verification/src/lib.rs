//! BLS12-381 signature verification circuit.
//!
//! In-circuit BLS attestation verification built on halo2-ecc, plus
//! off-circuit helper utilities in the [`helpers`] submodule.

use halo2_base::gates::flex_gate::threads::SinglePhaseCoreManager;
use halo2_base::gates::{GateInstructions, RangeChip, RangeInstructions};
use halo2_base::halo2_proofs::halo2curves::bls12_381::{G1Affine, G2Affine};
use halo2_base::utils::BigPrimeField;
use halo2_base::{AssignedValue, Context};
use halo2_ecc::bls12_381::bls_signature::BlsSignatureChip;
use halo2_ecc::bls12_381::pairing::PairingChip;
use halo2_ecc::bls12_381::{Fp2Chip, FpChip, G2Point};
use halo2_ecc::bigint::ProperCrtUint;
use halo2_ecc::ecc::{check_is_on_curve, EcPoint, EccChip};

// ---------------------------------------------------------------------------
// In-circuit BLS verification
// ---------------------------------------------------------------------------

/// Determines which signer threshold formula to enforce.
pub enum ThresholdMode {
    /// Primary finalization: `3 * n_signers >= 2 * n_pubkeys` (>= ceil(2n/3))
    Primary,
    /// Fallback finalization: `2 * n_signers > n_pubkeys` (> 50%)
    Fallback,
}

/// Load BK set pubkeys as assigned EC points and constrain each to lie on the BLS12-381 G1 curve.
///
/// Call this once to obtain `assigned_pks`, then pass them to one or more invocations of
/// [`verify_bls_attestation_with_assigned_msghash`] so that every verification provably
/// uses the same BK set.
pub fn load_bk_set_pubkeys<F: BigPrimeField>(
    ctx: &mut Context<F>,
    range: &RangeChip<F>,
    bk_set_pubkeys: &[G1Affine],
    limb_bits: usize,
    num_limbs: usize,
) -> Vec<EcPoint<F, ProperCrtUint<F>>> {
    let fp_chip = FpChip::<F>::new(range, limb_bits, num_limbs);
    let pairing_chip = PairingChip::new(&fp_chip);
    bk_set_pubkeys
        .iter()
        .map(|&pk| {
            let pt = pairing_chip.load_private_g1_unchecked(ctx, pk);
            check_is_on_curve::<F, FpChip<F>, G1Affine>(&fp_chip, ctx, &pt);
            pt
        })
        .collect()
}

/// Compute the sum of all BK set pubkeys.
///
/// This value is needed by [`verify_bls_attestation_with_assigned_msghash`] to correct
/// the shifted-scalar MSM.  When the same BK set is verified against multiple attestations
/// (e.g., fallback finalization), compute this once and pass it to each call to avoid
/// duplicating ~n_pubkeys EC additions.
pub fn compute_all_pub_sum<F: BigPrimeField>(
    ctx: &mut Context<F>,
    range: &RangeChip<F>,
    assigned_pks: &[EcPoint<F, ProperCrtUint<F>>],
    limb_bits: usize,
    num_limbs: usize,
) -> EcPoint<F, ProperCrtUint<F>> {
    let fp_chip = FpChip::<F>::new(range, limb_bits, num_limbs);
    let g1_chip = EccChip::new(&fp_chip);

    let mut all_pub_sum = assigned_pks[0].clone();
    for j in 1..assigned_pks.len() {
        // is_strict=true enforces x-coordinate inequality, preventing unsound
        // add_unequal results if two pubkeys coincide.
        all_pub_sum =
            g1_chip.add_unequal(ctx, all_pub_sum, assigned_pks[j].clone(), true);
    }
    all_pub_sum
}

/// Verify a BLS attestation signature in-circuit, using a pre-assigned `msg_hash` point
/// (e.g., from an in-circuit hash-to-curve computation) instead of loading it as a private witness.
///
/// Uses a **fixed-capacity** circuit with `max_signers` slots.  The actual number of
/// signers is loaded as a witness and constrained in-circuit, so the same verification
/// key works for any number of signers up to the capacity.
///
/// `assigned_pks` may be padded to `max_signers` for fixed circuit structure.
/// `actual_num_pubkeys` is the real BK set size, used for the threshold check and
/// signer index bounds.  The indicator/weights/MSM use `assigned_pks.len()` so that
/// padding pubkeys participate in the MSM (their zero weights + shift cancel via
/// the `all_pub_sum` correction).
///
/// Constrains:
/// - `n_signers` is in `[1, max_signers]`
/// - Threshold check determined by `threshold_mode`:
///   - Primary: `3 * n_signers >= 2 * actual_num_pubkeys`
///   - Fallback: `2 * n_signers > actual_num_pubkeys`
/// - Active slots (`i < n_signers`): index < actual_num_pubkeys, count >= 1, strictly increasing indices
/// - Inactive slots (`i >= n_signers`): index == 0, count == 0
/// - Aggregated pubkey is correctly composed from active-slot pubkeys × counts
/// - G2 signature lies on the BLS12-381 G2 curve
/// - BLS pairing check passes
///
/// Pubkeys must be pre-loaded via [`load_bk_set_pubkeys`], and `all_pub_sum` must be
/// pre-computed via [`compute_all_pub_sum`].
pub fn verify_bls_attestation_with_assigned_msghash<F: BigPrimeField>(
    pool: &mut SinglePhaseCoreManager<F>,
    range: &RangeChip<F>,
    signature: G2Affine,
    msghash_assigned: G2Point<F>,
    assigned_pks: &[EcPoint<F, ProperCrtUint<F>>],
    signers_data: &[(u16, u16)],
    max_signers: usize,
    limb_bits: usize,
    num_limbs: usize,
    threshold_mode: ThresholdMode,
    all_pub_sum: EcPoint<F, ProperCrtUint<F>>,
    actual_num_pubkeys: AssignedValue<F>,
) {
    let n_pubkeys = assigned_pks.len();
    let n_signers = signers_data.len();
    assert!(n_signers > 0, "must have at least one signer");
    assert!(n_signers <= max_signers, "n_signers exceeds max_signers");
    assert!(max_signers > 0, "max_signers must be positive");
    assert!(n_pubkeys > 0, "must have at least one pubkey in assigned_pks");

    // Bit-width sufficient to represent max_signers and n_pubkeys values.
    let idx_bits = (usize::BITS - max_signers.leading_zeros()).max(1) as usize;

    let fp_chip = FpChip::<F>::new(range, limb_bits, num_limbs);
    let pairing_chip = PairingChip::new(&fp_chip);
    let g1_chip = EccChip::new(&fp_chip);
    let gate = range.gate();

    // -----------------------------------------------------------------------
    // Phase 1: single-threaded work (load pubkeys, signer constraints, weights)
    // -----------------------------------------------------------------------
    let shifted_scalars = {
        let ctx = pool.main();

        // 1. Pad signers_data to max_signers slots (inactive slots get (0, 0)).
        let padded: Vec<(u16, u16)> = {
            let mut v = signers_data.to_vec();
            v.resize(max_signers, (0, 0));
            v
        };
        let idx_cells: Vec<AssignedValue<F>> = padded
            .iter()
            .map(|&(idx, _)| ctx.load_witness(F::from(idx as u64)))
            .collect();
        let count_cells: Vec<AssignedValue<F>> = padded
            .iter()
            .map(|&(_, count)| ctx.load_witness(F::from(count as u64)))
            .collect();

        // 3. Load n_signers as a witness and constrain 1 <= n_signers <= max_signers.
        let n_signers_cell = ctx.load_witness(F::from(n_signers as u64));
        let one = ctx.load_constant(F::from(1u64));
        let zero = ctx.load_constant(F::from(0u64));
        {
            let ns_minus_one = gate.sub(ctx, n_signers_cell, one);
            range.range_check(ctx, ns_minus_one, idx_bits);
            let max_s = ctx.load_constant(F::from(max_signers as u64));
            let slack = gate.sub(ctx, max_s, n_signers_cell);
            range.range_check(ctx, slack, idx_bits);
        }

        // 4. Constrain actual_num_pubkeys in [1, max_signers].
        let npk_minus_one = gate.sub(ctx, actual_num_pubkeys, one);
        range.range_check(ctx, npk_minus_one, idx_bits);
        {
            let max_s = ctx.load_constant(F::from(max_signers as u64));
            let npk_slack = gate.sub(ctx, max_s, actual_num_pubkeys);
            range.range_check(ctx, npk_slack, idx_bits);
        }

        // 5. Threshold check determined by threshold_mode (fixed at circuit build time).
        //
        // Primary: 3 * n_signers >= 2 * actual_num_pubkeys  (>= ceil(2n/3))
        //   Prove: (3 * n_signers - 2 * actual_num_pubkeys) >= 0  via range check.
        //
        // Fallback: 2 * n_signers > actual_num_pubkeys  (> 50%)
        //   Prove: (2 * n_signers - actual_num_pubkeys - 1) >= 0  via range check.
        {
            let check_val = match threshold_mode {
                ThresholdMode::Primary => {
                    let three = ctx.load_constant(F::from(3u64));
                    let two = ctx.load_constant(F::from(2u64));
                    let three_ns = gate.mul(ctx, three, n_signers_cell);
                    let two_npk = gate.mul(ctx, two, actual_num_pubkeys);
                    gate.sub(ctx, three_ns, two_npk)
                }
                ThresholdMode::Fallback => {
                    let two = ctx.load_constant(F::from(2u64));
                    let two_ns = gate.mul(ctx, two, n_signers_cell);
                    let diff = gate.sub(ctx, two_ns, actual_num_pubkeys);
                    gate.sub(ctx, diff, one)
                }
            };
            // max value: 3*300 = 900, fits in 16 bits.
            range.range_check(ctx, check_val, 16);
        }

        // 6. Compute is_active[i] = (i < n_signers_cell) for each slot.
        let is_active: Vec<AssignedValue<F>> = (0..max_signers)
            .map(|i| {
                let i_const = ctx.load_constant(F::from(i as u64));
                range.is_less_than(ctx, i_const, n_signers_cell, idx_bits + 1)
            })
            .collect();

        // 7. Constrain inactive slots: idx == 0 and count == 0.
        for i in 0..max_signers {
            let not_active = gate.sub(ctx, one, is_active[i]);
            let check_idx = gate.mul(ctx, not_active, idx_cells[i]);
            ctx.constrain_equal(&check_idx, &zero);
            let check_count = gate.mul(ctx, not_active, count_cells[i]);
            ctx.constrain_equal(&check_count, &zero);
        }

        // 8. Active-slot constraints.
        for i in 0..max_signers {
            // 7a. idx < actual_num_pubkeys (prevent referencing padding pubkeys).
            // Dynamic bound: constrain (actual_num_pubkeys - 1 - idx) fits in idx_bits.
            {
                let diff = gate.sub(ctx, npk_minus_one, idx_cells[i]);
                range.range_check(ctx, diff, idx_bits);
            }

            // 7b. Strict monotonicity for consecutive active slots.
            if i + 1 < max_signers {
                let both_active = gate.mul(ctx, is_active[i], is_active[i + 1]);
                let gap = gate.sub(ctx, idx_cells[i + 1], idx_cells[i]);
                let gap_minus_one = gate.sub(ctx, gap, one);
                let checked_val = gate.mul(ctx, both_active, gap_minus_one);
                range.range_check(ctx, checked_val, 16);
            }

            // 7c. Active slots must have count >= 1.
            let count_minus_one = gate.sub(ctx, count_cells[i], one);
            let active_count_check = gate.mul(ctx, is_active[i], count_minus_one);
            range.range_check(ctx, active_count_check, 16);
        }

        // 8a. Accumulate weights: weight[j] = sum_i indicator[i][j] * count[i].
        let mut weights: Vec<AssignedValue<F>> = (0..n_pubkeys).map(|_| zero).collect();
        for i in 0..max_signers {
            let indicator = gate.idx_to_indicator(ctx, idx_cells[i], n_pubkeys);
            for j in 0..n_pubkeys {
                let w_contrib = gate.mul(ctx, indicator[j], count_cells[i]);
                weights[j] = gate.add(ctx, weights[j], w_contrib);
            }
        }

        // 8b. Compute shifted scalars: shifted[j] = weight[j] + 1 for MSM safety.
        //     This guarantees all scalars >= 1, avoiding zero-scalar edge cases in MSM.
        //     We correct for the +1 shift by subtracting sum(pk[j]) after MSM.
        let shifted_scalars: Vec<Vec<AssignedValue<F>>> = weights
            .iter()
            .map(|&w| {
                let shifted = gate.add(ctx, w, one);
                vec![shifted]
            })
            .collect();

        shifted_scalars
    }; // ctx borrow ends here

    // -----------------------------------------------------------------------
    // Phase 2: MSM (requires &mut pool, not ctx)
    // -----------------------------------------------------------------------
    // MSM computes: sum(pk[j] * shifted[j]) = sum(pk[j] * (weight[j] + 1))
    //             = sum(pk[j] * weight[j]) + sum(pk[j])
    //             = agg_pk + all_pub_sum
    //
    // max_bits=17: weight can be up to 65536 = 2^16, shifted up to 65537 < 2^17.
    let msm_result =
        g1_chip.variable_base_msm::<G1Affine>(pool, assigned_pks, shifted_scalars, 17);

    // -----------------------------------------------------------------------
    // Phase 3: subtract correction, load signature, pairing check
    // -----------------------------------------------------------------------
    let ctx = pool.main();

    // 8d. Recover agg_pk = msm_result - all_pub_sum.
    let neg_all_pub_sum = g1_chip.negate(ctx, all_pub_sum);
    let acc = g1_chip.add_unequal(ctx, msm_result, neg_all_pub_sum, true);

    // 9. Load signature and verify BLS pairing check.
    let sig_assigned = pairing_chip.load_private_g2_unchecked(ctx, signature);
    // Constrain that the signature point lies on the G2 curve (y^2 = x^3 + b').
    let fp2_chip = Fp2Chip::new(&fp_chip);
    check_is_on_curve::<F, Fp2Chip<F>, G2Affine>(&fp2_chip, ctx, &sig_assigned);
    let bls_chip = BlsSignatureChip::new(&fp_chip, &pairing_chip);
    bls_chip.assert_valid_signature(ctx, sig_assigned, msghash_assigned, acc);
}

// ---------------------------------------------------------------------------
// Off-circuit BLS helpers
// ---------------------------------------------------------------------------

/// Off-circuit (native) BLS12-381 utilities.
///
/// Includes DST constant, point deserialization, hash-to-curve,
/// pubkey aggregation, and native pairing check.
pub mod helpers {
    use std::collections::HashMap;

    use halo2_base::halo2_proofs::halo2curves::bls12_381::{
        hash_to_curve::{ExpandMsgXmd, HashToCurve},
        multi_miller_loop, G1Affine, G2Affine, G2Prepared, Gt, G2,
    };
    use halo2_base::halo2_proofs::halo2curves::group::Curve;
    use sha2::Sha256;

    /// DST used by GoshBLS for BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_.
    /// Note: ends in `NUL_`, not `POP_`.
    pub const DST: &[u8] = b"BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_NUL_";

    /// Deserialize a 192-byte BLS signature into a G2Affine point.
    ///
    /// Tries big-endian uncompressed format first, then little-endian.
    pub fn deserialize_g2_signature(sig_bytes: &[u8]) -> G2Affine {
        assert_eq!(sig_bytes.len(), 192, "BLS signature must be 192 bytes");
        let bytes: [u8; 192] = sig_bytes.try_into().unwrap();

        let opt_be = G2Affine::from_uncompressed_be(&bytes);
        if bool::from(opt_be.is_some()) {
            return opt_be.unwrap();
        }

        let opt_le = G2Affine::from_uncompressed_le(&bytes);
        if bool::from(opt_le.is_some()) {
            return opt_le.unwrap();
        }

        panic!(
            "Failed to deserialize 192-byte BLS signature as G2Affine (tried BE and LE). \
             First 8 bytes: {:02x?}",
            &sig_bytes[..8]
        );
    }

    /// Deserialize a 48-byte compressed BLS public key into a G1Affine point.
    pub fn deserialize_g1_pubkey(pk_bytes: &[u8]) -> G1Affine {
        assert_eq!(pk_bytes.len(), 48, "BLS public key must be 48 bytes");
        let bytes: [u8; 48] = pk_bytes.try_into().unwrap();

        let opt_be = G1Affine::from_compressed_be(&bytes);
        if bool::from(opt_be.is_some()) {
            return opt_be.unwrap();
        }

        let opt_le = G1Affine::from_compressed_le(&bytes);
        if bool::from(opt_le.is_some()) {
            return opt_le.unwrap();
        }

        panic!(
            "Failed to deserialize 48-byte BLS pubkey as G1Affine. First 8 bytes: {:02x?}",
            &pk_bytes[..8]
        );
    }

    /// Compute the message hash via hash-to-curve (SHA-256 XMD, SSWU map to G2).
    pub fn compute_msg_hash(msg: &[u8]) -> G2Affine {
        <G2 as HashToCurve<ExpandMsgXmd<Sha256>>>::hash_to_curve(msg, DST).to_affine()
    }

    /// Resolve signer entries against a `bk_set` and return (pubkey, count) pairs.
    pub fn resolve_pubkeys(
        entries: &[(u16, u16)],
        bk_set: &HashMap<u16, Vec<u8>>,
    ) -> Vec<(G1Affine, u16)> {
        entries
            .iter()
            .map(|&(idx, count)| {
                let pk_bytes = bk_set
                    .get(&idx)
                    .unwrap_or_else(|| panic!("signer index {} not found in bk_set", idx));
                (deserialize_g1_pubkey(pk_bytes), count)
            })
            .collect()
    }

    /// Compute the aggregated public key: sum of (count * pubkey) for each entry.
    pub fn compute_agg_pubkey(pubkeys_with_counts: &[(G1Affine, u16)]) -> G1Affine {
        use halo2_base::halo2_proofs::halo2curves::bls12_381::G1;

        let mut agg = G1::identity();
        for &(pk, count) in pubkeys_with_counts {
            let pk_proj = G1::from(pk);
            for _ in 0..count {
                agg = agg + pk_proj;
            }
        }
        agg.to_affine()
    }

    /// Verify BLS signature off-circuit (native pairing check).
    ///
    /// Returns true iff `e(-G1, sig) * e(agg_pk, msg_hash) == 1`.
    pub fn verify_bls_native(
        signature: &G2Affine,
        agg_pubkey: &G1Affine,
        msg_hash: &G2Affine,
    ) -> bool {
        use std::ops::Neg;
        let g1_neg = G1Affine::generator().neg();
        let result = multi_miller_loop(&[
            (&g1_neg, &G2Prepared::from(*signature)),
            (agg_pubkey, &G2Prepared::from(*msg_hash)),
        ])
        .final_exponentiation();
        result == Gt::identity()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use helpers::{compute_msg_hash, verify_bls_native};
    use halo2_base::halo2_proofs::halo2curves::bls12_381::G1Affine;
    use halo2_base::utils::testing::base_test;

    // ---------------------------------------------------------------------------
    // Off-circuit tests using sk=1 trick:
    //   For sk=1: pk = G1::generator(), sig = H(msg)^sk = H(msg)
    //   BLS check: e(-G1, H(msg)) * e(G1, H(msg)) = 1  ✓
    // ---------------------------------------------------------------------------

    #[test]
    fn test_bls_offcircuit_sk1() {
        let msg = b"gosh bls test message";
        let msg_hash = compute_msg_hash(msg);
        // For sk=1: pk = G1 generator, sig = H(msg).
        let pk = G1Affine::generator();
        let signature = msg_hash;
        assert!(
            verify_bls_native(&signature, &pk, &msg_hash),
            "Off-circuit BLS (sk=1) should pass"
        );
    }

    #[test]
    fn test_bls_bad_signature_offcircuit() {
        let msg = b"gosh bls test message";
        let msg_hash = compute_msg_hash(msg);
        let bad_msg_hash = compute_msg_hash(b"corrupted message");
        let pk = G1Affine::generator();
        // Sign msg but verify against bad_msg_hash — should fail.
        let signature = msg_hash;
        assert!(
            !verify_bls_native(&signature, &pk, &bad_msg_hash),
            "Off-circuit BLS with wrong message should fail"
        );
    }

    // ---------------------------------------------------------------------------
    // In-circuit tests (ignored by default — require k=19 MockProver, ~minutes)
    // ---------------------------------------------------------------------------

    /// In-circuit BLS verification with sk=1 (1 pubkey = G1 generator, sig = H(msg)).
    #[test]
    #[ignore]
    fn test_bls_incircuit_sk1() {
        let msg = b"gosh bls test message";
        let msg_hash = compute_msg_hash(msg);
        let pk = G1Affine::generator();
        // For sk=1: sig = H(msg).
        let signature = msg_hash;

        let signers_data: &[(u16, u16)] = &[(0, 1)];
        let max_signers = 1;

        assert!(
            verify_bls_native(&signature, &pk, &msg_hash),
            "Off-circuit sanity check failed"
        );

        base_test().k(19).lookup_bits(18).run_builder(|pool, range| {
            let fp_chip = FpChip::<_>::new(range, 104, 5);
            let pairing_chip = PairingChip::new(&fp_chip);
            let ctx = pool.main();

            let msghash_assigned = pairing_chip.load_private_g2_unchecked(ctx, msg_hash);
            let assigned_pks = load_bk_set_pubkeys(ctx, range, &[pk], 104, 5);
            let all_pub_sum = compute_all_pub_sum(ctx, range, &assigned_pks, 104, 5);
            let actual_npk = ctx.load_witness(F::from(1u64));

            verify_bls_attestation_with_assigned_msghash(
                pool,
                range,
                signature,
                msghash_assigned,
                &assigned_pks,
                signers_data,
                max_signers,
                104,
                5,
                ThresholdMode::Primary,
                all_pub_sum,
                actual_npk,
            );
        });
        println!("In-circuit BLS (sk=1) MockProver passed!");
    }

    /// In-circuit BLS with wrong message hash — MockProver must reject it.
    #[test]
    #[ignore]
    #[should_panic]
    fn test_bls_bad_signature_incircuit() {
        let msg = b"gosh bls test message";
        let msg_hash = compute_msg_hash(msg);
        let bad_msg_hash = compute_msg_hash(b"corrupted message");
        let pk = G1Affine::generator();
        // Sign msg but pass bad_msg_hash to the circuit.
        let signature = msg_hash;

        let signers_data: &[(u16, u16)] = &[(0, 1)];
        let max_signers = 1;

        base_test().k(19).lookup_bits(18).run_builder(|pool, range| {
            let fp_chip = FpChip::<_>::new(range, 104, 5);
            let pairing_chip = PairingChip::new(&fp_chip);
            let ctx = pool.main();

            let msghash_assigned = pairing_chip.load_private_g2_unchecked(ctx, bad_msg_hash);
            let assigned_pks = load_bk_set_pubkeys(ctx, range, &[pk], 104, 5);
            let all_pub_sum = compute_all_pub_sum(ctx, range, &assigned_pks, 104, 5);
            let actual_npk = ctx.load_witness(F::from(1u64));

            verify_bls_attestation_with_assigned_msghash(
                pool,
                range,
                signature,
                msghash_assigned,
                &assigned_pks,
                signers_data,
                max_signers,
                104,
                5,
                ThresholdMode::Primary,
                all_pub_sum,
                actual_npk,
            );
        });
    }
}
