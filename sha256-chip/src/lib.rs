pub mod compression;

use compression::{sha256_compression, INIT_STATE, NUM_STATE_WORD};
use halo2_base::{
    gates::{
        flex_gate::threads::SinglePhaseCoreManager, GateInstructions, RangeChip, RangeInstructions,
    },
    halo2_proofs::plonk::Error,
    utils::BigPrimeField,
    AssignedValue, Context, QuantumCell,
};
use halo2_ecc::ecc::hash_to_curve::HashInstructions;
use sha2::{Digest, Sha256};

/// Native eDSL SHA-256 chip.
///
/// Performs SHA-256 entirely within the halo2-base eDSL (no vanilla custom gates,
/// no bridging). Uses bit decomposition for bitwise operations.
pub struct Sha256Chip<'a, F: BigPrimeField> {
    range: &'a RangeChip<F>,
}

impl<'a, F: BigPrimeField> Sha256Chip<'a, F> {
    pub fn new(range: &'a RangeChip<F>) -> Self {
        Self { range }
    }

    pub fn range(&self) -> &RangeChip<F> {
        self.range
    }

    /// Hash a byte array of known fixed length.
    ///
    /// `input_bytes` must already be assigned as field elements in [0, 255].
    /// Each input byte is range-checked to 8 bits.
    /// Returns 32 output bytes (big-endian) as assigned field elements.
    pub fn digest_bytes(
        &self,
        ctx: &mut Context<F>,
        input_bytes: &[AssignedValue<F>],
    ) -> Vec<AssignedValue<F>> {
        let input_len = input_bytes.len();

        // Range-check all input bytes.
        for &b in input_bytes {
            self.range.range_check(ctx, b, 8);
        }

        // Compute SHA-256 padding.
        let input_byte_size_with_9 = input_len + 9;
        let num_blocks = (input_byte_size_with_9 + 63) / 64;
        let padded_size = num_blocks * 64;
        let zero_padding_len = padded_size - input_byte_size_with_9;

        // Build padded input:
        // original_bytes || 0x80 || zeros || big-endian 64-bit bit length
        let mut padded: Vec<AssignedValue<F>> = input_bytes.to_vec();
        padded.push(ctx.load_constant(F::from(0x80u64)));
        for _ in 0..zero_padding_len {
            padded.push(ctx.load_constant(F::ZERO));
        }
        // Append 8-byte big-endian bit length.
        let bit_len = (input_len as u64) * 8;
        for i in (0..8).rev() {
            let byte_val = (bit_len >> (i * 8)) & 0xFF;
            padded.push(ctx.load_constant(F::from(byte_val)));
        }
        debug_assert_eq!(padded.len(), padded_size);

        // Initialize state with SHA-256 IV.
        let mut state: Vec<AssignedValue<F>> = INIT_STATE
            .iter()
            .map(|&h| ctx.load_constant(F::from(h as u64)))
            .collect();

        // Process each 64-byte block.
        for block in padded.chunks(64) {
            state = sha256_compression(ctx, self.range, block, &state);
        }

        // Decompose 8 state words into 32 big-endian bytes.
        self.state_to_bytes(ctx, &state)
    }

    /// Convert 8 u32 state words into 32 big-endian bytes with range checks
    /// and composition verification.
    pub fn state_to_bytes(
        &self,
        ctx: &mut Context<F>,
        state: &[AssignedValue<F>],
    ) -> Vec<AssignedValue<F>> {
        debug_assert_eq!(state.len(), NUM_STATE_WORD);
        let gate = self.range.gate();

        state
            .iter()
            .flat_map(|&word| {
                let raw = word.value().get_lower_32() as u64;
                let bytes: Vec<AssignedValue<F>> = (0..4)
                    .map(|i| {
                        let byte_val = ((raw >> (24 - 8 * i)) & 0xFF) as u64;
                        let b = ctx.load_witness(F::from(byte_val));
                        self.range.range_check(ctx, b, 8);
                        b
                    })
                    .collect();

                // Verify: word == bytes[0]*2^24 + bytes[1]*2^16 + bytes[2]*2^8 + bytes[3]
                let mut acc = bytes[0];
                for i in 1..4 {
                    acc = gate.mul_add(
                        ctx,
                        acc,
                        QuantumCell::Constant(F::from(256u64)),
                        bytes[i],
                    );
                }
                ctx.constrain_equal(&acc, &word);

                bytes
            })
            .collect()
    }
}

impl<'a, F: BigPrimeField> HashInstructions<F> for Sha256Chip<'a, F> {
    const BLOCK_SIZE: usize = 64;
    const DIGEST_SIZE: usize = 32;

    type CircuitBuilder = SinglePhaseCoreManager<F>;
    type Output = Vec<AssignedValue<F>>;

    fn digest(
        &self,
        thread_pool: &mut Self::CircuitBuilder,
        input: impl IntoIterator<Item = QuantumCell<F>>,
    ) -> Result<Vec<AssignedValue<F>>, Error> {
        let ctx = thread_pool.main();

        // Materialize input cells.
        let input_cells: Vec<AssignedValue<F>> = input
            .into_iter()
            .map(|q| match q {
                QuantumCell::Existing(v) => v,
                QuantumCell::Constant(f) => ctx.load_constant(f),
                QuantumCell::Witness(f) => ctx.load_witness(f),
                _ => unreachable!(),
            })
            .collect();

        // Compute SHA-256 natively to get the expected output (for witness generation).
        let input_bytes_native: Vec<u8> = input_cells
            .iter()
            .map(|av| av.value().get_lower_32() as u8)
            .collect();
        let expected_hash = Sha256::digest(&input_bytes_native);

        // Run in-circuit SHA-256.
        let result = self.digest_bytes(ctx, &input_cells);

        // Sanity check: verify witness values match native computation.
        #[cfg(debug_assertions)]
        for (i, &b) in result.iter().enumerate() {
            debug_assert_eq!(
                b.value().get_lower_32() as u8,
                expected_hash[i],
                "SHA256 witness mismatch at byte {}",
                i
            );
        }
        let _ = expected_hash;

        Ok(result)
    }

    fn digest_varlen(
        &self,
        _ctx: &mut Self::CircuitBuilder,
        _input: impl IntoIterator<Item = QuantumCell<F>>,
        _max_input_len: usize,
    ) -> Result<Self::Output, Error> {
        unimplemented!("digest_varlen not needed for fixed-length XMD inputs")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_base::halo2_proofs::halo2curves::bn256::Fr;
    use halo2_base::halo2_proofs::dev::MockProver;
    use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
    use halo2_base::gates::circuit::BaseCircuitParams;
    use halo2_base::utils::{testing::base_test, ScalarField};

    // ---------------------------------------------------------------------------
    // Positive tests — correct SHA-256 computation
    // ---------------------------------------------------------------------------

    #[test]
    fn test_sha256_empty() {
        let expected = hex::decode(
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        )
        .unwrap();

        base_test()
            .k(17)
            .lookup_bits(16)
            .run(|ctx: &mut Context<Fr>, range: &RangeChip<Fr>| {
                let chip = Sha256Chip::new(range);
                let result = chip.digest_bytes(ctx, &[]);
                for (i, &b) in result.iter().enumerate() {
                    assert_eq!(
                        b.value().get_lower_32() as u8,
                        expected[i],
                        "byte {} mismatch",
                        i
                    );
                }
            });
    }

    #[test]
    fn test_sha256_abc() {
        let input = b"abc";
        let expected = Sha256::digest(input);

        base_test()
            .k(17)
            .lookup_bits(16)
            .run(|ctx: &mut Context<Fr>, range: &RangeChip<Fr>| {
                let chip = Sha256Chip::new(range);
                let assigned_input: Vec<AssignedValue<Fr>> = input
                    .iter()
                    .map(|&b| ctx.load_witness(Fr::from(b as u64)))
                    .collect();
                let result = chip.digest_bytes(ctx, &assigned_input);
                for (i, &b) in result.iter().enumerate() {
                    assert_eq!(
                        b.value().get_lower_32() as u8,
                        expected[i],
                        "byte {} mismatch",
                        i
                    );
                }
            });
    }

    #[test]
    fn test_sha256_multiblock() {
        // 56 bytes of 0x01 requires 2 blocks after padding
        let input = vec![0x01u8; 56];
        let expected = Sha256::digest(&input);

        base_test()
            .k(17)
            .lookup_bits(16)
            .run(|ctx: &mut Context<Fr>, range: &RangeChip<Fr>| {
                let chip = Sha256Chip::new(range);
                let assigned_input: Vec<AssignedValue<Fr>> = input
                    .iter()
                    .map(|&b| ctx.load_witness(Fr::from(b as u64)))
                    .collect();
                let result = chip.digest_bytes(ctx, &assigned_input);
                for (i, &b) in result.iter().enumerate() {
                    assert_eq!(
                        b.value().get_lower_32() as u8,
                        expected[i],
                        "byte {} mismatch",
                        i
                    );
                }
            });
    }

    /// 55 bytes: maximum input that fits in a single 64-byte block after padding.
    /// Exercises the exact block-boundary edge case.
    #[test]
    fn test_sha256_55_byte_boundary() {
        let input = vec![0xABu8; 55];
        let expected = Sha256::digest(&input);

        base_test()
            .k(17)
            .lookup_bits(16)
            .run(|ctx: &mut Context<Fr>, range: &RangeChip<Fr>| {
                let chip = Sha256Chip::new(range);
                let assigned_input: Vec<AssignedValue<Fr>> = input
                    .iter()
                    .map(|&b| ctx.load_witness(Fr::from(b as u64)))
                    .collect();
                let result = chip.digest_bytes(ctx, &assigned_input);
                for (i, &b) in result.iter().enumerate() {
                    assert_eq!(
                        b.value().get_lower_32() as u8,
                        expected[i],
                        "byte {} mismatch",
                        i
                    );
                }
            });
    }

    /// 128 bytes: requires 3 blocks (128 + 1 + 0 + 8 = 137 → 192 bytes = 3 blocks).
    /// Exercises chaining across three compression calls and stresses mod_u32.
    #[test]
    fn test_sha256_three_blocks() {
        let input: Vec<u8> = (0..128).map(|i| (i & 0xFF) as u8).collect();
        let expected = Sha256::digest(&input);

        base_test()
            .k(18)
            .lookup_bits(17)
            .run(|ctx: &mut Context<Fr>, range: &RangeChip<Fr>| {
                let chip = Sha256Chip::new(range);
                let assigned_input: Vec<AssignedValue<Fr>> = input
                    .iter()
                    .map(|&b| ctx.load_witness(Fr::from(b as u64)))
                    .collect();
                let result = chip.digest_bytes(ctx, &assigned_input);
                for (i, &b) in result.iter().enumerate() {
                    assert_eq!(
                        b.value().get_lower_32() as u8,
                        expected[i],
                        "byte {} mismatch",
                        i
                    );
                }
            });
    }

    // ---------------------------------------------------------------------------
    // Negative tests — adversarial / invalid inputs
    // ---------------------------------------------------------------------------

    /// Loading an input byte with value 256 (> 0xFF) must fail the 8-bit range check.
    #[test]
    fn test_sha256_rejects_out_of_range_byte() {
        let k = 17u32;
        let lookup_bits = 16;

        let mut builder = BaseCircuitBuilder::<Fr>::new(false);
        builder.set_params(BaseCircuitParams {
            k: k as usize,
            num_instance_columns: 0,
            lookup_bits: Some(lookup_bits),
            ..Default::default()
        });
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let chip = Sha256Chip::new(&range);

        // Load two valid bytes and one out-of-range byte (256).
        let bad_byte = ctx.load_witness(Fr::from(256u64));
        let good_byte = ctx.load_witness(Fr::from(0x41u64));
        let input = vec![good_byte, bad_byte, good_byte];

        let _result = chip.digest_bytes(ctx, &input);

        let params = builder.calculate_params(Some(20));
        let mut builder2 = BaseCircuitBuilder::<Fr>::new(false);
        builder2.set_params(params);
        // Replay witness generation with the bad input in builder2.
        {
            let range2 = builder2.range_chip();
            let ctx2 = builder2.main(0);
            let chip2 = Sha256Chip::new(&range2);
            let bad_byte2 = ctx2.load_witness(Fr::from(256u64));
            let good_byte2 = ctx2.load_witness(Fr::from(0x41u64));
            let input2 = vec![good_byte2, bad_byte2, good_byte2];
            let _result2 = chip2.digest_bytes(ctx2, &input2);
        }

        let prover = MockProver::run(k, &builder2, vec![]).unwrap();
        // The 8-bit range check on the 256-valued byte should fail.
        assert!(
            prover.verify().is_err(),
            "MockProver should reject out-of-range input byte (256)"
        );
    }

    /// Constraining the SHA-256 output to a wrong value must be detected.
    #[test]
    fn test_sha256_rejects_wrong_hash_constraint() {
        let k = 17u32;
        let lookup_bits = 16;
        let input = b"abc";

        let mut builder = BaseCircuitBuilder::<Fr>::new(false);
        builder.set_params(BaseCircuitParams {
            k: k as usize,
            num_instance_columns: 0,
            lookup_bits: Some(lookup_bits),
            ..Default::default()
        });
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let chip = Sha256Chip::new(&range);

        let assigned_input: Vec<AssignedValue<Fr>> = input
            .iter()
            .map(|&b| ctx.load_witness(Fr::from(b as u64)))
            .collect();
        let hash = chip.digest_bytes(ctx, &assigned_input);

        // Constrain first hash byte to a wrong value (0x00 instead of actual 0xba).
        let wrong_byte = ctx.load_constant(Fr::from(0x00u64));
        ctx.constrain_equal(&hash[0], &wrong_byte);

        let params = builder.calculate_params(Some(20));
        // Rebuild with same bad constraint.
        let mut builder2 = BaseCircuitBuilder::<Fr>::new(false);
        builder2.set_params(params);
        {
            let range2 = builder2.range_chip();
            let ctx2 = builder2.main(0);
            let chip2 = Sha256Chip::new(&range2);
            let assigned_input2: Vec<AssignedValue<Fr>> = input
                .iter()
                .map(|&b| ctx2.load_witness(Fr::from(b as u64)))
                .collect();
            let hash2 = chip2.digest_bytes(ctx2, &assigned_input2);
            let wrong_byte2 = ctx2.load_constant(Fr::from(0x00u64));
            ctx2.constrain_equal(&hash2[0], &wrong_byte2);
        }

        let prover = MockProver::run(k, &builder2, vec![]).unwrap();
        assert!(
            prover.verify().is_err(),
            "MockProver should reject wrong hash constraint"
        );
    }
}
