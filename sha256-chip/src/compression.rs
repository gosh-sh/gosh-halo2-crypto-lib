use halo2_base::{
    gates::{GateInstructions, RangeChip, RangeInstructions},
    utils::BigPrimeField,
    AssignedValue, Context, QuantumCell,
};

pub const NUM_ROUND: usize = 64;
pub const NUM_STATE_WORD: usize = 8;

pub const ROUND_CONSTANTS: [u32; NUM_ROUND] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
    0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
    0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
    0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
    0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
    0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
    0xc67178f2,
];

pub const INIT_STATE: [u32; NUM_STATE_WORD] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];

// ---------------------------------------------------------------------------
// Bit manipulation primitives
// ---------------------------------------------------------------------------

/// Decompose a 32-bit field element into 32 individual bit cells (LSB first).
/// Each bit is range-checked to 1 bit, and the composition is verified.
fn u32_to_bits<F: BigPrimeField>(
    ctx: &mut Context<F>,
    range: &RangeChip<F>,
    val: AssignedValue<F>,
) -> Vec<AssignedValue<F>> {
    let gate = range.gate();
    let raw_val = val.value().get_lower_32() as u64;

    // Load 32 bit witnesses
    let bits: Vec<AssignedValue<F>> = (0..32)
        .map(|i| {
            let bit = ((raw_val >> i) & 1) as u64;
            let b = ctx.load_witness(F::from(bit));
            gate.assert_bit(ctx, b);
            b
        })
        .collect();

    // Verify composition: val == sum(bits[i] * 2^i)
    let composed = bits_to_u32(ctx, gate, &bits);
    ctx.constrain_equal(&composed, &val);

    bits
}

/// Compose 32 bit cells (LSB first) into a single field element.
fn bits_to_u32<F: BigPrimeField>(
    ctx: &mut Context<F>,
    gate: &impl GateInstructions<F>,
    bits: &[AssignedValue<F>],
) -> AssignedValue<F> {
    debug_assert_eq!(bits.len(), 32);
    // Use inner_product: sum of bits[i] * 2^i
    let powers: Vec<QuantumCell<F>> = (0..32)
        .map(|i| QuantumCell::Constant(F::from(1u64 << i)))
        .collect();
    gate.inner_product(
        ctx,
        bits.iter().copied(),
        powers,
    )
}

/// Rotate right: bits[n..] ++ bits[..n] (just index reordering, zero cost).
fn rotr<F: BigPrimeField>(bits: &[AssignedValue<F>], n: usize) -> Vec<AssignedValue<F>> {
    debug_assert_eq!(bits.len(), 32);
    (0..32).map(|i| bits[(i + n) % 32]).collect()
}

/// Shift right: bits[n..] ++ zeros.
fn shr<F: BigPrimeField>(
    ctx: &mut Context<F>,
    bits: &[AssignedValue<F>],
    n: usize,
) -> Vec<AssignedValue<F>> {
    debug_assert_eq!(bits.len(), 32);
    let zero = ctx.load_zero();
    (0..32)
        .map(|i| if i + n < 32 { bits[i + n] } else { zero })
        .collect()
}

/// Bitwise XOR of two 32-bit values represented as bits.
/// xor(a, b) = a + b - 2*a*b for each bit.
/// 3 gate ops per bit (mul, add, mul_add).
#[allow(dead_code)]
fn xor_bits<F: BigPrimeField>(
    ctx: &mut Context<F>,
    gate: &impl GateInstructions<F>,
    a: &[AssignedValue<F>],
    b: &[AssignedValue<F>],
) -> Vec<AssignedValue<F>> {
    debug_assert_eq!(a.len(), 32);
    debug_assert_eq!(b.len(), 32);
    let neg_two = F::from(2).neg();
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| {
            // a + b - 2*a*b = (a*b)*(-2) + (a + b)
            let ab = gate.mul(ctx, ai, bi);
            let sum = gate.add(ctx, ai, bi);
            gate.mul_add(ctx, ab, QuantumCell::Constant(neg_two), sum)
        })
        .collect()
}

/// Bitwise XOR of three 32-bit values represented as bits.
/// Direct formula: a + b + c - 2*(ab + ac + bc) + 4*abc.
/// 5 gate ops per bit vs 6 for cascading two xor_bits calls.
fn xor3_bits<F: BigPrimeField>(
    ctx: &mut Context<F>,
    gate: &impl GateInstructions<F>,
    a: &[AssignedValue<F>],
    b: &[AssignedValue<F>],
    c: &[AssignedValue<F>],
) -> Vec<AssignedValue<F>> {
    debug_assert_eq!(a.len(), 32);
    debug_assert_eq!(b.len(), 32);
    debug_assert_eq!(c.len(), 32);
    let neg_two = F::from(2).neg();
    let four = F::from(4);
    (0..32)
        .map(|i| {
            // ab = a*b
            let ab = gate.mul(ctx, a[i], b[i]);
            // abc = ab*c
            let abc = gate.mul(ctx, ab, c[i]);
            // s = ab + ac + bc = ab + c*(a + b)
            let a_plus_b = gate.add(ctx, a[i], b[i]);
            let s = gate.mul_add(ctx, c[i], a_plus_b, ab);
            // result = a + b + c - 2*s + 4*abc
            gate.inner_product(
                ctx,
                [a[i], b[i], c[i], s, abc],
                [
                    QuantumCell::Constant(F::ONE),
                    QuantumCell::Constant(F::ONE),
                    QuantumCell::Constant(F::ONE),
                    QuantumCell::Constant(neg_two),
                    QuantumCell::Constant(four),
                ],
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Modular arithmetic
// ---------------------------------------------------------------------------

/// Reduce a field element modulo 2^32 with a range check.
///
/// `hi_bits` is the number of bits required for the high part (x >> 32).
/// Callers must ensure `x < 2^(32 + hi_bits)`:
///   - Sums of 2 u32 values: hi < 2, use `hi_bits=2`
///   - Sums of 4 u32 values: hi < 4, use `hi_bits=3`  (message schedule)
///   - Sums of up to 7 u32 values: hi < 8, use `hi_bits=4`  (compression rounds)
fn mod_u32<F: BigPrimeField>(
    ctx: &mut Context<F>,
    range: &RangeChip<F>,
    x: AssignedValue<F>,
    hi_bits: usize,
) -> AssignedValue<F> {
    let gate = range.gate();
    let raw = x.value().get_lower_64();
    let lo_val = raw & 0xFFFF_FFFF;
    let hi_val = (raw >> 32) & 0xFFFF_FFFF;

    let lo = ctx.load_witness(F::from(lo_val));
    let hi = ctx.load_witness(F::from(hi_val));
    range.range_check(ctx, lo, 32);
    // Range-check hi to ensure unique decomposition in the field.
    // Without this, a malicious prover can pick alternative (hi', lo') pairs
    // satisfying x ≡ hi'*2^32 + lo' (mod p) with lo' ≠ x mod 2^32.
    range.range_check(ctx, hi, hi_bits);

    // Verify: x == hi * 2^32 + lo
    let composed = gate.mul_add(ctx, hi, QuantumCell::Constant(F::from(1u64 << 32)), lo);
    ctx.constrain_equal(&x, &composed);

    lo
}

// ---------------------------------------------------------------------------
// SHA-256 functions
// ---------------------------------------------------------------------------

/// Ch(e, f, g) = (e AND f) XOR (NOT(e) AND g)
fn ch<F: BigPrimeField>(
    ctx: &mut Context<F>,
    gate: &impl GateInstructions<F>,
    e_bits: &[AssignedValue<F>],
    f_bits: &[AssignedValue<F>],
    g_bits: &[AssignedValue<F>],
) -> Vec<AssignedValue<F>> {
    // ch = e*f + (1-e)*g = e*(f-g) + g
    debug_assert_eq!(e_bits.len(), 32);
    debug_assert_eq!(f_bits.len(), 32);
    debug_assert_eq!(g_bits.len(), 32);
    (0..32)
        .map(|i| {
            let f_sub_g = gate.sub(ctx, f_bits[i], g_bits[i]);
            gate.mul_add(ctx, e_bits[i], f_sub_g, g_bits[i])
        })
        .collect()
}

/// Maj(a, b, c) = (a AND b) XOR (a AND c) XOR (b AND c)
/// Equivalently: Maj(a,b,c) = (a*b) + (a*c) + (b*c) - 2*a*b*c
/// But simpler bit-by-bit: majority of 3 bits = a*b + a*c + b*c - 2*a*b*c
fn maj<F: BigPrimeField>(
    ctx: &mut Context<F>,
    gate: &impl GateInstructions<F>,
    a_bits: &[AssignedValue<F>],
    b_bits: &[AssignedValue<F>],
    c_bits: &[AssignedValue<F>],
) -> Vec<AssignedValue<F>> {
    debug_assert_eq!(a_bits.len(), 32);
    debug_assert_eq!(b_bits.len(), 32);
    debug_assert_eq!(c_bits.len(), 32);
    (0..32)
        .map(|i| {
            // maj = a*b + a*c + b*c - 2*a*b*c
            // = a*b + c*(a + b - 2*a*b)
            // = a*b + c*xor(a,b)
            let ab = gate.mul(ctx, a_bits[i], b_bits[i]);
            let a_xor_b = {
                let sum = gate.add(ctx, a_bits[i], b_bits[i]);
                let two_ab = gate.add(ctx, ab, ab);
                gate.sub(ctx, sum, two_ab)
            };
            gate.mul_add(ctx, c_bits[i], a_xor_b, ab)
        })
        .collect()
}

/// Sigma_upper_0(a) = ROTR(2, a) XOR ROTR(13, a) XOR ROTR(22, a)
fn sigma_upper0<F: BigPrimeField>(
    ctx: &mut Context<F>,
    gate: &impl GateInstructions<F>,
    a_bits: &[AssignedValue<F>],
) -> Vec<AssignedValue<F>> {
    let r2 = rotr(a_bits, 2);
    let r13 = rotr(a_bits, 13);
    let r22 = rotr(a_bits, 22);
    xor3_bits(ctx, gate, &r2, &r13, &r22)
}

/// Sigma_upper_1(e) = ROTR(6, e) XOR ROTR(11, e) XOR ROTR(25, e)
fn sigma_upper1<F: BigPrimeField>(
    ctx: &mut Context<F>,
    gate: &impl GateInstructions<F>,
    e_bits: &[AssignedValue<F>],
) -> Vec<AssignedValue<F>> {
    let r6 = rotr(e_bits, 6);
    let r11 = rotr(e_bits, 11);
    let r25 = rotr(e_bits, 25);
    xor3_bits(ctx, gate, &r6, &r11, &r25)
}

/// sigma_lower_0(w) = ROTR(7, w) XOR ROTR(18, w) XOR SHR(3, w)
fn sigma_lower0<F: BigPrimeField>(
    ctx: &mut Context<F>,
    gate: &impl GateInstructions<F>,
    w_bits: &[AssignedValue<F>],
) -> Vec<AssignedValue<F>> {
    let r7 = rotr(w_bits, 7);
    let r18 = rotr(w_bits, 18);
    let s3 = shr(ctx, w_bits, 3);
    xor3_bits(ctx, gate, &r7, &r18, &s3)
}

/// sigma_lower_1(w) = ROTR(17, w) XOR ROTR(19, w) XOR SHR(10, w)
fn sigma_lower1<F: BigPrimeField>(
    ctx: &mut Context<F>,
    gate: &impl GateInstructions<F>,
    w_bits: &[AssignedValue<F>],
) -> Vec<AssignedValue<F>> {
    let r17 = rotr(w_bits, 17);
    let r19 = rotr(w_bits, 19);
    let s10 = shr(ctx, w_bits, 10);
    xor3_bits(ctx, gate, &r17, &r19, &s10)
}

// ---------------------------------------------------------------------------
// SHA-256 compression function (single 64-byte block)
// ---------------------------------------------------------------------------

/// Process a single 64-byte block through the SHA-256 compression function.
///
/// `assigned_input_bytes`: 64 assigned byte cells (big-endian per word).
/// `pre_state_words`: 8 assigned u32 state words (h0..h7).
///
/// Returns 8 new state words (u32 values as field elements).
pub fn sha256_compression<F: BigPrimeField>(
    ctx: &mut Context<F>,
    range: &RangeChip<F>,
    assigned_input_bytes: &[AssignedValue<F>],
    pre_state_words: &[AssignedValue<F>],
) -> Vec<AssignedValue<F>> {
    debug_assert_eq!(assigned_input_bytes.len(), 64);
    debug_assert_eq!(pre_state_words.len(), 8);
    let gate = range.gate();

    // 1. Message schedule: compose bytes into 16 big-endian u32 words.
    let mut message_u32s: Vec<AssignedValue<F>> = assigned_input_bytes
        .chunks(4)
        .map(|bytes| {
            // Big-endian: word = bytes[0]*2^24 + bytes[1]*2^16 + bytes[2]*2^8 + bytes[3]
            let mut sum = bytes[0];
            for idx in 1..4 {
                sum = gate.mul_add(
                    ctx,
                    sum,
                    QuantumCell::Constant(F::from(256u64)),
                    bytes[idx],
                );
            }
            sum
        })
        .collect();

    // Decompose initial 16 words into bits for sigma operations.
    let mut message_bits: Vec<Vec<AssignedValue<F>>> = message_u32s
        .iter()
        .map(|&w| u32_to_bits(ctx, range, w))
        .collect();

    // 2. Expand to 64 words.
    for idx in 16..64 {
        let s0_bits = sigma_lower0(ctx, gate, &message_bits[idx - 15]);
        let s1_bits = sigma_lower1(ctx, gate, &message_bits[idx - 2]);
        let s0_val = bits_to_u32(ctx, gate, &s0_bits);
        let s1_val = bits_to_u32(ctx, gate, &s1_bits);

        let sum = gate.add(ctx, s1_val, message_u32s[idx - 7]);
        let sum = gate.add(ctx, sum, s0_val);
        let sum = gate.add(ctx, sum, message_u32s[idx - 16]);
        // Sum of 4 u32 values < 4·2^32, so hi < 4 fits in 3 bits.
        let new_w = mod_u32(ctx, range, sum, 3);

        let new_w_bits = u32_to_bits(ctx, range, new_w);
        message_u32s.push(new_w);
        message_bits.push(new_w_bits);
    }

    // 3. Initialize working variables.
    let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h) = (
        pre_state_words[0],
        pre_state_words[1],
        pre_state_words[2],
        pre_state_words[3],
        pre_state_words[4],
        pre_state_words[5],
        pre_state_words[6],
        pre_state_words[7],
    );
    let mut a_bits = u32_to_bits(ctx, range, a);
    let mut b_bits = u32_to_bits(ctx, range, b);
    let mut c_bits = u32_to_bits(ctx, range, c);
    let mut e_bits = u32_to_bits(ctx, range, e);
    let mut f_bits = u32_to_bits(ctx, range, f);
    let mut g_bits = u32_to_bits(ctx, range, g);

    // 4. 64 compression rounds.
    //
    // Performance optimization: T1 and T2 are NOT reduced mod 2^32 individually.
    // They are only used to compute new_e = (d + T1) mod 2^32 and
    // new_a = (T1 + T2) mod 2^32, so the intermediate reductions are redundant:
    //   (d + raw_T1) mod 2^32 = (d + (T1 mod 2^32)) mod 2^32
    // This eliminates 2 mod_u32 calls per round (128 total).
    for idx in 0..64 {
        // raw_T1 = h + Sigma1(e) + Ch(e,f,g) + K[idx] + W[idx]
        // Unreduced sum of 5 u32 values: raw_T1 < 5·2^32
        let raw_t1 = {
            let sigma_bits = sigma_upper1(ctx, gate, &e_bits);
            let sigma_val = bits_to_u32(ctx, gate, &sigma_bits);
            let ch_bits = ch(ctx, gate, &e_bits, &f_bits, &g_bits);
            let ch_val = bits_to_u32(ctx, gate, &ch_bits);

            let sum = gate.add(ctx, h, sigma_val);
            let sum = gate.add(ctx, sum, ch_val);
            let sum = gate.add(
                ctx,
                sum,
                QuantumCell::Constant(F::from(ROUND_CONSTANTS[idx] as u64)),
            );
            gate.add(ctx, sum, message_u32s[idx])
        };

        // raw_T2 = Sigma0(a) + Maj(a,b,c)
        // Unreduced sum of 2 u32 values: raw_T2 < 2·2^32
        let raw_t2 = {
            let sigma_bits = sigma_upper0(ctx, gate, &a_bits);
            let sigma_val = bits_to_u32(ctx, gate, &sigma_bits);
            let maj_bits = maj(ctx, gate, &a_bits, &b_bits, &c_bits);
            let maj_val = bits_to_u32(ctx, gate, &maj_bits);

            gate.add(ctx, sigma_val, maj_val)
        };

        // Update working variables.
        h = g;
        g = f;
        g_bits = f_bits;
        f = e;
        f_bits = e_bits;
        e = {
            // d + raw_T1 < 2^32 + 5·2^32 = 6·2^32, hi < 2^4
            let sum = gate.add(ctx, d, raw_t1);
            mod_u32(ctx, range, sum, 4)
        };
        e_bits = u32_to_bits(ctx, range, e);
        d = c;
        c = b;
        c_bits = b_bits;
        b = a;
        b_bits = a_bits;
        a = {
            // raw_T1 + raw_T2 < 5·2^32 + 2·2^32 = 7·2^32, hi < 2^4
            let sum = gate.add(ctx, raw_t1, raw_t2);
            mod_u32(ctx, range, sum, 4)
        };
        a_bits = u32_to_bits(ctx, range, a);
    }

    // 5. Compute new state: H_i = (old_state[i] + working_var[i]) mod 2^32
    // Sum of 2 u32 values < 2·2^32, so hi < 2 fits in 2 bits.
    let new_states = [a, b, c, d, e, f, g, h];
    new_states
        .iter()
        .zip(pre_state_words.iter())
        .map(|(&new, &old)| {
            let sum = gate.add(ctx, new, old);
            mod_u32(ctx, range, sum, 2)
        })
        .collect()
}
