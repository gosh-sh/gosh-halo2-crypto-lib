# `gosh-dense-balanced-tree` — depth scaling and benchmarks

This note answers a sizing question that comes up when the dense balanced
Merkle tree chip is reused outside its historical setting (depth-8 /
256-leaf history-proof trees): **how far does depth scale, and what does
it cost?** Concretely, is a depth-15 tree (~30,000 leaves) feasible, and
how do the numbers compare to the depth-8 shape in production today.

## TL;DR

- The chip has **no hardcoded depth**. `preprocess_dense_proof` and
  `dense_merkle_root_circuit` iterate over whatever `siblings.len()` is
  passed. The "256 leaves" number lives in the acki-nacki protocol, not in
  the chip.
- In-circuit cost grows **linearly in depth** (not in leaf count): going
  from 256 leaves to 32,768 leaves is **1.88× in-circuit**, 128× only in
  the one-time native tree build.
- Measured depth-15 real-prover run (k=14, SHPLONK + Blake2b, release):
  **~400 ms per proof, 2,304 B proof, ~1.9 ms verify**, 5-advice-column
  circuit shape.
- 11-link chain (`MAX_CHAIN_LEN`) at depth 15 is estimated ~15–30 s per
  proof and still fits at k=14 with ~22 columns (or k=15 with ~10).
- No chip code change needed to support larger trees. The existing
  `preprocess_dense_proof_padded` + `dense_merkle_root_circuit_padded`
  already handle variable real depth under a fixed VK.

## Where the depth lives

The chip's module-level constants are:

| Const | Value | Role |
|---|---|---|
| `T` | 3 | Poseidon state width |
| `RATE` | 2 | Poseidon absorption rate |
| `R_F` | 8 | Poseidon full rounds |
| `R_P` | 57 | Poseidon partial rounds |
| `MAX_CHAIN_LEN` | 11 | Number of trees chainable in `verify_chain_of_dense_proofs` |

There is **no `DEPTH` or `MAX_LEAVES`**. The depth of any single tree is
whatever the caller supplies as `siblings.len()`.

## Per-level cost (same at any depth)

One iteration of `dense_merkle_root_circuit` (`src/lib.rs:330–368`):

| Op | Advice cells | Lookup rows |
|---|---|---|
| witness loads (sibling, bit, c0, c1, c2, left_hi) | ~6 | — |
| `assert_bit` | ~1 | — |
| `cond_swap` (2× select) | ~4 | — |
| linking constraints (3× `mul_add`/`mul`/`sub`) | ~4 | — |
| `range_check` of 248, 240, 8, 16 bits at `lookup_bits=k-1` | ~22 | ~44 |
| `hash_fix_len_array` of 3 Fr (2 Poseidon permutations) | **~1,000–1,500** | — |
| **Total per level** | **~1,500–2,000** | **~44** |

Poseidon dominates (~85–95% of per-level cost). Everything else is byte
decomposition and linking.

## Depth-8 vs depth-15 at a glance

Per single Merkle path, assuming the same chip and the same Poseidon spec:

| Dimension | Depth 8 (256 leaves) | Depth 15 (32,768 leaves) | Ratio |
|---|---|---|---|
| Siblings in path | 8 × 32 B = 256 B | 15 × 32 B = 480 B | ×1.88 |
| In-circuit Poseidon permutations | 16 | 30 | ×1.88 |
| In-circuit range checks | 32 | 60 | ×1.88 |
| Lookup rows | ~352 | ~660 | ×1.88 |
| Advice cells / proof | ~16,000 | ~30,000 | ×1.88 |
| Native tree build (one-time) | ~50 ms | ~1.3 s (measured) | ×26 |
| Memory for tree bytes | ~16 KB | ~2 MB | ×128 |
| Suggested `k` | 13 | **14** | +1 |

The native tree build grows with leaf count. The proof itself does not.

## Measured numbers

Hardware: this workstation, release profile, SHPLONK + Blake2b transcript,
single instance column.

### Depth-7 reference (`test_dense_circuit_real_prover`, k=13, 128 leaves)

Already present in `tests/dense_tree_tests.rs:325`. Serves as a baseline
for the depth-15 extrapolation and comparison.

### Depth-15 (`test_dense_circuit_depth15_real_prover`, k=14, 32,768 leaves)

Added in this change at `tests/dense_tree_tests.rs:437`. Full output:

```
=== Depth-15 Real Prover (k=14, depth=15, leaves=32768) ===
Tree build: 32768 leaves, leaf_hash=664 ms, tree_build=621 ms
  Tree build (32768 leaves):   1.290 s
  SRS generation (k=14):         637 ms
  Circuit params:              k=14, num_advice=5, num_lookup_advice=1, num_fixed=1
  Keygen circuit build:           44 ms
  VK generation:                 161 ms
  PK generation:                  47 ms
  Total keygen:                  252 ms
--- Proving leaf at pos 0 ---
  Prover circuit build:          699 ms
  Proof generation:              397 ms
  Proof size:                   2304 bytes
  Proof verification:            1.9 ms
--- Proving leaf at pos 16384 ---
  Proof generation:              400 ms
  Proof size:                   2304 bytes
  Proof verification:            1.9 ms
--- Proving leaf at pos 32767 ---
  Proof generation:              405 ms
  Proof size:                   2304 bytes
  Proof verification:            1.9 ms
```

Reproduce with:

```
cargo test --release -p gosh-dense-balanced-tree \
    --test dense_tree_tests test_dense_circuit_depth15_real_prover \
    -- --ignored --nocapture
```

### Measurement vs estimate

The real numbers came in well under the a-priori estimates:

| Metric | Estimate | Measured |
|---|---|---|
| Tree build | 5–7 s | **1.29 s** |
| VK generation | 1–3 s | 161 ms |
| PK generation | 3–6 s | 47 ms |
| Proof generation | 2–5 s | **~400 ms** |
| Proof size | 5–8 KB | **2,304 B** |
| Verification | 5–10 ms | ~1.9 ms |

The gap is mainly `pse_poseidon`'s native throughput and the modest column
count (5 advice) picked by `calculate_params`.

## Scaling further (same chip, no code change)

Order-of-magnitude extrapolations from the measured depth-15 point.
Per-level cost is constant, so prove time is roughly linear in depth and
dominated by the `k` bump threshold.

| Leaves | Depth | Rec. `k` | Prove time (est.) | Notes |
|---|---|---|---|---|
| 128 | 7 | 13 | ~0.2 s | current `test_dense_circuit_real_prover` |
| 1,024 | 10 | 13 | ~0.3 s | |
| **32,768** | **15** | **14** | **~0.4 s (measured)** | **this test** |
| 262,144 | 18 | 14–15 | ~0.6–1.5 s | depth 18 still fits k=14 |
| 1,048,576 | 20 | 15 | ~1.5–3 s | SRS ~4 MB |
| 16,777,216 | 24 | 16 | ~5–10 s | native tree build becomes the bottleneck |

Beyond ~2²⁰ leaves the practical limit is native tree build and tree
memory, not the circuit.

## Chain of trees (`verify_chain_of_dense_proofs`)

The chain combinator at `src/lib.rs:493` iterates `MAX_CHAIN_LEN = 11`
links, each verifying a Merkle path in its own tree, where the output
root of link `j` becomes the leaf of link `j+1`. A fixed VK requires all
slots to share the same (worst-case) depth; shallower real trees are
padded via `preprocess_dense_proof_padded` and masked with
`dense_merkle_root_circuit_padded` + a `num_active_levels` witness.

Per-chain cost at various per-link depths:

| Chain config | Advice cells | Suggested `k` | Prove time (est.) |
|---|---|---|---|
| 11 × depth 8 (current production shape) | ~176,000 | k=14, ~15 cols | ~8–15 s |
| 6 × depth 15 | ~180,000 | k=14, ~14–18 cols | ~8–15 s |
| **11 × depth 15** | **~330,000** | **k=14 ~22 cols or k=15 ~10 cols** | **~15–30 s** |

The native side pays per-tree: 11 × 1.3 s ≈ ~15 s to build 11 depth-15
trees. Rayon-parallelize across independent trees to collapse that.

## Design implications

- **Standalone depth-15 proof is cheap.** ~400 ms per proof at 2.3 KB
  with sub-2 ms verify is comfortably in interactive territory.
- **Variable-depth under one VK is already supported.** No chip change is
  required to serve both depth-8 history trees and larger trees in the
  same deployment — pad shorter ones to a chosen `MAX_DEPTH` and gate
  with `dense_merkle_root_circuit_padded`.
- **The `MAX_CHAIN_LEN = 11` ceiling is where depth growth is felt most.**
  Chained depth-15 proofs still land under 30 s but widen the circuit to
  ~22 advice columns at k=14. Jumping to k=15 trades a slightly larger
  SRS for a narrower table.
- **Native tree build is the lurking cost at very large leaf counts.** At
  32k leaves it is already comparable to prove time (~1.3 s vs ~0.4 s);
  at 2²⁰ leaves it will dominate the pipeline unless cached or
  parallelized.

## Files touched

- `tests/dense_tree_tests.rs:437` — added `test_dense_circuit_depth15_real_prover`,
  gated with `#[ignore]` because of the ~1.3 s native tree build.
- `src/lib.rs` — unchanged.
