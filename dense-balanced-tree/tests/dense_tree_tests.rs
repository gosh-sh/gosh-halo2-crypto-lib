use std::time::Instant;

use dense_balanced_tree::{
    dense_leaf_hash, dense_merkle_proof, dense_merkle_root, dense_merkle_verify, PoseidonHasher,
};
use gosh_dense_balanced_tree::{
    bytes_to_fr, compute_root_native, dense_merkle_root_circuit, preprocess_dense_proof,
};
use halo2_base::{
    gates::{
        circuit::{builder::BaseCircuitBuilder, CircuitBuilderStage},
        RangeInstructions,
    },
    halo2_proofs::{
        dev::MockProver,
        halo2curves::bn256::{Bn256, Fr, G1Affine},
        plonk::{create_proof, keygen_pk, keygen_vk, verify_proof},
        poly::{
            commitment::ParamsProver,
            kzg::{
                commitment::{KZGCommitmentScheme, ParamsKZG},
                multiopen::{ProverSHPLONK, VerifierSHPLONK},
                strategy::SingleStrategy,
            },
        },
        transcript::{
            Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer,
            TranscriptWriterBuffer,
        },
    },
    poseidon::hasher::{spec::OptimizedPoseidonSpec, PoseidonHasher as Halo2PoseidonHasher},
    utils::fs::gen_srs,
};
use rand::{rngs::StdRng, SeedableRng};

const T: usize = 3;
const RATE: usize = 2;

/// Helper: build an n-leaf tree from deterministic data and return (leaves, root).
fn build_test_tree(n: usize) -> (Vec<[u8; 32]>, [u8; 32]) {
    let hasher = PoseidonHasher::new();

    let t0 = Instant::now();
    let leaves: Vec<[u8; 32]> = (0..n)
        .map(|i| {
            let mut key = [0u8; 32];
            key[0] = i as u8;
            key[1] = (i >> 8) as u8;
            let mut val = [0u8; 32];
            val[0] = (i as u8).wrapping_add(0x80);
            val[1] = (i >> 8) as u8;
            dense_leaf_hash(&hasher, &key, &val)
        })
        .collect();
    let leaf_time = t0.elapsed();

    let t1 = Instant::now();
    let root = dense_merkle_root(&hasher, &leaves);
    let tree_time = t1.elapsed();

    println!(
        "Tree build: {} leaves, leaf_hash={:?}, tree_build={:?}, root={}",
        n,
        leaf_time,
        tree_time,
        hex::encode(root)
    );
    (leaves, root)
}

/// Run the circuit test for a given leaf position in a tree, with timing.
fn run_dense_circuit_test(leaves: &[[u8; 32]], expected_root: &[u8; 32], pos: usize) {
    let hasher = PoseidonHasher::new();

    // --- Proof generation ---
    let t0 = Instant::now();
    let siblings = dense_merkle_proof(&hasher, leaves, pos);
    let proof_gen_time = t0.elapsed();

    let depth = siblings.len();
    let proof_size_bytes = 32 /* leaf */ + siblings.len() * 32 /* siblings */;

    // --- Native verification ---
    let t1 = Instant::now();
    assert!(
        dense_merkle_verify(&hasher, expected_root, &leaves[pos], pos, &siblings),
        "Native verification failed for pos {}",
        pos
    );
    let native_verify_time = t1.elapsed();

    // --- Preprocessing ---
    let t2 = Instant::now();
    let proof = preprocess_dense_proof(leaves[pos], &siblings, pos);
    let preprocess_time = t2.elapsed();

    let k = 13;

    // --- Circuit witness generation ---
    let t3 = Instant::now();
    let mut builder = BaseCircuitBuilder::new(false)
        .use_k(k)
        .use_lookup_bits(k - 1)
        .use_instance_columns(1);

    let range = builder.range_chip();

    let root_assigned = {
        let ctx = builder.main(0);

        let spec = OptimizedPoseidonSpec::<Fr, T, RATE>::new::<8, 57, 0>();
        let mut halo2_hasher = Halo2PoseidonHasher::<Fr, T, RATE>::new(spec);
        halo2_hasher.initialize_consts(ctx, range.gate());

        let leaf_fr = ctx.load_witness(bytes_to_fr(&proof.leaf));
        dense_merkle_root_circuit(ctx, &range, &halo2_hasher, &proof, leaf_fr)
    };

    builder.assigned_instances[0].push(root_assigned);
    builder.calculate_params(Some(20));
    let witness_gen_time = t3.elapsed();

    // --- MockProver run ---
    let t4 = Instant::now();
    let expected_root_fr = bytes_to_fr(expected_root);
    let prover =
        MockProver::<Fr>::run(k as u32, &builder, vec![vec![expected_root_fr]]).unwrap();
    let mock_run_time = t4.elapsed();

    // --- MockProver verify ---
    let t5 = Instant::now();
    prover.assert_satisfied();
    let mock_verify_time = t5.elapsed();

    let params = &builder.config_params;
    println!("--- Dense circuit test: pos={}, depth={} ---", pos, depth);
    println!("  Proof size:           {} bytes ({} siblings x 32B + 32B leaf)", proof_size_bytes, depth);
    println!("  Proof generation:     {:?}", proof_gen_time);
    println!("  Native verification:  {:?}", native_verify_time);
    println!("  Preprocessing:        {:?}", preprocess_time);
    println!("  Witness generation:   {:?}", witness_gen_time);
    println!("  MockProver::run:      {:?}", mock_run_time);
    println!("  MockProver::verify:   {:?}", mock_verify_time);
    println!("  Total circuit time:   {:?}", witness_gen_time + mock_run_time + mock_verify_time);
    println!("  Circuit params:       k={}, num_advice={}, num_lookup_advice={}, num_fixed={}",
        params.k, params.num_advice_per_phase[0], params.num_lookup_advice_per_phase[0], params.num_fixed);
    println!("------");
}

#[test]
fn test_native_preprocessing_consistency() {
    let hasher = PoseidonHasher::new();

    let t0 = Instant::now();
    let (leaves, root) = build_test_tree(128);
    let tree_build_time = t0.elapsed();
    println!("Total tree build time: {:?}", tree_build_time);

    for pos in [0, 1, 63, 64, 127] {
        let t1 = Instant::now();
        let siblings = dense_merkle_proof(&hasher, &leaves, pos);
        let proof_time = t1.elapsed();

        let t2 = Instant::now();
        let proof = preprocess_dense_proof(leaves[pos], &siblings, pos);
        let preprocess_time = t2.elapsed();

        let t3 = Instant::now();
        let computed_root = compute_root_native(&proof);
        let native_hash_time = t3.elapsed();

        let expected_root_fr = bytes_to_fr(&root);
        assert_eq!(
            computed_root, expected_root_fr,
            "Native preprocessing root mismatch for pos {}",
            pos
        );

        println!(
            "  pos {:>3}: proof_gen={:?}, preprocess={:?}, native_hash={:?}, siblings={}",
            pos, proof_time, preprocess_time, native_hash_time, siblings.len()
        );
    }
    println!("Native preprocessing consistency test passed for all positions");
}

#[test]
fn test_dense_circuit_single_leaf() {
    let (leaves, root) = build_test_tree(1);
    run_dense_circuit_test(&leaves, &root, 0);
}

#[test]
fn test_dense_circuit_128_leaves_pos0() {
    let (leaves, root) = build_test_tree(128);
    run_dense_circuit_test(&leaves, &root, 0);
}

#[test]
fn test_dense_circuit_128_leaves_pos63() {
    let (leaves, root) = build_test_tree(128);
    run_dense_circuit_test(&leaves, &root, 63);
}

#[test]
fn test_dense_circuit_128_leaves_pos127() {
    let (leaves, root) = build_test_tree(128);
    run_dense_circuit_test(&leaves, &root, 127);
}

#[test]
fn test_dense_circuit_negative_tampered_sibling() {
    let hasher = PoseidonHasher::new();
    let (leaves, root) = build_test_tree(128);
    let pos = 0;

    let mut siblings = dense_merkle_proof(&hasher, &leaves, pos);
    // Tamper with the first sibling
    siblings[0][0] ^= 0xFF;

    let proof = preprocess_dense_proof(leaves[pos], &siblings, pos);

    let k = 13;

    let mut builder = BaseCircuitBuilder::new(false)
        .use_k(k)
        .use_lookup_bits(k - 1)
        .use_instance_columns(1);

    let range = builder.range_chip();

    let root_assigned = {
        let ctx = builder.main(0);

        let spec = OptimizedPoseidonSpec::<Fr, T, RATE>::new::<8, 57, 0>();
        let mut halo2_hasher = Halo2PoseidonHasher::<Fr, T, RATE>::new(spec);
        halo2_hasher.initialize_consts(ctx, range.gate());

        let leaf_fr = ctx.load_witness(bytes_to_fr(&proof.leaf));
        dense_merkle_root_circuit(ctx, &range, &halo2_hasher, &proof, leaf_fr)
    };

    builder.assigned_instances[0].push(root_assigned);
    builder.calculate_params(Some(20));

    // The tampered proof should NOT produce the correct root
    let expected_root_fr = bytes_to_fr(&root);
    let prover =
        MockProver::<Fr>::run(k as u32, &builder, vec![vec![expected_root_fr]]).unwrap();

    // This should fail verification (instance mismatch)
    let t0 = Instant::now();
    let result = prover.verify();
    let verify_time = t0.elapsed();
    assert!(
        result.is_err(),
        "Tampered proof should fail verification"
    );
    println!("Negative test passed: tampered sibling correctly rejected (verify took {:?})", verify_time);
}

/// Build a keygen-stage circuit for a dense tree proof of the given depth.
/// Returns the builder with calculated params, ready for keygen.
fn build_keygen_circuit(k: usize, depth: usize) -> BaseCircuitBuilder<Fr> {
    // Use dummy all-zero data for keygen (shape only, values don't matter)
    let leaf = [0u8; 32];
    let siblings: Vec<[u8; 32]> = vec![[0u8; 32]; depth];
    let proof = preprocess_dense_proof(leaf, &siblings, 0);

    let mut builder = BaseCircuitBuilder::from_stage(CircuitBuilderStage::Keygen)
        .use_k(k)
        .use_lookup_bits(k - 1)
        .use_instance_columns(1);

    let range = builder.range_chip();

    let root_assigned = {
        let ctx = builder.main(0);
        let spec = OptimizedPoseidonSpec::<Fr, T, RATE>::new::<8, 57, 0>();
        let mut hasher = Halo2PoseidonHasher::<Fr, T, RATE>::new(spec);
        hasher.initialize_consts(ctx, range.gate());
        let leaf_fr = ctx.load_witness(bytes_to_fr(&proof.leaf));
        dense_merkle_root_circuit(ctx, &range, &hasher, &proof, leaf_fr)
    };

    builder.assigned_instances[0].push(root_assigned);
    builder.calculate_params(Some(20));
    builder
}

/// Build a prover-stage circuit for a specific Merkle proof.
/// Uses the config_params and break_points from the keygen circuit.
fn build_prover_circuit(
    keygen_builder: &BaseCircuitBuilder<Fr>,
    leaves: &[[u8; 32]],
    pos: usize,
) -> (BaseCircuitBuilder<Fr>, Fr) {
    let hasher = PoseidonHasher::new();
    let siblings = dense_merkle_proof(&hasher, leaves, pos);
    let proof = preprocess_dense_proof(leaves[pos], &siblings, pos);

    let mut builder = BaseCircuitBuilder::prover(
        keygen_builder.config_params.clone(),
        keygen_builder.break_points(),
    )
    .use_instance_columns(1);

    let range = builder.range_chip();

    let root_assigned = {
        let ctx = builder.main(0);
        let spec = OptimizedPoseidonSpec::<Fr, T, RATE>::new::<8, 57, 0>();
        let mut hasher = Halo2PoseidonHasher::<Fr, T, RATE>::new(spec);
        hasher.initialize_consts(ctx, range.gate());
        let leaf_fr = ctx.load_witness(bytes_to_fr(&proof.leaf));
        dense_merkle_root_circuit(ctx, &range, &hasher, &proof, leaf_fr)
    };

    builder.assigned_instances[0].push(root_assigned);

    let root_fr = compute_root_native(&proof);
    (builder, root_fr)
}

#[test]
fn test_dense_circuit_real_prover() {
    let k: usize = 13;
    let depth = 7; // log2(128)

    println!("=== Dense Tree Real Prover Test (k={}, depth={}) ===", k, depth);

    // --- Build tree ---
    let t0 = Instant::now();
    let (leaves, root_bytes) = build_test_tree(128);
    let tree_time = t0.elapsed();
    println!("  Tree build (128 leaves):  {:?}", tree_time);

    // --- Generate SRS ---
    let t1 = Instant::now();
    let params: ParamsKZG<Bn256> = gen_srs(k as u32);
    let srs_time = t1.elapsed();
    println!("  SRS generation (k={}):    {:?}", k, srs_time);

    // --- Keygen: build circuit with dummy data ---
    let t2 = Instant::now();
    let keygen_builder = build_keygen_circuit(k, depth);
    let config_params = &keygen_builder.config_params;
    println!("  Circuit params:           k={}, num_advice={}, num_lookup_advice={}, num_fixed={}",
        config_params.k, config_params.num_advice_per_phase[0],
        config_params.num_lookup_advice_per_phase[0], config_params.num_fixed);
    let keygen_circuit_time = t2.elapsed();
    println!("  Keygen circuit build:     {:?}", keygen_circuit_time);

    // --- Generate VK ---
    let t3 = Instant::now();
    let vk = keygen_vk(&params, &keygen_builder).expect("keygen_vk should not fail");
    let vk_time = t3.elapsed();
    println!("  VK generation:            {:?}", vk_time);

    // --- Generate PK ---
    let t4 = Instant::now();
    let pk = keygen_pk(&params, vk, &keygen_builder).expect("keygen_pk should not fail");
    let pk_time = t4.elapsed();
    println!("  PK generation:            {:?}", pk_time);
    println!("  Total keygen:             {:?}", keygen_circuit_time + vk_time + pk_time);

    // --- Test multiple positions ---
    for pos in [0, 63, 127] {
        println!("--- Proving leaf at pos {} ---", pos);

        // --- Build prover circuit ---
        let t5 = Instant::now();
        let (prover_builder, root_fr) = build_prover_circuit(&keygen_builder, &leaves, pos);
        let prover_circuit_time = t5.elapsed();

        let expected_root_fr = bytes_to_fr(&root_bytes);
        assert_eq!(root_fr, expected_root_fr, "Native root mismatch at pos {}", pos);

        // --- Create proof ---
        let t6 = Instant::now();
        let instances = vec![root_fr];
        let proof = {
            let rng = StdRng::seed_from_u64(0);
            let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
            create_proof::<
                KZGCommitmentScheme<Bn256>,
                ProverSHPLONK<'_, Bn256>,
                Challenge255<_>,
                _,
                Blake2bWrite<Vec<u8>, G1Affine, _>,
                _,
            >(
                &params,
                &pk,
                &[prover_builder],
                &[&[instances.as_slice()]],
                rng,
                &mut transcript,
            )
            .expect("proof generation should not fail");
            transcript.finalize()
        };
        let prove_time = t6.elapsed();

        // --- Verify proof ---
        let t7 = Instant::now();
        {
            let strategy = SingleStrategy::new(&params);
            let mut transcript =
                Blake2bRead::<_, _, Challenge255<_>>::init(proof.as_slice());
            verify_proof::<
                KZGCommitmentScheme<Bn256>,
                VerifierSHPLONK<'_, Bn256>,
                Challenge255<G1Affine>,
                Blake2bRead<&[u8], G1Affine, Challenge255<G1Affine>>,
                SingleStrategy<'_, Bn256>,
            >(
                params.verifier_params(),
                pk.get_vk(),
                strategy,
                &[&[instances.as_slice()]],
                &mut transcript,
            )
            .expect("proof verification should not fail");
        }
        let verify_time = t7.elapsed();

        println!("  Prover circuit build:   {:?}", prover_circuit_time);
        println!("  Proof generation:       {:?}", prove_time);
        println!("  Proof size:             {} bytes", proof.len());
        println!("  Proof verification:     {:?}", verify_time);
        println!("  Total (build+prove):    {:?}", prover_circuit_time + prove_time);
    }

    println!("=== Real prover test passed for all positions ===");
}
