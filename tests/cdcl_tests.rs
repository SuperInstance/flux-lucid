//! CDCL trace operations tests for flux-lucid
//!
//! Tests the CDCL (Conflict-Driven Clause Learning) pipeline:
//! 1. Intent encoding and classification
//! 2. Intent-directed compilation and constraint checking
//! 3. Beam-tolerance solver integration
//! 4. Head direction and navigation consistency
//! 5. Dream reconstruction accuracy tracking

use flux_lucid::intent::encode_intent;
use flux_lucid::intent_compilation::{
    classify_constraints, check_constraint, batch_check,
    Constraint, ConstraintClass, BatchResult,
};
use flux_lucid::intent_emitter::{
    check_with_precision, batch_check_directed, differential_test,
    IntentDirective, Precision, ClassificationStats,
};
use flux_lucid::navigation::Fitting;
use flux_lucid::beam_tolerance::{
    BeamMaterial, compute_tolerance, classify_precision,
    SoABatch, PrecisionClass,
};
use flux_lucid::head_direction::{
    HeadDirection, PositionedAgent, angular_coherence,
    consolidate_paths, FleetSnapshot, ConsolidatedTile,
};
use flux_lucid::dream::{
    DreamConfig, DreamFragment, DreamStyle,
    reconstruct, negative_reconstruction, dream_layer,
    amnesia_curve, style_resilience, compression_frontier,
};

// ===========================================================================
// CDCL Trace Operations — Intent Classification Pipeline
// ===========================================================================

#[test]
fn test_cdcl_encode_then_classify() {
    // Encode various messages and verify they classify correctly
    let urgent = encode_intent("Deploy urgently by Friday — safety-critical!");
    let casual = encode_intent("How's the weather?");
    let technical = encode_intent("API algorithm implementation research deploy");

    // Urgent should have high stakes
    assert!(urgent.get(flux_lucid::Channel::Stakes) > 0.5,
            "Urgent message should have high stakes");

    // Casual should NOT have high stakes
    assert!(casual.get(flux_lucid::Channel::Stakes) < 0.95,
            "Casual message should not have maximum stakes");

    // Technical should match pattern/code keywords
    assert!(technical.get(flux_lucid::Channel::Pattern) > 0.5,
            "Technical message should have high Pattern salience");

    // Check that both have valid classification
    let profiles = vec![urgent, casual, technical];
    let result = classify_constraints(&profiles);
    assert_eq!(result.total, 3);
    assert!(result.safety_critical + result.advisory + result.operational + result.technical == 3);
}

#[test]
fn test_cdcl_mixed_precision_check() {
    // Create a batch of constraints with mixed precision directives
    let values = vec![10, 50, 200, 50000, 100000];
    let lowers = vec![0, 0, 0, 0, 0];
    let uppers = vec![100, 100, 500, 60000, 200000];

    let directives = vec![
        IntentDirective { precision: Precision::INT8, constraint_idx: 0 },
        IntentDirective { precision: Precision::INT16, constraint_idx: 1 },
        IntentDirective { precision: Precision::INT16, constraint_idx: 2 },
        IntentDirective { precision: Precision::INT32, constraint_idx: 3 },
        IntentDirective { precision: Precision::DUAL, constraint_idx: 4 },
    ];

    let (passed, failed) = batch_check_directed(&values, &lowers, &uppers, &directives);
    assert_eq!(passed, 5, "All constraints should pass");
    assert_eq!(failed, 0);
}

#[test]
fn test_cdcl_differential_in_range() {
    // Verify mixed-precision checks match reference INT32 for in-range values
    let n = 100usize;
    let values: Vec<i32> = (0..n).map(|i| (i * 5) as i32).collect();
    let lowers: Vec<i32> = vec![0; n];
    let uppers: Vec<i32> = vec![500; n];

    let directives: Vec<IntentDirective> = (0..n)
        .map(|i| IntentDirective {
            precision: match i % 4 {
                0 => Precision::INT8,
                1 => Precision::INT16,
                2 => Precision::INT32,
                _ => Precision::DUAL,
            },
            constraint_idx: i,
        })
        .collect();

    let (mismatches, indices) = differential_test(&values, &lowers, &uppers, &directives);
    assert_eq!(mismatches, 0,
               "Expected zero mismatches, got {} at indices {:?}", mismatches, indices);
}

#[test]
fn test_cdcl_out_of_range_int8_saturates() {
    // INT8 truncation: values > 127 are saturated but should still fail correctly
    let result = check_with_precision(150, 0, 200, Precision::INT8);
    // 150 = -106 as i8... but -106 < 0 → false
    // Actually check: (-106 < 0) → false, so the test passes as expected
    assert!(!result, "150 as INT8 should fail (wraps to -106 which is < 0)");
}

#[test]
fn test_cdcl_empty_batch() {
    let (passed, failed) = batch_check_directed(&[], &[], &[], &[]);
    assert_eq!(passed, 0);
    assert_eq!(failed, 0);
}

#[test]
fn test_cdcl_classification_stats_av_mix() {
    // Simulate autonomous vehicle constraint mix
    let n = 1000usize;
    let directives: Vec<IntentDirective> = (0..n)
        .map(|i| {
            let precision = if i < 750 { Precision::INT8 }
                           else if i < 900 { Precision::INT16 }
                           else if i < 980 { Precision::INT32 }
                           else { Precision::DUAL };
            IntentDirective { precision, constraint_idx: i }
        })
        .collect();

    let stats = ClassificationStats::from_directives(&directives);
    assert_eq!(stats.int8_count, 750);
    assert_eq!(stats.int16_count, 150);
    assert_eq!(stats.int32_count, 80);
    assert_eq!(stats.dual_count, 20);
    assert!(stats.theoretical_throughput_gain > 3.0,
            "AV mix should have >3x throughput, got {:.2}x",
            stats.theoretical_throughput_gain);
}

// ===========================================================================
// CDCL Trace Operations — Intent Compilation
// ===========================================================================

#[test]
fn test_intent_compilation_throughput_av() {
    // Simulate autonomous vehicle constraints
    use flux_lucid::Channel;
    use flux_lucid::IntentVector;

    let profiles: Vec<IntentVector> = (0..1000)
        .map(|i| {
            let mut p = IntentVector::zero();
            if i < 20 { p.set(Channel::Stakes, 0.95); }
            else if i < 100 { p.set(Channel::Stakes, 0.6); }
            else if i < 250 { p.set(Channel::Stakes, 0.4); }
            else if i < 500 { p.set(Channel::Stakes, 0.2); }
            else { p.set(Channel::Stakes, 0.1); }
            p
        })
        .collect();

    let result = classify_constraints(&profiles);
    assert!(result.throughput_multiplier > 2.0,
            "Expected >2x throughput, got {:.2}x", result.throughput_multiplier);
}

#[test]
fn test_intent_compilation_batch_check() {
    let constraints: Vec<Constraint> = (0..50)
        .map(|i| Constraint { value: i * 2, lower: 0, upper: 99 })
        .collect();

    use flux_lucid::Channel;
    use flux_lucid::IntentVector;
    let profiles: Vec<IntentVector> = (0..50)
        .map(|i| {
            let mut p = IntentVector::zero();
            p.set(Channel::Stakes, 0.1 + i as f64 * 0.018);
            p
        })
        .collect();

    let result = batch_check(&constraints, &profiles);
    assert_eq!(result.total, 50);
    // Some should pass (value within bounds), some fail (value[25]=50 > 99)
    assert!(result.passed > 0);
}

#[test]
fn test_fitting_selection_pipeline() {
    // Verify the full pipeline: stakes → Fitting → ConstraintClass → precision
    assert_eq!(Fitting::from_stakes(0.1), Fitting::HoseClamp);
    assert_eq!(Fitting::from_stakes(0.3), Fitting::IndustrialFitting);
    assert_eq!(Fitting::from_stakes(0.6), Fitting::JicFitting);
    assert_eq!(Fitting::from_stakes(0.9), Fitting::DeepSeaSeal);

    assert_eq!(ConstraintClass::from_fitting(&Fitting::HoseClamp).bits_per_constraint(), 8);
    assert_eq!(ConstraintClass::from_fitting(&Fitting::DeepSeaSeal).redundancy(), 2);
}

// ===========================================================================
// Beam Tolerance Solver
// ===========================================================================

#[test]
fn test_beam_tolerance_material_hierarchy() {
    let steel = BeamMaterial::steel();
    let fiberglass = BeamMaterial::fiberglass();
    let rubber = BeamMaterial::rubber();

    // Stiffer material → tighter tolerance
    assert!(steel.max_tolerance(1.0) < fiberglass.max_tolerance(1.0),
            "Steel should be tighter than fiberglass");
    assert!(fiberglass.max_tolerance(1.0) < rubber.max_tolerance(1.0),
            "Fiberglass should be tighter than rubber");
}

#[test]
fn test_beam_tolerance_safety_factor() {
    let oak = BeamMaterial::oak();
    let tol_1x = oak.max_tolerance(1.0);
    let tol_2x = oak.max_tolerance(2.0);
    assert!(tol_2x <= tol_1x, "Higher safety factor should give tighter tolerance");
}

#[test]
fn test_dynamic_amplification_squat_effect() {
    let steel = BeamMaterial::steel();
    let rubber = BeamMaterial::rubber();

    // Steel is stiff → minimal dynamic amplification
    let da_steel = steel.dynamic_amplification(1.0);
    assert!(da_steel <= 2.0, "Steel DAF should be low, got {}", da_steel);

    // Rubber is flexible → significant dynamic amplification
    let da_rubber = rubber.dynamic_amplification(1.0);
    assert!(da_rubber >= 1.5, "Rubber DAF should be high, got {}", da_rubber);
}

#[test]
fn test_stakes_to_material_mapping() {
    assert_eq!(flux_lucid::beam_tolerance::stakes_to_material(0.8).youngs_modulus, 200.0);
    assert_eq!(flux_lucid::beam_tolerance::stakes_to_material(0.6).youngs_modulus, 30.0);
    assert_eq!(flux_lucid::beam_tolerance::stakes_to_material(0.3).youngs_modulus, 12.0);
    assert_eq!(flux_lucid::beam_tolerance::stakes_to_material(0.15).youngs_modulus, 6.0);
    assert_eq!(flux_lucid::beam_tolerance::stakes_to_material(0.05).youngs_modulus, 0.01);
}

#[test]
fn test_compute_tolerance_decreasing_with_stakes() {
    let low = compute_tolerance(0.1, 1.0);
    let high = compute_tolerance(0.9, 1.0);
    assert!(high <= low, "Higher stakes => tighter tolerance, got {} <= {}", high, low);
}

#[test]
fn test_classify_precision_all_levels() {
    assert_eq!(classify_precision(0.1, 10.0), PrecisionClass::INT8);
    assert_eq!(classify_precision(0.3, 100.0), PrecisionClass::INT16);
    assert_eq!(classify_precision(0.6, 1000.0), PrecisionClass::INT32);
    assert_eq!(classify_precision(0.8, 50000.0), PrecisionClass::DUAL);
}

#[test]
fn test_soa_batch_memory_savings_significant() {
    // 100 constraints: 50 INT8, 30 INT16, 15 INT32, 5 DUAL
    let mut constraints = Vec::new();
    for _ in 0..50 { constraints.push((5.0, 0.0, 10.0, 0.1)); }
    for _ in 0..30 { constraints.push((50.0, 0.0, 100.0, 0.3)); }
    for _ in 0..15 { constraints.push((500.0, 0.0, 1000.0, 0.6)); }
    for _ in 0..5 { constraints.push((5000.0, 0.0, 10000.0, 0.9)); }

    let batch = SoABatch::from_constraints(&constraints);
    let (actual, baseline) = batch.memory_stats();
    let savings = 1.0 - (actual as f64 / baseline as f64);
    assert!(savings > 0.4, "SoA batch should save >40% memory, got {:.1}%", savings * 100.0);
}

// ===========================================================================
// Head Direction & Navigation
// ===========================================================================

#[test]
fn test_head_direction_consistency() {
    // All agents facing same direction → perfect coherence
    let agents: Vec<PositionedAgent> = (0..10)
        .map(|i| PositionedAgent::new(i, i, HeadDirection::from_step(3), 10))
        .collect();

    let coherence = angular_coherence(&agents);
    assert!((coherence - 1.0).abs() < 1e-10,
            "Uniform heading should have coherence ~1.0, got {}", coherence);
}

#[test]
fn test_head_direction_maximal_dispersion() {
    // Agents spread uniformly across all 12 directions → minimal coherence
    let agents: Vec<PositionedAgent> = (0..12)
        .map(|i| PositionedAgent::new(i, 0, HeadDirection::from_step(i as u8), 10))
        .collect();

    let coherence = angular_coherence(&agents);
    // Uniform distribution over 12 directions → mean resultant length ≈ 0
    assert!(coherence < 0.3,
            "Uniformly dispersed agents should have low coherence, got {}", coherence);
}

#[test]
fn test_path_consolidation_threshold() {
    // Path with fewer visits than threshold → no tiles
    let snapshots: Vec<FleetSnapshot> = (0..2)
        .map(|t| FleetSnapshot::new(
            vec![PositionedAgent::new(t, t, HeadDirection::from_step(0), 10)],
            (t as u64) * 100,
        ))
        .collect();

    let tiles = consolidate_paths(&snapshots, 3);
    assert!(tiles.is_empty(), "Should not consolidate below threshold");
}

#[test]
fn test_consolidated_tile_span_and_duration() {
    let snapshots: Vec<FleetSnapshot> = (0..5)
        .map(|t| FleetSnapshot::new(
            vec![PositionedAgent::new(t, t, HeadDirection::from_step(0), 10)],
            (t as u64) * 100,
        ))
        .collect();

    let tiles = consolidate_paths(&snapshots, 3);
    assert_eq!(tiles.len(), 1);
    assert_eq!(tiles[0].start_q, 0);
    assert_eq!(tiles[0].start_r, 0);
    assert_eq!(tiles[0].end_q, 4);
    assert_eq!(tiles[0].end_r, 4);
    assert_eq!(tiles[0].visit_count, 5);
    assert_eq!(tiles[0].first_seen, 0);
    assert_eq!(tiles[0].last_seen, 400);
    assert_eq!(tiles[0].duration(), 400);
    assert_eq!(tiles[0].span(), 8); // (0,0)→(4,4): dq=4, dr=4, ds=8 → (4+4+8)/2 = 8
}

// ===========================================================================
// Dream Reconstruction
// ===========================================================================

#[test]
fn test_dream_amnesia_curve_interpolation() {
    // Verify interpolation between data points
    let at_100 = amnesia_curve(1.00);  // 0.975
    let at_75 = amnesia_curve(0.75);   // 0.775
    let interpolated = amnesia_curve(0.875); // halfway between 1.0 and 0.75
    assert!((interpolated - (at_100 + at_75) / 2.0).abs() < 0.02,
            "Interpolation at 87.5% should be midpoint");
}

#[test]
fn test_dream_reconstruction_with_fragments() {
    let config = DreamConfig::default();
    let fragments = vec![
        DreamFragment {
            content: "The baton passes through the constraint lattice.".into(),
            coverage: 0.50,
            style: DreamStyle::Literal,
            constraints_preserved: vec!["baton exists".into(), "lattice exists".into()],
            timestamp: 1000.0,
        },
        DreamFragment {
            content: "Holonomy deviation is within INT8 bounds.".into(),
            coverage: 0.40,
            style: DreamStyle::Abstract,
            constraints_preserved: vec!["INT8 bounds checked".into()],
            timestamp: 1005.0,
        },
    ];

    let result = reconstruct(&fragments, &config);
    assert!(!result.content.is_empty());
    assert!(result.accuracy_estimate > 0.0);
    assert!(result.confidence > 0.0);
    assert_eq!(result.style, DreamStyle::Literal); // dominant by coverage
}

#[test]
fn test_dream_hallucination_zone() {
    let config = DreamConfig::default();
    let fragments = vec![
        DreamFragment {
            content: "Faint memory...".into(),
            coverage: 0.03, // Below 10% amnesia cliff
            style: DreamStyle::Narrative,
            constraints_preserved: vec![],
            timestamp: 0.0,
        },
    ];

    let result = reconstruct(&fragments, &config);
    assert!(result.content.contains("HALLUCINATION"), "Should flag hallucination risk");
    assert!(result.accuracy_estimate < 0.1, "Accuracy should be very low");
}

#[test]
fn test_negative_reconstruction_inference() {
    let negs = vec!["no constraint violation".into(), "no cycle break".into(),
                     "no faulty tile".into()];
    let result = negative_reconstruction(&negs);
    assert_eq!(result.style, DreamStyle::Negative);
    assert_eq!(result.novel_inferences.len(), 3);
    // Each negation generates a positive constraint inference
    for inference in &result.novel_inferences {
        assert!(inference.contains("positive constraint"), "Each negative → positive: {}", inference);
    }
}

#[test]
fn test_dream_layer_creativity() {
    let config = DreamConfig::default();
    let text = "The consensus engine computes holonomy across 9 dimensions.";
    let literal_result = reconstruct(&[DreamFragment {
        content: text.into(),
        coverage: 1.0,
        style: DreamStyle::Literal,
        constraints_preserved: vec![],
        timestamp: 0.0,
    }], &config);

    let dream_result = dream_layer(text, &config);

    // Dream mode adds the ~dream~ markers
    assert!(dream_result.content.contains("~dream~"),
            "Dream mode should wrap content in markers");
    assert!(dream_result.accuracy_estimate < literal_result.accuracy_estimate,
            "Dream mode should have lower accuracy than literal");
}

#[test]
fn test_compression_frontier_extrapolation() {
    // Below smallest data point (22 chars) → should return ~0.025
    let accuracy = compression_frontier("test", 10);
    assert!(accuracy <= 0.1, "Very high compression should give low accuracy, got {}", accuracy);
    assert!(accuracy >= 0.0, "Accuracy should be non-negative");

    // Above largest data point (1145 chars) → should return ~0.775 (plateau)
    let accuracy_high = compression_frontier("test", 2000);
    assert!((accuracy_high - 0.775).abs() < 0.01,
            "Above max compression should plateau at 0.775, got {}", accuracy_high);
}

// ===========================================================================
// SoA Emitter Mixed Precision
// ===========================================================================

#[test]
fn test_soa_emitter_all_precision_classes() {
    use flux_lucid::soa_emitter::{SoAConstraintBatch, classify as soa_classify};

    let constraints = vec![
        (5.0, 0.0, 10.0, 0.1),     // INT8
        (50.0, 0.0, 100.0, 0.3),   // INT16
        (500.0, 0.0, 1000.0, 0.6), // INT32
        (5000.0, 0.0, 10000.0, 0.8), // DUAL
    ];

    let batch = SoAConstraintBatch::from_constraints(&constraints);
    let results = batch.check_all();
    assert_eq!(results.len(), 4);
    assert!(results.iter().all(|&r| r), "All constraints should pass");

    let stats = batch.stats();
    assert_eq!(stats.int8_count, 1);
    assert_eq!(stats.int16_count, 1);
    assert_eq!(stats.int32_count, 1);
    assert_eq!(stats.dual_count, 1);
}

#[test]
fn test_soa_emitter_differential_precision() {
    use flux_lucid::soa_emitter::SoAConstraintBatch;

    // Same value at different precisions should agree for in-range values
    let val = 5.0;
    let lo = 0.0;
    let hi = 10.0;

    let int8_batch = SoAConstraintBatch::from_constraints(&[(val, lo, hi, 0.1)]);
    let int16_batch = SoAConstraintBatch::from_constraints(&[(val, lo, hi, 0.3)]);
    let int32_batch = SoAConstraintBatch::from_constraints(&[(val, lo, hi, 0.6)]);

    assert_eq!(int8_batch.check_all()[0], true);
    assert_eq!(int16_batch.check_all()[0], true);
    assert_eq!(int32_batch.check_all()[0], true);
}

// ===========================================================================
// Spectral Conservation Integration
// ===========================================================================

#[test]
fn test_fleet_conservation_monitor_creation() {
    use flux_lucid::spectral::FleetConservationMonitor;

    let monitor = FleetConservationMonitor::new(&["agent1", "agent2", "agent3"]);
    // Just verify construction works
    let _ = monitor;
}
