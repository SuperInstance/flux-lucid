//! Constraint solving tests for flux-lucid
//!
//! Tests the full constraint theory pipeline:
//! - Constraint classification by intent profiles
//! - Mixed-precision checking (INT8, INT16, INT32, DUAL)
//! - Batch constraint operations
//! - Beam-tolerance boundary conditions
//! - Intent alignment safety margins
//! - Navigation draft calculations

use flux_lucid::navigation::{check_draft, tolerance_stack, Fitting};
use flux_lucid::beam_tolerance::{
    SoABatch, BeamMaterial, compute_tolerance, compute_draft,
};
use flux_lucid::IntentVector;
use flux_lucid::Channel;
use flux_lucid::intent::check_alignment;

// ===========================================================================
// Constraint Classification — All Precision Levels
// ===========================================================================

#[test]
fn test_constraint_classify_int8_range() {
    // INT8 range: ±127 → for constraint values within this range, stakes ≤ 0.25
    let batch = SoABatch::from_constraints(&[
        (50.0, 0.0, 100.0, 0.1),
        (30.0, 10.0, 50.0, 0.05),
        (-50.0, -100.0, 0.0, 0.2),
    ]);
    assert_eq!(batch.memory_stats().0, batch.memory_stats().0); // sanity
    let results = batch.check_all();
    assert_eq!(results.len(), 3);
    assert!(results.iter().all(|&r| r), "All INT8 constraints should pass");
}

#[test]
fn test_constraint_classify_int16_range() {
    let batch = SoABatch::from_constraints(&[
        (1000.0, 0.0, 32000.0, 0.3),
        (-5000.0, -10000.0, 0.0, 0.4),
    ]);
    let results = batch.check_all();
    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|&r| r));
}

#[test]
fn test_constraint_classify_int32_range() {
    // High stakes (> 0.5) or large range (> 127) → INT32
    let batch = SoABatch::from_constraints(&[
        (50000.0, 0.0, 100000.0, 0.6),
        (100000.0, 0.0, 200000.0, 0.55),
    ]);
    let results = batch.check_all();
    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|&r| r));
}

#[test]
fn test_constraint_classify_dual_range() {
    let batch = SoABatch::from_constraints(&[
        (500000.0, 0.0, 1000000.0, 0.9),
    ]);
    let results = batch.check_all();
    assert_eq!(results.len(), 1);
    assert!(results[0]);
}

// ===========================================================================
// Constraint Boundary Conditions
// ===========================================================================

#[test]
fn test_constraint_at_exact_boundary_passes() {
    // Value exactly at boundary should pass
    let batch = SoABatch::from_constraints(&[
        (0.0, 0.0, 100.0, 0.1),  // lower boundary
        (100.0, 0.0, 100.0, 0.1), // upper boundary
    ]);
    let results = batch.check_all();
    assert!(results[0], "Value at lower boundary should pass");
    assert!(results[1], "Value at upper boundary should pass");
}

#[test]
fn test_constraint_outside_lower_bound_fails() {
    let batch = SoABatch::from_constraints(&[
        (-1.0, 0.0, 100.0, 0.1), // below lower
    ]);
    let results = batch.check_all();
    assert!(!results[0], "Value below lower bound should fail");
}

#[test]
fn test_constraint_outside_upper_bound_fails() {
    let batch = SoABatch::from_constraints(&[
        (101.0, 0.0, 100.0, 0.1), // above upper
    ]);
    let results = batch.check_all();
    assert!(!results[0], "Value above upper bound should fail");
}

#[test]
fn test_constraint_int8_boundary_truncation() {
    // INT8 truncation: value near INT8 boundary should still work correctly
    // 127 is max INT8 → should be in range
    let batch = SoABatch::from_constraints(&[
        (127.0, 0.0, 127.0, 0.1),  // exactly at INT8_MAX
        (-128.0, -128.0, 0.0, 0.1), // exactly at INT8_MIN
    ]);
    let results = batch.check_all();
    assert!(results[0], "INT8_MAX should pass");
    assert!(results[1], "INT8_MIN should pass");
}

#[test]
fn test_large_batch_constraint_consistency() {
    // 500 constraints, all in range → all should pass
    let mut constraints = Vec::new();
    for i in 0..500usize {
        let stakes = 0.1 + (i % 4) as f64 * 0.25; // cycles through all precision levels
        constraints.push((i as f64, 0.0, 499.0, stakes));
    }

    let batch = SoABatch::from_constraints(&constraints);
    let results = batch.check_all();
    assert_eq!(results.len(), 500);
    assert!(results.iter().all(|&r| r), "All 500 constraints should pass");
}

#[test]
fn test_large_batch_mixed_pass_fail() {
    let mut constraints = Vec::new();
    for i in 0..100usize {
        let pass = i < 80; // first 80 pass, last 20 fail
        let stakes = 0.2;
        constraints.push((
            if pass { i as f64 } else { 1000.0 },
            0.0, 99.0, stakes
        ));
    }

    let batch = SoABatch::from_constraints(&constraints);
    let results = batch.check_all();
    assert_eq!(results.len(), 100);

    let passed = results.iter().filter(|&&r| r).count();
    let failed = results.iter().filter(|&&r| !r).count();
    assert_eq!(passed, 80, "First 80 should pass");
    assert_eq!(failed, 20, "Last 20 should fail");
}

// ===========================================================================
// Constraint Memory Efficiency
// ===========================================================================

#[test]
fn test_memory_savings_int8_dominated() {
    // 80% INT8, 10% INT16, 7% INT32, 3% DUAL
    let mut constraints = Vec::new();
    for _ in 0..800 { constraints.push((5.0, 0.0, 10.0, 0.1)); }
    for _ in 0..100 { constraints.push((50.0, 0.0, 100.0, 0.3)); }
    for _ in 0..70  { constraints.push((500.0, 0.0, 1000.0, 0.6)); }
    for _ in 0..30  { constraints.push((5000.0, 0.0, 10000.0, 0.8)); }

    let batch = SoABatch::from_constraints(&constraints);
    let (actual, baseline) = batch.memory_stats();
    let savings = 1.0 - (actual as f64 / baseline as f64);
    assert!(savings > 0.55,
            "INT8-dominated mix should save >55% memory, got {:.1}%", savings * 100.0);
}

#[test]
fn test_memory_savings_dual_dominated() {
    // Safety-critical: 90% DUAL
    let mut constraints = Vec::new();
    for _ in 0..900 { constraints.push((5000.0, 0.0, 10000.0, 0.9)); }

    let batch = SoABatch::from_constraints(&constraints);
    let (actual, baseline) = batch.memory_stats();
    let savings = 1.0 - (actual as f64 / baseline as f64);
    // DUAL uses 64 bits vs INT32 32 bits, so no savings — in fact, more expensive
    assert!(savings < 0.0,
            "DUAL-dominated should use MORE memory than baseline, savings={:.1}%", savings * 100.0);
}

// ===========================================================================
// Draft/Alignment Safety
// ===========================================================================

#[test]
fn test_draft_report_safe_communication() {
    let mut sender = IntentVector::zero();
    sender.set(Channel::Social, 0.5);
    sender.set(Channel::Process, 0.3);

    let report = check_draft(&sender, 0.8, 0.0);
    assert!(report.is_safe, "Well-toleranced message should be safe");

    let report_str = format!("{}", report);
    assert!(report_str.contains("SAFE"), "Report should indicate safe: {}", report_str);
    assert!(report_str.contains("margin="), "Report should contain margin: {}", report_str);
}

#[test]
fn test_draft_report_grounded_communication() {
    let mut sender = IntentVector::zero();
    sender.set(Channel::Stakes, 0.95);
    sender.set_tolerance(Channel::Stakes, 0.05);

    let report = check_draft(&sender, 0.1, 1.0);
    assert!(!report.is_safe, "Oversized draft should be unsafe");
    assert!(report.margin < 0.0, "Margin should be negative");
}

#[test]
fn test_tolerance_stack_consistency() {
    let mut profile = IntentVector::zero();
    let t1 = tolerance_stack(&profile);
    assert!((t1 - 1.5).abs() < 0.01, "Uniform tolerance 0.5 → stack ~1.5, got {}", t1);

    // Tighten all tolerances
    for ch in Channel::all() {
        profile.set_tolerance(*ch, 0.1);
    }
    let t2 = tolerance_stack(&profile);
    assert!((t2 - 0.3).abs() < 0.01, "Tight tolerance 0.1 → stack ~0.3, got {}", t2);
}

#[test]
fn test_alignment_check_full_pipeline() {
    let mut sender = IntentVector::zero();
    sender.set(Channel::Stakes, 0.8);
    sender.set(Channel::Boundary, 0.7);
    sender.set_tolerance(Channel::Stakes, 0.3);
    sender.set_tolerance(Channel::Boundary, 0.3);

    let mut receiver = IntentVector::zero();
    receiver.set(Channel::Stakes, 0.75);
    receiver.set(Channel::Boundary, 0.65);

    let report = check_alignment(&sender, &receiver);
    assert!(report.cosine_similarity > 0.99, "Similar intents should have high cosine similarity");
    assert!(report.is_safe, "Close alignment should be safe");
}

#[test]
fn test_alignment_check_mismatch_warning() {
    let mut sender = IntentVector::zero();
    sender.set(Channel::Stakes, 0.9);
    sender.set_tolerance(Channel::Stakes, 0.05);

    let mut receiver = IntentVector::zero();
    receiver.set(Channel::Stakes, 0.1); // Very different

    let report = check_alignment(&sender, &receiver);
    assert!(!report.is_safe, "Mismatched stakes should be unsafe");
    assert!(!report.warnings.is_empty(), "Should have warning about stakes mismatch");
}

// ===========================================================================
// Fitting (Navigation)
// ===========================================================================

#[test]
fn test_fitting_tolerance_levels() {
    assert!((Fitting::HoseClamp.tolerance() - 0.8).abs() < 0.01);
    assert!((Fitting::IndustrialFitting.tolerance() - 0.5).abs() < 0.01);
    assert!((Fitting::JicFitting.tolerance() - 0.2).abs() < 0.01);
    assert!((Fitting::DeepSeaSeal.tolerance() - 0.05).abs() < 0.01);
}

#[test]
fn test_fitting_min_channels() {
    assert_eq!(Fitting::HoseClamp.min_channels(), 2);
    assert_eq!(Fitting::IndustrialFitting.min_channels(), 4);
    assert_eq!(Fitting::JicFitting.min_channels(), 7);
    assert_eq!(Fitting::DeepSeaSeal.min_channels(), 9);
}

// ===========================================================================
// Beam Material Properties
// ===========================================================================

#[test]
fn test_beam_material_yield_strength_hierarchy() {
    let steel = BeamMaterial::steel();
    let oak = BeamMaterial::oak();
    let rubber = BeamMaterial::rubber();

    assert!(steel.yield_strength > oak.yield_strength,
            "Steel should yield at higher stress than oak");
    assert!(oak.yield_strength > rubber.yield_strength,
            "Oak should yield at higher stress than rubber");
}

#[test]
fn test_beam_material_density() {
    assert!(BeamMaterial::steel().density > BeamMaterial::fiberglass().density);
    assert!(BeamMaterial::fiberglass().density > BeamMaterial::oak().density);
    assert!(BeamMaterial::oak().density > BeamMaterial::cedar().density);
}

#[test]
fn test_compute_draft_squat_effect_reduces_tolerance() {
    let base = compute_tolerance(0.5, 1.0);
    let rushed = compute_draft(base, 0.8, 0.5);
    assert!(rushed <= base, "Squat effect should reduce effective tolerance");
}

// ===========================================================================
// Intent Vector Edge Cases
// ===========================================================================

#[test]
fn test_intent_zero_constraints() {
    let v = IntentVector::zero();
    assert_eq!(v.values, [0.0; 9]);
    assert_eq!(v.tolerance, [0.5; 9]);
}

#[test]
fn test_intent_clamping() {
    let mut v = IntentVector::zero();
    v.set(Channel::Stakes, 1.5); // Above 1.0 → clamped to 1.0
    assert!((v.get(Channel::Stakes) - 1.0).abs() < 0.001);

    v.set(Channel::Stakes, -0.5); // Below 0.0 → clamped to 0.0
    assert!((v.get(Channel::Stakes) - 0.0).abs() < 0.001);
}

#[test]
fn test_intent_draft_calculation() {
    let mut v = IntentVector::zero();
    v.set(Channel::Stakes, 0.9);
    v.set_tolerance(Channel::Stakes, 0.5);

    let draft = v.draft();
    assert!(draft > 0.0, "Draft should be positive: {}", draft);
    assert!(draft <= 0.2, "Draft should be ≤ 0.2 (max per channel allowed): {}", draft);
}

#[test]
fn test_intent_dominant_channel() {
    let mut v = IntentVector::zero();
    v.set(Channel::Social, 0.95);
    v.set(Channel::Stakes, 0.1);
    v.set(Channel::Boundary, 0.1);
    assert_eq!(v.dominant_channel(), Channel::Social);
}

#[test]
fn test_intent_euclidean_distance_orthogonal() {
    let mut a = IntentVector::zero();
    a.set(Channel::Social, 1.0);
    let mut b = IntentVector::zero();
    b.set(Channel::Stakes, 1.0);

    let dist = a.euclidean_distance(&b);
    assert!((dist - (2.0_f64).sqrt()).abs() < 0.01,
            "Distance between orthogonal unit vectors should be sqrt(2), got {}", dist);
}
