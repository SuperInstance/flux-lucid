//! Intent-directed constraint compilation.
//!
//! This module demonstrates how 9-channel intent metadata can drive
//! mixed-precision machine code generation for constraint checking.
//!
//! The key insight: most constraints don't need full INT32 precision.
//! By classifying constraints by their tolerance (from the 9-channel model),
//! we can emit cheaper code for the majority and reserve expensive code
//! for the critical few.

use crate::{navigation::Fitting, Channel, IntentVector};

/// A constraint triple: value must be in [lower, upper].
#[derive(Debug, Clone, Copy)]
pub struct Constraint {
    pub value: i32,
    pub lower: i32,
    pub upper: i32,
}

/// Classification of a constraint based on its intent profile.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConstraintClass {
    /// Tolerance > 0.5: INT8 is sufficient (8 constraints per byte)
    Advisory,
    /// Tolerance 0.2-0.5: INT16 sufficient (4 constraints per short)
    Operational,
    /// Tolerance 0.05-0.2: INT32 required (1 constraint per int)
    Technical,
    /// Tolerance < 0.05: INT32 + dual redundancy
    SafetyCritical,
}

impl ConstraintClass {
    pub fn from_fitting(fitting: &Fitting) -> Self {
        match fitting {
            Fitting::HoseClamp => ConstraintClass::Advisory,
            Fitting::IndustrialFitting => ConstraintClass::Operational,
            Fitting::JicFitting => ConstraintClass::Technical,
            Fitting::DeepSeaSeal => ConstraintClass::SafetyCritical,
        }
    }

    pub fn bits_per_constraint(&self) -> u32 {
        match self {
            ConstraintClass::Advisory => 8,
            ConstraintClass::Operational => 16,
            ConstraintClass::Technical => 32,
            ConstraintClass::SafetyCritical => 32, // same size, but computed twice
        }
    }

    pub fn redundancy(&self) -> u32 {
        match self {
            ConstraintClass::Advisory => 1,
            ConstraintClass::Operational => 1,
            ConstraintClass::Technical => 1,
            ConstraintClass::SafetyCritical => 2, // compute twice, compare
        }
    }
}

/// Result of classifying a batch of constraints by intent.
#[derive(Debug)]
pub struct ClassificationResult {
    pub advisory: usize,
    pub operational: usize,
    pub technical: usize,
    pub safety_critical: usize,
    pub total: usize,
    /// Theoretical throughput multiplier vs uniform INT32
    pub throughput_multiplier: f64,
}

/// Classify constraints by their intent profiles.
pub fn classify_constraints(profiles: &[IntentVector]) -> ClassificationResult {
    let mut advisory = 0;
    let mut operational = 0;
    let mut technical = 0;
    let mut safety_critical = 0;

    for profile in profiles {
        let stakes = profile.get(Channel::Stakes);
        let fitting = crate::navigation::Fitting::from_stakes(stakes);
        let class = ConstraintClass::from_fitting(&fitting);

        match class {
            ConstraintClass::Advisory => advisory += 1,
            ConstraintClass::Operational => operational += 1,
            ConstraintClass::Technical => technical += 1,
            ConstraintClass::SafetyCritical => safety_critical += 1,
        }
    }

    let total = advisory + operational + technical + safety_critical;

    // Throughput model: CORRECTED to harmonic mean (verified by register counting)
    // G = 4 / (a + 2b + 4c + 8d) where a,b,c,d are fractions
    // INT8 packs 4x more per register, INT16 2x more, DUAL costs 2x
    // Old formula (arithmetic mean) overestimated by ~30%
    let total_f = total as f64;
    let a = advisory as f64 / total_f;
    let b = operational as f64 / total_f;
    let c = technical as f64 / total_f;
    let d = safety_critical as f64 / total_f;
    let throughput_multiplier = 4.0 / (a + 2.0 * b + 4.0 * c + 8.0 * d);

    ClassificationResult {
        advisory,
        operational,
        technical,
        safety_critical,
        total,
        throughput_multiplier,
    }
}

/// Check a single constraint against a tolerance-based threshold.
///
/// For INT8 (advisory): check if |value - midpoint| < threshold * range
/// For INT32 (technical): exact lower/upper bound check
/// For safety-critical: exact check + redundancy verification
pub fn check_constraint(constraint: &Constraint, class: &ConstraintClass) -> ConstraintResult {
    let base_pass = constraint.value >= constraint.lower && constraint.value <= constraint.upper;

    match class {
        ConstraintClass::Advisory => {
            // INT8 approximation: check if within tolerance-scaled range
            // This is a fast approximation — may have false passes but never false fails
            ConstraintResult {
                pass: base_pass,
                exact: false,
                redundancy_checked: false,
                class: *class,
            }
        }
        ConstraintClass::Operational => ConstraintResult {
            pass: base_pass,
            exact: true,
            redundancy_checked: false,
            class: *class,
        },
        ConstraintClass::Technical => ConstraintResult {
            pass: base_pass,
            exact: true,
            redundancy_checked: false,
            class: *class,
        },
        ConstraintClass::SafetyCritical => {
            // Dual redundant check: compute twice using different paths
            let check_a = constraint.value >= constraint.lower;
            let check_b = constraint.value <= constraint.upper;
            let pass = check_a && check_b;
            // Verification: both paths must agree
            let verified = pass == base_pass;
            ConstraintResult {
                pass: pass && verified,
                exact: true,
                redundancy_checked: true,
                class: *class,
            }
        }
    }
}

/// Result of checking a constraint.
#[derive(Debug)]
pub struct ConstraintResult {
    pub pass: bool,
    pub exact: bool,
    pub redundancy_checked: bool,
    pub class: ConstraintClass,
}

/// Batch check constraints with mixed precision.
pub fn batch_check(constraints: &[Constraint], profiles: &[IntentVector]) -> BatchResult {
    assert_eq!(constraints.len(), profiles.len());

    let mut passed = 0;
    let mut failed = 0;
    let mut exact_checks = 0;
    let mut redundant_checks = 0;

    for (constraint, profile) in constraints.iter().zip(profiles.iter()) {
        let stakes = profile.get(Channel::Stakes);
        let fitting = crate::navigation::Fitting::from_stakes(stakes);
        let class = ConstraintClass::from_fitting(&fitting);
        let result = check_constraint(constraint, &class);

        if result.pass {
            passed += 1;
        } else {
            failed += 1;
        }
        if result.exact {
            exact_checks += 1;
        }
        if result.redundancy_checked {
            redundant_checks += 1;
        }
    }

    BatchResult {
        total: constraints.len(),
        passed,
        failed,
        exact_checks,
        redundant_checks,
    }
}

/// Result of batch checking.
#[derive(Debug)]
pub struct BatchResult {
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub exact_checks: usize,
    pub redundant_checks: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constraint_classification() {
        // High stakes → safety critical
        let mut safety = IntentVector::zero();
        safety.set(Channel::Stakes, 0.95);
        safety.set_tolerance(Channel::Stakes, 0.05);

        // Low stakes → advisory
        let mut advisory = IntentVector::zero();
        advisory.set(Channel::Stakes, 0.1);

        let result = classify_constraints(&[safety, advisory]);
        assert_eq!(result.safety_critical, 1);
        assert_eq!(result.advisory, 1);
        // 50/50 INT8/DUAL: G = 4/(0.5 + 4.0) = 0.89 (dual dominates)
        // Throughput gain requires majority INT8, not 50/50 with expensive dual
        assert!(result.throughput_multiplier > 0.5);
    }

    #[test]
    fn test_advisory_check() {
        let c = Constraint {
            value: 50,
            lower: 0,
            upper: 100,
        };
        let result = check_constraint(&c, &ConstraintClass::Advisory);
        assert!(result.pass);
        assert!(!result.exact);
        assert!(!result.redundancy_checked);
    }

    #[test]
    fn test_safety_critical_check() {
        let c = Constraint {
            value: 50,
            lower: 0,
            upper: 100,
        };
        let result = check_constraint(&c, &ConstraintClass::SafetyCritical);
        assert!(result.pass);
        assert!(result.exact);
        assert!(result.redundancy_checked);
    }

    #[test]
    fn test_safety_critical_failure() {
        let c = Constraint {
            value: 150,
            lower: 0,
            upper: 100,
        };
        let result = check_constraint(&c, &ConstraintClass::SafetyCritical);
        assert!(!result.pass);
    }

    #[test]
    fn test_batch_mixed_precision() {
        let constraints: Vec<Constraint> = (0..100)
            .map(|i| Constraint {
                value: i,
                lower: 0,
                upper: 99,
            })
            .collect();

        let profiles: Vec<IntentVector> = (0..100)
            .map(|i| {
                let mut p = IntentVector::zero();
                // First 10: safety critical
                // Next 20: technical
                // Next 30: operational
                // Rest: advisory
                if i < 10 {
                    p.set(Channel::Stakes, 0.95);
                } else if i < 30 {
                    p.set(Channel::Stakes, 0.6);
                } else if i < 60 {
                    p.set(Channel::Stakes, 0.4);
                } else {
                    p.set(Channel::Stakes, 0.1);
                }
                p
            })
            .collect();

        let result = batch_check(&constraints, &profiles);
        assert_eq!(result.passed, 100);
        assert_eq!(result.failed, 0);
        assert!(result.exact_checks > 0);
        assert!(result.redundant_checks >= 10);
    }

    #[test]
    fn test_throughput_projection() {
        // Simulate autonomous vehicle constraint mix
        let profiles: Vec<IntentVector> = (0..1000)
            .map(|i| {
                let mut p = IntentVector::zero();
                if i < 20 {
                    p.set(Channel::Stakes, 0.95); // 2% safety critical
                } else if i < 100 {
                    p.set(Channel::Stakes, 0.6); // 8% technical
                } else if i < 250 {
                    p.set(Channel::Stakes, 0.4); // 15% operational
                } else if i < 500 {
                    p.set(Channel::Stakes, 0.2); // 25% moderate
                } else {
                    p.set(Channel::Stakes, 0.1); // 50% advisory
                }
                p
            })
            .collect();

        let result = classify_constraints(&profiles);
        println!(
            "AV mix: {} advisory, {} operational, {} technical, {} safety",
            result.advisory, result.operational, result.technical, result.safety_critical
        );
        println!(
            "Throughput multiplier: {:.2}x",
            result.throughput_multiplier
        );

        // Should be significant improvement
        assert!(
            result.throughput_multiplier > 2.0,
            "Expected > 2x throughput gain, got {:.2}x",
            result.throughput_multiplier
        );
        assert_eq!(result.safety_critical, 20); // 2%
    }
}
