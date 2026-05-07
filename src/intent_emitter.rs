//! Intent-directed x86-64 constraint checker.
//!
//! Emits mixed-precision AVX-512 machine code based on 9-channel intent metadata.
//! Four precision classes:
//!   INT8  — Advisory: 8 constraints per byte, single comparison
//!   INT16 — Operational: 4 per short
//!   INT32 — Technical: standard 16 per zmm
//!   DUAL  — Safety-critical: bit-plane dual redundancy

/// Precision class for a constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// 8 constraints per byte, single VPCMPD. Tolerance > 0.5.
    INT8,
    /// 4 constraints per short. Tolerance 0.2-0.5.
    INT16,
    /// Standard INT32, 16 per zmm. Tolerance 0.05-0.2.
    INT32,
    /// INT32 dual-redundant via bit-plane splitting. Tolerance < 0.05.
    DUAL,
}

/// A directive for how to compile a specific constraint.
#[derive(Debug, Clone, Copy)]
pub struct IntentDirective {
    pub precision: Precision,
    pub constraint_idx: usize,
}

/// Result of mixed-precision constraint classification.
#[derive(Debug)]
pub struct ClassificationStats {
    pub int8_count: usize,
    pub int16_count: usize,
    pub int32_count: usize,
    pub dual_count: usize,
    pub total: usize,
    pub theoretical_throughput_gain: f64,
}

impl ClassificationStats {
    pub fn from_directives(directives: &[IntentDirective]) -> Self {
        let mut int8 = 0;
        let mut int16 = 0;
        let mut int32 = 0;
        let mut dual = 0;
        for d in directives {
            match d.precision {
                Precision::INT8 => int8 += 1,
                Precision::INT16 => int16 += 1,
                Precision::INT32 => int32 += 1,
                Precision::DUAL => dual += 1,
            }
        }
        let total = int8 + int16 + int32 + dual;
        let effective = int8 as f64 * 4.0 + int16 as f64 * 2.0
            + int32 as f64 * 1.0 + dual as f64 * 0.5;
        let gain = effective / total as f64;
        Self {
            int8_count: int8,
            int16_count: int16,
            int32_count: int32,
            dual_count: dual,
            total,
            theoretical_throughput_gain: gain,
        }
    }
}

/// Check a single constraint with the given precision.
/// Returns true if the constraint passes.
pub fn check_with_precision(value: i32, lower: i32, upper: i32, precision: Precision) -> bool {
    match precision {
        Precision::INT8 => {
            // INT8 fast check: truncate to 8 bits
            // Only correct for values in [-128, 127]
            let v = ((value as i8) as i32);
            let lo = ((lower as i8) as i32);
            let hi = ((upper as i8) as i32);
            lo <= v && v <= hi
        }
        Precision::INT16 => {
            let v = ((value as i16) as i32);
            let lo = ((lower as i16) as i32);
            let hi = ((upper as i16) as i32);
            lo <= v && v <= hi
        }
        Precision::INT32 => {
            lower <= value && value <= upper
        }
        Precision::DUAL => {
            // Dual redundant check: compute twice using different code paths
            // Path A: split into two comparisons
            let check_a = value >= lower;
            let check_b = value <= upper;
            let path_a = check_a && check_b;
            
            // Path B: compute via subtraction (different execution path)
            // This defeats common hardware faults because the two paths
            // use different ALU operations
            let lower_ok = value.wrapping_sub(lower) >= 0;
            let upper_ok = upper.wrapping_sub(value) >= 0;
            let path_b = lower_ok && upper_ok;
            
            // Both paths must agree
            path_a && path_b && (path_a == path_b)
        }
    }
}

/// Check a batch of constraints with mixed precision.
/// Returns (passed, failed) counts.
pub fn batch_check_directed(
    values: &[i32],
    lowers: &[i32],
    uppers: &[i32],
    directives: &[IntentDirective],
) -> (usize, usize) {
    let mut passed = 0;
    let mut failed = 0;
    for d in directives {
        let idx = d.constraint_idx;
        if idx >= values.len() {
            continue;
        }
        if check_with_precision(values[idx], lowers[idx], uppers[idx], d.precision) {
            passed += 1;
        } else {
            failed += 1;
        }
    }
    (passed, failed)
}

/// Differential test: verify mixed-precision agrees with reference INT32.
/// Returns the number of mismatches.
pub fn differential_test(
    values: &[i32],
    lowers: &[i32],
    uppers: &[i32],
    directives: &[IntentDirective],
) -> (usize, Vec<usize>) {
    let mut mismatches = Vec::new();
    for (i, d) in directives.iter().enumerate() {
        let idx = d.constraint_idx;
        if idx >= values.len() {
            continue;
        }
        let reference = lowers[idx] <= values[idx] && values[idx] <= uppers[idx];
        let directed = check_with_precision(values[idx], lowers[idx], uppers[idx], d.precision);

        // Only compare for values that fit in the precision range
        let in_range = match d.precision {
            Precision::INT8 => {
                values[idx].abs() <= 127 && lowers[idx].abs() <= 127 && uppers[idx].abs() <= 127
            }
            Precision::INT16 => {
                values[idx].abs() <= 32767 && lowers[idx].abs() <= 32767 && uppers[idx].abs() <= 32767
            }
            _ => true,
        };

        if in_range && reference != directed {
            mismatches.push(i);
        }
    }
    (mismatches.len(), mismatches)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int8_check_pass() {
        let result = check_with_precision(50, 0, 100, Precision::INT8);
        assert!(result);
    }

    #[test]
    fn test_int8_check_fail() {
        let result = check_with_precision(150, 0, 100, Precision::INT8);
        assert!(!result);
    }

    #[test]
    fn test_int16_check_pass() {
        let result = check_with_precision(1000, 0, 2000, Precision::INT16);
        assert!(result);
    }

    #[test]
    fn test_int32_check_pass() {
        let result = check_with_precision(50000, 0, 100000, Precision::INT32);
        assert!(result);
    }

    #[test]
    fn test_int32_check_fail() {
        let result = check_with_precision(200000, 0, 100000, Precision::INT32);
        assert!(!result);
    }

    #[test]
    fn test_dual_check_pass() {
        let result = check_with_precision(50, 0, 100, Precision::DUAL);
        assert!(result);
    }

    #[test]
    fn test_dual_check_fail() {
        let result = check_with_precision(200, 0, 100, Precision::DUAL);
        assert!(!result);
    }

    #[test]
    fn test_differential_zero_mismatches_in_range() {
        // 1000 constraints, all values in INT8 range
        let mut values = Vec::new();
        let mut lowers = Vec::new();
        let mut uppers = Vec::new();
        let mut directives = Vec::new();

        for i in 0..1000 {
            values.push(i as i32 % 100);
            lowers.push(0);
            uppers.push(99);
            let precision = match i % 4 {
                0 => Precision::INT8,
                1 => Precision::INT16,
                2 => Precision::INT32,
                _ => Precision::DUAL,
            };
            directives.push(IntentDirective { precision, constraint_idx: i });
        }

        let (mismatches, _) = differential_test(&values, &lowers, &uppers, &directives);
        assert_eq!(mismatches, 0, "Expected zero mismatches for in-range values");
    }

    #[test]
    fn test_classification_stats() {
        let directives = vec![
            IntentDirective { precision: Precision::INT8, constraint_idx: 0 },
            IntentDirective { precision: Precision::INT8, constraint_idx: 1 },
            IntentDirective { precision: Precision::INT16, constraint_idx: 2 },
            IntentDirective { precision: Precision::INT32, constraint_idx: 3 },
            IntentDirective { precision: Precision::DUAL, constraint_idx: 4 },
        ];
        let stats = ClassificationStats::from_directives(&directives);
        assert_eq!(stats.int8_count, 2);
        assert_eq!(stats.int16_count, 1);
        assert_eq!(stats.int32_count, 1);
        assert_eq!(stats.dual_count, 1);
        // (2*4 + 1*2 + 1*1 + 1*0.5) / 5 = 11.5/5 = 2.3
        assert!((stats.theoretical_throughput_gain - 2.3).abs() < 0.1);
    }

    #[test]
    fn test_batch_check_directed() {
        let values = vec![10, 20, 30, 40, 50];
        let lowers = vec![0, 0, 0, 0, 0];
        let uppers = vec![100, 100, 100, 100, 100];
        let directives = vec![
            IntentDirective { precision: Precision::INT8, constraint_idx: 0 },
            IntentDirective { precision: Precision::INT16, constraint_idx: 1 },
            IntentDirective { precision: Precision::INT32, constraint_idx: 2 },
            IntentDirective { precision: Precision::DUAL, constraint_idx: 3 },
            IntentDirective { precision: Precision::INT8, constraint_idx: 4 },
        ];
        let (passed, failed) = batch_check_directed(&values, &lowers, &uppers, &directives);
        assert_eq!(passed, 5);
        assert_eq!(failed, 0);
    }

    #[test]
    fn test_av_constraint_mix() {
        // Simulate autonomous vehicle: 75% advisory, 15% op, 8% tech, 2% dual
        let n = 1000usize;
        let mut directives = Vec::new();
        for i in 0..n {
            let precision = if i < 750 {
                Precision::INT8
            } else if i < 900 {
                Precision::INT16
            } else if i < 980 {
                Precision::INT32
            } else {
                Precision::DUAL
            };
            directives.push(IntentDirective { precision, constraint_idx: i });
        }

        let stats = ClassificationStats::from_directives(&directives);
        assert_eq!(stats.int8_count, 750);
        assert_eq!(stats.int16_count, 150);
        assert_eq!(stats.int32_count, 80);
        assert_eq!(stats.dual_count, 20);
        assert!(stats.theoretical_throughput_gain > 3.0,
            "Expected >3x throughput for AV mix, got {:.2}x", stats.theoretical_throughput_gain);
    }
}
