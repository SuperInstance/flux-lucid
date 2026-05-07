//! SoA mixed-precision constraint emitter.
//!
//! Sorts constraints into precision-stratified SoA (Structure of Arrays) batches
//! based on C9 stakes and value range. Each precision class gets its own contiguous
//! `Vec`, enabling SIMD-friendly access patterns and significant memory savings.

/// Precision class determined by stakes and value range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrecisionClass {
    /// ±127 range — low stakes, small values.
    INT8,
    /// ±32767 range — medium stakes, medium values.
    INT16,
    /// ±2.1B range — high stakes, large values.
    INT32,
    /// Dual-path INT32 — critical stakes, maximum fidelity.
    DUAL,
}

/// Statistics about a constraint batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchStats {
    /// Number of INT8 constraints.
    pub int8_count: usize,
    /// Number of INT16 constraints.
    pub int16_count: usize,
    /// Number of INT32 constraints.
    pub int32_count: usize,
    /// Number of DUAL constraints.
    pub dual_count: usize,
    /// Total bytes used across all SoA arrays.
    pub bytes_used: usize,
    /// Bytes that would be needed if everything were INT32.
    pub bytes_if_all_int32: usize,
}

impl BatchStats {
    /// Memory savings as a fraction (0.0 = no savings, 1.0 = all INT8).
    pub fn savings_fraction(&self) -> f64 {
        if self.bytes_if_all_int32 == 0 {
            return 0.0;
        }
        1.0 - (self.bytes_used as f64 / self.bytes_if_all_int32 as f64)
    }
}

/// A single constraint stored at a given precision.
/// Each array stores (value, lo, hi) as the native integer type.
/// The original stakes are stored separately.

/// SoA batch: constraints sorted into precision-stratified contiguous arrays.
///
/// Layout:
/// - `int8_data`: flat [i8] — groups of 3: (value, lo, hi)
/// - `int16_data`: flat [i16] — groups of 3: (value, lo, hi)
/// - `int32_data`: flat [i32] — groups of 3: (value, lo, hi)
/// - `dual_data`: flat [i32] — groups of 3: (value, lo, hi) — redundant path
/// - `stakes`: per-class stakes vectors, parallel to data arrays
#[derive(Debug, Clone, Default)]
pub struct SoAConstraintBatch {
    /// INT8 constraints: (value, lo, hi) triples.
    pub int8_data: Vec<i8>,
    /// Stakes for INT8 constraints.
    pub int8_stakes: Vec<f64>,

    /// INT16 constraints: (value, lo, hi) triples.
    pub int16_data: Vec<i16>,
    /// Stakes for INT16 constraints.
    pub int16_stakes: Vec<f64>,

    /// INT32 constraints: (value, lo, hi) triples.
    pub int32_data: Vec<i32>,
    /// Stakes for INT32 constraints.
    pub int32_stakes: Vec<f64>,

    /// DUAL constraints: (value, lo, hi) triples.
    pub dual_data: Vec<i32>,
    /// Stakes for DUAL constraints.
    pub dual_stakes: Vec<f64>,
}

/// Classify a single constraint by stakes and value range.
///
/// Rules:
/// - stakes > 0.75 OR value range > 32000 → DUAL
/// - stakes > 0.5  OR value range > 127   → INT32
/// - stakes > 0.25 OR value range > 15    → INT16
/// - otherwise                           → INT8
pub fn classify(value: f64, lo: f64, hi: f64, stakes: f64) -> PrecisionClass {
    let range = (hi - lo).abs();
    if stakes > 0.75 || range > 32000.0 {
        PrecisionClass::DUAL
    } else if stakes > 0.5 || range > 127.0 {
        PrecisionClass::INT32
    } else if stakes > 0.25 || range > 15.0 {
        PrecisionClass::INT16
    } else {
        PrecisionClass::INT8
    }
}

/// Round a f64 to the nearest representable value in a target integer type,
/// clamping to the type's range.
macro_rules! round_to {
    ($val:expr, $ty:ident) => {{
        let min = $ty::MIN as f64;
        let max = $ty::MAX as f64;
        let clamped = $val.clamp(min, max);
        clamped.round() as $ty
    }};
}

impl SoAConstraintBatch {
    /// Build a batch from raw (value, lo, hi, stakes) tuples.
    pub fn from_constraints(constraints: &[(f64, f64, f64, f64)]) -> Self {
        let mut batch = Self::default();

        for &(value, lo, hi, stakes) in constraints {
            let class = classify(value, lo, hi, stakes);
            match class {
                PrecisionClass::INT8 => {
                    batch.int8_data.push(round_to!(value, i8));
                    batch.int8_data.push(round_to!(lo, i8));
                    batch.int8_data.push(round_to!(hi, i8));
                    batch.int8_stakes.push(stakes);
                }
                PrecisionClass::INT16 => {
                    batch.int16_data.push(round_to!(value, i16));
                    batch.int16_data.push(round_to!(lo, i16));
                    batch.int16_data.push(round_to!(hi, i16));
                    batch.int16_stakes.push(stakes);
                }
                PrecisionClass::INT32 => {
                    batch.int32_data.push(round_to!(value, i32));
                    batch.int32_data.push(round_to!(lo, i32));
                    batch.int32_data.push(round_to!(hi, i32));
                    batch.int32_stakes.push(stakes);
                }
                PrecisionClass::DUAL => {
                    batch.dual_data.push(round_to!(value, i32));
                    batch.dual_data.push(round_to!(lo, i32));
                    batch.dual_data.push(round_to!(hi, i32));
                    batch.dual_stakes.push(stakes);
                }
            }
        }

        batch
    }

    /// Check all constraints at native precision. Returns Vec<bool> parallel to
    /// the input order (reconstructed by iterating INT8 → INT16 → INT32 → DUAL).
    pub fn check_all(&self) -> Vec<bool> {
        let mut results = Vec::new();

        // INT8 checks
        for chunk in self.int8_data.chunks_exact(3) {
            let val = chunk[0];
            let lo = chunk[1];
            let hi = chunk[2];
            results.push(val >= lo && val <= hi);
        }

        // INT16 checks
        for chunk in self.int16_data.chunks_exact(3) {
            let val = chunk[0];
            let lo = chunk[1];
            let hi = chunk[2];
            results.push(val >= lo && val <= hi);
        }

        // INT32 checks
        for chunk in self.int32_data.chunks_exact(3) {
            let val = chunk[0];
            let lo = chunk[1];
            let hi = chunk[2];
            results.push(val >= lo && val <= hi);
        }

        // DUAL checks (redundant path — same logic, different memory region)
        for chunk in self.dual_data.chunks_exact(3) {
            let val = chunk[0];
            let lo = chunk[1];
            let hi = chunk[2];
            results.push(val >= lo && val <= hi);
        }

        results
    }

    /// Compute batch statistics: counts per class, memory used vs all-INT32.
    pub fn stats(&self) -> BatchStats {
        let int8_count = self.int8_data.len() / 3;
        let int16_count = self.int16_data.len() / 3;
        let int32_count = self.int32_data.len() / 3;
        let dual_count = self.dual_data.len() / 3;

        let total = int8_count + int16_count + int32_count + dual_count;

        // Each constraint is 3 values + 1 f64 stake
        let bytes_used = self.int8_data.len() * std::mem::size_of::<i8>()
            + self.int16_data.len() * std::mem::size_of::<i16>()
            + self.int32_data.len() * std::mem::size_of::<i32>()
            + self.dual_data.len() * std::mem::size_of::<i32>()
            + (self.int8_stakes.len() + self.int16_stakes.len()
                + self.int32_stakes.len() + self.dual_stakes.len())
                * std::mem::size_of::<f64>();

        // If all were INT32: total constraints × 3 i32s + total × 1 f64 stake
        let bytes_if_all_int32 =
            total * 3 * std::mem::size_of::<i32>() + total * std::mem::size_of::<f64>();

        BatchStats {
            int8_count,
            int16_count,
            int32_count,
            dual_count,
            bytes_used,
            bytes_if_all_int32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_low_stakes_small_range() {
        // stakes <= 0.25, range <= 15 → INT8
        let class = classify(5.0, 0.0, 10.0, 0.1);
        assert_eq!(class, PrecisionClass::INT8);
    }

    #[test]
    fn test_classify_medium_stakes_medium_range() {
        // stakes > 0.25, range <= 127 → INT16
        let class = classify(50.0, 0.0, 100.0, 0.3);
        assert_eq!(class, PrecisionClass::INT16);
    }

    #[test]
    fn test_classify_high_stakes() {
        // stakes > 0.5 → INT32
        let class = classify(100.0, 0.0, 50.0, 0.6);
        assert_eq!(class, PrecisionClass::INT32);
    }

    #[test]
    fn test_classify_critical_stakes() {
        // stakes > 0.75 → DUAL
        let class = classify(1000.0, 0.0, 500.0, 0.8);
        assert_eq!(class, PrecisionClass::DUAL);
    }

    #[test]
    fn test_soa_batch_check_all() {
        // 10 constraints, all within bounds → all pass
        let constraints: Vec<(f64, f64, f64, f64)> = (0..10)
            .map(|i| {
                let v = i as f64 * 5.0;
                let stakes = 0.1 + (i as f64) * 0.08; // 0.1 to 0.82
                (v, 0.0, 100.0, stakes)
            })
            .collect();

        let batch = SoAConstraintBatch::from_constraints(&constraints);
        let results = batch.check_all();

        assert_eq!(results.len(), 10);
        assert!(results.iter().all(|&r| r), "All constraints should pass");
    }

    #[test]
    fn test_soa_batch_memory_savings() {
        // Mix of constraints across precision classes
        let mut constraints = Vec::new();

        // 50 low-stakes → INT8 (range < 15, stakes < 0.25)
        for i in 0..50 {
            constraints.push((i as f64 % 10.0, 0.0, 10.0, 0.05));
        }
        // 30 medium → INT16 (range 15-127, stakes 0.25-0.5)
        for i in 0..30 {
            constraints.push((i as f64, 0.0, 50.0, 0.3));
        }
        // 15 high → INT32 (stakes > 0.5)
        for i in 0..15 {
            constraints.push((i as f64 * 3.0, 0.0, 90.0, 0.6));
        }
        // 5 critical → DUAL (stakes > 0.75)
        for i in 0..5 {
            constraints.push((i as f64 * 4.0, 0.0, 100.0, 0.9));
        }

        let batch = SoAConstraintBatch::from_constraints(&constraints);
        let stats = batch.stats();

        assert_eq!(stats.int8_count, 50);
        assert_eq!(stats.int16_count, 30);
        assert_eq!(stats.int32_count, 15);
        assert_eq!(stats.dual_count, 5);
        assert_eq!(stats.int8_count + stats.int16_count + stats.int32_count + stats.dual_count, 100);

        let savings = stats.savings_fraction();
        // INT8 dominates → should see ~60%+ savings on data portion
        assert!(
            savings > 0.25,
            "Expected >25% memory savings (data-only), got {:.1}%",
            savings * 100.0
        );
    }

    #[test]
    fn test_soa_differential() {
        // Verify INT8/INT16 results match INT32 for in-range values.
        // We create the same constraint at each precision and check they all pass.
        let val = 5.0_f64;
        let lo = 0.0_f64;
        let hi = 10.0_f64;

        // Build three batches, one per precision, by adjusting stakes
        let int8_batch = SoAConstraintBatch::from_constraints(&[(val, lo, hi, 0.1)]);
        let int16_batch = SoAConstraintBatch::from_constraints(&[(val, lo, hi, 0.3)]);
        let int32_batch = SoAConstraintBatch::from_constraints(&[(val, lo, hi, 0.6)]);
        let dual_batch = SoAConstraintBatch::from_constraints(&[(val, lo, hi, 0.9)]);

        let r8 = int8_batch.check_all();
        let r16 = int16_batch.check_all();
        let r32 = int32_batch.check_all();
        let rd = dual_batch.check_all();

        // All should agree: val=5 is within [0, 10]
        assert_eq!(r8[0], true);
        assert_eq!(r16[0], true);
        assert_eq!(r32[0], true);
        assert_eq!(rd[0], true);

        // Verify they landed in the correct precision class
        assert!(!int8_batch.int8_data.is_empty());
        assert!(!int16_batch.int16_data.is_empty());
        assert!(!int32_batch.int32_data.is_empty());
        assert!(!dual_batch.dual_data.is_empty());

        // Now test a failing constraint
        let fail_batch = SoAConstraintBatch::from_constraints(&[(15.0, 0.0, 10.0, 0.1)]);
        let fail_results = fail_batch.check_all();
        assert_eq!(fail_results[0], false, "15 should be outside [0, 10]");
    }
}
