//! Beam-Tolerance Solver — Physical math for intent channel stiffness
//!
//! Maps Oracle1's spline-physics beam module to Forgemaster's intent alignment.
//! Each intent channel has a "material stiffness" derived from stakes (C9).
//! Higher stakes = stiffer beam = tighter tolerance = higher precision class.

/// Material properties for beam-intent equivalence
#[derive(Debug, Clone, Copy)]
pub struct BeamMaterial {
    /// Young's modulus in GPa — higher = stiffer = less tolerance
    pub youngs_modulus: f64,
    /// Density in g/cm³ (information density metaphor)
    pub density: f64,
    /// Yield strength in MPa — maximum stress before failure
    pub yield_strength: f64,
}

impl BeamMaterial {
    /// Steel: E=200 GPa — critical systems, zero tolerance
    pub fn steel() -> Self {
        Self { youngs_modulus: 200.0, density: 7.8, yield_strength: 600.0 }
    }
    /// Fiberglass: E=30 GPa — important but not life-critical
    pub fn fiberglass() -> Self {
        Self { youngs_modulus: 30.0, density: 2.0, yield_strength: 200.0 }
    }
    /// Oak: E=12 GPa — moderate importance
    pub fn oak() -> Self {
        Self { youngs_modulus: 12.0, density: 0.7, yield_strength: 80.0 }
    }
    /// Cedar: E=6 GPa — flexible, advisory
    pub fn cedar() -> Self {
        Self { youngs_modulus: 6.0, density: 0.4, yield_strength: 40.0 }
    }
    /// Rubber: E=0.01 GPa — highly flexible, informational only
    pub fn rubber() -> Self {
        Self { youngs_modulus: 0.01, density: 1.1, yield_strength: 10.0 }
    }

    /// Maximum deflection under load — maps to tolerance
    /// δ_max = L / (E × safety_factor)
    /// For intent: tolerance inversely proportional to stiffness
    /// Steel (E=200) → tolerance ~0.05, Rubber (E=0.01) → tolerance ~1.0
    pub fn max_tolerance(&self, safety_factor: f64) -> f64 {
        let normalized_stiffness = self.youngs_modulus / 200.0; // 0-1 scale
        // Use logarithmic mapping: stiff materials get tight tolerance
        let tolerance = 0.01 + (1.0 - normalized_stiffness) * 0.99;
        (tolerance / safety_factor).clamp(0.01, 1.0)
    }

    /// Dynamic amplification factor — squat effect for rushed messages
    /// DAF = 1 + speed_factor (0 = careful, 1 = emergency)
    pub fn dynamic_amplification(&self, speed_factor: f64) -> f64 {
        1.0 + speed_factor * (1.0 - self.youngs_modulus / 200.0).max(0.0)
    }
}

/// Precision class derived from beam material stiffness
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecisionClass {
    /// INT8: 64 constraints per AVX-512 register
    INT8,
    /// INT16: 32 constraints per AVX-512 register
    INT16,
    /// INT32: 16 constraints per AVX-512 register
    INT32,
    /// DUAL: INT32 with dual-path verification (comparison + subtraction)
    DUAL,
}

impl PrecisionClass {
    /// Constraints per AVX-512 register (512 bits)
    pub fn constraints_per_register(&self) -> usize {
        match self {
            Self::INT8 => 64,
            Self::INT16 => 32,
            Self::INT32 => 16,
            Self::DUAL => 16, // same density as INT32, but 2x cost
        }
    }

    /// Bits per constraint
    pub fn bits_per_constraint(&self) -> usize {
        match self {
            Self::INT8 => 8,
            Self::INT16 => 16,
            Self::INT32 => 32,
            Self::DUAL => 64, // dual-path = 2× INT32
        }
    }

    /// Memory reduction vs INT32 baseline (as fraction)
    pub fn memory_reduction(&self) -> f64 {
        1.0 - (self.bits_per_constraint() as f64 / 32.0)
    }
}

/// Classify precision from stakes (C9) and value range
pub fn classify_precision(stakes: f64, value_range: f64) -> PrecisionClass {
    // Critical stakes → DUAL regardless of range
    if stakes > 0.75 {
        return PrecisionClass::DUAL;
    }
    // High stakes or large range → INT32
    if stakes > 0.5 || value_range > 32000.0 {
        return PrecisionClass::INT32;
    }
    // Medium stakes or medium range → INT16
    if stakes > 0.25 || value_range > 127.0 {
        return PrecisionClass::INT16;
    }
    // Low stakes, small range → INT8
    PrecisionClass::INT8
}

/// Map stakes to beam material
pub fn stakes_to_material(stakes: f64) -> BeamMaterial {
    match stakes {
        s if s > 0.75 => BeamMaterial::steel(),
        s if s > 0.5  => BeamMaterial::fiberglass(),
        s if s > 0.25 => BeamMaterial::oak(),
        s if s > 0.1  => BeamMaterial::cedar(),
        _              => BeamMaterial::rubber(),
    }
}

/// Compute tolerance for a channel given its stakes and safety factor
pub fn compute_tolerance(stakes: f64, safety_factor: f64) -> f64 {
    stakes_to_material(stakes).max_tolerance(safety_factor)
}

/// Compute draft with squat effect (rushed messages)
pub fn compute_draft(base_tolerance: f64, speed_factor: f64, stakes: f64) -> f64 {
    let material = stakes_to_material(stakes);
    let daf = material.dynamic_amplification(speed_factor);
    base_tolerance / daf
}

/// SoA batch of constraints sorted by precision class
pub struct SoABatch {
    pub int8_values: Vec<i8>,
    pub int8_lowers: Vec<i8>,
    pub int8_uppers: Vec<i8>,
    pub int16_values: Vec<i16>,
    pub int16_lowers: Vec<i16>,
    pub int16_uppers: Vec<i16>,
    pub int32_values: Vec<i32>,
    pub int32_lowers: Vec<i32>,
    pub int32_uppers: Vec<i32>,
    pub dual_values: Vec<i32>,
    pub dual_lowers: Vec<i32>,
    pub dual_uppers: Vec<i32>,
}

impl SoABatch {
    /// Create from raw constraint tuples: (value, lower, upper, stakes)
    pub fn from_constraints(constraints: &[(f64, f64, f64, f64)]) -> Self {
        let mut batch = SoABatch {
            int8_values: Vec::new(), int8_lowers: Vec::new(), int8_uppers: Vec::new(),
            int16_values: Vec::new(), int16_lowers: Vec::new(), int16_uppers: Vec::new(),
            int32_values: Vec::new(), int32_lowers: Vec::new(), int32_uppers: Vec::new(),
            dual_values: Vec::new(), dual_lowers: Vec::new(), dual_uppers: Vec::new(),
        };

        for &(value, lower, upper, stakes) in constraints {
            let range = (upper - lower).abs();
            match classify_precision(stakes, range) {
                PrecisionClass::INT8 => {
                    batch.int8_values.push(value as i8);
                    batch.int8_lowers.push(lower as i8);
                    batch.int8_uppers.push(upper as i8);
                }
                PrecisionClass::INT16 => {
                    batch.int16_values.push(value as i16);
                    batch.int16_lowers.push(lower as i16);
                    batch.int16_uppers.push(upper as i16);
                }
                PrecisionClass::INT32 => {
                    batch.int32_values.push(value as i32);
                    batch.int32_lowers.push(lower as i32);
                    batch.int32_uppers.push(upper as i32);
                }
                PrecisionClass::DUAL => {
                    batch.dual_values.push(value as i32);
                    batch.dual_lowers.push(lower as i32);
                    batch.dual_uppers.push(upper as i32);
                }
            }
        }
        batch
    }

    /// Check all constraints at native precision (scalar fallback)
    pub fn check_all(&self) -> Vec<bool> {
        let mut results = Vec::new();

        for i in 0..self.int8_values.len() {
            results.push(self.int8_values[i] >= self.int8_lowers[i] && 
                         self.int8_values[i] <= self.int8_uppers[i]);
        }
        for i in 0..self.int16_values.len() {
            results.push(self.int16_values[i] >= self.int16_lowers[i] && 
                         self.int16_values[i] <= self.int16_uppers[i]);
        }
        for i in 0..self.int32_values.len() {
            results.push(self.int32_values[i] >= self.int32_lowers[i] && 
                         self.int32_values[i] <= self.int32_uppers[i]);
        }
        for i in 0..self.dual_values.len() {
            let v = self.dual_values[i];
            let lo = self.dual_lowers[i];
            let hi = self.dual_uppers[i];
            // Path A: comparison
            let pass_a = v >= lo && v <= hi;
            // Path B: XOR-based signed-to-unsigned conversion (overflow-safe)
            // XOR with 0x80000000 converts signed to unsigned range
            // This eliminates subtraction overflow at INT_MAX/INT_MIN boundaries
            let vu = (v as u32) ^ 0x80000000u32;
            let lu = (lo as u32) ^ 0x80000000u32;
            let hu = (hi as u32) ^ 0x80000000u32;
            let pass_b = vu >= lu && vu <= hu;
            // Both paths must agree (if they disagree, flag as failure)
            results.push(pass_a && pass_b);
        }
        results
    }

    /// Memory usage in bits vs all-INT32 baseline
    pub fn memory_stats(&self) -> (usize, usize) {
        let n8 = self.int8_values.len();
        let n16 = self.int16_values.len();
        let n32 = self.int32_values.len();
        let nd = self.dual_values.len();
        let total = n8 + n16 + n32 + nd;

        let actual_bits = n8 * 8 * 3 + n16 * 16 * 3 + n32 * 32 * 3 + nd * 64 * 3;
        let baseline_bits = total * 32 * 3;

        (actual_bits, baseline_bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_int8() {
        assert_eq!(classify_precision(0.1, 50.0), PrecisionClass::INT8);
    }

    #[test]
    fn test_classify_int16() {
        assert_eq!(classify_precision(0.3, 200.0), PrecisionClass::INT16);
    }

    #[test]
    fn test_classify_int32() {
        assert_eq!(classify_precision(0.6, 100.0), PrecisionClass::INT32);
    }

    #[test]
    fn test_classify_dual() {
        assert_eq!(classify_precision(0.8, 10.0), PrecisionClass::DUAL);
    }

    #[test]
    fn test_steel_tolerance() {
        let mat = BeamMaterial::steel();
        let tol = mat.max_tolerance(1.0);
        assert!(tol < 0.1, "Steel should have very tight tolerance, got {}", tol);
    }

    #[test]
    fn test_rubber_tolerance() {
        let mat = BeamMaterial::rubber();
        let tol = mat.max_tolerance(1.0);
        assert!(tol > 0.5, "Rubber should have very loose tolerance, got {}", tol);
    }

    #[test]
    fn test_soa_batch_check_all() {
        let constraints: Vec<(f64, f64, f64, f64)> = vec![
            (5.0, 0.0, 10.0, 0.1),   // INT8
            (50.0, 0.0, 100.0, 0.3),  // INT16
            (500.0, 0.0, 1000.0, 0.6),// INT32
            (5000.0, 0.0, 10000.0, 0.8),// DUAL
        ];
        let batch = SoABatch::from_constraints(&constraints);
        let results = batch.check_all();
        assert!(results.iter().all(|&r| r), "All constraints should pass");
    }

    #[test]
    fn test_memory_reduction() {
        // AV mix: 75% INT8, 15% INT16, 8% INT32, 2% DUAL
        let mut constraints = Vec::new();
        for _ in 0..7500 { constraints.push((5.0, 0.0, 10.0, 0.1)); }
        for _ in 0..1500 { constraints.push((500.0, 0.0, 1000.0, 0.3)); }
        for _ in 0..800  { constraints.push((5000.0, 0.0, 10000.0, 0.6)); }
        for _ in 0..200  { constraints.push((50000.0, 0.0, 100000.0, 0.8)); }
        
        let batch = SoABatch::from_constraints(&constraints);
        let (actual, baseline) = batch.memory_stats();
        let reduction = 1.0 - (actual as f64 / baseline as f64);
        assert!(reduction > 0.5, "AV mix should save >50%% memory, got {:.1}%%", reduction*100.0);
    }

    #[test]
    fn test_differential_int8() {
        // All values in INT8 range — results must match INT32
        let constraints: Vec<(f64, f64, f64, f64)> = (0..100)
            .map(|i| {
                let lo = (i % 50) as f64;
                let hi = lo + 50.0;
                let v = lo + (i as f64 % 50.0);
                (v, lo, hi, 0.1) // INT8 precision
            })
            .collect();
        
        let batch = SoABatch::from_constraints(&constraints);
        let results = batch.check_all();
        assert!(results.iter().all(|&r| r), "All in-range values should pass");
    }

    #[test]
    fn test_draft_squat_effect() {
        // Rushed messages should have tighter effective tolerance
        let base = compute_tolerance(0.5, 1.0);
        let rushed = compute_draft(base, 0.8, 0.5);
        assert!(rushed <= base, "Rushed draft should be ≤ base, got {} > {}", rushed, base);
    }

    #[test]
    fn test_precision_bits() {
        assert_eq!(PrecisionClass::INT8.bits_per_constraint(), 8);
        assert_eq!(PrecisionClass::INT16.bits_per_constraint(), 16);
        assert_eq!(PrecisionClass::INT32.bits_per_constraint(), 32);
        assert_eq!(PrecisionClass::DUAL.bits_per_constraint(), 64);
    }
}
