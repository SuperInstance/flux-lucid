//! flux-lucid: Unified constraint theory ecosystem.
//!
//! Single dependency that pulls together:
//! - **constraint-theory-llvm**: CDCL → LLVM IR → AVX-512 compilation
//! - **holonomy-consensus**: GL(9) zero-holonomy consensus for fleet coordination
//! - **9-channel intent encoding**: The A2A polyglot communication framework
//!
//! # The Navigation Metaphors
//!
//! This library is built on five principles from nautical navigation:
//!
//! 1. **Splines in the Ether**: The 9 communication channels are anchor points
//!    on a continuous intent curve. The curve between them is irreducible.
//! 2. **Fair Curve First**: Sight the intent first, find measurements second.
//! 3. **Where the Rocks Aren't**: Negative knowledge (absence of danger) is primary.
//! 4. **Draft Determines Truth**: The same message is safe or deadly depending on
//!    the receiver's context depth.
//! 5. **Speed Beats Truth**: In real-time domains, survival forbids accurate models.
//!    A satisficer in 50ms beats an optimizer in 2000ms.

pub mod intent;
pub mod intent_compilation;
pub mod intent_emitter;
pub mod beam_tolerance;
pub mod soa_emitter;
pub mod navigation;
pub mod head_direction;

// Re-export core types from sub-crates
pub use constraint_theory_llvm as llvm;
pub use holonomy_consensus as consensus;

/// The 9 communication channels.
///
/// These are the Pythagorean anchor points of the intent curve.
/// The actual intent flows between them — continuous, irreducible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Channel {
    /// C1: What are we talking about?
    Boundary,
    /// C2: How do pieces connect?
    Pattern,
    /// C3: What's happening over time?
    Process,
    /// C4: How sure am I?
    Knowledge,
    /// C5: Who cares and why?
    Social,
    /// C6: What's really being said?
    DeepStructure,
    /// C7: What tools are available?
    Instrument,
    /// C8: What model of thought?
    Paradigm,
    /// C9: What matters vs what doesn't?
    Stakes,
}

impl Channel {
    /// All 9 channels in order.
    pub fn all() -> &'static [Channel; 9] {
        &[
            Channel::Boundary,
            Channel::Pattern,
            Channel::Process,
            Channel::Knowledge,
            Channel::Social,
            Channel::DeepStructure,
            Channel::Instrument,
            Channel::Paradigm,
            Channel::Stakes,
        ]
    }

    /// The polyglot question this channel answers.
    pub fn question(&self) -> &'static str {
        match self {
            Channel::Boundary => "What are we talking about?",
            Channel::Pattern => "How do pieces connect?",
            Channel::Process => "What's happening over time?",
            Channel::Knowledge => "How sure am I?",
            Channel::Social => "Who cares and why?",
            Channel::DeepStructure => "What's really being said?",
            Channel::Instrument => "What tools are available?",
            Channel::Paradigm => "What model of thought?",
            Channel::Stakes => "What matters vs what doesn't?",
        }
    }

    /// Short label for this channel.
    pub fn label(&self) -> &'static str {
        match self {
            Channel::Boundary => "Boundary",
            Channel::Pattern => "Pattern",
            Channel::Process => "Process",
            Channel::Knowledge => "Knowledge",
            Channel::Social => "Social",
            Channel::DeepStructure => "Deep Structure",
            Channel::Instrument => "Instrument",
            Channel::Paradigm => "Paradigm",
            Channel::Stakes => "Stakes",
        }
    }
}

/// A 9-dimensional intent vector.
///
/// Each dimension is a f64 in [0, 1] representing the salience of that channel.
/// This is an anchor point on the continuous intent curve — the curve between
/// any two profiles is irreducible.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IntentVector {
    /// Salience values for each of the 9 channels [0, 1].
    pub values: [f64; 9],
    /// Per-channel tolerance (how much deviation is acceptable).
    pub tolerance: [f64; 9],
}

impl IntentVector {
    /// Create a zero intent vector with default tolerance 0.5.
    pub fn zero() -> Self {
        Self {
            values: [0.0; 9],
            tolerance: [0.5; 9],
        }
    }

    /// Set a channel's salience value.
    pub fn set(&mut self, channel: Channel, value: f64) {
        self.values[channel as usize] = value.clamp(0.0, 1.0);
    }

    /// Set a channel's tolerance.
    pub fn set_tolerance(&mut self, channel: Channel, tol: f64) {
        self.tolerance[channel as usize] = tol.max(0.001);
    }

    /// Get a channel's value.
    pub fn get(&self, channel: Channel) -> f64 {
        self.values[channel as usize]
    }

    /// Cosine similarity between two intent vectors.
    pub fn cosine_similarity(&self, other: &IntentVector) -> f64 {
        let dot: f64 = self.values.iter().zip(other.values.iter()).map(|(a, b)| a * b).sum();
        let norm_a: f64 = self.values.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = other.values.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 { 0.0 } else { dot / (norm_a * norm_b) }
    }

    /// Euclidean distance between two intent vectors.
    pub fn euclidean_distance(&self, other: &IntentVector) -> f64 {
        self.values.iter().zip(other.values.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Draft: how deep this intent's requirements are.
    /// Higher draft = more shared context needed for safe communication.
    pub fn draft(&self) -> f64 {
        let total: f64 = self.values.iter().zip(self.tolerance.iter())
            .map(|(v, t)| v / t.max(0.001))
            .sum();
        (total / 9.0).min(2.0) / 10.0
    }

    /// The dominant channel (highest salience).
    pub fn dominant_channel(&self) -> Channel {
        let idx = self.values.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        Channel::all()[idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nine_channels() {
        assert_eq!(Channel::all().len(), 9);
    }

    #[test]
    fn test_intent_zero() {
        let v = IntentVector::zero();
        assert_eq!(v.values, [0.0; 9]);
        assert_eq!(v.tolerance, [0.5; 9]);
    }

    #[test]
    fn test_intent_set_get() {
        let mut v = IntentVector::zero();
        v.set(Channel::Stakes, 0.9);
        assert!((v.get(Channel::Stakes) - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity() {
        let mut a = IntentVector::zero();
        let mut b = IntentVector::zero();
        a.set(Channel::Stakes, 0.9);
        a.set(Channel::Process, 0.8);
        b.set(Channel::Stakes, 0.85);
        b.set(Channel::Process, 0.75);
        let sim = a.cosine_similarity(&b);
        assert!(sim > 0.99);
    }

    #[test]
    fn test_dominant_channel() {
        let mut v = IntentVector::zero();
        v.set(Channel::Social, 0.95);
        assert_eq!(v.dominant_channel(), Channel::Social);
    }
}
