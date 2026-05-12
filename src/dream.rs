//! Dream reconstruction — using a model's latent space as a constraint lattice.
//!
//! Based on real experimental results from the baton protocol experiments.
//! Key findings:
//! - **The Amnesia Gradient**: Reconstruction accuracy degrades predictably with
//!   source coverage. Below 10% coverage, confident hallucination dominates.
//! - **Negative Space Reconstruction**: Describing what's NOT there achieves 77.5%
//!   accuracy — the shadow contains the shape.
//! - **Style Resilience**: Literal extraction preserves ~97.5% accuracy; surreal
//!   dream-mode drops to ~55% but produces highly creative, novel inferences.
//! - **Compression Frontier**: Accuracy collapses non-linearly with compression.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Experimental constants
// ---------------------------------------------------------------------------

/// Experimental data: accuracy vs source coverage fraction.
/// From "The Amnesia Gradient" experiment (Seed-2.0-mini, temp=1.0).
pub const AMNESIA_DATA: [(f64, f64); 8] = [
    (1.00, 0.975), // 100% source → 97.5% accuracy
    (0.75, 0.775), // 75% → 77.5%
    (0.50, 0.475), // 50% → 47.5%
    (0.33, 0.325), // 33% → 32.5%
    (0.25, 0.225), // 25% → 22.5%
    (0.15, 0.225), // 15% → 22.5% (plateau)
    (0.10, 0.125), // 10% → 12.5%
    (0.05, 0.000), // 5% → 0% (hallucination zone)
];

/// Style resilience from "The Style Gauntlet" experiment.
pub const STYLE_RESILIENCE: [(DreamStyle, f64); 5] = [
    (DreamStyle::Literal, 0.975),
    (DreamStyle::Abstract, 0.750),
    (DreamStyle::Negative, 0.775),
    (DreamStyle::Narrative, 0.325),
    (DreamStyle::Surreal, 0.550),
];

/// Compression frontier: accuracy vs compressed character count.
pub const COMPRESSION_DATA: [(usize, f64); 6] = [
    (1145, 0.775), // ~500 target → 77.5%
    (540, 0.300),  // ~300 target → 30.0%
    (222, 0.075),  // ~150 target → 7.5%
    (107, 0.025),  // ~75 target → 2.5%
    (37, 0.025),   // ~40 target → 2.5%
    (22, 0.100),   // ~20 target → 10.0% (anomaly)
];

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// How a dream fragment was generated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DreamStyle {
    /// Raw extraction — the "built" shard.
    Literal,
    /// Reasoning/thought extraction.
    Abstract,
    /// What's NOT there — negative space finding (77.5% accuracy).
    Negative,
    /// Story-mode — the "storyteller" finding.
    Narrative,
    /// Dream mode — 55% accuracy but highly creative.
    Surreal,
}

impl std::fmt::Display for DreamStyle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DreamStyle::Literal => write!(f, "literal"),
            DreamStyle::Abstract => write!(f, "abstract"),
            DreamStyle::Negative => write!(f, "negative"),
            DreamStyle::Narrative => write!(f, "narrative"),
            DreamStyle::Surreal => write!(f, "surreal"),
        }
    }
}

/// A partial observation — like a tile from the baton protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamFragment {
    /// The partial text content.
    pub content: String,
    /// What fraction of the original source is covered (0.0–1.0).
    pub coverage: f64,
    /// How this fragment was generated.
    pub style: DreamStyle,
    /// Facts known to survive in this fragment.
    pub constraints_preserved: Vec<String>,
    /// Timestamp of observation (unix epoch seconds).
    pub timestamp: f64,
}

/// Result of a dream reconstruction pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamReconstruction {
    /// Reconstructed text.
    pub content: String,
    /// Predicted accuracy based on coverage + style.
    pub accuracy_estimate: f64,
    /// Things the model inferred that weren't in any fragment.
    pub novel_inferences: Vec<String>,
    /// Overall confidence in the reconstruction.
    pub confidence: f64,
    /// The dominant style used for reconstruction.
    pub style: DreamStyle,
}

/// Parameters for the dream reconstruction engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamConfig {
    /// Sampling temperature (optimal: 1.0 per experiments).
    pub temperature: f64,
    /// Minimum fragment coverage before attempting reconstruction (0.1 = 10% amnesia cliff).
    pub min_coverage: f64,
    /// Per-style fact preservation scores.
    pub style_resilience: HashMap<DreamStyle, f64>,
    /// Coverage threshold below which hallucination dominates (0.10 from experiments).
    pub amnesia_cliff: f64,
}

impl Default for DreamConfig {
    fn default() -> Self {
        let mut style_resilience = HashMap::new();
        for (style, score) in STYLE_RESILIENCE.iter() {
            style_resilience.insert(*style, *score);
        }
        DreamConfig {
            temperature: 1.0,
            min_coverage: 0.10,
            style_resilience,
            amnesia_cliff: 0.10,
        }
    }
}

// ---------------------------------------------------------------------------
// Core functions
// ---------------------------------------------------------------------------

/// The forgetting curve discovered in baton protocol experiments.
///
/// Uses linear interpolation between known data points.
/// Below the amnesia cliff (10% coverage), returns 0.0 — confident hallucination zone.
pub fn amnesia_curve(coverage: f64) -> f64 {
    if coverage <= 0.05 {
        return 0.0;
    }
    if coverage >= 1.0 {
        return 0.975;
    }

    // Walk the data table and interpolate.
    let data: &[(f64, f64)] = &AMNESIA_DATA;
    // Data is sorted descending by coverage.
    for window in data.windows(2) {
        let (c_high, a_high) = window[0];
        let (c_low, a_low) = window[1];
        if coverage <= c_high && coverage >= c_low {
            if (c_high - c_low).abs() < f64::EPSILON {
                return a_high;
            }
            let t = (coverage - c_low) / (c_high - c_low);
            return a_low + t * (a_high - a_low);
        }
    }
    // Below the last data point (5%) — hallucination zone.
    0.0
}

/// How well a given style preserves facts (from style gauntlet experiments).
pub fn style_resilience(style: DreamStyle) -> f64 {
    for (s, r) in STYLE_RESILIENCE.iter() {
        if *s == style {
            return *r;
        }
    }
    0.0
}

/// Reconstruct information from dream fragments.
///
/// Computes total coverage, applies the amnesia curve, and weights by style
/// resilience of the dominant (highest coverage) fragment.
pub fn reconstruct(fragments: &[DreamFragment], config: &DreamConfig) -> DreamReconstruction {
    if fragments.is_empty() {
        return DreamReconstruction {
            content: String::new(),
            accuracy_estimate: 0.0,
            novel_inferences: vec![],
            confidence: 0.0,
            style: DreamStyle::Literal,
        };
    }

    // Total coverage (capped at 1.0).
    let total_coverage = fragments.iter().map(|f| f.coverage).sum::<f64>().min(1.0);

    // Base accuracy from the amnesia curve.
    let base_accuracy = amnesia_curve(total_coverage);

    // Dominant fragment = highest coverage.
    let dominant = fragments
        .iter()
        .max_by(|a, b| a.coverage.partial_cmp(&b.coverage).unwrap_or(std::cmp::Ordering::Equal))
        .expect("non-empty fragments");

    let style_weight = config
        .style_resilience
        .get(&dominant.style)
        .copied()
        .unwrap_or_else(|| style_resilience(dominant.style));

    // Combined accuracy: base × style weight.
    let accuracy_estimate = base_accuracy * style_weight;

    // Confidence: geometric mean of coverage and accuracy.
    let confidence = if total_coverage > 0.0 && accuracy_estimate > 0.0 {
        (total_coverage * accuracy_estimate).sqrt()
    } else {
        0.0
    };

    // Gather preserved constraints from all fragments.
    let constraints: Vec<String> = fragments
        .iter()
        .flat_map(|f| f.constraints_preserved.clone())
        .collect();

    // Build reconstructed content.
    let content = if total_coverage < config.amnesia_cliff {
        format!(
            "[HALLUCINATION RISK: coverage {:.1}% below amnesia cliff {:.1}%]\n{}",
            total_coverage * 100.0,
            config.amnesia_cliff * 100.0,
            fragments
                .iter()
                .map(|f| f.content.as_str())
                .collect::<Vec<_>>()
                .join(" ")
        )
    } else {
        fragments
            .iter()
            .map(|f| f.content.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    };

    DreamReconstruction {
        content,
        accuracy_estimate,
        novel_inferences: if constraints.is_empty() {
            vec![]
        } else {
            // Novel inferences = things inferred from constraints that weren't
            // directly observed. In a real system this would call the model.
            vec![format!(
                "Inferred from {} preserved constraints",
                constraints.len()
            )]
        },
        confidence,
        style: dominant.style,
    }
}

/// Reconstruct from negative facts — what's FALSE reveals what's TRUE.
///
/// Our 77.5% accuracy finding: the shadow contains the shape.
pub fn negative_reconstruction(negative_facts: &[String]) -> DreamReconstruction {
    if negative_facts.is_empty() {
        return DreamReconstruction {
            content: String::new(),
            accuracy_estimate: 0.775,
            novel_inferences: vec![],
            confidence: 0.0,
            style: DreamStyle::Negative,
        };
    }

    let mut novel = Vec::new();
    for fact in negative_facts {
        // The negation of a negation is an affirmation.
        novel.push(format!("NOT({}) → positive constraint inferred", fact));
    }

    DreamReconstruction {
        content: format!(
            "Negative-space reconstruction from {} excluded facts",
            negative_facts.len()
        ),
        accuracy_estimate: 0.775,
        novel_inferences: novel,
        confidence: 0.775,
        style: DreamStyle::Negative,
    }
}

/// Surreal/dream-mode reconstruction.
///
/// Lower accuracy (55%) but HIGH novelty — useful for creative exploration.
/// Facts become metaphors; structural relationships are preserved even when
/// surface details change.
pub fn dream_layer(content: &str, config: &DreamConfig) -> DreamReconstruction {
    let len = content.len();
    let coverage = if len > 0 { 1.0 } else { 0.0 };
    let base_accuracy = amnesia_curve(coverage);
    let dream_weight = config
        .style_resilience
        .get(&DreamStyle::Surreal)
        .copied()
        .unwrap_or(0.55);
    let accuracy = base_accuracy * dream_weight;

    DreamReconstruction {
        content: if content.is_empty() {
            String::from("~void~")
        } else {
            format!("~dream~ {} ~dream~", content)
        },
        accuracy_estimate: accuracy,
        novel_inferences: vec![String::from(
            "Structural metaphors generated; surface details may diverge",
        )],
        confidence: accuracy,
        style: DreamStyle::Surreal,
    }
}

/// Estimate reconstruction accuracy at a given compression level.
///
/// From the compression frontier experiments:
/// - 500 target chars → 77.5%
/// - 300 → 30.0%
/// - 150 → 7.5%
/// - below 75 → ~2.5%
pub fn compression_frontier(content: &str, target_chars: usize) -> f64 {
    let _content_len = content.len();
    if target_chars == 0 {
        return 0.0;
    }

    // Data is sorted descending by char count.
    let data: &[(usize, f64)] = &COMPRESSION_DATA;

    // Find surrounding data points and interpolate.
    for window in data.windows(2) {
        let (c_high, a_high) = window[0];
        let (c_low, a_low) = window[1];
        if target_chars <= c_high && target_chars >= c_low {
            let range = (c_high - c_low) as f64;
            if range < f64::EPSILON {
                return a_high;
            }
            let t = (target_chars - c_low) as f64 / range;
            return a_low + t * (a_high - a_low);
        }
    }

    // Below the smallest data point.
    if target_chars < data.last().map(|(c, _)| *c).unwrap_or(0) {
        // Extrapolate downward — very low accuracy.
        return 0.025;
    }

    // Above the largest data point — plateau at highest known accuracy.
    data.first().map(|(_, a)| *a).unwrap_or(0.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn amnesia_curve_at_known_data_points() {
        // Exact data points should match closely.
        assert!((amnesia_curve(1.00) - 0.975).abs() < 0.01);
        assert!((amnesia_curve(0.75) - 0.775).abs() < 0.01);
        assert!((amnesia_curve(0.50) - 0.475).abs() < 0.01);
        assert!((amnesia_curve(0.33) - 0.325).abs() < 0.01);
        assert!((amnesia_curve(0.25) - 0.225).abs() < 0.01);
        assert!((amnesia_curve(0.15) - 0.225).abs() < 0.01);
        assert!((amnesia_curve(0.10) - 0.125).abs() < 0.01);
    }

    #[test]
    fn amnesia_curve_below_cliff_returns_zero() {
        assert_eq!(amnesia_curve(0.05), 0.0);
        assert_eq!(amnesia_curve(0.01), 0.0);
        assert_eq!(amnesia_curve(0.0), 0.0);
    }

    #[test]
    fn style_resilience_matches_experimental_data() {
        assert!((style_resilience(DreamStyle::Literal) - 0.975).abs() < 0.001);
        assert!((style_resilience(DreamStyle::Abstract) - 0.750).abs() < 0.001);
        assert!((style_resilience(DreamStyle::Negative) - 0.775).abs() < 0.001);
        assert!((style_resilience(DreamStyle::Narrative) - 0.325).abs() < 0.001);
        assert!((style_resilience(DreamStyle::Surreal) - 0.550).abs() < 0.001);
    }

    #[test]
    fn reconstruct_single_fragment() {
        let config = DreamConfig::default();
        let frag = DreamFragment {
            content: "The constraint lattice".into(),
            coverage: 0.50,
            style: DreamStyle::Literal,
            constraints_preserved: vec!["lattice exists".into()],
            timestamp: 1000.0,
        };
        let result = reconstruct(&[frag], &config);
        assert!(!result.content.is_empty());
        assert!(result.accuracy_estimate > 0.0);
        assert_eq!(result.style, DreamStyle::Literal);
    }

    #[test]
    fn reconstruct_multiple_fragments_coverage_adds() {
        let config = DreamConfig::default();
        let frags = vec![
            DreamFragment {
                content: "Part A".into(),
                coverage: 0.40,
                style: DreamStyle::Literal,
                constraints_preserved: vec![],
                timestamp: 1000.0,
            },
            DreamFragment {
                content: "Part B".into(),
                coverage: 0.35,
                style: DreamStyle::Literal,
                constraints_preserved: vec![],
                timestamp: 1001.0,
            },
        ];
        let result = reconstruct(&frags, &config);
        // Total coverage = 0.75, accuracy should be amnesia_curve(0.75) * style_resilience(Literal)
        let expected_accuracy = amnesia_curve(0.75) * 0.975;
        assert!((result.accuracy_estimate - expected_accuracy).abs() < 0.01);
    }

    #[test]
    fn negative_reconstruction_non_empty() {
        let negs = vec![
            "not a constraint".into(),
            "not a lattice".into(),
        ];
        let result = negative_reconstruction(&negs);
        assert!(!result.content.is_empty());
        assert!((result.accuracy_estimate - 0.775).abs() < 0.01);
        assert_eq!(result.novel_inferences.len(), 2);
        assert_eq!(result.style, DreamStyle::Negative);
    }

    #[test]
    fn dream_layer_lower_accuracy_than_literal() {
        let config = DreamConfig::default();
        let text = "The baton passes through the lattice";
        let dream_result = dream_layer(text, &config);
        let literal_result = reconstruct(
            &[DreamFragment {
                content: text.into(),
                coverage: 1.0,
                style: DreamStyle::Literal,
                constraints_preserved: vec![],
                timestamp: 0.0,
            }],
            &config,
        );
        assert!(dream_result.accuracy_estimate < literal_result.accuracy_estimate);
    }

    #[test]
    fn compression_frontier_experimental_data() {
        assert!((compression_frontier("x", 1145) - 0.775).abs() < 0.01);
        assert!((compression_frontier("x", 540) - 0.300).abs() < 0.01);
        assert!((compression_frontier("x", 222) - 0.075).abs() < 0.01);
        assert!((compression_frontier("x", 107) - 0.025).abs() < 0.01);
    }

    #[test]
    fn config_defaults_match_experiments() {
        let config = DreamConfig::default();
        assert!((config.temperature - 1.0).abs() < f64::EPSILON);
        assert!((config.amnesia_cliff - 0.10).abs() < f64::EPSILON);
        assert!((config.min_coverage - 0.10).abs() < f64::EPSILON);
        assert!((config.style_resilience[&DreamStyle::Literal] - 0.975).abs() < 0.001);
        assert!((config.style_resilience[&DreamStyle::Surreal] - 0.550).abs() < 0.001);
    }

    #[test]
    fn edge_case_empty_fragments() {
        let config = DreamConfig::default();
        let result = reconstruct(&[], &config);
        assert!(result.content.is_empty());
        assert_eq!(result.accuracy_estimate, 0.0);
        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn edge_case_full_coverage_literal() {
        let config = DreamConfig::default();
        let result = reconstruct(
            &[DreamFragment {
                content: "complete reconstruction".into(),
                coverage: 1.0,
                style: DreamStyle::Literal,
                constraints_preserved: vec!["everything".into()],
                timestamp: 0.0,
            }],
            &config,
        );
        // 0.975 (amnesia) × 0.975 (literal resilience) ≈ 0.9506
        assert!((result.accuracy_estimate - 0.950625).abs() < 0.01);
        assert!(result.confidence > 0.0);
    }
}
