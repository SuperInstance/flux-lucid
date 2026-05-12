//! Intent encoding and alignment utilities.

use crate::{Channel, IntentVector};

/// Encode a simple intent description into a 9-channel profile.
///
/// This is a heuristic encoder. For production use, replace with
/// a model-based encoder.
pub fn encode_intent(description: &str) -> IntentVector {
    let mut profile = IntentVector::zero();
    let lower = description.to_lowercase();

    // Heuristic keyword matching
    let patterns: &[(&[&str], Channel, f64)] = &[
        (
            &["deadline", "urgent", "asap", "hurry", "rush"],
            Channel::Stakes,
            0.9,
        ),
        (
            &["risk", "danger", "unsafe", "critical", "hazard"],
            Channel::Stakes,
            0.95,
        ),
        (
            &["safety", "safe", "verify", "certif", "compliance"],
            Channel::Boundary,
            0.9,
        ),
        (
            &["team", "together", "collaborate", "we need"],
            Channel::Social,
            0.9,
        ),
        (
            &["code", "api", "system", "algorithm", "implement"],
            Channel::Pattern,
            0.85,
        ),
        (
            &["research", "study", "hypothesis", "experiment"],
            Channel::Paradigm,
            0.9,
        ),
        (
            &["deploy", "ship", "release", "launch"],
            Channel::Process,
            0.85,
        ),
        (
            &["design", "architect", "plan", "blueprint"],
            Channel::DeepStructure,
            0.85,
        ),
    ];

    let mut matched = false;
    for (keywords, channel, weight) in patterns {
        if keywords.iter().any(|kw| lower.contains(kw)) {
            profile.set(*channel, *weight);
            matched = true;
        }
    }

    if !matched {
        profile.set(Channel::Boundary, 0.5);
        profile.set(Channel::Stakes, 0.4);
    }

    profile
}

/// Check alignment between sender and receiver intent vectors.
///
/// Uses the draft-tolerance equation:
/// Communication Tolerance = Receiver Context - Sender Draft
pub struct AlignmentReport {
    pub cosine_similarity: f64,
    pub euclidean_distance: f64,
    pub draft_margin: f64,
    pub is_safe: bool,
    pub warnings: Vec<String>,
}

impl std::fmt::Display for AlignmentReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let status = if self.is_safe {
            "✓ SAFE"
        } else {
            "✗ GROUNDED"
        };
        write!(
            f,
            "{} (sim={:.3}, dist={:.3}, margin={:.3})",
            status, self.cosine_similarity, self.euclidean_distance, self.draft_margin
        )
    }
}

/// Check alignment between sender and receiver.
pub fn check_alignment(sender: &IntentVector, receiver: &IntentVector) -> AlignmentReport {
    let cosine = sender.cosine_similarity(receiver);
    let distance = sender.euclidean_distance(receiver);

    let mut warnings = Vec::new();
    for (i, ch) in Channel::all().iter().enumerate() {
        let dist = (sender.values[i] - receiver.values[i]).abs();
        if dist > sender.tolerance[i] + 0.1 {
            warnings.push(format!(
                "C{} ({}): distance {:.3} exceeds tolerance {:.3}",
                i + 1,
                ch.label(),
                dist,
                sender.tolerance[i]
            ));
        }
    }

    let receiver_capacity = 1.0 - receiver.draft();
    let draft_margin = receiver_capacity - sender.draft();
    let is_safe = warnings.is_empty() && draft_margin > 0.0;

    AlignmentReport {
        cosine_similarity: cosine,
        euclidean_distance: distance,
        draft_margin,
        is_safe,
        warnings,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_urgent() {
        let p = encode_intent("Deploy urgently by Friday");
        assert!(p.get(Channel::Stakes) > 0.5);
        assert!(p.get(Channel::Process) > 0.5);
    }

    #[test]
    fn test_alignment_safe() {
        let mut sender = IntentVector::zero();
        sender.set(Channel::Stakes, 0.9);
        sender.set_tolerance(Channel::Stakes, 0.5);

        let mut receiver = IntentVector::zero();
        receiver.set(Channel::Stakes, 0.8);
        receiver.set_tolerance(Channel::Stakes, 0.5);

        let report = check_alignment(&sender, &receiver);
        assert!(report.cosine_similarity > 0.99);
    }
}
