//! Navigation metaphor utilities.
//!
//! Draft determines truth. Tolerance stacks. Fairness checks.
//! The physical world already solved these problems.

use crate::{Channel, IntentVector};

/// Hydraulic fitting selection: right-size tooling to pressure.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Fitting {
    /// 50 PSI — casual communication (emoji, reactions)
    HoseClamp,
    /// 300 PSI — work communication (emails, standups)
    IndustrialFitting,
    /// 2500 PSI — technical specs (API contracts, safety specs)
    JicFitting,
    /// 10000 PSI — safety-critical (DO-178C, ISO 26262)
    DeepSeaSeal,
}

impl Fitting {
    /// Select fitting based on stakes level [0, 1].
    pub fn from_stakes(stakes: f64) -> Self {
        if stakes < 0.25 { Fitting::HoseClamp }
        else if stakes < 0.5 { Fitting::IndustrialFitting }
        else if stakes < 0.75 { Fitting::JicFitting }
        else { Fitting::DeepSeaSeal }
    }

    /// Required tolerance for this fitting.
    pub fn tolerance(&self) -> f64 {
        match self {
            Fitting::HoseClamp => 0.8,
            Fitting::IndustrialFitting => 0.5,
            Fitting::JicFitting => 0.2,
            Fitting::DeepSeaSeal => 0.05,
        }
    }

    /// Minimum channels that must be anchored.
    pub fn min_channels(&self) -> usize {
        match self {
            Fitting::HoseClamp => 2,
            Fitting::IndustrialFitting => 4,
            Fitting::JicFitting => 7,
            Fitting::DeepSeaSeal => 9,
        }
    }
}

/// Draft analysis for a message.
///
/// The squat effect: rushed messages have MORE draft.
/// speed_factor: 0.0 = careful, 0.5 = rushed, 1.0 = emergency.
pub struct DraftReport {
    pub base_draft: f64,
    pub effective_draft: f64,
    pub receiver_capacity: f64,
    pub margin: f64,
    pub is_safe: bool,
}

impl std::fmt::Display for DraftReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let status = if self.is_safe { "SAFE" } else { "GROUNDED" };
        write!(
            f,
            "{} (draft={:.3}, margin={:.3})",
            status, self.effective_draft, self.margin
        )
    }
}

/// Check draft compatibility.
pub fn check_draft(
    sender: &IntentVector,
    receiver_capacity: f64,
    speed_factor: f64,
) -> DraftReport {
    let base = sender.draft();
    let effective = base * (1.0 + speed_factor);
    let margin = receiver_capacity - effective;
    DraftReport {
        base_draft: base,
        effective_draft: effective,
        receiver_capacity,
        margin,
        is_safe: margin > 0.0,
    }
}

/// Tolerance stack across all channels.
///
/// ε_total = √(ε₁² + ε₂² + ... + ε₉²)
pub fn tolerance_stack(profile: &IntentVector) -> f64 {
    profile.tolerance.iter().map(|t| t * t).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fitting_selection() {
        assert_eq!(Fitting::from_stakes(0.1), Fitting::HoseClamp);
        assert_eq!(Fitting::from_stakes(0.9), Fitting::DeepSeaSeal);
    }

    #[test]
    fn test_draft_safe() {
        let mut sender = IntentVector::zero();
        sender.set(Channel::Stakes, 0.3);
        let report = check_draft(&sender, 0.8, 0.0);
        assert!(report.is_safe);
    }

    #[test]
    fn test_draft_grounded() {
        let mut sender = IntentVector::zero();
        sender.set(Channel::Stakes, 0.95);
        sender.set_tolerance(Channel::Stakes, 0.05);
        let report = check_draft(&sender, 0.1, 1.0);
        assert!(!report.is_safe);
    }

    #[test]
    fn test_tolerance_stack() {
        let profile = IntentVector::zero();
        let total = tolerance_stack(&profile);
        assert!(total > 0.0);
        // With all tolerances at 0.5: sqrt(9 * 0.25) = sqrt(2.25) = 1.5
        assert!((total - 1.5).abs() < 0.01);
    }
}
