//! Spectral conservation integration for the constraint theory ecosystem.
//!
//! Re-exports the spectral-conservation crate and provides fleet-specific
//! extensions for monitoring conservation across agent coupling.

pub use spectral_conservation;

use spectral_conservation::{ConservationMonitor, SpectralState, spectral_state, commutator_norm};
use nalgebra::{DMatrix, DVector};

/// Fleet-level conservation health report.
#[derive(Debug, Clone)]
pub struct FleetConservationReport {
    /// Per-agent conservation status
    pub agents: Vec<AgentConservation>,
    /// Fleet-level coupling matrix conservation
    pub fleet_cv: f64,
    /// Fleet regime
    pub fleet_healthy: bool,
}

/// Per-agent conservation status.
#[derive(Debug, Clone)]
pub struct AgentConservation {
    /// Agent identifier
    pub agent_id: String,
    /// Current I value
    pub invariant: f64,
    /// CV over recent history
    pub cv: f64,
    /// Alert level
    pub alert: spectral_conservation::Alert,
}

/// Monitor fleet-level spectral conservation.
pub struct FleetConservationMonitor {
    /// Per-agent monitors
    agent_monitors: Vec<(String, ConservationMonitor)>,
    /// Fleet-level monitor
    fleet_monitor: ConservationMonitor,
}

impl FleetConservationMonitor {
    /// Create a fleet monitor for the given agents.
    pub fn new(agent_ids: &[&str]) -> Self {
        let agent_monitors = agent_ids
            .iter()
            .map(|&id| (id.to_string(), ConservationMonitor::default_threshold()))
            .collect();

        Self {
            agent_monitors,
            fleet_monitor: ConservationMonitor::default_threshold(),
        }
    }

    /// Update with a fleet coupling matrix.
    pub fn step(&mut self, fleet_coupling: &DMatrix<f64>) -> FleetConservationReport {
        let fleet_state = spectral_state(fleet_coupling).ok();
        let fleet_status = if let Some(ref state) = fleet_state {
            self.fleet_monitor.step(state)
        } else {
            return FleetConservationReport {
                agents: vec![],
                fleet_cv: f64::MAX,
                fleet_healthy: false,
            };
        };

        let agents = self.agent_monitors.iter().map(|(id, _)| {
            // In a real fleet, each agent would have its own coupling matrix
            AgentConservation {
                agent_id: id.clone(),
                invariant: fleet_state.as_ref().map(|s| s.invariant).unwrap_or(0.0),
                cv: fleet_status.cv,
                alert: fleet_status.alert,
            }
        }).collect();

        FleetConservationReport {
            agents,
            fleet_cv: fleet_status.cv,
            fleet_healthy: fleet_status.cv < 0.03,
        }
    }
}
