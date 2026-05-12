//! Head direction encoding for fleet agents.
//!
//! Mammalian brains maintain two independent spatial systems:
//! - **Grid cells** (entorhinal cortex) — hexagonal metric tiling (our Eisenstein lattice)
//! - **Head direction cells** (retrosplenial cortex, Taube 2007) — orientation frame
//!   independent of position
//!
//! Our constraint system had position (E12 coordinates) but no head direction encoding.
//! A fleet of agents in motion loses angular coherence because there's no way to say
//! "I'm at (q,r) facing 60°." Position tells WHERE but not WHICH WAY.
//!
//! This module fills that gap with 12 discrete orientations (every 30°), fitting in 4 bits
//! and aligning with the dodecet nibble system.
//!
//! # Example
//!
//! ```
//! use flux_lucid::head_direction::{HeadDirection, PositionedAgent, angular_coherence};
//!
//! let heading = HeadDirection::from_radians(std::f64::consts::FRAC_PI_2); // 90°
//! assert_eq!(heading.to_radians(), std::f64::consts::FRAC_PI_2);
//!
//! let rotated = heading.rotate_by(3); // 90° + 90° = 180°
//! assert_eq!(rotated, HeadDirection::from_radians(std::f64::consts::PI));
//!
//! let agent = PositionedAgent::new(5, 10, heading, 100);
//! assert_eq!(agent.q, 5);
//! assert_eq!(agent.r, 10);
//! ```

/// A discrete heading in one of 12 directions (every 30°).
///
/// Encoded as a `u8` value 0–11, where `n` represents `n × 30°`.
/// This fits in 4 bits, aligning with the dodecet nibble encoding.
///
/// The 12 directions cover both square (0°, 90°, 180°, 270°) and
/// hexagonal (0°, 60°, 120°, 180°, 240°, 300°) grid orientations.
///
/// # Layout
///
/// ```text
///  0 =   0° (East)
///  1 =  30°
///  2 =  60° (ENE on hex grid)
///  3 =  90° (North)
///  4 = 120° (NNW on hex grid)
///  5 = 150°
///  6 = 180° (West)
///  7 = 210°
///  8 = 240° (SSW on hex grid)
///  9 = 270° (South)
/// 10 = 300° (SSE on hex grid)
/// 11 = 330°
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct HeadDirection(u8);

impl HeadDirection {
    /// Number of discrete directions.
    pub const COUNT: u8 = 12;

    /// Angular step in radians (30° = π/6).
    pub const STEP_RAD: f64 = std::f64::consts::FRAC_PI_6;

    /// Create a heading from a discrete step value (0–11).
    ///
    /// Values wrap modulo 12, so `from_step(12) == from_step(0)`.
    ///
    /// # Example
    ///
    /// ```
    /// use flux_lucid::head_direction::HeadDirection;
    ///
    /// let h = HeadDirection::from_step(3); // 90°
    /// assert_eq!(h.to_degrees(), 90.0);
    /// ```
    pub fn from_step(step: u8) -> Self {
        HeadDirection(step % Self::COUNT)
    }

    /// Create a heading from an angle in radians.
    ///
    /// Snaps to the nearest discrete direction. Angles are normalized
    /// to [0, 2π) before snapping.
    ///
    /// # Example
    ///
    /// ```
    /// use flux_lucid::head_direction::HeadDirection;
    ///
    /// // 45° snaps to 30° (step 1)
    /// let h = HeadDirection::from_radians(0.785);
    /// assert_eq!(h.step(), 1);
    ///
    /// // 90° snaps exactly to step 3
    /// let h2 = HeadDirection::from_radians(std::f64::consts::FRAC_PI_2);
    /// assert_eq!(h2.step(), 3);
    /// ```
    pub fn from_radians(theta: f64) -> Self {
        let two_pi = 2.0 * std::f64::consts::PI;
        let normalized = ((theta % two_pi) + two_pi) % two_pi;
        let step = (normalized / Self::STEP_RAD).round() as u8 % Self::COUNT;
        HeadDirection(step)
    }

    /// Create a heading from an angle in degrees.
    ///
    /// # Example
    ///
    /// ```
    /// use flux_lucid::head_direction::HeadDirection;
    ///
    /// let h = HeadDirection::from_degrees(180.0);
    /// assert_eq!(h.step(), 6);
    /// ```
    pub fn from_degrees(deg: f64) -> Self {
        Self::from_radians(deg * std::f64::consts::PI / 180.0)
    }

    /// Get the discrete step value (0–11).
    #[inline]
    pub fn step(&self) -> u8 {
        self.0
    }

    /// Recover the angle in radians.
    ///
    /// # Example
    ///
    /// ```
    /// use flux_lucid::head_direction::HeadDirection;
    ///
    /// let h = HeadDirection::from_step(6);
    /// assert!((h.to_radians() - std::f64::consts::PI).abs() < 1e-10);
    /// ```
    pub fn to_radians(&self) -> f64 {
        self.0 as f64 * Self::STEP_RAD
    }

    /// Recover the angle in degrees.
    pub fn to_degrees(&self) -> f64 {
        self.0 as f64 * 30.0
    }

    /// Rotate heading by `n` steps (each step = 30°).
    ///
    /// Positive values rotate counterclockwise, negative clockwise.
    /// Wraps around at 12.
    ///
    /// # Example
    ///
    /// ```
    /// use flux_lucid::head_direction::HeadDirection;
    ///
    /// let east = HeadDirection::from_step(0);
    /// let north = east.rotate_by(3); // +90°
    /// assert_eq!(north.step(), 3);
    ///
    /// // Wrapping: 11 + 1 = 0
    /// let almost_east = HeadDirection::from_step(11);
    /// let back_to_east = almost_east.rotate_by(1);
    /// assert_eq!(back_to_east.step(), 0);
    /// ```
    pub fn rotate_by(&self, steps: i8) -> Self {
        let raw = (self.0 as i16 + steps as i16).rem_euclid(Self::COUNT as i16) as u8;
        HeadDirection(raw)
    }

    /// Signed difference in steps from `self` to `other`.
    ///
    /// Returns the shortest angular path: result is in [-6, 6].
    /// Positive means `other` is counterclockwise from `self`.
    ///
    /// # Example
    ///
    /// ```
    /// use flux_lucid::head_direction::HeadDirection;
    ///
    /// let east = HeadDirection::from_step(0);
    /// let north = HeadDirection::from_step(3);
    ///
    /// // North is 3 steps counterclockwise from East
    /// assert_eq!(east.relative_to(&north), 3);
    /// assert_eq!(north.relative_to(&east), -3);
    /// ```
    pub fn relative_to(&self, other: &HeadDirection) -> i8 {
        let diff = other.0 as i8 - self.0 as i8;
        // Wrap to [-6, 6]
        if diff > 6 {
            diff - 12
        } else if diff < -6 {
            diff + 12
        } else {
            diff
        }
    }

    /// All 12 discrete headings in order.
    pub fn all() -> [HeadDirection; 12] {
        [
            HeadDirection(0),
            HeadDirection(1),
            HeadDirection(2),
            HeadDirection(3),
            HeadDirection(4),
            HeadDirection(5),
            HeadDirection(6),
            HeadDirection(7),
            HeadDirection(8),
            HeadDirection(9),
            HeadDirection(10),
            HeadDirection(11),
        ]
    }

    /// Pack the heading into the low 4 bits of a u8.
    #[inline]
    pub fn to_nibble(&self) -> u8 {
        self.0 & 0x0F
    }

    /// Unpack a heading from the low 4 bits of a u8.
    pub fn from_nibble(nibble: u8) -> Self {
        HeadDirection(nibble & 0x0F)
    }
}

impl std::fmt::Display for HeadDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}°", self.to_degrees() as u32)
    }
}

/// An agent with position, heading, and speed.
///
/// Packs into 48 bits (6 bytes): `q` (16) + `r` (16) + `heading` (4) + `speed` (8) + spare (4).
/// Can be packed into a single `u64` with 16 bits to spare.
///
/// # Example
///
/// ```
/// use flux_lucid::head_direction::{HeadDirection, PositionedAgent};
///
/// let agent = PositionedAgent::new(10, 20, HeadDirection::from_step(3), 50);
/// assert_eq!(agent.q, 10);
/// assert_eq!(agent.r, 20);
/// assert_eq!(agent.heading.step(), 3);
/// assert_eq!(agent.speed, 50);
///
/// // Pack and unpack
/// let packed = agent.to_u64();
/// let unpacked = PositionedAgent::from_u64(packed);
/// assert_eq!(agent, unpacked);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PositionedAgent {
    /// Axial coordinate q (dodecet).
    pub q: u16,
    /// Axial coordinate r (dodecet).
    pub r: u16,
    /// Which way the agent is facing (4 bits).
    pub heading: HeadDirection,
    /// Current speed 0–255 (8 bits).
    pub speed: u8,
}

impl PositionedAgent {
    /// Create a new positioned agent.
    pub fn new(q: u16, r: u16, heading: HeadDirection, speed: u8) -> Self {
        PositionedAgent {
            q,
            r,
            heading,
            speed,
        }
    }

    /// Create an agent at origin facing East with zero speed.
    pub fn origin() -> Self {
        PositionedAgent {
            q: 0,
            r: 0,
            heading: HeadDirection::default(),
            speed: 0,
        }
    }

    /// Pack into a single `u64`.
    ///
    /// Layout: `[q: 16 bits | r: 16 bits | heading: 4 bits | speed: 8 bits | spare: 20 bits]`
    pub fn to_u64(&self) -> u64 {
        let q = self.q as u64;
        let r = self.r as u64;
        let h = self.heading.to_nibble() as u64;
        let s = self.speed as u64;
        (q << 48) | (r << 32) | (h << 28) | (s << 20)
    }

    /// Unpack from a `u64`.
    pub fn from_u64(val: u64) -> Self {
        let q = (val >> 48) as u16;
        let r = ((val >> 32) & 0xFFFF) as u16;
        let heading = HeadDirection::from_nibble(((val >> 28) & 0xF) as u8);
        let speed = ((val >> 20) & 0xFF) as u8;
        PositionedAgent {
            q,
            r,
            heading,
            speed,
        }
    }

    /// Check if two agents are at the same position regardless of heading/speed.
    pub fn same_position(&self, other: &PositionedAgent) -> bool {
        self.q == other.q && self.r == other.r
    }

    /// Axial distance (hex grid) between two agents.
    pub fn hex_distance(&self, other: &PositionedAgent) -> u32 {
        let dq = (self.q as i32 - other.q as i32).abs();
        let dr = (self.r as i32 - other.r as i32).abs();
        let ds = ((self.q as i32 + self.r as i32) - (other.q as i32 + other.r as i32)).abs();
        ((dq + dr + ds) / 2) as u32
    }
}

/// A point-in-time snapshot of all fleet agents.
///
/// Used for temporal replay and sharp-wave-ripple consolidation.
///
/// # Example
///
/// ```
/// use flux_lucid::head_direction::{
///     HeadDirection, PositionedAgent, FleetSnapshot,
/// };
///
/// let agents = vec![
///     PositionedAgent::new(0, 0, HeadDirection::from_step(0), 10),
///     PositionedAgent::new(1, 0, HeadDirection::from_step(0), 10),
/// ];
/// let snapshot = FleetSnapshot::new(agents, 1000);
/// assert_eq!(snapshot.agents.len(), 2);
/// assert_eq!(snapshot.timestamp, 1000);
/// ```
#[derive(Debug, Clone)]
pub struct FleetSnapshot {
    /// All agents at this point in time.
    pub agents: Vec<PositionedAgent>,
    /// Monotonic clock value.
    pub timestamp: u64,
}

impl FleetSnapshot {
    /// Create a new fleet snapshot.
    pub fn new(agents: Vec<PositionedAgent>, timestamp: u64) -> Self {
        FleetSnapshot { agents, timestamp }
    }

    /// Empty snapshot at a given time.
    pub fn empty(timestamp: u64) -> Self {
        FleetSnapshot {
            agents: Vec::new(),
            timestamp,
        }
    }

    /// Number of agents in the snapshot.
    pub fn len(&self) -> usize {
        self.agents.len()
    }

    /// Whether the snapshot has no agents.
    pub fn is_empty(&self) -> bool {
        self.agents.is_empty()
    }
}

/// Check if a cluster of agents maintains coherent heading.
///
/// Returns a coherence score in [0.0, 1.0]:
/// - **1.0** when all agents face the same direction.
/// - **0.0** when headings are maximally dispersed (uniform across all 12 directions).
///
/// The score is computed as `1.0 - circular_variance`, where circular variance
/// is the normalized spread of the heading angles on the unit circle.
///
/// # Example
///
/// ```
/// use flux_lucid::head_direction::{
///     HeadDirection, PositionedAgent, angular_coherence,
/// };
///
/// // All facing the same way → perfect coherence
/// let coherent = vec![
///     PositionedAgent::new(0, 0, HeadDirection::from_step(3), 10),
///     PositionedAgent::new(1, 0, HeadDirection::from_step(3), 10),
///     PositionedAgent::new(0, 1, HeadDirection::from_step(3), 10),
/// ];
/// assert!(angular_coherence(&coherent) > 0.99);
///
/// // Facing opposite directions → low coherence
/// let dispersed = vec![
///     PositionedAgent::new(0, 0, HeadDirection::from_step(0), 10),
///     PositionedAgent::new(1, 0, HeadDirection::from_step(6), 10),
/// ];
/// let score = angular_coherence(&dispersed);
/// assert!(score < 0.2);
/// ```
pub fn angular_coherence(agents: &[PositionedAgent]) -> f64 {
    if agents.is_empty() {
        return 1.0;
    }
    if agents.len() == 1 {
        return 1.0;
    }

    // Mean resultant vector on the unit circle
    let n = agents.len() as f64;
    let sum_cos: f64 = agents.iter().map(|a| a.heading.to_radians().cos()).sum();
    let sum_sin: f64 = agents.iter().map(|a| a.heading.to_radians().sin()).sum();
    let mean_resultant = (sum_cos * sum_cos + sum_sin * sum_sin).sqrt() / n;

    // Mean resultant is in [0, 1]. When 1, all face same direction.
    // This IS the coherence score.
    mean_resultant
}

/// A tile produced by sharp-wave-ripple consolidation.
///
/// Compressed representation of a path segment that was traversed
/// repeatedly, suitable for caching or precomputation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConsolidatedTile {
    /// Starting position (q, r).
    pub start_q: u16,
    pub start_r: u16,
    /// Ending position (q, r).
    pub end_q: u16,
    pub end_r: u16,
    /// Dominant heading during traversal.
    pub heading: HeadDirection,
    /// Number of snapshots consolidated into this tile.
    pub visit_count: usize,
    /// Timestamp of the first snapshot in the consolidation window.
    pub first_seen: u64,
    /// Timestamp of the last snapshot in the consolidation window.
    pub last_seen: u64,
}

impl ConsolidatedTile {
    /// Create a new consolidated tile.
    pub fn new(
        start_q: u16,
        start_r: u16,
        end_q: u16,
        end_r: u16,
        heading: HeadDirection,
        visit_count: usize,
        first_seen: u64,
        last_seen: u64,
    ) -> Self {
        ConsolidatedTile {
            start_q,
            start_r,
            end_q,
            end_r,
            heading,
            visit_count,
            first_seen,
            last_seen,
        }
    }

    /// Duration spanned by this tile in time units.
    pub fn duration(&self) -> u64 {
        self.last_seen.saturating_sub(self.first_seen)
    }

    /// Hex distance from start to end.
    pub fn span(&self) -> u32 {
        let start = PositionedAgent::new(self.start_q, self.start_r, self.heading, 0);
        let end = PositionedAgent::new(self.end_q, self.end_r, self.heading, 0);
        start.hex_distance(&end)
    }
}

/// During idle periods, replay recent fleet paths to consolidate
/// into compressed tiles.
///
/// Inspired by hippocampal sharp-wave ripples (SWRs): during rest,
/// the brain replays recent trajectories and consolidates them into
/// long-term memory tiles.
///
/// Only consolidates paths that have been traversed at least
/// `tile_threshold` times.
///
/// # Example
///
/// ```
/// use flux_lucid::head_direction::{
///     HeadDirection, PositionedAgent, FleetSnapshot, consolidate_paths,
/// };
///
/// let snapshots = vec![
///     FleetSnapshot::new(vec![
///         PositionedAgent::new(0, 0, HeadDirection::from_step(3), 10),
///     ], 0),
///     FleetSnapshot::new(vec![
///         PositionedAgent::new(0, 1, HeadDirection::from_step(3), 10),
///     ], 100),
///     FleetSnapshot::new(vec![
///         PositionedAgent::new(0, 2, HeadDirection::from_step(3), 10),
///     ], 200),
/// ];
///
/// // Consolidate with threshold 2 (need at least 2 visits to form a tile)
/// let tiles = consolidate_paths(&snapshots, 2);
/// assert!(!tiles.is_empty());
/// ```
pub fn consolidate_paths(recent: &[FleetSnapshot], tile_threshold: usize) -> Vec<ConsolidatedTile> {
    if recent.len() < 2 || tile_threshold < 2 {
        return Vec::new();
    }

    // Group consecutive positions into path segments per agent index.
    // For simplicity, we consolidate based on position sequence,
    // treating each agent index as a tracked identity.
    let max_agents = recent.iter().map(|s| s.agents.len()).max().unwrap_or(0);

    let mut tiles = Vec::new();

    for agent_idx in 0..max_agents {
        // Extract this agent's trajectory
        let trajectory: Vec<_> = recent
            .iter()
            .filter_map(|snap| snap.agents.get(agent_idx).map(|a| (*a, snap.timestamp)))
            .collect();

        if trajectory.len() < tile_threshold {
            continue;
        }

        // Find the most common heading (dominant heading)
        let mut heading_counts = [0usize; 12];
        for (agent, _) in &trajectory {
            heading_counts[agent.heading.step() as usize] += 1;
        }
        let dominant_step = heading_counts
            .iter()
            .enumerate()
            .max_by_key(|(_, &count)| count)
            .map(|(step, _)| step as u8)
            .unwrap_or(0);
        let dominant_heading = HeadDirection::from_step(dominant_step);

        let first = &trajectory[0];
        let last = &trajectory.last().unwrap();

        tiles.push(ConsolidatedTile::new(
            first.0.q,
            first.0.r,
            last.0.q,
            last.0.r,
            dominant_heading,
            trajectory.len(),
            first.1,
            last.1,
        ));
    }

    tiles
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- HeadDirection ---

    #[test]
    fn test_from_step_basic() {
        assert_eq!(HeadDirection::from_step(0).step(), 0);
        assert_eq!(HeadDirection::from_step(6).step(), 6);
        assert_eq!(HeadDirection::from_step(11).step(), 11);
    }

    #[test]
    fn test_from_step_wrapping() {
        assert_eq!(HeadDirection::from_step(12).step(), 0);
        assert_eq!(HeadDirection::from_step(13).step(), 1);
        assert_eq!(HeadDirection::from_step(255).step(), 255 % 12);
    }

    #[test]
    fn test_from_radians_cardinal() {
        assert_eq!(HeadDirection::from_radians(0.0).step(), 0);
        assert_eq!(
            HeadDirection::from_radians(std::f64::consts::FRAC_PI_2).step(),
            3
        );
        assert_eq!(HeadDirection::from_radians(std::f64::consts::PI).step(), 6);
        assert_eq!(
            HeadDirection::from_radians(3.0 * std::f64::consts::FRAC_PI_2).step(),
            9
        );
    }

    #[test]
    fn test_from_radians_hex() {
        assert_eq!(
            HeadDirection::from_radians(std::f64::consts::FRAC_PI_3).step(),
            2
        );
        assert_eq!(
            HeadDirection::from_radians(2.0 * std::f64::consts::FRAC_PI_3).step(),
            4
        );
        assert_eq!(
            HeadDirection::from_radians(4.0 * std::f64::consts::FRAC_PI_3).step(),
            8
        );
        assert_eq!(
            HeadDirection::from_radians(5.0 * std::f64::consts::FRAC_PI_3).step(),
            10
        );
    }

    #[test]
    fn test_from_radians_negative() {
        let h = HeadDirection::from_radians(-std::f64::consts::FRAC_PI_6);
        assert_eq!(h.step(), 11); // -30° → 330°
    }

    #[test]
    fn test_from_radians_large() {
        let h = HeadDirection::from_radians(7.0 * std::f64::consts::PI);
        assert_eq!(h.step(), 6); // 7π → π mod 2π → 180°
    }

    #[test]
    fn test_from_degrees() {
        assert_eq!(HeadDirection::from_degrees(0.0).step(), 0);
        assert_eq!(HeadDirection::from_degrees(90.0).step(), 3);
        assert_eq!(HeadDirection::from_degrees(180.0).step(), 6);
        assert_eq!(HeadDirection::from_degrees(270.0).step(), 9);
        assert_eq!(HeadDirection::from_degrees(300.0).step(), 10);
    }

    #[test]
    fn test_to_radians_roundtrip() {
        for step in 0u8..12 {
            let h = HeadDirection::from_step(step);
            let h2 = HeadDirection::from_radians(h.to_radians());
            assert_eq!(h, h2, "roundtrip failed for step {}", step);
        }
    }

    #[test]
    fn test_to_degrees() {
        assert!((HeadDirection::from_step(3).to_degrees() - 90.0).abs() < 1e-10);
        assert!((HeadDirection::from_step(6).to_degrees() - 180.0).abs() < 1e-10);
    }

    #[test]
    fn test_rotate_by_positive() {
        let h = HeadDirection::from_step(0);
        assert_eq!(h.rotate_by(3).step(), 3);
        assert_eq!(h.rotate_by(6).step(), 6);
        assert_eq!(h.rotate_by(12).step(), 0);
    }

    #[test]
    fn test_rotate_by_negative() {
        let h = HeadDirection::from_step(3);
        assert_eq!(h.rotate_by(-3).step(), 0);
        assert_eq!(h.rotate_by(-6).step(), 9);
    }

    #[test]
    fn test_rotate_by_wrap() {
        let h = HeadDirection::from_step(11);
        assert_eq!(h.rotate_by(1).step(), 0);
        assert_eq!(h.rotate_by(-11).step(), 0);
    }

    #[test]
    fn test_relative_to_same() {
        let h = HeadDirection::from_step(3);
        assert_eq!(h.relative_to(&h), 0);
    }

    #[test]
    fn test_relative_to_ccw() {
        let east = HeadDirection::from_step(0);
        let north = HeadDirection::from_step(3);
        assert_eq!(east.relative_to(&north), 3);
    }

    #[test]
    fn test_relative_to_cw() {
        let north = HeadDirection::from_step(3);
        let east = HeadDirection::from_step(0);
        assert_eq!(north.relative_to(&east), -3);
    }

    #[test]
    fn test_relative_to_shortest_path() {
        // Step 1 and step 11: difference is 2 steps (not 10)
        let a = HeadDirection::from_step(1);
        let b = HeadDirection::from_step(11);
        assert_eq!(a.relative_to(&b), -2);
        assert_eq!(b.relative_to(&a), 2);
    }

    #[test]
    fn test_relative_to_opposite() {
        let a = HeadDirection::from_step(0);
        let b = HeadDirection::from_step(6);
        // Exact opposite: +6 or -6, we pick +6
        assert_eq!(a.relative_to(&b).abs(), 6);
    }

    #[test]
    fn test_nibble_roundtrip() {
        for step in 0u8..12 {
            let h = HeadDirection::from_step(step);
            assert_eq!(HeadDirection::from_nibble(h.to_nibble()), h);
        }
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", HeadDirection::from_step(3)), "90°");
        assert_eq!(format!("{}", HeadDirection::from_step(0)), "0°");
    }

    #[test]
    fn test_all_twelve() {
        let all = HeadDirection::all();
        assert_eq!(all.len(), 12);
        for (i, h) in all.iter().enumerate() {
            assert_eq!(h.step(), i as u8);
        }
    }

    // --- PositionedAgent ---

    #[test]
    fn test_agent_new() {
        let agent = PositionedAgent::new(5, 10, HeadDirection::from_step(3), 100);
        assert_eq!(agent.q, 5);
        assert_eq!(agent.r, 10);
        assert_eq!(agent.heading.step(), 3);
        assert_eq!(agent.speed, 100);
    }

    #[test]
    fn test_agent_origin() {
        let agent = PositionedAgent::origin();
        assert_eq!(agent.q, 0);
        assert_eq!(agent.r, 0);
        assert_eq!(agent.heading.step(), 0);
        assert_eq!(agent.speed, 0);
    }

    #[test]
    fn test_agent_pack_unpack_roundtrip() {
        let agent = PositionedAgent::new(12345, 54321, HeadDirection::from_step(7), 200);
        let packed = agent.to_u64();
        let unpacked = PositionedAgent::from_u64(packed);
        assert_eq!(agent, unpacked);
    }

    #[test]
    fn test_agent_pack_unpack_edge_cases() {
        // Max values
        let max_agent =
            PositionedAgent::new(u16::MAX, u16::MAX, HeadDirection::from_step(11), u8::MAX);
        let packed = max_agent.to_u64();
        let unpacked = PositionedAgent::from_u64(packed);
        assert_eq!(max_agent, unpacked);

        // Zero values
        let zero_agent = PositionedAgent::origin();
        assert_eq!(PositionedAgent::from_u64(zero_agent.to_u64()), zero_agent);
    }

    #[test]
    fn test_same_position() {
        let a = PositionedAgent::new(3, 7, HeadDirection::from_step(0), 10);
        let b = PositionedAgent::new(3, 7, HeadDirection::from_step(6), 99);
        assert!(a.same_position(&b));

        let c = PositionedAgent::new(3, 8, HeadDirection::from_step(0), 10);
        assert!(!a.same_position(&c));
    }

    #[test]
    fn test_hex_distance_adjacent() {
        let origin = PositionedAgent::new(0, 0, HeadDirection::default(), 0);
        let neighbor = PositionedAgent::new(1, 0, HeadDirection::default(), 0);
        assert_eq!(origin.hex_distance(&neighbor), 1);
    }

    #[test]
    fn test_hex_distance_same() {
        let a = PositionedAgent::new(5, 5, HeadDirection::default(), 0);
        assert_eq!(a.hex_distance(&a), 0);
    }

    #[test]
    fn test_hex_distance_diagonal() {
        let a = PositionedAgent::new(0, 0, HeadDirection::default(), 0);
        let b = PositionedAgent::new(3, 3, HeadDirection::default(), 0);
        // Axial: dq=3, dr=3, ds=6 → (3+3+6)/2 = 6
        assert_eq!(a.hex_distance(&b), 6);
    }

    // --- FleetSnapshot ---

    #[test]
    fn test_snapshot_new() {
        let agents = vec![PositionedAgent::origin()];
        let snap = FleetSnapshot::new(agents.clone(), 42);
        assert_eq!(snap.len(), 1);
        assert_eq!(snap.timestamp, 42);
        assert!(!snap.is_empty());
    }

    #[test]
    fn test_snapshot_empty() {
        let snap = FleetSnapshot::empty(0);
        assert!(snap.is_empty());
        assert_eq!(snap.len(), 0);
    }

    // --- angular_coherence ---

    #[test]
    fn test_coherence_empty() {
        let score = angular_coherence(&[]);
        assert_eq!(score, 1.0);
    }

    #[test]
    fn test_coherence_single() {
        let agents = vec![PositionedAgent::origin()];
        assert_eq!(angular_coherence(&agents), 1.0);
    }

    #[test]
    fn test_coherence_perfect() {
        let agents = vec![
            PositionedAgent::new(0, 0, HeadDirection::from_step(3), 10),
            PositionedAgent::new(1, 0, HeadDirection::from_step(3), 20),
            PositionedAgent::new(0, 1, HeadDirection::from_step(3), 30),
            PositionedAgent::new(2, 0, HeadDirection::from_step(3), 40),
        ];
        assert!((angular_coherence(&agents) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_opposite() {
        // Two agents facing 180° apart: cos(0)+cos(π)=0, sin(0)+sin(π)=0 → 0.0
        let agents = vec![
            PositionedAgent::new(0, 0, HeadDirection::from_step(0), 10),
            PositionedAgent::new(1, 0, HeadDirection::from_step(6), 10),
        ];
        assert!(angular_coherence(&agents) < 0.01);
    }

    #[test]
    fn test_coherence_partial() {
        // 3 agents at 0°, 0°, 90° — partial coherence
        let agents = vec![
            PositionedAgent::new(0, 0, HeadDirection::from_step(0), 10),
            PositionedAgent::new(1, 0, HeadDirection::from_step(0), 10),
            PositionedAgent::new(0, 1, HeadDirection::from_step(3), 10),
        ];
        let score = angular_coherence(&agents);
        assert!(score > 0.3 && score < 0.9);
    }

    // --- consolidate_paths ---

    #[test]
    fn test_consolidate_empty() {
        let tiles = consolidate_paths(&[], 3);
        assert!(tiles.is_empty());
    }

    #[test]
    fn test_consolidate_below_threshold() {
        let snapshots = vec![
            FleetSnapshot::new(vec![PositionedAgent::origin()], 0),
            FleetSnapshot::new(
                vec![PositionedAgent::new(0, 1, HeadDirection::default(), 0)],
                100,
            ),
        ];
        // Threshold is 3, only 2 snapshots → no tiles
        let tiles = consolidate_paths(&snapshots, 3);
        assert!(tiles.is_empty());
    }

    #[test]
    fn test_consolidate_basic() {
        let snapshots = vec![
            FleetSnapshot::new(
                vec![PositionedAgent::new(0, 0, HeadDirection::from_step(3), 10)],
                0,
            ),
            FleetSnapshot::new(
                vec![PositionedAgent::new(0, 1, HeadDirection::from_step(3), 10)],
                100,
            ),
            FleetSnapshot::new(
                vec![PositionedAgent::new(0, 2, HeadDirection::from_step(3), 10)],
                200,
            ),
        ];
        let tiles = consolidate_paths(&snapshots, 2);
        assert_eq!(tiles.len(), 1);
        assert_eq!(tiles[0].start_q, 0);
        assert_eq!(tiles[0].start_r, 0);
        assert_eq!(tiles[0].end_q, 0);
        assert_eq!(tiles[0].end_r, 2);
        assert_eq!(tiles[0].heading.step(), 3);
        assert_eq!(tiles[0].visit_count, 3);
        assert_eq!(tiles[0].first_seen, 0);
        assert_eq!(tiles[0].last_seen, 200);
    }

    #[test]
    fn test_consolidate_multi_agent() {
        let make_snap = |t: u64, q0: u16, r0: u16, q1: u16, r1: u16| {
            FleetSnapshot::new(
                vec![
                    PositionedAgent::new(q0, r0, HeadDirection::from_step(0), 5),
                    PositionedAgent::new(q1, r1, HeadDirection::from_step(3), 5),
                ],
                t,
            )
        };

        let snapshots = vec![
            make_snap(0, 0, 0, 10, 10),
            make_snap(100, 1, 0, 10, 11),
            make_snap(200, 2, 0, 10, 12),
        ];
        let tiles = consolidate_paths(&snapshots, 2);
        assert_eq!(tiles.len(), 2);
    }

    #[test]
    fn test_consolidated_tile_span() {
        let tile = ConsolidatedTile::new(0, 0, 3, 3, HeadDirection::from_step(3), 5, 0, 100);
        // (0,0)→(3,3): dq=3, dr=3, ds=6 → distance = 6
        assert_eq!(tile.span(), 6);
        assert_eq!(tile.duration(), 100);
    }
}
