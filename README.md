# flux-lucid

**Unified constraint theory ecosystem — one dependency to rule them all.**

Pulls together constraint compilation (CDCL → LLVM IR → AVX-512), fleet coordination (GL(9) zero-holonomy consensus), and 9-channel intent communication into a single crate.

## Quick Start

```rust
use flux_lucid::{Channel, IntentVector, intent, navigation};

// Encode intent
let mut sender = IntentVector::zero();
sender.set(Channel::Stakes, 0.9);
sender.set(Channel::Process, 0.8);

// Check alignment
let mut receiver = IntentVector::zero();
receiver.set(Channel::Stakes, 0.85);
let report = intent::check_alignment(&sender, &receiver);
println!("{}", report); // ✓ SAFE

// Check draft
let draft = navigation::check_draft(&sender, 0.8, 0.0);
println!("{}", draft); // SAFE

// Select hydraulic fitting
let fitting = navigation::Fitting::from_stakes(0.9);
println!("{:?}", fitting); // DeepSeaSeal
```

## The Navigation Metaphors

This library implements five principles from nautical navigation:

1. **Splines in the Ether** — The 9 channels are anchor points. Intent between them is irreducible.
2. **Fair Curve First** — Sight intent first, find measurements second.
3. **Where the Rocks Aren't** — Negative knowledge is primary.
4. **Draft Determines Truth** — Same message, different safety per receiver.
5. **Speed Beats Truth** — Satisficing beats optimizing in real-time.

## Components

- `constraint-theory-llvm` — CDCL → LLVM IR → AVX-512
- `holonomy-consensus` — GL(9) zero-holonomy consensus

## Module Reference

### `beam_tolerance` — Physical Math for Intent Stiffness

Maps beam physics to intent alignment. Each channel has a "material stiffness" derived from stakes (C9):

```rust
use flux_lucid::beam_tolerance::{BeamMaterial, classify_precision, compute_tolerance, compute_draft};

// Steel (stakes > 0.75): E=200 GPa → tolerance ~0.05
let tol = compute_tolerance(0.9, 1.0);
assert!(tol < 0.1);

// Rubber (stakes < 0.1): E=0.01 GPa → tolerance ~1.0
let tol = compute_tolerance(0.05, 1.0);
assert!(tol > 0.5);

// Draft with squat effect (rushed messages)
let draft = compute_draft(base_tol, 0.8, 0.5);
```

Beam materials map to precision classes:

| Stakes | Material | Precision | Bits/Constraint |
|--------|----------|-----------|-----------------|
| > 0.75 | Steel | DUAL | 64 |
| 0.5–0.75 | Fiberglass | INT32 | 32 |
| 0.25–0.50 | Oak | INT16 | 16 |
| < 0.25 | Rubber/Cedar | INT8 | 8 |

### `soa_emitter` — Struct-of-Arrays Mixed-Precision Batch

Groups constraints by precision class for cache-friendly AVX-512 execution:

```rust
use flux_lucid::soa_emitter::SoABatch;

let constraints = vec![
    (5.0, 0.0, 10.0, 0.1),    // INT8
    (500.0, 0.0, 1000.0, 0.6), // INT32
    (5000.0, 0.0, 10000.0, 0.9), // DUAL
];
let batch = SoABatch::from_constraints(&constraints);
let results = batch.check_all(); // Vec<bool>
let (actual_bits, baseline_bits) = batch.memory_stats();
```

Typical sensor mixes save **50–70% memory** vs uniform INT32.

### `intent_emitter` — Intent-Directed Constraint Emission

Bridges 9-channel intent profiles to constraint compilation:

```rust
use flux_lucid::intent_emitter::emit_constraints;

// Takes IntentVector + constraint spec → classified constraint batch
let batch = emit_constraints(&profile, &specs);
```

### `intent_compilation` — Precision Classification

Classifies precision from C9 stakes with epsilon-aware verification:

```rust
use flux_lucid::intent_compilation::{classify_precision, Precision, check_with_precision};

let prec = classify_precision(&profile);
let passed = check_with_precision(50.0, 0.0, 100.0, Precision::INT32);
```

### XOR Dual-Path Verification

DUAL-classified constraints use two independent execution paths:

1. **Path A**: Direct comparison (`v >= lo && v <= hi`)
2. **Path B**: XOR-based signed→unsigned conversion (`v ^ 0x80000000`)

Both paths must agree. This catches silicon-level errors (rowhammer, cosmic ray bit flips) without doubling execution time — the XOR trick is branchless and pipeline-friendly.

## Cargo Features

| Feature | Description |
|---------|-------------|
| `x86-64-emitter` | Direct AVX-512 emission (bypasses LLVM) |
| `jit` | JIT compilation via Cranelift |
| `fleet` | Full fleet coordination features |

## License

Apache-2.0
