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

## License

Apache-2.0
