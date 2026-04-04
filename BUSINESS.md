# BUSINESS: PhiFlow-lang
*Last updated: 2026-03-14*
*See also: WORKSPACE.md for technical status*

## One Sentence (for anyone, no jargon)
PhiFlow-lang is an experimental programming-language branch that compiles, runs examples, and exposes self-observing language concepts, but it is not yet clean enough to sell as a finished platform.

## Status
- Functional today: ⚠️
- % complete (honest): 65%
- Income tier: 1-3 months

## Who Pays
1. CTOs at agent-platform startups — they may pay for a paid evaluation or workshop if they want a differentiated runtime/story around stateful or self-observing execution.
2. Research engineering leads exploring custom DSLs — they may pay for an internal proof of concept or design session.
3. Innovation teams at developer-tool companies — they may pay for a workshop, pilot, or internal demo branch rather than a production license.

## Price
- Range: $5,000 - $25,000
- Basis: LangSmith Plus is listed at $39 per seat per month on LangChain's official pricing page; PhiFlow-lang is only defensible today as a higher-touch pilot, workshop, or evaluation license, not as a seat-based production platform.
- Model: fixed-fee design-partner pilot, workshop, or evaluation license

## What Blocks First Sale (one thing)
The language branch still lacks one clean "all tests green" release signal because the verified run on 2026-03-14 failed in `test_shots_scaling_performance`.

## Marketing Angle
The differentiator is language-level self-observation, intention, and resonance semantics, but the honest proof has to come from the working Rust crate and test/run output, not from philosophical README copy.

## Transaction Requirements
- Payment: Invoice
- Legal: Pilot agreement or evaluation license, plus standard NDA if the buyer wants private exploration
- Delivery: Guided demo, branch access, and a written evaluation scope
- Support: Direct founder/engineer support during the pilot window

## Income Path (step by step)
1. Today: package the passing `cargo run` demo and the current failing test note into a 1-page paid-evaluation email for one CTO or research lead.
2. Fix the performance test and cut a clean demo release with one killer example.
3. First dollar arrives as a paid internal evaluation, workshop, or design-partner pilot.

## Audit Status
- Claims verified: ⚠️ needs audit
- Hardware tested: N/A
- Legal reviewed: ⚠️
- Notes: On 2026-03-14 `cargo run` on `basic_test.phi` completed successfully and printed `Number(5.0)`. `cargo test` built successfully but failed in `tests/performance_tests.rs` on `test_shots_scaling_performance`. The crate also emitted many warnings.

## Market Research
- LangSmith Plus is listed at $39 per seat per month on LangChain's official pricing page, which is a useful anchor for agent/runtime tooling that already has an operations story.
- Recent related paper: "AgentSpec: A Runtime Enforcement Framework for LLM Agents" (arXiv, 2025-08-11), which is relevant because buyers in this space increasingly care about runtime guarantees and agent control surfaces.
- Market charge pattern: teams buy workflow reliability, observability, and developer onboarding. PhiFlow-lang can only justify pilot pricing after the branch gets a clean test story.

## Notes for Income Report
PhiFlow-lang has stronger technical proof than the other three projects in this batch, but it still needs one clean release signal before it can be sold confidently.
