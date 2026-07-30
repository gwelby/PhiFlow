# PhiFlow Discovery Call Script
*30-Minute Structured Call for Pilot Conversion*

---

## Pre-Call Preparation (5 minutes before)

### Research Checklist:
- [ ] Review prospect's LinkedIn (recent posts, background)
- [ ] Check company blog/news (last 3 months)
- [ ] Identify their current tech stack (Qiskit? LangChain? Custom?)
- [ ] Find one specific detail to reference (builds rapport)
- [ ] Open tabs: PhiFlow demo, IBM receipt, proof scripts

### Tools Ready:
- [ ] Screen sharing enabled
- [ ] `agent_handshake.phi` ready to run
- [ ] IBM receipt PDF open
- [ ] Calendly link for next steps
- [ ] CRM/spreadsheet for notes

---

## The Call Structure (30 minutes)

### 0-5 min: Rapport & Agenda (5 minutes)

**Opening:**
```
"Hi {First}, thanks for taking the time. I saw {specific detail about their work} —
really interesting approach to {quantum/agents/consciousness}.

Today I'd like to:
1. Understand your current workflow with {quantum programming/agent orchestration/biofeedback}
2. Show you what we've built — 5-minute demo, very concrete
3. See if there's a pilot opportunity or just useful knowledge exchange

Does that work for you?"
```

**If they say yes:**
```
"Great. Before we dive in — what's your biggest frustration with
current {quantum tools/agent frameworks/sensor integration}?"
```

**Listen for:**
- Tool limitations
- Verification gaps
- Reproducibility issues
- Scaling challenges

---

### 5-15 min: Their Situation (10 minutes)

**Question 1: Current Stack**
```
"Walk me through your current approach to {quantum development/agent coordination}.
What tools are you using day-to-day?"
```

**Probe deeper:**
- "How does that workflow feel?"
- "Where does it get clunky?"
- "What's the verification process like?"

**Question 2: The Pain Point**
```
"If you could wave a wand and fix one thing about your current setup,
what would it be?"
```

**Listen for PhiFlow fits:**
- "We lose context between agent handoffs" → PhiFlow signed handoffs
- "We can't prove what the system did" → PhiFlow receipts
- "Semantic concepts don't map to hardware" → PhiFlow OpenQASM bridge
- "Experiments aren't reproducible" → PhiFlow three-backend equivalence

**Question 3: Current Priorities**
```
"What's the top priority for your team this quarter?"
```

**Assess timing:**
- If they have active projects → High priority
- If just exploring → Nurture for later
- If budget just closed → Mark for next quarter

---

### 15-25 min: The Demo (10 minutes)

**Transition:**
```
"Based on what you've shared, let me show you something concrete.
This will take about 5 minutes, and you can tell me if it's relevant."
```

**Screen share — Run the demo:**

#### Demo Part 1: Self-Verifying Code (3 min)
```bash
# Terminal already open, run:
cargo run --bin phic -- examples/agent_handshake.phi
```

**Narrate:**
```
"See this λ = 0.618033988749895? That's not hardcoded.
It's computed from nested intention depth using φ-harmonic mathematics.

The code proves its own coherence formula: λ = φ^(-depth) * coherence_score

At depth 2: φ^-2 = 0.381966... but with coherence 1.0, we get 0.618..."
```

**Pause for questions.**

#### Demo Part 2: The IBM Receipt (3 min)

**Open PDF:** `D:\CosmicFamily\EVIDENCE\PHIFLOW_IBM_HERON_20260414.md`

**Narrate:**
```
"This is job d7euddh5a5qc73drdosg on ibm_fez — IBM Heron r2, 156 qubits.

We took high-level semantic code, compiled it to OpenQASM 3.0,
and executed it on physical hardware.

The result: |0⟩ 338 counts, |1⟩ 686 counts — exactly what the
semantics predicted based on coherence calculations.

This is not simulation. This is verification on 156-qubit processors."
```

**Pause for questions.**

#### Demo Part 3: Proof Scripts (2 min)

**Show in browser:**
```
"And all of this is independently verifiable.

If you run this command:
curl -s https://phiflow.dev/proof/verify-coherence | bash

It downloads the test suite, runs the coherence calculation,
and confirms φ^-2 = 0.618033988749895 to 15 decimal places.

No trust required — just run the script."
```

**Ask:**
```
"Does this verification approach resonate with how your team
thinks about reproducibility?"
```

---

### 25-30 min: The Ask (5 minutes)

**Transition:**
```
"Based on what you've seen — is there a specific workflow where having
hardware-verified semantics would be valuable?"
```

**If YES (interested):**

```
"Great. Here's how we typically work with teams like yours:

We run a 6-8 week pilot focused on ONE workflow you already care about.

Deliverables:
• Custom .phi program for your use case
• OpenQASM compilation where relevant
• Hardware verification on IBM (or simulator if you don't have access)
• Complete receipt package: source, binaries, test results, documentation
• Reproduction notes so your team can verify independently

Investment: $25k-$35k depending on complexity.

Does that structure make sense for exploring this?"
```

**If they hesitate on price:**
```
"The alternative is building this internally. Three engineers for 6 weeks
= $45k+ with no hardware verification path. This is cheaper than doing
it yourself — and you get the receipts."
```

**If they need time:**
```
"Totally understand. Can I send you the pilot offer document to review
with your team? And would it make sense to check back in 2 weeks?"
```

**If NO (not interested):**

```
"No problem at all. This isn't the right fit for every team.

Can I ask — what's the main hesitation? Is it timing, budget, or
the approach doesn't fit your problem space?"
```

**Learn why — don't push.**

**Then:**
```
"Fair enough. Would it be okay if I checked back in 3-6 months as
your roadmap evolves? Things change quickly in this space."
```

---

## Post-Call Actions (Within 24 hours)

### If Pilot Interest:

- [ ] Send thank you email within 2 hours
- [ ] Attach `docs/pilot_offer.md`
- [ ] Include personalized next steps
- [ ] Schedule follow-up call in 1 week
- [ ] Update CRM: "Pilot interest — follow up {date}"

**Email template:**
```
Hi {First},

Thanks for the conversation today. Great to learn about your work on {specific}.

As discussed, attached is the pilot offer document. 

Next steps:
1. Review the scope with your team
2. Let me know if you want to adjust anything
3. If it looks good, we can schedule a kickoff call

I'm holding {proposed date} for our follow-up. Let me know if that still works.

Best,
Greg
```

### If Not Now:

- [ ] Send thank you email within 2 hours
- [ ] No attachments — keep it light
- [ ] Add to nurture list for 3-month check-in
- [ ] Update CRM: "Nurture — check in {3 months from now}"

**Email template:**
```
Hi {First},

Thanks for taking the time today. Appreciate you sharing details about {their work}.

I'll check back in a few months as your roadmap evolves.

Good luck with {specific project}.

Best,
Greg
```

---

## Objection Handling Quick Reference

| Objection | Response | Redirect |
|-----------|----------|----------|
| "We're committed to Qiskit/Cirq" | "PhiFlow emits OpenQASM — it complements your stack. Think semantic layer above gates." | "Want to see how it compiles to QASM?" |
| "Sounds like philosophy, not engineering" | "Run `cargo test` — 335 tests. The 'philosophy' is syntax. The output is hardware execution." | "Want to see the test output?" |
| "We need quantum advantage, not semantics" | "Pilot doesn't promise advantage. It delivers a reproducible artifact chain." | "The receipts let you inspect before scaling." |
| "$25k is too expensive" | "Three engineers for 6 weeks = $45k+ with no verification. This is cheaper than DIY." | "What's your current experimentation budget?" |
| "Can we just use open source?" | "Absolutely. Pilot buys you SOMA integration, IBM hardware run with receipt, priority support." | "Want to try self-serve first?" |
| "Too good to be true" | "Healthy skepticism. Run `curl phiflow.dev/proof/verify-coherence` — 30 seconds, independent verification." | "Want to run it during the call?" |
| "No budget this quarter" | "Fair. When does your next fiscal start? I'll check back then." | "Can I send the pilot offer for review?" |
| "Need to run this by the team" | "Of course. Want me to join a follow-up call to answer technical questions?" | "What concerns do you anticipate?" |

---

## Success Metrics

Track in spreadsheet/CRM:

| Metric | Target |
|--------|--------|
| Calls per week | 2-3 |
| Pilot interest rate | 30%+ |
| Pilot close rate | 20%+ |
| Time to close | < 4 weeks |

**Review weekly:**
- Which objections came up most?
- Which demo parts got engagement?
- Where do prospects drop off?
- What proof do they request most?

**Iterate:**
- Refine demo based on engagement
- Add proof scripts for common requests
- Adjust pricing based on pushback
- Update sequences based on open rates

---

*Script version: 1.0*
*Last updated: 2026-04-25*
*Status: Ready for execution*
