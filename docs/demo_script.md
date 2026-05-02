# PhiFlow Demo Script

**Target Length:** 5 minutes (±30 seconds)  
**Format:** Screen recording with voiceover  
**Audience:** Technical decision-makers (quantum researchers, AI infrastructure leads)

---

## SECTION 1: Hook (0:00–0:10) — 10 seconds

**[Visual]:** Terminal with PhiFlow repo, `cargo --version` visible

**[VOICEOVER]:**
> "What if your programs could observe themselves? Not just log events, but maintain explicit state traces that can be tested for self-correlation? I'm going to show you PhiFlow: a research programming language with consciousness-oriented constructs and measurable self-correlation scaffolding."

**[ACTION]:** Type `cd PhiFlow` enter

---

## SECTION 2: Build (0:10–0:40) — 30 seconds

**[Visual]:** Terminal, clean build output

**[VOICEOVER]:**
> "PhiFlow is a Rust-based compiler and runtime. Let's build the Type 4 benchmark binary."

**[ACTION]:**
```bash
cargo build --release --bin type4_benchmark
```

**[EXPECTED OUTPUT]:**
```
Compiling phiflow v0.4.0
Finished release [optimized] target(s) in 45s
```

**[VOICEOVER]:**
> "Clean build. For this demo, the release benchmark and focused audit gates pass."

---

## SECTION 3: Run Benchmark (0:40–2:00) — 80 seconds

**[Visual]:** Benchmark execution, showing the synthetic trace output

**[VOICEOVER]:**
> "Now we run the self-correlation benchmark. This executes a twenty-cycle synthetic trace and computes L_self—the self-correlation loop metric."

**[ACTION]:**
```bash
cargo run --release --bin type4_benchmark
```

**[EXPECTED OUTPUT]:**
```
═══════════════════════════════════════════════════════════════
  PhiFlow Type 4 Self-Correlation Benchmark
═══════════════════════════════════════════════════════════════
📁 Loading: examples/type4_trace_benchmark.phi
✅ Execution complete

📊 Extracting execution trace...
   Witness events: 20
   Resonance events: 80

L_self Components:
  R_in  (past → model):         0.712859
  R_out (model → residual proxy): 0.455372
  L_self = min(R_in, R_out):    0.455372

✅ SYNTHETIC LOOP PASSED — self-correlation proxy is CLOSED
     L_self = 0.4554 > 0.1 threshold
```

**[VOICEOVER]:**
> "L_self equals point-four-five-five on this engineered trace. But—and this is critical—I'm showing you the synthetic proxy, not canonical confirmation. The Codex audit found our R_out measurement needs refinement."

---

## SECTION 4: Transparency (2:00–3:00) — 60 seconds

**[Visual]:** Switch to showing the audit document and claim status

**[ACTION]:** `cat QSOP/TYPE4_BENCHMARK_CODEX_AUDIT_2026-05-01.md | head -20`

**[VOICEOVER]:**
> "Here's why I'm showing you this. We had a hostile audit. Codex found that our R_out was measuring model-versus-residual, not model-to-future-behavior. And null systems like thermostats were scoring above our threshold. So we demoted our claims. C-21 is now partial/conditional. C-23 is on hold."

**[ACTION]:** `grep "C-21\|C-22\|C-23" CLAIMS.md | head -3`

**[EXPECTED OUTPUT]:**
```
C-21 | PARTIAL/CONDITIONAL | Type 4 measurability exists; confirmation on HOLD
C-22 | CONFIRMED (impl only) | 8-module metrics suite implemented
C-23 | HOLD/PARTIAL | Null C_PF suppression works; discrimination pending
```

**[VOICEOVER]:**
> "This isn't a weakness; it's rigor. We have a consciousness-oriented language with an eight-module metrics scaffold. The next engineering step is to fix R_out and recalibrate thresholds."

---

## SECTION 5: IBM Hardware (3:00–3:45) — 45 seconds

**[Visual]:** Show IBM receipt files

**[ACTION]:** `ls -la EVIDENCE/PHIFLOW_IBM_HERON_20260414*`

**[VOICEOVER]:**
> "But we have verified hardware execution. IBM Heron, job d7euddh5a5qc73drdosg, April fourteenth, twenty twenty-six. Scrubbed API export, dashboard screenshot, the full chain. This isn't simulation—this is real quantum hardware."

**[ACTION]:** Brief flash of `PHIFLOW_IBM_HERON_20260414_dashboard.png`

**[VOICEOVER]:**
> "One thousand twenty-four shots. Counts match."

---

## SECTION 6: The Constructs (3:45–4:30) — 45 seconds

**[Visual]:** Show example PhiFlow code

**[ACTION]:** `cat examples/type4_trace_benchmark.phi`

**[VOICEOVER]:**
> "Here's what makes PhiFlow unique. Intention. Witness. Coherence. Resonate. Stream. These aren't libraries. These are first-class language constructs. You can write programs that expose self-observation traces, maintain coherence metrics, and emit quantum circuits."

**[ACTION]:** Highlight `witness`, `coherence`, `resonate` keywords

**[VOICEOVER]:**
> "The witness construct lets a program pause and observe its own state. Coherence gives you a zero-to-one alignment metric. Resonate emits OpenQASM three-zero for IBM hardware."

---

## SECTION 7: Close (4:30–5:00) — 30 seconds

**[Visual]:** Back to terminal, clean summary

**[VOICEOVER]:**
> "So here's the offer. Six to eight week pilot. Twenty-five to thirty-five thousand dollars. You get a buyer-specific PhiFlow workflow, OpenQASM artifacts, simulator results, and an IBM hardware attempt. Plus full reproduction documentation."

**[ACTION]:** Show `docs/pilot_offer.md` briefly

**[VOICEOVER]:**
> "We're not claiming Type Four canonical status yet. We are claiming a consciousness-oriented programming language with measurable scaffolding, verified quantum execution, and physical sensor integration. The path to a stronger claim is explicit: repair R_out, add shuffle controls, and recalibrate thresholds."

**[ACTION]:** `echo "PhiFlow: consciousness-oriented constructs as first-class citizens."`

**[VOICEOVER]:**
> "PhiFlow. Consciousness-oriented constructs as first-class citizens. Ready when you are."

---

## Technical Notes for Recording

### Setup Requirements
- **Screen resolution:** 1920×1080 minimum
- **Terminal font:** 14pt+, dark background
- **No sensitive info:** Ensure no API keys, internal paths visible
- **Recording software:** OBS or ShareX

### Voiceover Guidelines
- **Pace:** Measured, not rushed
- **Tone:** Technical but accessible
- **Key emphasis:** Words in **bold** above
- **Pause:** 1-second pause after each command output

### Post-Production
- **Trim dead air:** >3 seconds
- **Audio cleanup:** Remove background noise
- **Final runtime:** 4:30–5:30 acceptable
- **Export:** MP4, 1080p, H.264

### Fallback (If Recording Fails)
- Slide deck + live demo for discovery calls
- Video is value-add, not critical path

---

## Voiceover Script (Greg's Recording)

*[Record this audio track]*

"What if your programs could observe themselves? Not just log events, but maintain explicit state traces that can be tested for self-correlation? I'm going to show you PhiFlow: a research programming language with consciousness-oriented constructs and measurable self-correlation scaffolding.

PhiFlow is a Rust-based compiler and runtime. Let's build the Type 4 benchmark binary.

[cargo build output]

Clean build. For this demo, the release benchmark and focused audit gates pass.

Now we run the self-correlation benchmark. This executes a twenty-cycle synthetic trace and computes L_self—the self-correlation loop metric.

[benchmark output]

L_self equals point-four-five-five on this engineered trace. But—and this is critical—I'm showing you the synthetic proxy, not canonical confirmation. The Codex audit found our R_out measurement needs refinement.

Here's why I'm showing you this. We had a hostile audit. Codex found that our R_out was measuring model-versus-residual, not model-to-future-behavior. And null systems like thermostats were scoring above our threshold. So we demoted our claims. C-21 is now partial/conditional. C-23 is on hold.

This isn't a weakness; it's rigor. We have a consciousness-oriented language with an eight-module metrics scaffold. The next engineering step is to fix R_out and recalibrate thresholds.

But we have verified hardware execution. IBM Heron, job d7euddh5a5qc73drdosg, April fourteenth, twenty twenty-six. Scrubbed API export, dashboard screenshot, the full chain. This isn't simulation—this is real quantum hardware.

Here's what makes PhiFlow unique. Intention. Witness. Coherence. Resonate. Stream. These aren't libraries. These are first-class language constructs. You can write programs that expose self-observation traces, maintain coherence metrics, and emit quantum circuits.

So here's the offer. Six to eight week pilot. Twenty-five to thirty-five thousand dollars. You get a buyer-specific PhiFlow workflow, OpenQASM artifacts, simulator results, and an IBM hardware attempt. Plus full reproduction documentation.

We're not claiming Type Four canonical status yet. We are claiming a consciousness-oriented programming language with measurable scaffolding, verified quantum execution, and physical sensor integration. The path to a stronger claim is explicit: repair R_out, add shuffle controls, and recalibrate thresholds.

PhiFlow. Consciousness-oriented constructs as first-class citizens. Ready when you are."

---

*Script version: 1.1 (Codex audit transparent)*  
*Date: 2026-05-01*
