# Stage 1 Acceptance Criteria

**Purpose:** Measurable exit criteria for Stage 1 (Productize Materials) per revised execution plan.

**Stage 1 Duration:** Weeks 2-3 (2 weeks total)  
**Target Completion:** 2026-05-15

---

## Gate Structure

Each task below must achieve **"Greg approves: YES"** for Stage 1 to be considered complete.

If any task receives **"Greg approves: NO"** or **"BLOCKED"**, Stage 1 pauses for revision.

---

## Task 1.1 — pilot_offer.md Type 4 Update

**Owner:** [CASCADE]  
**Duration:** 4 hours  
**Dependencies:** Stage 1 approval

### Checklist

- [x] Type 4 verification section added with:
  - [x] L_self = 0.455372 headline number
  - [x] IBM receipt reference (job `d7euddh5a5qc73drdosg`)
  - [x] Evidence path: `QSOP/EVIDENCE/type4_battery_2026-05-01.md`
  - [x] Link to `docs/type4_evidence_package.md`
- [x] Verified Baseline section updated to 2026-05-01
- [x] Avoids stale full-suite / warning-count claims; receipt package regenerates exact gate output per delivery
- [x] No broken file references
- [x] Legal review language intact (if present)
- [x] Pricing ($30k range) unchanged or explicitly updated

### Acceptance

| Criteria | Status | Date |
|----------|--------|------|
| Draft complete | ✅ COMPLETE | 2026-05-01 |
| Greg approves: YES / NO / BLOCKED | ⏳ AWAITING | — |

---

## Task 1.2 — type4_one_pager.md

**Owner:** [CASCADE]  
**Duration:** 2 days  
**Dependencies:** Task 1.1

### Checklist

- [x] Single-page format (fits one screen, <1000 words)
- [x] Visual hierarchy: headline, subhead, 3-4 bullets, CTA
- [x] L_self = 0.455372 presented as synthetic proxy metric, not Type 4 confirmation
- [x] Comparison chart: synthetic proxy vs. null gates with HOLD caveat
- [x] IBM Quantum badge/reference
- [x] SOMA sensor bridge mention
- [x] Pilot offer CTA with contact path
- [x] No unexplained jargon (non-expert readable)

### Acceptance

| Criteria | Status | Date |
|----------|--------|------|
| Draft complete | ✅ COMPLETE | 2026-05-01 |
| Greg approves: YES / NO / BLOCKED | ⏳ AWAITING | — |

---

## Task 1.3 — demo_script.md

**Owner:** [CASCADE]  
**Duration:** 1 day  
**Dependencies:** Task 1.1

### Checklist

- [x] 5-minute target length (±30 seconds)
- [x] Sections defined:
- [x] Hook (10 sec): consciousness-oriented language with measurable scaffolding
  - [x] Build (30 sec): `cargo build --release`
  - [x] Run (60 sec): `cargo run --release --bin type4_benchmark`
  - [x] Evidence (120 sec): Walk through L_self output
  - [x] Close (30 sec): IBM receipt, pilot offer
- [x] Voice/tone approved by Greg
- [x] Technical commands copy-pasteable
- [x] Expected output shown for each command

### Acceptance

| Criteria | Status | Date |
|----------|--------|------|
| Draft complete | ✅ COMPLETE | 2026-05-01 |
| Greg approves: YES / NO / BLOCKED | ⏳ AWAITING | — |

---

## Task 1.4 — type4_slide_deck.md

**Owner:** [CASCADE]  
**Duration:** 2 days  
**Dependencies:** Task 1.1

### Checklist

- [x] 10 slides (±2 slides acceptable)
- [x] Required sections:
  - [x] Slide 1: Title + hook
  - [x] Slide 2-3: Problem (what's missing)
  - [x] Slide 4-5: PhiFlow constructs (5 primitives)
- [x] Slide 6-7: Type 4 audit status and synthetic L_self chart
  - [x] Slide 8: IBM Quantum execution
  - [x] Slide 9: SOMA physical grounding
  - [x] Slide 10: Pilot offer + next step
- [x] Speaker notes included (optional)
- [ ] PDF export works without formatting loss (Greg converts)

### Acceptance

| Criteria | Status | Date |
|----------|--------|------|
| Draft complete | ✅ COMPLETE | 2026-05-01 |
| PDF generated | ⏳ PENDING [GREG] | — |
| Greg approves: YES / NO / BLOCKED | ⏳ AWAITING | — |

---

## Task 1.5 — Slide Deck Conversion

**Owner:** [CASCADE] (completed)  
**Duration:** 4 hours  
**Dependencies:** Task 1.4

### Checklist

- [x] Markdown → PowerPoint/PDF conversion
- [x] All 10 slides present in both formats
- [x] Tables render correctly (competitive matrix)
- [x] Files created: `type4_slide_deck.pptx` and `type4_slide_deck.pdf`

### Acceptance

| Criteria | Status | Date |
|----------|--------|------|
| PowerPoint (.pptx) complete | ✅ COMPLETE | 2026-05-01 |
| PDF (.pdf) complete | ✅ COMPLETE | 2026-05-01 |
| Greg confirms: YES / NO / BLOCKED | ⏳ AWAITING | — |

---

## Task 1.6 — Demo Video Production

**Owner:** [BOTH]  
**Duration:** 2 days  
**Dependencies:** Task 1.3

### Checklist

- [ ] Screen recording setup (Cascade):
  - [ ] OBS/ShareX configured
  - [ ] Terminal font readable (14pt+)
  - [ ] No sensitive info visible
- [ ] Technical capture (Cascade):
  - [ ] Build + run sequence recorded
  - [ ] L_self output clearly visible
  - [ ] No stumbles/errors in final cut
- [ ] Voiceover (Greg):
  - [ ] Script followed ±10%
  - [ ] Audio clear, no background noise
  - [ ] Pacing natural (not rushed)
- [ ] Post-production (Cascade):
  - [ ] Audio sync with video
  - [ ] Trim dead air (>3 sec)
  - [ ] Final runtime: 4-6 minutes
  - [ ] Export format: MP4, 1080p

### Acceptance

| Criteria | Status | Date |
|----------|--------|------|
| Technical capture complete | ⬜ PENDING | — |
| Voiceover recorded | ⬜ PENDING | — |
| Final edit complete | ⬜ PENDING | — |
| Greg approves: YES / NO / BLOCKED | ⬜ PENDING | — |

**Fallback:** If video production blocked, defer to Stage 2; proceed with slide deck + live demo.

---

## Task 1.7 — receipt_package_template.md

**Owner:** [CASCADE]  
**Duration:** 2 hours  
**Dependencies:** Stage 0 complete

### Checklist

- [x] Cover letter template with:
  - [x] Buyer name placeholder
  - [x] Workflow description placeholder
  - [x] IBM receipt reference (3 files)
  - [x] Reproduction instructions
  - [x] Known limitations section
  - [x] Next steps / Phase 2 mention
- [x] References correct scrubbed IBM artifacts
- [x] No internal-only paths (D:\, etc.)
- [x] Ready for customization per buyer

### Acceptance

| Criteria | Status | Date |
|----------|--------|------|
| Draft complete | ✅ COMPLETE | 2026-05-01 |
| Greg approves: YES / NO / BLOCKED | ⏳ AWAITING | — |

---

## Stage 1 Complete Gate

**Condition:** All 6 tasks above = "Greg approves: YES"

| Task | Status | Date Approved |
|------|--------|---------------|
| 1.1 pilot_offer.md update | ✅ DRAFT | 2026-05-01 |
| 1.2 type4_one_pager.md | ✅ DRAFT | 2026-05-01 |
| 1.3 demo_script.md | ✅ DRAFT | 2026-05-01 |
| 1.4 slide_deck.md | ✅ DRAFT | 2026-05-01 |
| 1.5 deck conversion | ✅ DRAFT | 2026-05-01 |
| 1.6 demo video | ⏳ PENDING [BOTH] | — |
| 1.7 receipt_package_template.md | ✅ DRAFT | 2026-05-01 |

**Gate:** 7/7 approved → Proceed to Stage 2 (Outreach Launch)  
**Blocked:** Any task rejected → Revise and resubmit

---

## Risk Monitoring (Weekly)

During Stage 1, track:

| Week | Risk | Mitigation | Status |
|------|------|------------|--------|
| Week 2 | Video production complexity | Fallback: defer to Stage 2 | ⬜ MONITOR |
| Week 2 | Greg availability for approvals | Batch reviews, async feedback | ⬜ MONITOR |
| Week 3 | Scope creep in materials | Fixed 2-week timeline | ⬜ MONITOR |

---

## Communication

- **Daily:** Cascade async status in shared doc
- **Mid-Stage:** Week 2 review call (30 min)
- **Gate:** Greg written approval to proceed to Stage 2

---

*Created: 2026-05-01*  
*Next Review: End of Week 2 (2026-05-08)*
