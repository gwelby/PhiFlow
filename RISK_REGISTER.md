# PhiFlow Risk Register

**Purpose:** Track top 10 risks across Stages 1-4 per revised execution plan.  
**Last Updated:** 2026-05-01  
**Next Review:** Weekly (Tuesdays)

---

## Risk Legend

| Status | Meaning |
|--------|---------|
| 🟢 OPEN | Risk identified, monitoring active |
| 🟡 MONITORING | Risk elevated, mitigation in progress |
| 🔴 TRIGGERED | Risk event occurred, action required |
| ⚫ CLOSED | Risk resolved or no longer applicable |

---

## Top 10 Risks (Revised Plan)

### 1. Zero Responses to Outreach

| Field | Value |
|-------|-------|
| **ID** | R-001 |
| **Description** | Cold outreach to quantum labs yields no responses |
| **Probability** | 40% |
| **Impact** | CRITICAL |
| **Status** | 🟢 OPEN |
| **Owner** | [BOTH] |
| **Mitigation** | Multiple buyer profiles; flexible pricing; strong Type 4 differentiator |
| **Contingency** | Pivot to agent infrastructure market (50/50 split) by Week 6 |
| **Trigger** | Zero responses by Week 6 (2026-06-12) |
| **Last Reviewed** | 2026-05-01 |
| **Next Review** | 2026-05-08 |
| **Notes** | Quantum labs move slowly (academic calendars, grant cycles). Expect 7-14 day response time. |

---

### 2. Pricing Rejection ($30k)

| Field | Value |
|-------|-------|
| **ID** | R-002 |
| **Description** | Buyers reject $30k pilot price as too high |
| **Probability** | 30% |
| **Impact** | HIGH |
| **Status** | 🟢 OPEN |
| **Owner** | [GREG] |
| **Mitigation** | Flex range $25-35k; academic/non-profit discounts |
| **Contingency** | Survey buyers; adjust to $20-25k range if 3+ rejections |
| **Trigger** | 3+ explicit price objections |
| **Last Reviewed** | 2026-05-01 |
| **Next Review** | 2026-05-22 (after initial outreach) |
| **Notes** | Comparable quantum tools (Qiskit, Cirq) are free. Value proposition must be clear. |

---

### 3. IBM Hardware Access Failure

| Field | Value |
|-------|-------|
| **ID** | R-003 |
| **Description** | Cannot execute IBM hardware run during pilot (account/credits/queue issues) |
| **Probability** | 20% |
| **Impact** | HIGH |
| **Status** | 🟢 OPEN |
| **Owner** | [CASCADE] |
| **Mitigation** | Fallback: Qiskit Aer simulator with hardware noise model |
| **Contingency** | Buyer provides their own IBM access if needed |
| **Trigger** | IBM job submission fails or queue time >2 weeks |
| **Last Reviewed** | 2026-05-01 |
| **Next Review** | When Stage 4 begins |
| **Notes** | Fallback explicitly defined in pilot_offer.md. Not a blocker if simulator produces valid results. |

---

### 4. Greg Unavailability During Critical Week

| Field | Value |
|-------|-------|
| **ID** | R-004 |
| **Description** | Greg unavailable for approvals/calls during key decision points |
| **Probability** | 25% |
| **Impact** | HIGH |
| **Status** | 🟢 OPEN |
| **Owner** | [GREG] |
| **Mitigation** | 1-week buffer added to timeline; async handoff plan |
| **Contingency** | Pause timeline; notify buyer with new dates |
| **Trigger** | Greg unavailable >1 week |
| **Last Reviewed** | 2026-05-01 |
| **Next Review** | Weekly |
| **Notes** | Discovery calls, contract negotiation, and final handoff require Greg. Cascade cannot substitute. |

---

### 5. Scope Creep in Pilot

| Field | Value |
|-------|-------|
| **ID** | R-005 |
| **Description** | Buyer requests features outside defined pilot scope |
| **Probability** | 35% |
| **Impact** | MEDIUM |
| **Status** | 🟢 OPEN |
| **Owner** | [BOTH] |
| **Mitigation** | Fixed-scope contract; Phase 2 option clause; clear exclusions |
| **Contingency** | Renegotiate timeline/price for expanded scope as Phase 2 |
| **Trigger** | Scope expansion >25% of original estimate |
| **Last Reviewed** | 2026-05-01 |
| **Next Review** | During Stage 3 (negotiation) and Stage 4 (implementation) |
| **Notes** | Phase 2 pricing = $15-25k depending on scope. Document all scope change requests. |

---

### 6. Video Production Blocked

| Field | Value |
|-------|-------|
| **ID** | R-006 |
| **Description** | 5-minute demo video cannot be completed within Stage 1 timeline |
| **Probability** | 20% |
| **Impact** | MEDIUM |
| **Status** | 🟢 OPEN |
| **Owner** | [BOTH] |
| **Mitigation** | 2-day allocation (realistic); Cascade handles technical setup |
| **Contingency** | Defer video to Stage 2; lead with slide deck + live demo |
| **Trigger** | Video not complete by end of Week 3 |
| **Last Reviewed** | 2026-05-01 |
| **Next Review** | End of Week 2 (2026-05-08) |
| **Notes** | Video is value-add, not critical path. Slide deck + live demo sufficient for outreach. |

---

### 7. Receipt Credibility Challenged

| Field | Value |
|-------|-------|
| **ID** | R-007 |
| **Description** | Buyer questions authenticity of IBM receipt evidence |
| **Probability** | 15% |
| **Impact** | MEDIUM |
| **Status** | 🟢 OPEN |
| **Owner** | [CASCADE] |
| **Mitigation** | Validation checklist; raw JSON + dashboard screenshot included |
| **Contingency** | Offer live demo run during pilot engagement |
| **Trigger** | Buyer requests additional verification |
| **Last Reviewed** | 2026-05-01 |
| **Next Review** | During Stage 3 (discovery calls) |
| **Notes** | Receipt includes job ID, backend, timestamp, counts. All verifiable via IBM Quantum dashboard. |

---

### 8. Competitor Emerges

| Field | Value |
|-------|-------|
| **ID** | R-008 |
| **Description** | Competing Type 4 or consciousness-language project enters market |
| **Probability** | 10% |
| **Impact** | MEDIUM |
| **Status** | 🟢 OPEN |
| **Owner** | [GREG] |
| **Mitigation** | Move fast; emphasize hardware receipts + verified metrics (unique) |
| **Contingency** | Emphasize IBM Heron receipts + physical sensor integration (SOMA) as differentiators |
| **Trigger** | News of competing project with similar claims |
| **Last Reviewed** | 2026-05-01 |
| **Next Review** | Monthly |
| **Notes** | Hardware verification + physical grounding are hard to replicate quickly. First-mover advantage significant. |

---

### 9. Compiler Warning Persists

| Field | Value |
|-------|-------|
| **ID** | R-009 |
| **Description** | Unused import warning reappears or new warnings introduced |
| **Probability** | 10% |
| **Impact** | LOW |
| **Status** | ⚫ CLOSED |
| **Owner** | [CASCADE] |
| **Mitigation** | CI check for zero warnings; pre-commit hook |
| **Contingency** | Document as known issue; fix post-pilot |
| **Trigger** | `cargo build --release` shows any warning |
| **Last Reviewed** | 2026-05-01 |
| **Next Review** | Weekly build verification |
| **Notes** | ✅ RESOLVED: Fixed unused `ConsciousnessMetrics` import in `src/bin/type4_benchmark.rs`. Build now shows zero warnings. |

---

### 10. Academic Buyer Requires Publication

| Field | Value |
|-------|-------|
| **ID** | R-010 |
| **Description** | Discovery call reveals buyer requires peer-reviewed publication before adoption |
| **Probability** | 15% |
| **Impact** | LOW |
| **Status** | 🟢 OPEN |
| **Owner** | [CASCADE] |
| **Mitigation** | Evidence package includes all data needed for paper; optional fast-track arxiv |
| **Contingency** | Fast-track arxiv preprint (2-3 weeks) if required |
| **Trigger** | Buyer explicitly states publication requirement |
| **Last Reviewed** | 2026-05-01 |
| **Next Review** | During Stage 3 (discovery calls) |
| **Notes** | Publication deferred to post-pilot per plan. Can be accelerated if buyer is co-author candidate. |

---

## Risk Summary Dashboard

| Status | Count | Risks |
|--------|-------|-------|
| 🟢 OPEN | 9 | R-001, R-002, R-003, R-004, R-005, R-006, R-007, R-008, R-010 |
| 🟡 MONITORING | 0 | — |
| 🔴 TRIGGERED | 0 | — |
| ⚫ CLOSED | 1 | R-009 (compiler warning fixed) |
| **Total** | **10** | — |

---

## Review History

| Date | Action | Risks Updated |
|------|--------|---------------|
| 2026-05-01 | Created | All 10 risks initialized |
| — | — | — |

---

## Next Review

**Date:** 2026-05-08 (Tuesday)  
**Format:** Weekly check-in per `docs/weekly_checkin_template.md`

---

*Source: Revised execution plan `phiflow-type4-revised-execution-plan-c19981.md`*  
*Audit: Bob (Advanced Mode) hostile audit 2026-05-01*
