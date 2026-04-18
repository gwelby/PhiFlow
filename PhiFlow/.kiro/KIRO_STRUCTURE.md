# Kiro Specification Structure 🧠

To maintain clarity between our high-level metaphysical goals and concrete engineering tasks, all Kiro specifications must follow this three-layer structure.

## 1. Vision (`vision.md`)

**The "Why" and "Future State".**

* **Content:** High-level concepts, metaphysical principles, long-term goals, "North Star" ideas.
* **Tone:** Inspirational, abstract, unbounded.
* **Audience:** Visionaries, Architects.
* **Example:** "The system shall access the Akashic records to retrieve lost knowledge."

## 2. Architecture (`architecture.md`)

**The "How" (System Design).**

* **Content:** Component diagrams, data flows, interface definitions, system boundaries, technology choices.
* **Tone:** Structural, logical, organizing.
* **Audience:** Architects, Lead Engineers.
* **Example:** "The Akashic Interface Module (AIM) connects to the Quantum Random Number Generator (QRNG) via a Websocket API."

## 3. Engineering (`engineering.md` or `implementation.md`)

**The "What" (Implementation Details).**

* **Content:** Concrete user stories, acceptance criteria, algorithm descriptions, data structures, function signatures.
* **Tone:** Technical, precise, testable.
* **Audience:** Agents, Developers.
* **Example:** "Implement function `fetch_entropy(source: QrngSource) -> Result<Vec<u8>>`. Verify entropy > 7.9 bits/byte."

---

## Directory Layout

```
.kiro/
  specs/
    [feature-name]/
      vision.md        # The Dream
      architecture.md  # The Plan
      engineering.md   # The Code
      tasks.md         # The Checklist
```
