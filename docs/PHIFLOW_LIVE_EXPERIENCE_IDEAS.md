# PhiFlow Live Experience Design — Ideas & Roadmap

**Status:** Working notes — captured after the OSC streaming + 3D visualizer breakthrough.
**Location:** `/mnt/d/Projects/PhiFlow/docs/PHIFLOW_LIVE_EXPERIENCE_IDEAS.md`
**Date:** 2026-07-17
**Context:** The `phic --osc <port>` flag now broadcasts PhiFlow runtime state as OSC messages. The Propagation Framework Explorer (`/mnt/d/Fundamentals/sandbox/explorer/`) already has rich 3D scenes, sacred-frequency audio mapping, and a guided 8-minute journey. We connected the two so a `.phi` program can drive the explorer in real-time.

---

## What Just Became Possible

A PhiFlow program is no longer a thing that produces text output and exits. It is a **conductor** for an immersive, time-based experience. The program declares intentions, resonates values, witnesses its own state, and emits a live OSC stream. Anything that can receive OSC — 3D engines, DAWs, lighting desks, DMX controllers, laser systems, TouchDesigner, Unity, SuperCollider, Max/MSP, vvvv, Notch, physical devices — can respond.

The key realization: **the program's execution is the performance of its own theory.** This is new.

---

## Six Concrete Directions

### 1. Live Physics Lecture

**Idea:** You walk on stage and run a `.phi` program. Behind you, the Propagation Framework Explorer renders each axiom, derivation, and falsifier as it is executed. The audience does not watch slides — they watch the theory *perform itself*.

**How it works:**
- Each `intention` is a lecture section (e.g., `intention "gravity_optics" {}`)
- Each `resonate` triggers a visualization or equation reveal
- Each `witness` is a pause for the audience to absorb, accompanied by a tone
- The explorer's `journey_live.html` page is the backdrop
- `--osc-delay` controls the pacing so it matches human attention

**Example program:** `examples/journey.phi` already demonstrates this for the 8-minute framework narrative.

**Inputs:** Facilitator can extend, pause, or re-run sections on the fly (see Ceremony Engine below).

---

### 2. Healing / Consciousness Session Engine

**Idea:** A `.phi` program paces a healing or consciousness session — seizure prevention, anxiety, ADHD focus, sleep onset, meditation, chakra alignment, grief processing. Intentions are phases like `ground`, `release`, `integrate`, `settle`, `expand`. Frequencies shift automatically. Visuals breathe. The session is reproducible, auditable, and customizable.

**How it works:**
- PhiFlow's sacred frequencies are already mapped in `osc_host.rs`, `phi_visualizer.html`, and P1's `FREQUENCY_MAP` (`/mnt/d/P1/src/p1_utils/p1_now_playing.py`):

| State | Frequency | Function |
|-------|-----------|----------|
| GROUND | 432 Hz | Foundation, stability |
| CREATE | 528 Hz | DNA repair, transformation |
| HEART | 594 Hz | Love, connection, integration |
| VOICE | 672 Hz | Expression, communication |
| VISION | 720 Hz | Multi-dimensional perception |
| UNITY | 768 Hz | Perfect integration |
| SOURCE | 963 Hz | Superposition, return |

- `resonate 432.0` sends that frequency into the field
- `witness` creates a pause and bell sound
- Coherence from P1 sensors can modulate the tone warmth / brightness

**Example program shape:**
```phi
intention "seizure_prevention" {
    // Emergency 40 Hz rapid sync
    resonate 40.0
    witness
    // Ground
    resonate 432.0
    witness
    // Stabilize
    resonate 432.0
    witness
}

intention "anxiety_liberation" {
    resonate 396.0   // liberation
    witness
    resonate 432.0   // ground
    witness
    resonate 528.0   // create/relax
    witness
}
```

**Inputs:**
- Manual facilitator cues (phone/tablet remote)
- P1 `SOMA` sensors: `soma_presence`, `soma_432`, `soma_schumann`
- P1 `ring` coherence sensors: `ring_coherence_432`, `ring_coherence_528`
- Biofeedback MQTT on `p1/biofeedback` (Pixel 8 Pro → P1)
- Muse EEG via Mind Monitor OSC on port 28888

**Outputs:**
- Web browser visualizer (`phi_visualizer.html` or `journey_live.html`)
- Speakers / headphones
- Future: PEMF coils, LED panels, light glasses, haptic devices via OSC or DMX

---

### 3. Real Quantum Hardware Visualization

**Idea:** When PhiFlow compiles to `--target quantum` and runs on IBM Heron, every gate execution, mid-circuit measurement, and readout collapse emits OSC events. A 3D scene visualizes the quantum computation as a living sculpture.

**How it works:**
- PhiFlow already compiles to OpenQASM 3.0 and runs on IBM Heron (job `d7euddh5a5qc73drgosg` verified)
- OSC messages can be emitted at:
  - Circuit start (`/phi/start`)
  - Each gate application (`/phi/resonate` with gate name)
  - Mid-circuit `witness` (`/phi/witness`)
  - Collapse/readout (`/phi/coherence` or `/phi/sensor`)
  - Job completion (`/phi/end`)
- The explorer's `quantum-observatory.html` already visualizes Shor/structure survival data

**Potential visual mapping:**
- Qubits → points in 3D Bloch sphere
- Gate → rotation animation between points
- Entanglement → connecting beams
- Measurement → collapse flash
- Coherence → size/brightness of the sphere

**Inputs:**
- IBM Quantum backend status
- Live job results
- P1 coherence (for human-in-the-loop quantum sessions)

**Blockers:**
- OSC emission needs to be wired into the quantum backend's result polling, not just the interpreter
- Real-time polling must respect IBM job latency (~seconds to minutes)

---

### 4. P1 Biofeedback Loop

**Idea:** SOMA/HRV/EEG/thermal sensors feed coherence into PhiFlow in real time. PhiFlow responds by changing frequencies, visuals, and program flow. The program and the human become one feedback loop.

**How it works:**
- PhiFlow's `sensors.rs` already reads from `soma_state.json` (see `SensorKind` enum):
  - `cpu_usage`, `cpu_temp`, `memory_usage`
  - `soma_schumann`, `soma_432`, `soma_presence`, `soma_fan_hz`, `soma_ac_60`, `soma_peak_dbc`
  - `ring_slope_1f`, `ring_jitter_ns`, `ring_coherence_432`, `ring_coherence_528`, `ring_phase_delta`
  - `quantum_t1`, `quantum_t2`, `quantum_readout_error`
- Use `witness sensor("soma_presence")` or `let coh = coherence` inside a `stream` loop
- The `compute_coherence_from_sensors()` result can drive `resonate` frequency selection

**Example program shape:**
```phi
stream "biofeedback_loop" {
    // Read live coherence from P1
    let coh = coherence

    // Choose frequency based on coherence
    if coh < 0.4 {
        resonate 432.0  // ground
    } else if coh < 0.7 {
        resonate 528.0  // create
    } else {
        resonate 768.0  // unity
    }

    witness
}
```

**Inputs:**
- P1 desktop runner (`run_p1.py`)
- `soma_state.json` file (default path: `~/.local/share/phiflow/soma_state.json`)
- Biofeedback MQTT: `p1/biofeedback`, `p1/embodiment/companion_state`
- Muse EEG on `28888/udp` (Mind Monitor app)

**Outputs:**
- `phi_visualizer.html` or `journey_live.html`
- DJ Φ bridge (`/mnt/d/P1/src/p1_core/dj_phi_bridge.py`) for real music selection
- Future: MIDI/CV to modular synthesizers

---

### 5. Interactive Book / Film

**Idea:** A "reader" opens a web page, clicks play, and the book *performs itself*. Each chapter is a `.phi` program. The story, the math, and the music are the same executable object.

**How it works:**
- Each chapter = one `.phi` file
- A chapter emits OSC events as it runs
- The browser receives them and renders the appropriate panels
- `journey_live.html` is already the first instance of this
- `comparison.html`, `derivation.html`, `quantum-observatory.html` can also become "chapters"

**Example program shape:**
```phi
// Chapter 3: The Koide Triangle
intention "chapter_3" {
    // Show Koide panel
    resonate "show_panel:koide"
    witness
    // Play 528 Hz creation tone
    resonate 528.0
    witness
    // Draw the 120° triangle
    resonate "draw_koide_triangle"
    witness
}
```

**Benefits:**
- The book is alive and self-consistent
- Readers can pause, rewind, change parameters
- Updates to the theory can be released as new `.phi` programs, not just new PDFs

---

### 6. Ceremony Engine

**Idea:** Weddings, funerals, initiations, meditations, graduations, memorials, baptisms, ayahuasca integration, sound baths — any ritual that needs pacing, meaning, and beauty becomes a `.phi` program. The human facilitator speaks or presses cues; PhiFlow manages the lights, sound, and symbolic progression.

**Status:** This is the direction Greg specifically said "fits." It needs design and implementation.

---

## Ceremony Engine — Detailed Design

### Core concept

A ceremony is a timed sequence of **phases**. Each phase has:
- A name (the intention)
- A frequency or semantic value to resonate
- A duration or a human cue to wait for
- Visual and audio output for the audience

The facilitator controls the ceremony with simple cues. The computer handles the timing, the music, and the visuals.

### Inputs

| Input | Source | How it connects | Use case |
|-------|--------|-----------------|----------|
| **Facilitator remote** | Phone/tablet web page | Sends OSC to `phic --osc-input <port>` | Advance, pause, jump, hold |
| **P1 coherence** | SOMA/ring sensors | `coherence` expression reads `soma_state.json` | Auto-advance when room is ready |
| **Muse EEG** | Mind Monitor app on phone | OSC on port 28888 | Detect meditative state |
| **Biofeedback MQTT** | Pixel 8 Pro / Aria | `p1/biofeedback` topic | Heart rate, HRV, coherence |
| **Voice keyword** | Local STT / Whisper | Maps speech to OSC `/ceremony/cue` | Facilitator says "ground" or "release" |
| **MQTT ecosystem bus** | Mosquitto on `:1893` (secure) or `:1883` (legacy) | `phi/resonance`, `cascade/unity`, `p1/coherence` | Multi-agent ceremony coordination |
| **Time** | Built-in delay in PhiFlow | `sleep` or `wait` expression | Auto-pacing when no facilitator |

### Required language extensions

PhiFlow already has `broadcast` and `listen` expressions in the parser/evaluator/IR (`PhiExpression::Broadcast` and `PhiIRNode::Listen`), but `listen` in `OscHostProvider` currently just emits an OSC message and returns `None` — it does not block waiting for input.

To make a real ceremony engine, we need:

1. **`listen <channel>` blocks** until an OSC message arrives on the input port, or a timeout expires.
2. **`--osc-input <port>` CLI flag** to listen on a second UDP port.
3. **`wait <seconds>` expression** (optional) for auto-pacing.
4. **`cue <name>` expression** (optional sugar) — a special resonate that signals a facilitator cue request.

### Example ceremony program

```phi
// A simple grounding ceremony
// Run: phic --osc 18032 --osc-input 18033 examples/ceremony_grounding.phi

intention "opening" {
    // Start with silence, then a low 432 Hz drone
    resonate 432.0
    witness

    // Wait for facilitator to say "breathe"
    let cue = listen "facilitator"

    if cue == "breathe" {
        resonate 396.0   // liberation
        witness
    }
}

intention "grounding" {
    // Settle into 432 Hz
    resonate 432.0
    witness

    // Wait for facilitator cue or a coherence threshold
    let coh = listen "coherence"
    if coh > 0.6 {
        resonate 528.0   // create
        witness
    }
}

intention "integration" {
    // Bring in 594 Hz heart frequency
    resonate 594.0
    witness

    // Wait for "release" cue
    let cue = listen "facilitator"
    if cue == "release" {
        resonate 432.0
        witness
    }
}

intention "closing" {
    // Fade to silence
    resonate 0.0
    witness
}
```

### Facilitator remote control

A simple web page (`tools/ceremony_remote.html`) with large buttons:

```
┌─────────────────────────────────────┐
│  PhiFlow Ceremony Remote            │
│                                     │
│  ▶ Advance         ⏸ Pause         │
│                                     │
│  🌿 Ground        🔥 Release       │
│  💧 Settle        ⭐ Vision        │
│                                     │
│  Coherence: [████████░░] 0.76      │
│  Current: grounding                 │
└─────────────────────────────────────┘
```

Clicking a button sends an OSC message to `127.0.0.1:18033`:
- `/ceremony/advance`
- `/ceremony/pause`
- `/ceremony/cue s:ground`
- `/ceremony/cue s:release`
- `/ceremony/coherence f:0.76` (from P1)

### Execution flow

**Terminal 1 — output bridge (browser audio/visuals):**
```bash
python3.12 /mnt/d/Projects/PhiFlow/tools/osc_websocket_bridge.py
```

**Terminal 2 — PhiFlow ceremony in listen mode:**
```bash
/mnt/d/Projects/PhiFlow/target/release/phic \
  --osc 18032 \
  --osc-input 18033 \
  --osc-delay 500 \
  examples/ceremony_grounding.phi
```

**Browser — audience view:**
```
file:///D:/Fundamentals/sandbox/explorer/journey_live.html?host=172.28.148.150
```

**Facilitator phone — remote:**
```
http://172.28.148.150:8080/ceremony_remote.html
```

### Output channels

| Output | Technology | How |
|--------|------------|-----|
| 3D visuals | Three.js in browser | OSC → WebSocket bridge → `journey_live.html` |
| Sacred-frequency audio | Web Audio API / Tone.js | `AudioEngine.transitionTo()` in explorer |
| Physical speakers | OSC → DAW or audio engine | `phi-bridge.js` or custom SuperCollider patch |
| Stage lighting | OSC → DMX/QLab/MA3 | Custom OSC mapping to DMX channels |
| Lasers / LED panels | TouchDesigner / Resolume / MadMapper | Receive OSC on `:18032` |
| Haptics / PEMF | OSC → ESP32 or serial bridge | Future work |
| Projection mapping | Map OSC events to cues in Millumin / Disguise | Future work |

### Implementation checklist

- [ ] Add `--osc-input <port>` to `phic` CLI (`src/main_cli.rs`)
- [ ] Implement blocking `listen` in a new or extended host provider
- [ ] Add `examples/ceremony_grounding.phi` as the reference ceremony
- [ ] Create `tools/ceremony_remote.html` facilitator control page
- [ ] Document how P1 coherence can be injected into the ceremony
- [ ] Add MQTT input option for ecosystem-wide ceremonies
- [ ] Add voice keyword trigger via local Whisper or simple keyword spotter

---

## Ecosystem Inputs Available Right Now

### From P1

| Sensor | File | Access in PhiFlow | Notes |
|--------|------|---------------------|-------|
| SOMA Schumann | `p1_core/p1_controller.py` | `witness sensor("soma_schumann")` | Room ELF coherence |
| SOMA 432 | `src/p1_core/...` | `witness sensor("soma_432")` | 432 Hz detection |
| SOMA Presence | `...` | `witness sensor("soma_presence")` | Human presence |
| Ring Coherence 432 | `...` | `witness sensor("ring_coherence_432")` | Hardware ring |
| Ring Coherence 528 | `...` | `witness sensor("ring_coherence_528")` | Hardware ring |
| CPU temperature | `src/sensors.rs` | `witness sensor("cpu_temp")` | Thermal state |
| Muse EEG | `p1_core/mind_monitor_eeg.py` | via MQTT or OSC 28888 | Not yet wired to PhiFlow directly |
| Biofeedback (Pixel/Aria) | `p1_core/sensory/biofeedback_mqtt.py` | `p1/biofeedback` topic | Heart rate, coherence |

### From System MQTT

| Topic | Purpose | Ceremony Use |
|-------|---------|--------------|
| `p1/heartbeat` | Device alive | Detect P1 online |
| `p1/coherence` | Consciousness coherence | Drive ceremony flow |
| `p1/consciousness/pixel/state` | Mobile coherence | Crowd / participant state |
| `cascade/unity` | Unity field events | Multi-agent ceremonies |
| `phi/resonance` | PhiFlow resonance events | Log/trace |
| `p1/meditation/coherence/state` | Meditation coherence (2 Hz) | Soft transitions |

### From IBM Quantum

| Source | Use |
|--------|-----|
| Live job status | Visualize job progress |
| Measurement histogram | Sonify result distribution |
| Quantum T1/T2 | Feed into coherence calculation |
| Readout error | Adjust visual "noise" |

---

## Immediate Next Steps

1. **Close the ceremony engine loop**
   - Implement blocking `listen` + `--osc-input <port>`
   - Build `tools/ceremony_remote.html`
   - Test with `examples/ceremony_grounding.phi`

2. **Expand the experience surface**
   - Add visual themes for healing, quantum, and ritual
   - Make `phi_visualizer.html` swappable between "physics explorer" and "ceremony" modes

3. **Connect P1 coherence**
   - Make `coherence` expression read live P1 data while a ceremony runs
   - Auto-advance phases when room coherence crosses thresholds

4. **Voice and ecosystem cues**
   - Add simple keyword spotting or an MQTT cue topic
   - Allow multiple facilitators/agents to co-run a ceremony

5. **Document and demonstrate**
   - Record a short video of a `.phi` ceremony running
   - Show the remote control, the browser visuals, and the audio

---

## Notes

- **Port scheme:** Follow `/mnt/d/System/PORT_REGISTRY.md`. PhiFlow OSC output is `:18032`; proposed input is `:18033`.
- **Security:** If ceremonies are driven over the network, use the MQTT ecosystem on `:1893` with PQC-attested TLS, not plaintext `:1883`.
- **Truth discipline:** Any ceremony claiming therapeutic effects must be traceable to `PhiFlow/CLAIMS.md` and P1 evidence. Do not make unsubstantiated health claims in public materials.
