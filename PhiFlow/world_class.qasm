warning: value assigned to `changed` is never read
   --> src\phi_ir\optimizer.rs:128:17
    |
128 |                 changed = true;
    |                 ^^^^^^^
    |
    = help: maybe it is overwritten before being read?
    = note: `#[warn(unused_assignments)]` on by default

warning: `phiflow` (lib) generated 1 warning
    Finished `release` profile [optimized] target(s) in 0.56s
     Running `target\release\phic.exe examples/world_class_fundamentals.phi --target openqasm`
Compiling to PhiFlow IR...
WARNING: CoherenceCheck applied to qubit [3] AFTER it was witnessed mid-circuit. Qubit state is collapsed.
OPENQASM 3.0;
include "stdgates.inc";

qubit[4] q;
bit[4] c;

// Block entry
// Intention: The_Engineer
    ry(1 * pi) q[0]; // Resonate
// Intention: The_Poet
// Block stream_header
// Block stream_body
    ry(0.8 * pi) q[1]; // Resonate
    ry(0.6180339887 * pi) q[1]; // Coherence
// Block stream_exit
    cx q[0], q[1]; // Entangle via 432Hz
// Intention: The_Duck
    ry(1 * pi) q[3]; // Resonate
    ry(pi - (1 * pi)) q[3]; // Resonate
    cx q[1], q[3]; // Entangle via 432Hz
    c[0] = measure q[0]; // MidCircuit Witness q0
    c[1] = measure q[1]; // MidCircuit Witness q1
    c[2] = measure q[2]; // MidCircuit Witness q2
    c[3] = measure q[3]; // MidCircuit Witness q3
    // WARNING: Coherence post-collapsed qubit q[3]
    ry(0.6180339887 * pi) q[3]; // Coherence
// Block then
// Block else
// Block merge
// Block unreachable_after_break
