Compiling to PhiFlow IR...
OPENQASM 3.0;
include "stdgates.inc";

qubit[8] q;
bit[8] c;

// Block entry
// Intention: Q0
    ry(1 * pi) q[0]; // Resonate
// Intention: Q1
    ry(1 * pi) q[1]; // Resonate
    cx q[0], q[1]; // Entangle via 432Hz
// Intention: Q2
    ry(1 * pi) q[2]; // Resonate
    cx q[0], q[2]; // Entangle via 432Hz
// Intention: Q3
    ry(1 * pi) q[3]; // Resonate
    cx q[1], q[3]; // Entangle via 432Hz
// Intention: Q4
    ry(1 * pi) q[4]; // Resonate
    cx q[1], q[4]; // Entangle via 432Hz
// Intention: Q5
    ry(1 * pi) q[5]; // Resonate
    cx q[2], q[5]; // Entangle via 432Hz
// Intention: Q6
    ry(1 * pi) q[6]; // Resonate
    cx q[2], q[6]; // Entangle via 432Hz
// Intention: Q7
    ry(1 * pi) q[7]; // Resonate
    cx q[3], q[7]; // Entangle via 432Hz
    c[0] = measure q[0]; // Witness q0
    c[1] = measure q[1]; // Witness q1
    c[2] = measure q[2]; // Witness q2
    c[3] = measure q[3]; // Witness q3
    c[4] = measure q[4]; // Witness q4
    c[5] = measure q[5]; // Witness q5
    c[6] = measure q[6]; // Witness q6
    c[7] = measure q[7]; // Witness q7
