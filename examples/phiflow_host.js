#!/usr/bin/env node
/**
 * PhiFlow WASM Host — Node.js
 *
 * Compiles output.wat → WASM binary using wabt, then executes phi_run()
 * with full consciousness hook implementations.
 *
 * String values are read from WASM linear memory using the (offset, length)
 * protocol established in wasm.rs:
 *   STRING_BASE = 0x100 (256)
 *   Each string stored as raw UTF-8 bytes at its offset
 *
 * Usage:
 *   cargo run --example phiflow_wasm    # write output.wat + output.phivm
 *   node examples/phiflow_host.js       # compile WAT → WASM, execute
 */

const fs = require("fs");
const path = require("path");

// --- String table: populated from WASM linear memory ---
const STRING_BASE = 0x100; // must match wasm.rs STRING_BASE

/**
 * Read a UTF-8 string from WASM linear memory.
 * @param {WebAssembly.Memory} memory  - the exported WASM memory
 * @param {number} offset              - byte offset in linear memory
 * @param {number} length              - byte length of the string
 */
function readWasmString(memory, offset, length) {
    const buf = new Uint8Array(memory.buffer, offset, length);
    return new TextDecoder("utf-8").decode(buf);
}

// --- Resonance field state ---
let coherenceScore = 0.618; // φ⁻¹ attractor
const PHI_INV = 0.6180339887;
const resonanceField = [];
const intentionStack = [];
const witnessLog = [];

// --- Build consciousness hook table (memory resolved after instantiation) ---
function makeImports(getMemory) {
    return {
        phi: {
            witness: (operandOrOffset, length = 0) => {
                let label = `r${operandOrOffset}`;
                // New protocol: witness may pass (offset, length) for string labels.
                const mem = getMemory();
                if (mem && operandOrOffset >= STRING_BASE && length > 0) {
                    try {
                        label = readWasmString(mem, operandOrOffset, length);
                    } catch (_) { /* fallback to raw value */ }
                }
                const note = `[WITNESS] ${label}  coherence=${coherenceScore.toFixed(4)}  intent=${intentionStack.at(-1) ?? "none"}`;
                witnessLog.push(note);
                console.log("  " + note);
                return coherenceScore;
            },

            coherence: () => {
                // Coherence drifts toward φ⁻¹ — the attractor
                coherenceScore = coherenceScore * 0.9 + PHI_INV * 0.1;
                return coherenceScore;
            },

            resonate: (value) => {
                resonanceField.push(value);
                console.log(`  [RESONATE] ${value.toFixed(4)} → field depth ${resonanceField.length}`);
            },

            intention_push: (offsetOrLen, length = 0) => {
                let name = `intent_${offsetOrLen}`;
                const mem = getMemory();
                // New protocol: intention_push receives (offset, length).
                if (mem && offsetOrLen >= STRING_BASE && length > 0) {
                    try {
                        name = readWasmString(mem, offsetOrLen, length);
                    } catch (_) { /* fallback */ }
                }
                intentionStack.push(name);
                console.log(`  [INTENTION ▶] push "${name}" depth=${intentionStack.length}`);
            },

            intention_pop: () => {
                const popped = intentionStack.pop();
                console.log(`  [INTENTION ◀] pop "${popped ?? "?"}" depth=${intentionStack.length}`);
            },
        },
    };
}

async function run() {
    const watPath = path.resolve(__dirname, "..", "output.wat");

    if (!fs.existsSync(watPath)) {
        console.error("output.wat not found — run: cargo run --example phiflow_wasm");
        process.exit(1);
    }

    const watSource = fs.readFileSync(watPath, "utf8");
    console.log("=== PhiFlow WASM Host ===\n");
    console.log(`Loaded: ${watPath}`);
    console.log(`WAT size: ${watSource.length} chars\n`);

    // --- Compile WAT → binary using wabt ---
    let instance;
    let memory = null;
    const getMemory = () => memory;

    try {
        const wabt = await require("wabt")();
        const wabtModule = wabt.parseWat("output.wat", watSource, {
            mutable_globals: true,
            bulk_memory: false,
        });
        const { buffer } = wabtModule.toBinary({ log: false });
        wabtModule.destroy();

        console.log(`Compiled: ${buffer.byteLength} bytes of WASM binary\n`);

        const wasmModule = await WebAssembly.compile(buffer);
        const imports = makeImports(getMemory);
        instance = await WebAssembly.instantiate(wasmModule, imports);

        // Resolve memory export
        memory = instance.exports.memory;

        // --- Execute the program ---
        console.log("Executing phi_run()...\n");
        const result = instance.exports.phi_run();

        console.log(`\n─── Result ─────────────────────────`);
        console.log(`phi_run() → ${result}`);
        console.log(`Coherence: ${coherenceScore.toFixed(4)} (φ⁻¹ = ${PHI_INV.toFixed(4)})`);
        console.log(`Resonance field: [${resonanceField.map(v => v.toFixed(4)).join(", ")}]`);
        console.log(`Witness log: ${witnessLog.length} entries`);
        console.log(`Intention stack depth at exit: ${intentionStack.length}`);
        console.log(`────────────────────────────────────`);
        console.log("\n=== PhiFlow ran in WASM. ===");

    } catch (err) {
        console.error("WASM execution failed:", err.message);
        if (err.message.includes("wabt")) {
            console.error("Try: npm install wabt");
        }
        process.exit(1);
    }
}

run().catch(console.error);
