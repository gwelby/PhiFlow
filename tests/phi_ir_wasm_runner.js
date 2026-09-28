const fs = require("fs");

async function main() {
  const watPath = process.argv[2];
  const watSource = fs.readFileSync(watPath, "utf8");

  const wabt = await require("wabt")();
  const module = wabt.parseWat(watPath, watSource, {
    mutable_globals: true,
    bulk_memory: false,
  });
  const { buffer } = module.toBinary({ log: false });
  module.destroy();

  const PHI = 1.618033988749895;
  const TAU = 2.0 * Math.PI;
  let intentionDepth = 0;
  let resonanceCount = 0;
  const resonanceField = [];   // all resonated values
  const witnessLog = [];       // coherence values from witness calls
  const kvStore = new Map();   // remember/recall key-value store
  const channels = new Map();  // broadcast/listen channels
  const sensorValues = new Map([
    [0, 12.5],   // default sensor
    [1, 55.0],   // default sensor
    [2, 62.0],   // default sensor
    [3, 7.83],   // soma_schumann (Schumann resonance)
    [4, 432.0],  // soma_432 (432 Hz tone)
    [5, 0.75],   // soma_presence (presence detector)
  ]);

  function coherence() {
    // Canonical formula: base(depth) * phase(k), clamped to [0, 1]
    const base = intentionDepth === 0 ? 0.0 : 1.0 - Math.pow(PHI, -intentionDepth);
    let phase;
    if (resonanceCount <= 1) {
      phase = 1.0;
    } else {
      phase = Math.max(0.0, 1.0 - Math.log(resonanceCount) / Math.log(TAU));
    }
    return Math.min(Math.max(base * phase, 0.0), 1.0);
  }

  function fieldCoherence() {
    if (resonanceField.length === 0) return coherence();
    return resonanceField.reduce((a, b) => a + b, 0) / resonanceField.length;
  }

  function dissonance() {
    if (witnessLog.length < 2) return 0.0;
    const last = witnessLog[witnessLog.length - 1];
    const prev = witnessLog[witnessLog.length - 2];
    const delta = last - prev;
    return Math.max(-1.0, Math.min(1.0, delta * 10.0));
  }

  const imports = {
    phi: {
      witness: (operand) => {
        const c = coherence();
        witnessLog.push(c);
        return c;
      },
      sensor: (sensorId) => {
        if (!sensorValues.has(sensorId)) {
          throw new Error(`invalid sensor id ${sensorId}`);
        }
        return sensorValues.get(sensorId);
      },
      resonate: (value) => {
        resonanceCount += 1;
        resonanceField.push(value);
      },
      coherence: () => coherence(),
      intention_push: (_offset) => {
        intentionDepth += 1;
      },
      intention_pop: () => {
        if (intentionDepth > 0) intentionDepth -= 1;
      },
      field_coherence: () => fieldCoherence(),
      dissonance: () => dissonance(),
      coherence_of: (_nameIdx) => {
        // Return last resonated value, or current coherence if empty
        return resonanceField.length > 0
          ? resonanceField[resonanceField.length - 1]
          : coherence();
      },
      remember: (keyIdx, value) => {
        kvStore.set(keyIdx, value);
      },
      recall: (keyIdx) => {
        return kvStore.has(keyIdx) ? kvStore.get(keyIdx) : 0.0;
      },
      broadcast: (channelIdx, value) => {
        channels.set(channelIdx, value);
      },
      listen: (channelIdx) => {
        return channels.has(channelIdx) ? channels.get(channelIdx) : 0.0;
      },
      void_depth: () => 0.0,
    },
  };

  const { instance } = await WebAssembly.instantiate(buffer, imports);
  const result = instance.exports.phi_run();

  // Handle BSEI NaN-boxing for Void type
  const buffer_arr = new ArrayBuffer(8);
  const f64_view = new Float64Array(buffer_arr);
  const u32_view = new Uint32Array(buffer_arr);
  f64_view[0] = result;
  // TAG_VOID is 0x7FF80003_00000000 -> little endian uint32s: [0x00000000, 0x7FF80003]
  if (u32_view[1] === 0x7FF80003 && u32_view[0] === 0) {
    // If it's a void return, we don't output anything, or we could output 0.0 or whatever the evaluator produces.
    // The evaluator returns the last resonated value if there's no explicit return!
    // But since the runner currently prints NaN when it hits this, we'll try pulling from resonanceField
    if (resonanceField.length > 0) {
      process.stdout.write(String(resonanceField[resonanceField.length - 1]));
    } else {
      process.stdout.write("NaN");
    }
  } else {
    process.stdout.write(String(result));
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
