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
  const sensorValues = new Map([
    [0, 12.5],
    [1, 55.0],
    [2, 62.0],
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

  const imports = {
    phi: {
      witness: () => coherence(),
      sensor: (sensorId) => {
        if (!sensorValues.has(sensorId)) {
          throw new Error(`invalid sensor id ${sensorId}`);
        }
        return sensorValues.get(sensorId);
      },
      resonate: (_value) => {
        resonanceCount += 1;
      },
      coherence: () => coherence(),
      intention_push: (_offset) => {
        intentionDepth += 1;
      },
      intention_pop: () => {
        if (intentionDepth > 0) intentionDepth -= 1;
      },
    },
  };

  const { instance } = await WebAssembly.instantiate(buffer, imports);
  const result = instance.exports.phi_run();
  process.stdout.write(String(result));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
