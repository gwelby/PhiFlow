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
  let intentionDepth = 0;
  let resonanceCount = 0;

  function coherence() {
    if (intentionDepth === 0 && resonanceCount === 0) return 0.0;
    const intention = intentionDepth > 0 ? (1.0 - Math.pow(PHI, -intentionDepth)) : 0.0;
    const bonus = Math.min(resonanceCount * 0.05, 0.2);
    return Math.min(intention + bonus, 1.0);
  }

  const imports = {
    phi: {
      witness: () => coherence(),
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
