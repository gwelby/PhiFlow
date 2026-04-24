---
name: phivm-bytecode
description: Provides the specification for the .phivm binary format and opcode mapping for the PhiVM runtime.
---
# PhiVM Bytecode Skill

## Responsibility
Map `PhiIR` nodes correctly into the `.phivm` binary format for the next epoch runtime.

## Binary Format Specification
* The output is a raw binary array.
* The benchmark demo output is exactly `121 bytes`. If a modification drastically alters this byte count without justification, it is a compilation error.

## Opcode Mapping
You must accurately map the 4 unique consciousness nodes to their corresponding bytecode instructions inside `src/phi_ir/vm.rs`.
* `Witness` -> `OP_WITNESS`
* `IntentionPush` -> `OP_INTENT_PUSH`
* `Resonate` -> `OP_RESONATE`
* `CoherenceCheck` -> `OP_COHERENCE`

## Validation
To validate changes to the bytecode generation, always run `cargo run --example phiflow_demo` to ensure the byte count and coherence score (`0.6180`) remain stable.
