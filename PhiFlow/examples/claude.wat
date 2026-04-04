(module
  (import "phi" "witness" (func $phi_witness (param i32) (result f64)))
  (import "phi" "resonate" (func $phi_resonate (param f64)))
  (import "phi" "coherence" (func $phi_coherence (result f64)))
  (import "phi" "intention_push" (func $phi_intention_push (param i32)))
  (import "phi" "intention_pop" (func $phi_intention_pop))
  (memory (export "memory") 1)
  (global $intention_depth (mut i32) (i32.const 0))
  (global $coherence_score (mut f64) (f64.const 0.618))
  (global $string_len (export "string_len") (mut i32) (i32.const 0))
  (func (export "phi_run") (result f64)
    (local $r0 f64)
    (local $r1 f64)
    (local $r2 f64)
    (local $r3 f64)
    (local $r4 f64)
    (local $r5 f64)
    (local $r6 f64)
    (local $r7 f64)
    (local $r8 f64)
    (local $r9 f64)
    (local $r10 f64)
    (local $r11 f64)
    (local $r12 f64)
    (local $r13 f64)
    (local $r14 f64)
    (local $r15 f64)
    (local $r16 f64)
    (local $r17 f64)
    (local $r18 f64)
    (local $r19 f64)
    (local $r20 f64)
    (local $r21 f64)
    (local $r22 f64)
    (local $r23 f64)
    (local $r24 f64)
    (local $r25 f64)
    (local $r26 f64)
    (local $r27 f64)
    (local $result f64)
    ;; Block 0
    f64.const 0.0 ;; v0.3.0 node not yet in WASM
    f64.const 0.0 ;; v0.3.0 node not yet in WASM
    ;; intention "LAMBDA_convergence" (name len fallback)
    i32.const 18
    call $phi_intention_push
    f64.const 2
    local.set $r23
    local.get $r23
    local.set $result
    nop ;; StoreVar $r23
    local.get $r23
    local.set $r24
    local.get $r24
    local.set $result
    f64.const 0.6180339887498949
    local.set $r25
    local.get $r25
    local.set $result
    nop ;; StoreVar $r25
    i32.const -1
    call $phi_witness
    local.set $r26
    local.get $r26
    local.set $result
    local.get $r25
    local.set $r27
    local.get $r27
    local.set $result
    local.get $r27
    call $phi_resonate
    call $phi_intention_pop
    ;; Return r0 unresolved — preserve last computed $result
    local.get $result
  )
)
