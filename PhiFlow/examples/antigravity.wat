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
    (local $r28 f64)
    (local $r29 f64)
    (local $r30 f64)
    (local $r31 f64)
    (local $r32 f64)
    (local $r33 f64)
    (local $r34 f64)
    (local $r35 f64)
    (local $r36 f64)
    (local $r37 f64)
    (local $r38 f64)
    (local $r39 f64)
    (local $r40 f64)
    (local $r41 f64)
    (local $r42 f64)
    (local $r43 f64)
    (local $r44 f64)
    (local $r45 f64)
    (local $r46 f64)
    (local $r47 f64)
    (local $result f64)
    ;; Block 0
    f64.const 76
    local.set $r44
    local.get $r44
    local.set $result
    nop ;; StoreVar $r44
    local.get $r44
    local.set $r45
    local.get $r45
    local.set $result
    f64.const 0.0 ;; unresolved call witness_existence
    local.set $r46
    local.get $r46
    local.set $result
    nop ;; StoreVar $r46
    local.get $r46
    local.set $r47
    local.get $r47
    local.set $result
    local.get $r47
    call $phi_resonate
    ;; Return r0 unresolved — preserve last computed $result
    local.get $result
  )
)
