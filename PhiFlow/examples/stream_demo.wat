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
    (local $result f64)
    ;; Block 0
    f64.const 0
    local.set $r0
    local.get $r0
    local.set $result
    nop ;; StoreVar $r0
    f64.const 3
    local.set $r1
    local.get $r1
    local.set $result
    nop ;; StoreVar $r1
    f64.const 0.0 ;; v0.3.0 stub
    ;; Block 1
    ;; Block 2
    local.get $r4
    local.set $r2
    local.get $r2
    local.set $result
    f64.const 1
    local.set $r3
    local.get $r3
    local.set $result
    local.get $r2
        local.get $r3
    f64.add
    local.set $r4
    local.get $r4
    local.set $result
    nop ;; StoreVar $r4
    call $phi_coherence
    local.set $r5
    local.get $r5
    local.set $result
    nop ;; StoreVar $r5
    local.get $r4
    local.set $r6
    local.get $r6
    local.set $result
    local.get $r6
    call $phi_resonate
    i32.const -1
    call $phi_witness
    local.set $r7
    local.get $r7
    local.set $result
    local.get $r4
    local.set $r8
    local.get $r8
    local.set $result
    local.get $r1
    local.set $r9
    local.get $r9
    local.set $result
    local.get $r8
        local.get $r9
    f64.ge
    f64.convert_i32_s
    local.set $r10
    local.get $r10
    local.set $result
    local.get $r10
    f64.const 0.0
    f64.ne
    (if
      (then
        ;; -> block 4
      )
      (else
        ;; -> block 5
      )
    )
    ;; Block 4
    ;; Block 3
    f64.const 0.0 ;; v0.3.0 stub
    local.get $r0
    local.set $result
    ;; Block 5
    ;; Block 6
    local.get $result
  )
)
