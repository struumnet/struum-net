//! A tiny hand-written WebAssembly module for self-tests and examples.
//!
//! Entry points:
//! - `double`: `y[i] = 2 * x[i] + (offset + i)`, so with `x[i] = i` the result is `3 * i`
//!   (checks that shard offsets are applied correctly)
//! - `trap`: always traps; `fail`: returns status 7; `spin`: loops forever

use struum_types::Buffer;

use crate::CpuTask;

/// WAT source (wasmtime accepts WAT text as well as `.wasm` binaries).
pub const SELFTEST_WAT: &str = r#"
    (module
      (memory (export "memory") 1)
      (global $heap (mut i32) (i32.const 1024))
      (func (export "struum_alloc") (param $size i32) (param $align i32) (result i32)
        (local $p i32) (local $need i32)
        (local.set $p (i32.and
          (i32.add (global.get $heap) (i32.sub (local.get $align) (i32.const 1)))
          (i32.sub (i32.const 0) (local.get $align))))
        (global.set $heap (i32.add (local.get $p) (local.get $size)))
        (local.set $need (i32.sub (global.get $heap) (i32.mul (memory.size) (i32.const 65536))))
        (if (i32.gt_s (local.get $need) (i32.const 0))
          (then (drop (memory.grow (i32.add (i32.div_u (local.get $need) (i32.const 65536)) (i32.const 1))))))
        (local.get $p))
      (func (export "double") (param $d i32) (result i32)
        (local $i i32) (local $n i32) (local $off i32) (local $x i32) (local $y i32)
        (local.set $off (i32.load (local.get $d)))
        (local.set $n (i32.load offset=4 (local.get $d)))
        (local.set $x (i32.load offset=12 (local.get $d)))
        (local.set $y (i32.load offset=24 (local.get $d)))
        (block $done (loop $l
          (br_if $done (i32.ge_u (local.get $i) (local.get $n)))
          (f32.store (i32.add (local.get $y) (i32.shl (local.get $i) (i32.const 2)))
            (f32.add
              (f32.mul (f32.load (i32.add (local.get $x) (i32.shl (local.get $i) (i32.const 2)))) (f32.const 2))
              (f32.convert_i32_u (i32.add (local.get $off) (local.get $i)))))
          (local.set $i (i32.add (local.get $i) (i32.const 1)))
          (br $l)))
        (i32.const 0))
      ;; acc[0] += x[i], acc[1] += 1   (buffer 2 = acc, a SumF32 accumulator)
      (func (export "sum") (param $d i32) (result i32)
        (local $i i32) (local $n i32) (local $x i32) (local $acc i32)
        (local.set $n (i32.load offset=4 (local.get $d)))
        (local.set $x (i32.load offset=12 (local.get $d)))
        (local.set $acc (i32.load offset=36 (local.get $d)))
        (block $done (loop $l
          (br_if $done (i32.ge_u (local.get $i) (local.get $n)))
          (f32.store (local.get $acc) (f32.add (f32.load (local.get $acc))
            (f32.load (i32.add (local.get $x) (i32.shl (local.get $i) (i32.const 2))))))
          (f32.store offset=4 (local.get $acc) (f32.add (f32.load offset=4 (local.get $acc)) (f32.const 1)))
          (local.set $i (i32.add (local.get $i) (i32.const 1)))
          (br $l)))
        (i32.const 0))
      (func (export "trap") (param i32) (result i32) unreachable)
      (func (export "fail") (param i32) (result i32) (i32.const 7))
      (func (export "spin") (param i32) (result i32) (loop $l (br $l)) (i32.const 0))
    )"#;

/// A `double` task over `n` items with `x[i] = i`.
pub fn task(n: usize) -> CpuTask {
    task_with_entry("double", n)
}

/// Same data as [`task`] but calling another entry point.
pub fn task_with_entry(entry: &str, n: usize) -> CpuTask {
    let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let mut t = CpuTask::new(SELFTEST_WAT.as_bytes(), entry);
    t.add_buffer(Buffer::new("x", &x));
    t.add_buffer(Buffer::empty::<f32>("y", n));
    t.set_work_buffer("x");
    t
}

/// Expected `y` for [`task`]`(n)`.
pub fn expected(n: usize) -> Vec<f32> {
    (0..n).map(|i| 3.0 * i as f32).collect()
}
