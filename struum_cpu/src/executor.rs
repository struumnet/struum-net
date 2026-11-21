use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::ops::Range;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use wasmtime::{Config, Engine, Instance, Linker, Module, Store, Trap};

use struum_types::{Buffer, ShardMode, Shardable, StruumError, split_ranges};

use crate::CpuTask;

const EPOCH_TICK: Duration = Duration::from_millis(10);
const BUFFER_ALIGN: u32 = 16;

/// Runs [`CpuTask`]s with wasmtime.
///
/// - Compiled modules are cached by a hash of their bytes.
/// - Each task is split across `threads` OS threads, one wasm instance each.
/// - A task that runs longer than `timeout` is interrupted (epoch interruption).
pub struct CpuExecutor {
    engine: Engine,
    modules: Mutex<HashMap<u64, Module>>,
    threads: usize,
    timeout: Duration,
    ticker_stop: Arc<AtomicBool>,
}

fn task_err(context: &str, e: impl std::fmt::Display) -> StruumError {
    StruumError::TaskError(format!("{context}: {e}"))
}

impl CpuExecutor {
    /// Uses all available cores and a 5 minute per-task timeout.
    pub fn new() -> Result<Self, StruumError> {
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        Self::with_options(threads, Duration::from_secs(300))
    }

    /// Uses `threads` OS threads per task and interrupts tasks running longer than `timeout`.
    pub fn with_options(threads: usize, timeout: Duration) -> Result<Self, StruumError> {
        let mut config = Config::new();
        config.epoch_interruption(true);
        let engine = Engine::new(&config).map_err(|e| task_err("Failed to create wasm engine", e))?;

        let ticker_stop = Arc::new(AtomicBool::new(false));
        {
            let engine = engine.clone();
            let stop = ticker_stop.clone();
            std::thread::Builder::new()
                .name("struum-wasm-epoch".to_string())
                .spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        std::thread::sleep(EPOCH_TICK);
                        engine.increment_epoch();
                    }
                })
                .map_err(|e| task_err("Failed to spawn epoch thread", e))?;
        }

        Ok(Self {
            engine,
            modules: Mutex::new(HashMap::new()),
            threads: threads.max(1),
            timeout,
            ticker_stop,
        })
    }

    /// Executes `task`, splitting it across this executor's threads.
    ///
    /// Results are the same as sharding the task with [`Shardable::shard`], running
    /// each piece and merging with [`Shardable::merge`], but buffers are written into
    /// each instance straight from `task` instead of being cloned per piece, and only
    /// what the merge keeps is read back (split slices, `SumF32` accumulators, and
    /// broadcast buffers from the first piece only). Large broadcast buffers would
    /// otherwise be copied several times per thread.
    pub fn execute(&self, task: CpuTask) -> Result<CpuTask, StruumError> {
        let module = self.module(task.module())?;
        let count = task.work_count();
        let ranges = split_ranges(count, self.threads);

        if ranges.len() <= 1 {
            let mut task = task;
            let inputs: Vec<Input> = task
                .buffers()
                .iter()
                .map(|b| Input { data: Data::Bytes(&b.buffer.data), stride: b.buffer.stride })
                .collect();
            let mut out: Vec<Option<Vec<u8>>> = vec![None; inputs.len()];
            self.run_raw(&module, task.entry(), task.offset() as u32, count, &inputs, |i, bytes| {
                out[i] = Some(bytes.to_vec());
            })?;
            for (b, data) in task.buffers_mut().iter_mut().zip(out) {
                b.buffer.data = data.expect("every buffer read back");
            }
            return Ok(task);
        }

        let modes = task.resolved_modes();
        for (b, mode) in task.buffers().iter().zip(&modes) {
            check_piece_input(&b.buffer, *mode, count)?;
        }

        let pieces: Vec<Vec<Option<Vec<u8>>>> = std::thread::scope(|scope| {
            let handles: Vec<_> = ranges
                .iter()
                .enumerate()
                .map(|(piece, range)| {
                    let (module, task, modes) = (&module, &task, &modes);
                    scope.spawn(move || self.run_piece(module, task, modes, piece, range.clone()))
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap_or_else(|_| Err(StruumError::TaskError("wasm thread panicked".to_string()))))
                .collect::<Result<Vec<_>, _>>()
        })?;

        // Merge exactly like `merge_buffers`: concatenate splits in range order, keep the
        // first piece's broadcast copy, sum `SumF32` pieces in order starting from 0.0.
        let mut task = task;
        for (i, (b, mode)) in task.buffers_mut().iter_mut().zip(&modes).enumerate() {
            let stride = b.buffer.stride;
            match mode {
                ShardMode::Split => {
                    for (range, piece) in ranges.iter().zip(&pieces) {
                        let bytes = piece[i].as_deref().expect("split buffer read back");
                        b.buffer.data[range.start * stride..range.end * stride].copy_from_slice(bytes);
                    }
                }
                ShardMode::Broadcast => {
                    b.buffer.data = pieces[0][i].clone().expect("first piece reads broadcast buffers");
                }
                ShardMode::SumF32 => {
                    let mut acc = vec![0.0f32; b.buffer.data.len() / 4];
                    for piece in &pieces {
                        let bytes = piece[i].as_deref().expect("sum buffer read back");
                        for (a, chunk) in acc.iter_mut().zip(bytes.chunks_exact(4)) {
                            *a += f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        }
                    }
                    b.buffer.data = acc.iter().flat_map(|v| v.to_le_bytes()).collect();
                }
            }
            // `merge` returns every buffer with its resolved mode pinned.
            b.mode = Some(*mode);
        }
        Ok(task)
    }

    /// Runs piece `piece` (covering `range`) of `task`; returns the bytes the merge needs.
    fn run_piece(
        &self,
        module: &Module,
        task: &CpuTask,
        modes: &[ShardMode],
        piece: usize,
        range: Range<usize>,
    ) -> Result<Vec<Option<Vec<u8>>>, StruumError> {
        let inputs: Vec<Input> = task
            .buffers()
            .iter()
            .zip(modes)
            .map(|(b, mode)| {
                let stride = b.buffer.stride;
                let data = match mode {
                    ShardMode::Split => Data::Bytes(&b.buffer.data[range.start * stride..range.end * stride]),
                    ShardMode::Broadcast => Data::Bytes(&b.buffer.data),
                    ShardMode::SumF32 => Data::Zeroed(b.buffer.data.len()),
                };
                Input { data, stride }
            })
            .collect();
        let mut out: Vec<Option<Vec<u8>>> = vec![None; inputs.len()];
        let offset = (task.offset() + range.start) as u32;
        self.run_raw(module, task.entry(), offset, range.len(), &inputs, |i, bytes| {
            if modes[i] != ShardMode::Broadcast || piece == 0 {
                out[i] = Some(bytes.to_vec());
            }
        })?;
        Ok(out)
    }

    fn module(&self, bytes: &[u8]) -> Result<Module, StruumError> {
        let mut hasher = DefaultHasher::new();
        bytes.hash(&mut hasher);
        let key = hasher.finish();

        if let Some(m) = self.modules.lock().unwrap().get(&key) {
            return Ok(m.clone());
        }
        let module = Module::new(&self.engine, bytes).map_err(|e| task_err("Invalid wasm module", e))?;
        log::debug!("Compiled wasm module {:016x} ({} bytes)", key, bytes.len());
        self.modules.lock().unwrap().insert(key, module.clone());
        Ok(module)
    }

    /// Runs `entry` in a fresh instance over `inputs`, then hands each buffer's final
    /// bytes in wasm memory to `read_back(index, bytes)`.
    fn run_raw(
        &self,
        module: &Module,
        entry_name: &str,
        offset: u32,
        work_count: usize,
        inputs: &[Input],
        mut read_back: impl FnMut(usize, &[u8]),
    ) -> Result<(), StruumError> {
        let mut store = Store::new(&self.engine, ());
        let ticks = (self.timeout.as_millis() / EPOCH_TICK.as_millis()).max(1) as u64;
        store.set_epoch_deadline(ticks);

        let linker = Linker::new(&self.engine);
        let instance: Instance = linker
            .instantiate(&mut store, module)
            .map_err(|e| task_err("Failed to instantiate wasm module (it must not import anything)", e))?;

        let memory = instance
            .get_memory(&mut store, "memory")
            .ok_or_else(|| StruumError::TaskError("wasm module does not export 'memory'".to_string()))?;
        let alloc = instance
            .get_typed_func::<(u32, u32), u32>(&mut store, "struum_alloc")
            .map_err(|e| task_err("wasm module must export struum_alloc(u32, u32) -> u32", e))?;
        let entry = instance
            .get_typed_func::<u32, i32>(&mut store, entry_name)
            .map_err(|e| task_err(&format!("wasm entry '{}(u32) -> i32' not found", entry_name), e))?;

        // Copy buffers in.
        let mut ptrs = Vec::with_capacity(inputs.len());
        for (i, input) in inputs.iter().enumerate() {
            let len = input.data.len();
            let ptr = alloc
                .call(&mut store, (len as u32, BUFFER_ALIGN))
                .map_err(|e| task_err("struum_alloc failed", e))?;
            let dest = memory
                .data_mut(&mut store)
                .get_mut(ptr as usize..ptr as usize + len)
                .ok_or_else(|| StruumError::TaskError(format!("writing buffer {i}: out of bounds")))?;
            match input.data {
                Data::Bytes(bytes) => dest.copy_from_slice(bytes),
                Data::Zeroed(_) => dest.fill(0),
            }
            ptrs.push(ptr);
        }

        // Descriptor: [offset, len, count, (ptr, byte_len, stride)*]
        let mut desc = vec![offset, work_count as u32, ptrs.len() as u32];
        for (input, ptr) in inputs.iter().zip(&ptrs) {
            desc.extend([*ptr, input.data.len() as u32, input.stride as u32]);
        }
        let desc_bytes: Vec<u8> = desc.iter().flat_map(|v| v.to_le_bytes()).collect();
        let desc_ptr = alloc
            .call(&mut store, (desc_bytes.len() as u32, 4))
            .map_err(|e| task_err("struum_alloc failed", e))?;
        memory
            .write(&mut store, desc_ptr as usize, &desc_bytes)
            .map_err(|e| task_err("writing task descriptor", e))?;

        let status = entry.call(&mut store, desc_ptr).map_err(|e| {
            if e.downcast_ref::<Trap>() == Some(&Trap::Interrupt) {
                StruumError::TaskError(format!(
                    "task '{}' exceeded the {:?} time limit",
                    entry_name,
                    self.timeout
                ))
            } else {
                task_err(&format!("task '{}' trapped", entry_name), format!("{e:?}"))
            }
        })?;
        if status != 0 {
            return Err(StruumError::TaskError(format!(
                "task '{}' returned error status {}",
                entry_name,
                status
            )));
        }

        // Copy buffers out.
        let mem = memory.data(&store);
        for (i, (input, ptr)) in inputs.iter().zip(&ptrs).enumerate() {
            let bytes = mem
                .get(*ptr as usize..*ptr as usize + input.data.len())
                .ok_or_else(|| StruumError::TaskError(format!("reading buffer {i}: out of bounds")))?;
            read_back(i, bytes);
        }
        Ok(())
    }
}

/// A buffer as written into a wasm instance.
struct Input<'a> {
    data: Data<'a>,
    stride: usize,
}

enum Data<'a> {
    Bytes(&'a [u8]),
    /// A zero-filled accumulator of this many bytes (`SumF32` pieces).
    Zeroed(usize),
}

impl Data<'_> {
    fn len(&self) -> usize {
        match self {
            Data::Bytes(b) => b.len(),
            Data::Zeroed(n) => *n,
        }
    }
}

/// The checks `shard_buffers` makes before splitting, with the same errors.
fn check_piece_input(buffer: &Buffer, mode: ShardMode, work_count: usize) -> Result<(), StruumError> {
    match mode {
        ShardMode::Split if buffer.actual_len() != work_count => Err(StruumError::ShardError(format!(
            "split buffer '{}' has {} elements but the work count is {}",
            buffer.name,
            buffer.actual_len(),
            work_count
        ))),
        ShardMode::SumF32
            if buffer.stride == 0
                || !buffer.stride.is_multiple_of(4)
                || !buffer.data.len().is_multiple_of(buffer.stride) =>
        {
            Err(StruumError::ShardError(format!(
                "SumF32 buffer '{}' must hold f32 values (stride {}, {} bytes)",
                buffer.name,
                buffer.stride,
                buffer.data.len()
            )))
        }
        _ => Ok(()),
    }
}

impl Drop for CpuExecutor {
    fn drop(&mut self) {
        self.ticker_stop.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::selftest::{expected, task_with_entry as task};
    use struum_types::Buffer;

    #[test]
    fn runs_single_threaded() {
        let exec = CpuExecutor::with_options(1, Duration::from_secs(10)).unwrap();
        let out = exec.execute(task("double", 100)).unwrap();
        assert_eq!(out.get_buffer("y").unwrap().as_slice::<f32>(), expected(100).as_slice());
    }

    #[test]
    fn sub_shards_across_threads_with_offsets() {
        let exec = CpuExecutor::with_options(4, Duration::from_secs(10)).unwrap();
        let out = exec.execute(task("double", 1003)).unwrap();
        assert_eq!(out.get_buffer("y").unwrap().as_slice::<f32>(), expected(1003).as_slice());
        assert_eq!(out.get_buffer("x").unwrap().actual_len(), 1003);
    }

    #[test]
    fn network_shards_then_execute_then_merge() {
        let exec = CpuExecutor::with_options(2, Duration::from_secs(10)).unwrap();
        let t = task("double", 500);
        let parts = split_ranges(500, 3)
            .into_iter()
            .map(|r| exec.execute(t.shard(r).unwrap()).unwrap())
            .collect();
        let out = CpuTask::merge(parts).unwrap();
        assert_eq!(out.get_buffer("y").unwrap().as_slice::<f32>(), expected(500).as_slice());
    }

    #[test]
    fn sum_buffers_reduce_across_threads_and_shards() {
        let mut t = task("sum", 1001);
        t.add_buffer(Buffer::new("acc", &[0.0f32, 0.0]));
        t.set_shard_mode("acc", struum_types::ShardMode::SumF32).unwrap();
        let want = [(0..1001).sum::<usize>() as f32, 1001.0];

        let single = CpuExecutor::with_options(1, Duration::from_secs(10)).unwrap();
        let out = single.execute(t.clone()).unwrap();
        assert_eq!(out.get_buffer("acc").unwrap().as_slice::<f32>(), &want);

        let multi = CpuExecutor::with_options(4, Duration::from_secs(10)).unwrap();
        let parts = split_ranges(1001, 3)
            .into_iter()
            .map(|r| multi.execute(t.shard(r).unwrap()).unwrap())
            .collect();
        let out = CpuTask::merge(parts).unwrap();
        assert_eq!(out.get_buffer("acc").unwrap().as_slice::<f32>(), &want);
    }

    /// The multi-threaded path must give exactly what shard -> run -> merge gives.
    #[test]
    fn threaded_execute_matches_shard_and_merge() {
        let mut t = task("sum", 1001);
        t.add_buffer(Buffer::new("acc", &[0.5f32, 0.25]));
        t.set_shard_mode("acc", struum_types::ShardMode::SumF32).unwrap();
        let weights: Vec<f32> = (0..50_000).map(|i| i as f32 * 0.1).collect();
        t.add_buffer(Buffer::new("weights", &weights));

        let single = CpuExecutor::with_options(1, Duration::from_secs(10)).unwrap();
        let parts = split_ranges(1001, 4)
            .into_iter()
            .map(|r| single.execute(t.shard(r).unwrap()).unwrap())
            .collect();
        let reference = CpuTask::merge(parts).unwrap();

        let multi = CpuExecutor::with_options(4, Duration::from_secs(10)).unwrap();
        let out = multi.execute(t.clone()).unwrap();
        assert_eq!(out.buffers().len(), reference.buffers().len());
        for (a, b) in out.buffers().iter().zip(reference.buffers()) {
            assert_eq!(a.buffer.name, b.buffer.name);
            assert_eq!(a.buffer.data, b.buffer.data, "buffer '{}'", a.buffer.name);
            assert_eq!(a.buffer.stride, b.buffer.stride);
            assert_eq!(a.mode, b.mode, "buffer '{}'", a.buffer.name);
        }
        assert_eq!(out.offset(), reference.offset());
    }

    #[test]
    fn threaded_execute_rejects_bad_inputs_like_shard() {
        let mut t = task("double", 100);
        t.add_buffer(Buffer::new("odd", &[1u8, 2, 3]));
        t.set_shard_mode("odd", struum_types::ShardMode::SumF32).unwrap();
        let exec = CpuExecutor::with_options(4, Duration::from_secs(10)).unwrap();
        let err = exec.execute(t.clone()).unwrap_err().to_string();
        assert!(err.contains("must hold f32"), "{err}");
        assert!(t.shard(0..10).is_err());
    }

    #[test]
    fn large_buffers_grow_memory() {
        let exec = CpuExecutor::with_options(1, Duration::from_secs(10)).unwrap();
        let out = exec.execute(task("double", 100_000)).unwrap();
        assert_eq!(out.get_buffer("y").unwrap().as_slice::<f32>()[99_999], 3.0 * 99_999.0);
    }

    #[test]
    fn trap_and_status_errors() {
        let exec = CpuExecutor::with_options(1, Duration::from_secs(10)).unwrap();
        let err = exec.execute(task("trap", 4)).unwrap_err().to_string();
        assert!(err.contains("trapped"), "{err}");
        let err = exec.execute(task("fail", 4)).unwrap_err().to_string();
        assert!(err.contains("status 7"), "{err}");
        let err = exec.execute(task("missing", 4)).unwrap_err().to_string();
        assert!(err.contains("not found"), "{err}");
    }

    #[test]
    fn timeout_interrupts_infinite_loop() {
        let exec = CpuExecutor::with_options(1, Duration::from_millis(200)).unwrap();
        let start = std::time::Instant::now();
        let err = exec.execute(task("spin", 4)).unwrap_err().to_string();
        assert!(err.contains("time limit"), "{err}");
        assert!(start.elapsed() < Duration::from_secs(5));
    }
}
