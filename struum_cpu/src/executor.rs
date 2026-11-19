use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use wasmtime::{Config, Engine, Instance, Linker, Module, Store, Trap};

use struum_types::{Shardable, StruumError, split_ranges};

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
    pub fn execute(&self, task: CpuTask) -> Result<CpuTask, StruumError> {
        let module = self.module(task.module())?;
        let count = task.work_count();
        let ranges = split_ranges(count, self.threads);

        if ranges.len() <= 1 {
            let mut task = task;
            self.run_instance(&module, &mut task)?;
            return Ok(task);
        }

        let mut pieces = ranges
            .into_iter()
            .map(|r| task.shard(r))
            .collect::<Result<Vec<_>, _>>()?;

        std::thread::scope(|scope| {
            let handles: Vec<_> = pieces
                .iter_mut()
                .map(|piece| {
                    let module = &module;
                    scope.spawn(move || self.run_instance(module, piece))
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap_or_else(|_| Err(StruumError::TaskError("wasm thread panicked".to_string()))))
                .collect::<Result<Vec<_>, _>>()
        })?;

        CpuTask::merge(pieces)
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

    /// Runs one piece of work in a fresh instance and copies the buffers back.
    fn run_instance(&self, module: &Module, task: &mut CpuTask) -> Result<(), StruumError> {
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
            .get_typed_func::<u32, i32>(&mut store, task.entry())
            .map_err(|e| task_err(&format!("wasm entry '{}(u32) -> i32' not found", task.entry()), e))?;

        // Copy buffers in.
        let mut ptrs = Vec::with_capacity(task.buffers().len());
        for b in task.buffers() {
            let len = b.buffer.data.len() as u32;
            let ptr = alloc
                .call(&mut store, (len, BUFFER_ALIGN))
                .map_err(|e| task_err("struum_alloc failed", e))?;
            memory
                .write(&mut store, ptr as usize, &b.buffer.data)
                .map_err(|e| task_err(&format!("writing buffer '{}'", b.buffer.name), e))?;
            ptrs.push(ptr);
        }

        // Descriptor: [offset, len, count, (ptr, byte_len, stride)*]
        let mut desc = vec![task.offset() as u32, task.work_count() as u32, ptrs.len() as u32];
        for (b, ptr) in task.buffers().iter().zip(&ptrs) {
            desc.extend([*ptr, b.buffer.data.len() as u32, b.buffer.stride as u32]);
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
                    task.entry(),
                    self.timeout
                ))
            } else {
                task_err(&format!("task '{}' trapped", task.entry()), format!("{e:?}"))
            }
        })?;
        if status != 0 {
            return Err(StruumError::TaskError(format!(
                "task '{}' returned error status {}",
                task.entry(),
                status
            )));
        }

        // Copy buffers out.
        for (b, ptr) in task.buffers_mut().iter_mut().zip(&ptrs) {
            memory
                .read(&store, *ptr as usize, &mut b.buffer.data)
                .map_err(|e| task_err(&format!("reading buffer '{}'", b.buffer.name), e))?;
        }
        Ok(())
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
