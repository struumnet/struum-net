//! GPU kernels: GLSL compute code plus its buffers, ready to ship to a worker.
//!
//! A [`Kernel`] holds a GLSL function, the buffers it uses and which buffer defines the
//! work range. [`Kernel::pack`] turns it into a complete compute shader, which the
//! OpenGL backend (`struum_opengl`) compiles and runs. Kernels are serializable and
//! [`Shardable`]: a node can split one across peers and merge the results.
//!
//! Buffer element types implement [`GPUType`]: `f32`, `u32`, `i32` and friends, or your
//! own structs with `#[gpu_type]` from `struum_macros`.
//!
//! ```
//! use struum_kernel::Kernel;
//! use struum_types::Buffer;
//!
//! let mut kernel = Kernel::new("void compute(uint i) { y[i] = 2.0 * x[i]; }", "compute");
//! kernel.add_buffer::<f32>(Buffer::new("x", &[1.0f32, 2.0, 3.0]));
//! kernel.add_buffer::<f32>(Buffer::empty::<f32>("y", 3));
//! kernel.set_work_buffer("x");
//! kernel.pack()?;
//!
//! assert_eq!(kernel.get_work_count(), Some(3));
//! assert!(kernel.get_gpu_source().unwrap().contains("void main()"));
//! # Ok::<(), struum_types::StruumError>(())
//! ```

use regex::Regex;
use serde_derive::{Deserialize, Serialize};
use std::ops::Range;
use struum_types::{Buffer, GPUType, ShardMode, Shardable, StruumError, is_builtin_gpu_type};

// TODO(slok): Support multiple backend types

// TODO(slok): Currently the local size is hard coded
// 128 is the minimum GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS guaranteed by
// OpenGL ES 3.1, so kernels run on mobile (Android) GPUs as well as desktop.
// If it creates a bottleneck then figure something out
const DEFAULT_LOCAL_SIZE: u32 = 128;

/// A buffer attached to a [`Kernel`], with the GLSL type of its elements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferBinding {
    /// The data; its name is the array name in the kernel source.
    pub buffer: Buffer,
    /// GLSL element type, e.g. `float` or a struct name (from [`GPUType::glsl_type`]).
    pub gpu_type: String,
    /// GLSL struct definition for struct element types; empty for built-in types.
    pub gpu_struct: String,
    /// How this buffer is distributed when the kernel is sharded.
    /// `None` infers it: one element per work item => `Split`, else `Broadcast`.
    pub shard_mode: Option<ShardMode>,
}

/// A GPU compute job: a GLSL function plus the buffers it reads and writes.
///
/// The source defines a function called with the index of one work item. [`Kernel::pack`]
/// wraps it into a full compute shader: buffer declarations (one `std430` array per buffer,
/// in the order added), the struct definitions, and a `main` that calls the entry point once
/// per element of the work buffer. Run it locally through a node's scheduler or across
/// peers with `Node::run_sharded`.
///
/// The global index of item `i` is `STRUUM_OFFSET + i` (non-zero only in shards).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kernel {
    source: String,
    entry_point: String,
    buffer_bindings: Vec<BufferBinding>,

    local_size: u32,
    work_buffer: Option<String>,
    work_count: Option<u32>,
    group_size: Option<u32>,
    gpu_source: Option<String>,
    /// Global index of this kernel's first work item (non-zero for shards).
    /// Exposed to GLSL as `STRUUM_OFFSET`.
    offset: u32,
}

impl Kernel {
    /// Creates a kernel from GLSL `source` whose function `entry_point(uint i)` runs once
    /// per work item.
    pub fn new(source: &str, entry_point: &str) -> Self {
        Self {
            source: source.to_string(),
            entry_point: entry_point.to_string(),
            buffer_bindings: Vec::new(),
            local_size: DEFAULT_LOCAL_SIZE,
            work_buffer: None,
            work_count: None,
            group_size: None,
            gpu_source: None,
            offset: 0,
        }
    }

    /// Adds a buffer of `T` elements, bound at the next binding index. Its name is the
    /// array name the source uses.
    pub fn add_buffer<T: GPUType>(&mut self, buffer: Buffer) {
        self.buffer_bindings.push(BufferBinding {
            buffer: buffer,
            gpu_type: T::glsl_type().to_string(),
            gpu_struct: T::glsl(),
            shard_mode: None,
        });
    }

    /// Overrides how buffer `name` is distributed when the kernel is sharded.
    ///
    /// # Errors
    /// [`StruumError::NotFound`] if no buffer has that name.
    pub fn set_shard_mode(&mut self, name: &str, mode: ShardMode) -> Result<(), StruumError> {
        let binding = self
            .buffer_bindings
            .iter_mut()
            .find(|b| b.buffer.name == name)
            .ok_or_else(|| StruumError::NotFound(format!("Buffer '{}' not found", name)))?;
        binding.shard_mode = Some(mode);
        Ok(())
    }

    /// Global index of this kernel's first work item.
    pub fn get_offset(&self) -> u32 {
        self.offset
    }

    fn work_buffer_len(&self) -> Result<usize, StruumError> {
        let name = self
            .work_buffer
            .as_ref()
            .ok_or(StruumError::NotFound("Work Buffer Empty".to_string()))?;
        self.get_buffer(name)
            .map(|b| b.actual_len())
            .ok_or(StruumError::NotFound("Work Buffer not found".to_string()))
    }

    fn resolved_mode(&self, binding: &BufferBinding, work_count: usize) -> ShardMode {
        if self.work_buffer.as_deref() == Some(binding.buffer.name.as_str()) {
            return ShardMode::Split;
        }
        binding
            .shard_mode
            .unwrap_or_else(|| ShardMode::infer(&binding.buffer, work_count))
    }

    /// Sets the buffer whose element count is the number of work items. It is always
    /// split when sharding.
    pub fn set_work_buffer(&mut self, name: &str) {
        self.work_buffer = Some(name.to_string());
    }

    /// Computes the dispatch size and generates the full GLSL source. Call it after adding
    /// buffers and setting the work buffer, before running the kernel.
    ///
    /// # Errors
    /// [`StruumError::NotFound`] if the work buffer is unset or not one of the buffers.
    pub fn pack(&mut self) -> Result<(), StruumError> {
        self.compute_group_size()?;
        self.generate_glsl()?;

        Ok(())
    }

    /// Number of work items; `None` until [`Kernel::pack`].
    pub fn get_work_count(&self) -> Option<u32> {
        self.work_count
    }

    /// Number of work groups to dispatch; `None` until [`Kernel::pack`].
    pub fn get_group_size(&self) -> Option<u32> {
        self.group_size
    }

    /// The generated compute shader; `None` until [`Kernel::pack`].
    pub fn get_gpu_source(&self) -> Option<&str> {
        self.gpu_source.as_deref()
    }

    /// All buffers, in binding order.
    pub fn get_buffer_bindings(&self) -> &[BufferBinding] {
        &self.buffer_bindings
    }

    /// All buffers, in binding order, mutably.
    pub fn get_buffer_bindings_mut(&mut self) -> &mut [BufferBinding] {
        &mut self.buffer_bindings
    }

    /// The buffer called `name`.
    pub fn get_buffer(&self, name: &str) -> Option<&Buffer> {
        self.buffer_bindings
            .iter()
            .find(|binding| binding.buffer.name == name)
            .map(|binding| &binding.buffer)
    }

    /// The buffer called `name`, mutably.
    pub fn get_buffer_mut(&mut self, name: &str) -> Option<&mut Buffer> {
        self.buffer_bindings
            .iter_mut()
            .find(|binding| binding.buffer.name == name)
            .map(|binding| &mut binding.buffer)
    }

    fn compute_group_size(&mut self) -> Result<(), StruumError> {
        let work_buffer_name = self
            .work_buffer
            .as_ref()
            .ok_or(StruumError::NotFound("Work Buffer Empty".to_string()))?;

        let binding = self
            .buffer_bindings
            .iter()
            .find(|b| &b.buffer.name == work_buffer_name)
            .ok_or(StruumError::NotFound("Work Buffer not found".to_string()))?;

        let count = binding.buffer.actual_len();
        self.work_count = Some(count as u32);
        self.group_size = Some(
            (count as u32 + self.local_size - 1) / self.local_size
        );

        Ok(())
    }

    fn generate_glsl(&mut self) -> Result<(), StruumError> {
        let mut source = String::new();

        source.push_str("#version 430\n\n");

        source.push_str(&format!("layout(local_size_x = {}) in;\n\n", self.local_size));

        source.push_str(&format!("const uint STRUUM_OFFSET = {}u;\n\n", self.offset));

        // Keep track to struct so no duplication occurs
        let mut structs = std::collections::HashSet::new();

        // Generating structs
        for binding in &self.buffer_bindings {
            if structs.insert(binding.gpu_type.clone()) {
                source.push_str(&binding.gpu_struct);
                source.push_str("\n\n");
            }
        }

        // Generating buffer bindings
        for (i, binding) in self.buffer_bindings.iter().enumerate() {
            let buffer_name = &binding.buffer.name;
            let gpu_type = &binding.gpu_type;

            source.push_str(&format!(
                "layout(std430, binding = {}) buffer _struum_{}Buffer_{}_ {{\n",
                i,
                gpu_type,
                i,
            ));

            source.push_str(&format!("    {} {}[];\n", gpu_type, buffer_name,));

            source.push_str("};\n\n");
        }

        // Append kernel source
        source.push_str(&self.source);
        source.push_str("\n\n");

        // Generate main()
        source.push_str(&format!(
            r#"
void main() {{
    uint id = gl_GlobalInvocationID.x;
    if (id >= {}u) return;
    {}(id);
}}"#,
            self.work_count
                .ok_or(StruumError::NotFound("Work Count Empty".to_string()))?,
            self.entry_point
        ));

        // TODO(slok): This finds and replaces the original variables
        // Can change the strings content of shader but i dont think its imp
        // So find a better way if possible
        source = self.mangle_source(&source);

        self.gpu_source = Some(source);

        Ok(())
    }

    fn mangle_source(&self, source: &str) -> String {
        let mut mangled = source.to_string();

        for binding in &self.buffer_bindings {
            mangled = Self::mangle_identifier(
                &mangled,
                &binding.buffer.name,
                &format!("_struum_{}", binding.buffer.name),
            );
        }

        for binding in &self.buffer_bindings {
            // Donot mangle the primitive types
            if !is_builtin_gpu_type(&binding.gpu_type) {
                mangled = Self::mangle_identifier(
                    &mangled,
                    &binding.gpu_type,
                    &format!("_struum_{}", binding.gpu_type),
                );
            }
        }

        mangled
    }

    fn mangle_identifier(
        source: &str,
        identifier: &str,
        replacement: &str,
    ) -> String {
        let re = Regex::new(
            &format!(r"\b{}\b", regex::escape(identifier))
        ).unwrap();

        let mut result = String::new();
        let mut last = 0;

        for m in re.find_iter(source) {
            // Don't replace if this identifier is preceded by '.'
            if m.start() > 0 {
                let previous = source.as_bytes()[m.start() - 1];

                if previous == b'.' {
                    continue;
                }
            }

            result.push_str(&source[last..m.start()]);
            result.push_str(replacement);

            last = m.end();
        }

        result.push_str(&source[last..]);

        result
    }
}

impl Shardable for Kernel {
    fn work_count(&self) -> usize {
        self.work_buffer_len().unwrap_or(0)
    }

    fn offset(&self) -> usize {
        self.offset as usize
    }

    fn shard(&self, range: Range<usize>) -> Result<Self, StruumError> {
        let work_count = self.work_buffer_len()?;
        let mut shard = self.clone();
        for binding in shard.buffer_bindings.iter_mut() {
            let mode = self.resolved_mode(binding, work_count);
            if mode == ShardMode::SumF32 {
                // Summing into one buffer from many GPU threads would need float atomics.
                return Err(StruumError::ShardError(format!(
                    "buffer '{}': SumF32 is only supported for CPU tasks",
                    binding.buffer.name
                )));
            }
            if mode == ShardMode::Split {
                if binding.buffer.actual_len() != work_count {
                    return Err(StruumError::ShardError(format!(
                        "split buffer '{}' has {} elements but the work count is {}",
                        binding.buffer.name,
                        binding.buffer.actual_len(),
                        work_count
                    )));
                }
                binding.buffer = binding.buffer.slice(range.clone())?;
            }
            // Pin the mode so the shard doesn't re-infer with its smaller count.
            binding.shard_mode = Some(mode);
        }
        shard.offset = self.offset + range.start as u32;
        shard.pack()?;
        Ok(shard)
    }

    fn merge(parts: Vec<Self>) -> Result<Self, StruumError> {
        let first = parts
            .first()
            .ok_or_else(|| StruumError::ShardError("cannot merge zero shards".to_string()))?;
        let mut merged = first.clone();
        let first_count = first.work_buffer_len()?;
        for (i, binding) in merged.buffer_bindings.iter_mut().enumerate() {
            if first.resolved_mode(&first.buffer_bindings[i], first_count) == ShardMode::Split {
                let pieces = parts
                    .iter()
                    .map(|k| {
                        k.buffer_bindings
                            .get(i)
                            .map(|b| b.buffer.clone())
                            .ok_or_else(|| StruumError::ShardError("shards have different buffer layouts".to_string()))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                binding.buffer = Buffer::concat(&pieces)?;
            }
        }
        merged.pack()?;
        Ok(merged)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kernel() -> Kernel {
        let x: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let mut k = Kernel::new("void compute(uint i) { y[i] = p[0] * x[i]; }", "compute");
        k.add_buffer::<f32>(Buffer::new("p", &[2.0f32]));
        k.add_buffer::<f32>(Buffer::new("x", &x));
        k.add_buffer::<f32>(Buffer::empty::<f32>("y", x.len()));
        k.set_work_buffer("x");
        k.pack().unwrap();
        k
    }

    #[test]
    fn shard_slices_split_and_keeps_broadcast() {
        let k = kernel();
        let s = k.shard(4..10).unwrap();
        assert_eq!(s.get_work_count(), Some(6));
        assert_eq!(s.get_offset(), 4);
        assert_eq!(s.get_buffer("x").unwrap().as_slice::<f32>()[0], 4.0);
        assert_eq!(s.get_buffer("p").unwrap().actual_len(), 1);
        assert!(s.get_gpu_source().unwrap().contains("STRUUM_OFFSET = 4u"));
        assert!(s.get_gpu_source().unwrap().contains("id >= 6u"));
    }

    #[test]
    fn shard_merge_round_trip() {
        let k = kernel();
        let parts: Vec<Kernel> = struum_types::split_ranges(10, 3)
            .into_iter()
            .map(|r| k.shard(r).unwrap())
            .collect();
        let merged = Kernel::merge(parts).unwrap();
        assert_eq!(merged.get_work_count(), Some(10));
        assert_eq!(
            merged.get_buffer("x").unwrap().as_slice::<f32>(),
            k.get_buffer("x").unwrap().as_slice::<f32>()
        );
        assert_eq!(merged.get_buffer("p").unwrap().as_slice::<f32>(), &[2.0]);
    }

    #[test]
    fn nested_shards_keep_modes() {
        // A 1-element shard must not re-infer the 1-element param buffer as split.
        let k = kernel();
        let s = k.shard(0..1).unwrap().shard(0..1).unwrap();
        assert_eq!(s.get_buffer("p").unwrap().as_slice::<f32>(), &[2.0]);
    }
}
