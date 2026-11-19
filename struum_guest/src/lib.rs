//! Guest-side helpers for writing struum CPU tasks in Rust.
//!
//! Build your task crate as a `cdylib` for `wasm32-unknown-unknown`:
//!
//! ```ignore
//! use struum_guest::TaskContext;
//!
//! // y[i] = 2 * x[i]   (buffer 0 = x, buffer 1 = y, in the order the requester added them)
//! fn double(ctx: &TaskContext) {
//!     let x = ctx.input::<f32>(0);
//!     let y = ctx.output::<f32>(1);
//!     for i in 0..ctx.len() {
//!         y[i] = 2.0 * x[i];
//!     }
//! }
//!
//! struum_guest::tasks!(double);
//! ```
//!
//! `ctx.len()` is the number of work items in this piece and `ctx.offset()`
//! its first global index (`Split` buffers are already sliced to this piece).
//! A panic inside a task traps the instance and is reported as a task error.

use std::cell::Cell;

pub use bytemuck;

/// Alignment the host uses for buffers (enough for any `#[repr(C)]` Pod struct).
pub const BUFFER_ALIGN: usize = 16;

/// Allocates guest memory for the host. Exported by [`tasks!`].
#[doc(hidden)]
pub fn alloc(size: u32, align: u32) -> u32 {
    let layout = std::alloc::Layout::from_size_align(size.max(1) as usize, align.max(1) as usize)
        .expect("invalid allocation layout");
    // Memory is reclaimed when the host drops the instance after the task.
    unsafe { std::alloc::alloc_zeroed(layout) as u32 }
}

#[derive(Clone, Copy)]
struct RawBuffer {
    ptr: u32,
    byte_len: u32,
    stride: u32,
}

/// View of one piece of work handed to a task by the host.
pub struct TaskContext {
    offset: u32,
    len: u32,
    buffers: Vec<RawBuffer>,
    /// Bit i set => buffer i has been borrowed mutably via `output`.
    borrowed: Cell<u128>,
}

impl TaskContext {
    /// # Safety
    /// `desc` must point to a descriptor written by the struum host.
    #[doc(hidden)]
    pub unsafe fn from_desc(desc: u32) -> Self {
        let words = desc as *const u32;
        let (offset, len, count) = unsafe { (*words, *words.add(1), *words.add(2)) };
        let buffers = (0..count as usize)
            .map(|i| unsafe {
                let base = words.add(3 + 3 * i);
                RawBuffer { ptr: *base, byte_len: *base.add(1), stride: *base.add(2) }
            })
            .collect();
        Self { offset, len, buffers, borrowed: Cell::new(0) }
    }

    /// Global index of this piece's first work item.
    pub fn offset(&self) -> usize {
        self.offset as usize
    }

    /// Number of work items in this piece.
    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of buffers passed to the task.
    pub fn buffer_count(&self) -> usize {
        self.buffers.len()
    }

    fn raw(&self, index: usize, elem_size: usize) -> RawBuffer {
        let raw = *self
            .buffers
            .get(index)
            .unwrap_or_else(|| panic!("buffer {index} does not exist ({} buffers)", self.buffers.len()));
        assert!(
            raw.stride as usize == elem_size,
            "buffer {index} has stride {} but was accessed as a {}-byte type",
            raw.stride,
            elem_size
        );
        raw
    }

    /// Read-only view of buffer `index`.
    pub fn input<T: bytemuck::Pod>(&self, index: usize) -> &[T] {
        assert!(
            self.borrowed.get() & (1 << index) == 0,
            "buffer {index} is already borrowed as output"
        );
        let raw = self.raw(index, std::mem::size_of::<T>());
        let bytes = unsafe { std::slice::from_raw_parts(raw.ptr as *const u8, raw.byte_len as usize) };
        bytemuck::cast_slice(bytes)
    }

    /// Mutable view of buffer `index`. Each buffer can be taken as output once.
    #[allow(clippy::mut_from_ref)]
    pub fn output<T: bytemuck::Pod>(&self, index: usize) -> &mut [T] {
        let mask = 1u128 << index;
        assert!(self.borrowed.get() & mask == 0, "buffer {index} is already borrowed as output");
        let raw = self.raw(index, std::mem::size_of::<T>());
        self.borrowed.set(self.borrowed.get() | mask);
        let bytes = unsafe { std::slice::from_raw_parts_mut(raw.ptr as *mut u8, raw.byte_len as usize) };
        bytemuck::cast_slice_mut(bytes)
    }
}

/// Exports `struum_alloc` and one wasm entry point per task function.
/// Each function must have the signature `fn(&TaskContext)`.
#[macro_export]
macro_rules! tasks {
    ($($name:ident),+ $(,)?) => {
        #[unsafe(no_mangle)]
        pub extern "C" fn struum_alloc(size: u32, align: u32) -> u32 {
            $crate::alloc(size, align)
        }

        #[doc(hidden)]
        mod __struum_task_exports {
            $(
                #[unsafe(export_name = stringify!($name))]
                pub extern "C" fn $name(desc: u32) -> i32 {
                    let ctx = unsafe { $crate::TaskContext::from_desc(desc) };
                    super::$name(&ctx);
                    0
                }
            )+
        }
    };
}
