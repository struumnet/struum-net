use regex::Regex;
use struum_types::{Buffer, GPUType, StruumError};

// TODO(slok): Support multiple backend types

// TODO(slok): Currently the local size is hard coded
// 256 might be enough of invocations needed per work group
// but if it created a bottleneck then figure something out

pub struct BufferBinding {
    pub buffer: Buffer,
    pub gpu_type: &'static str,
    pub gpu_struct: String,
}

pub struct Kernel {
    source: String,
    entry_point: String,
    buffer_bindings: Vec<BufferBinding>,

    local_size: u32,
    work_buffer: Option<String>,
    group_size: Option<u32>,
    gpu_source: Option<String>,
}

impl Kernel {
    pub fn new(source: &str, entry_point: &str) -> Self {
        Self {
            source: source.to_string(),
            entry_point: entry_point.to_string(),
            buffer_bindings: Vec::new(),
            local_size: 256,
            work_buffer: None,
            group_size: None,
            gpu_source: None,
        }
    }

    pub fn add_buffer<T: GPUType>(&mut self, buffer: Buffer) {
        self.buffer_bindings.push(BufferBinding {
            buffer: buffer,
            gpu_type: T::glsl_type(),
            gpu_struct: T::glsl(),
        });
    }

    pub fn set_work_buffer(&mut self, name: &str) {
        self.work_buffer = Some(name.to_string());
    }

    pub fn pack(&mut self) -> Result<(), StruumError> {
        self.compute_group_size()?;
        self.generate_glsl();

        Ok(())
    }

    pub fn get_group_size(&self) -> Option<u32> {
        self.group_size
    }

    pub fn get_gpu_source(&self) -> Option<&str> {
        self.gpu_source.as_deref()
    }

    pub fn get_buffer_bindings(&self) -> &[BufferBinding] {
        &self.buffer_bindings
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
        self.group_size = Some((count as u32 + self.local_size - 1) / self.local_size);

        Ok(())
    }

    fn generate_glsl(&mut self) {
        let mut source = String::new();

        source.push_str("#version 430\n\n");

        source.push_str(&format!("layout(local_size_x = {}) in;\n", self.local_size));

        // Keep track to struct so no duplication occurs
        let mut structs = std::collections::HashSet::new();

        // Generating structs
        for binding in &self.buffer_bindings {
            if structs.insert(binding.gpu_type) {
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
            "void main() {{\n    uint id = gl_GlobalInvocationID.x;\n    {}(id);\n}}\n",
            self.entry_point
        ));

        // TODO(slok): This finds and replaces the original variables
        // Can change the strings content of shader but i dont think its imp
        // So find a better way if possible
        source = self.mangle_source(&source);

        self.gpu_source = Some(source);
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
            mangled = Self::mangle_identifier(
                &mangled,
                &binding.gpu_type,
                &format!("_struum_{}", binding.gpu_type),
            );
        }

        mangled
    }

    fn mangle_identifier(source: &str, original: &str, mangled: &str) -> String {
        let pattern = format!(r"\b{}\b", regex::escape(original));

        Regex::new(&pattern)
            .unwrap()
            .replace_all(source, mangled)
            .into_owned()
    }
}
