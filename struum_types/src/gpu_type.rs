// TODO(slok): Implement other gpu types like vecs

/// A Rust type that can be a GPU buffer element, with its GLSL equivalent.
///
/// Implemented for the scalar types; derive it for structs with `#[gpu_type]` from
/// `struum_macros`.
pub trait GPUType {
    /// GLSL declarations the type needs: a `struct` definition, or empty for built-ins.
    fn glsl() -> String;
    /// The GLSL type name, e.g. `float` or the struct's name.
    fn glsl_type() -> &'static str;
}

/// Whether `name` is a built-in GLSL scalar type (needs no struct definition).
pub fn is_builtin_gpu_type(name: &str) -> bool {
    matches!(
        name,
        "float"
            | "double"
            | "int"
            | "uint"
            | "bool"
    )
}

impl GPUType for f32 {
    fn glsl() -> String {
        "".to_string()
    }

    fn glsl_type() -> &'static str {
        "float"
    }
}

impl GPUType for f64 {
    fn glsl() -> String {
        "".to_string()
    }

    fn glsl_type() -> &'static str {
        "double"
    }
}

impl GPUType for i32 {
    fn glsl() -> String {
        "".to_string()
    }

    fn glsl_type() -> &'static str {
        "int"
    }
}

impl GPUType for u32 {
    fn glsl() -> String {
        "".to_string()
    }

    fn glsl_type() -> &'static str {
        "uint"
    }
}

impl GPUType for bool {
    fn glsl() -> String {
        "".to_string()
    }

    fn glsl_type() -> &'static str {
        "bool"
    }
}
