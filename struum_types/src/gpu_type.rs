// TODO(slok): Implement other gpu types like vecs

pub trait GPUType {
    fn glsl() -> String;
    fn glsl_type() -> &'static str;
}

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
