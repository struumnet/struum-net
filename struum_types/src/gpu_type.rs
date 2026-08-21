pub trait GPUType {
    fn glsl() -> String;
    fn glsl_type() -> &'static str;
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
