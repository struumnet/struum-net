use struum_opengl::OpenglBackend;
use struum_types::{
    StruumError,
    BufferRole,
    Buffer,
};


fn main() {
    let shader = r#"
        #version 430

        layout(local_size_x = 1) in;

        layout(std430, binding = 0) buffer Data {
            float values[];
        };

        void main()
        {
            uint i = gl_GlobalInvocationID.x;
            values[i] *= 2.0;
        }
    "#;

    let buffers = vec![
        Buffer::new(BufferRole::Input, vec![1,2,3]),
    ];


    let backend = OpenglBackend::new().unwrap();
    backend.execute(shader, &buffers).unwrap();
}
