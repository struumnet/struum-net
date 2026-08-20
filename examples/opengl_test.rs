use struum_opengl::OpenglBackend;
use struum_types::{
    BufferRole,
    Buffer,
};


fn main() {
    let shader = r#"
        #version 430

        layout(local_size_x = 3) in;

        layout(std430, binding = 0) buffer Input {
            float x[];
        };

        layout(std430, binding = 1) buffer Output {
            float y[];
        };

        void main() {
            uint i = gl_GlobalInvocationID.x;
            y[i] = x[i] * 2.0;
        }
    "#;

    let input: [f32; 3] = [1.0, 2.0, 3.0];
    let buffers = vec![
        Buffer::new(BufferRole::Input, &input),
        Buffer::empty::<f32>(BufferRole::Output, 3),
    ];

    let backend = OpenglBackend::new().unwrap();
    let output = backend.execute(shader, &buffers).unwrap();

    for out in output {
        println!("{:#?}", out.as_slice::<f32>());
    }
}
