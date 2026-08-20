use struum_opengl::OpenglBackend;
use struum_types::Buffer;

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
        Buffer::new("x", &input),
        Buffer::empty::<f32>("y", 3),
    ];

    let mut backend = OpenglBackend::new(shader, &buffers).unwrap();
    backend.execute();

    let out = backend.read_buffer::<f32>("y").unwrap();
    println!("{:#?}", out);
}
