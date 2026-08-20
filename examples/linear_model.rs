use struum_opengl::OpenglBackend;
use struum_types::Buffer;

fn main() {
    let shader = r#"
        #version 430

        // We have 2 invocations:
        // invocation 0 -> calculate gradient for m
        // invocation 1 -> calculate gradient for c
        layout(local_size_x = 2) in;

        layout(std430, binding = 0) buffer Parameters {
            float params[];
        };

        layout(std430, binding = 1) buffer X {
            float x[];
        };

        layout(std430, binding = 2) buffer Y {
            float y[];
        };

        void main()
        {
            uint parameter = gl_GlobalInvocationID.x;

            float m = params[0];
            float c = params[1];

            float dm = 0.0;
            float dc = 0.0;

            // Number of training samples.
            uint n = x.length();

            for (uint i = 0; i < n; i++) {
                float prediction = m * x[i] + c;
                float error = prediction - y[i];

                dm += 2.0 * x[i] * error;
                dc += 2.0 * error;
            }

            // Mean gradient
            dm /= float(n);
            dc /= float(n);

            float learning_rate = 0.01;

            if (parameter == 0) {
                params[0] = m - learning_rate * dm;
            }

            if (parameter == 1) {
                params[1] = c - learning_rate * dc;
            }
        }
    "#;

    // Training data:
    //
    // y = 2x + 1
    //
    let x: Vec<f32> = (0..5)
        .map(|i| i as f32)
        .collect();

    let y: Vec<f32> = x
        .iter()
        .map(|&x| 2.0 * x + 1.0)
        .collect();

    // Initial model:
    //
    // y = 0x + 0
    //
    let buffers = vec![
        Buffer::new("params", &[0.0f32, 0.0f32]),
        Buffer::new("x", &x),
        Buffer::new("y", &y),
    ];

    let mut backend = OpenglBackend::new(shader, &buffers).unwrap();

    for iteration in 0..1000 {
        backend.execute();

        if iteration % 100 == 0 {
            println!("iteration {}", iteration);
        }
    }

    let result = backend.read_buffer::<f32>("params").unwrap();

    println!();
    println!("Final model:");
    println!("m = {}", result[0]);
    println!("c = {}", result[1]);
}
