use struum_kernel::Kernel;
use struum_macros::gpu_type;
use struum_opengl::OpenglBackend;
use struum_types::Buffer;

#[gpu_type]
#[derive(Debug)]
struct Param {
    m: f32,
    c: f32,
}

#[gpu_type]
#[derive(Debug)]
struct Gradient {
    dm: f32,
    dc: f32,
}

fn main() {
    // Training data:
    //
    // y = 2x + 1
    //
    let x: Vec<f32> = (0..5).map(|i| i as f32).collect();
    let y: Vec<f32> = x.iter().map(|&x| 2.0 * x + 1.0).collect();

    let source = r#"
        void compute(uint i) {
            Param p = param[0];

            float prediction = p.m * x[i] + p.c;
            float error = prediction - y[i];

            gradients[i] = Gradient(
                2.0 * x[i] * error,
                2.0 * error
            );
        }
    "#;

    let mut kernel = Kernel::new(source, "compute");

    // Model parameters.
    kernel.add_buffer::<Param>(
        Buffer::new(
            "param",
            &[Param {
                m: 0.0,
                c: 0.0,
            }],
        ),
    );

    // Training data.
    kernel.add_buffer::<f32>(
        Buffer::new("x", &x),
    );

    kernel.add_buffer::<f32>(
        Buffer::new("y", &y),
    );

    // One gradient per training sample.
    kernel.add_buffer::<Gradient>(
        Buffer::empty::<Gradient>("gradients", x.len()),
    );

    // One invocation per x/y sample.
    kernel.set_work_buffer("x");

    kernel.pack().unwrap();

    let src = kernel.get_gpu_source().unwrap();
    println!("Generated GLSL");
    println!("{}", src);

    let mut backend =
        OpenglBackend::from_kernel(&kernel).unwrap();

    println!("Computation..");
    for iteration in 0..1000 {
        backend.execute();

        // Read the per-sample gradients.
        let gradients = backend.read_buffer::<Gradient>("gradients").unwrap();

        // Reduce gradients on CPU for now.
        let mut dm = 0.0;
        let mut dc = 0.0;

        for gradient in &gradients {
            dm += gradient.dm;
            dc += gradient.dc;
        }

        dm /= x.len() as f32;
        dc /= x.len() as f32;

        // Read current parameters.
        let mut params = backend.read_buffer::<Param>("param").unwrap();

        params[0].m -= 0.01 * dm;
        params[0].c -= 0.01 * dc;

        // Write back the new parameters
        backend.write_buffer("param", &params).unwrap();

        if iteration % 100 == 0 {
            println!(
                "iteration {}: m={}, c={}",
                iteration,
                params[0].m,
                params[0].c
            );
        }
    }

    let result = backend.read_buffer::<Param>("param").unwrap();

    println!();
    println!("Final model:");
    println!("m = {}", result[0].m);
    println!("c = {}", result[0].c);
}
