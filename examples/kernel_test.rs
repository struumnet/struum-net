use struum_kernel::*;

use struum_macros::gpu_type;
use struum_types::Buffer;
use struum_opengl::OpenglBackend;

#[gpu_type]
#[derive(Debug)]
struct Vec2 {
    x: f32,
    y: f32,
}

#[gpu_type]
#[derive(Debug)]
struct Param {
    m: f32,
    c: f32,
}

fn main() {
    let source = r#"
        void compute(uint i) {
            Param p = param[0];
            Vec2 x = pos[i];
            output[i] = Vec2(
                p.m * x.x,
                p.c * x.y
            );
        }
    "#;

    let mut kernel = Kernel::new(source, "compute");

    kernel.add_buffer::<Param>(Buffer::new("param", &[Param { m: 2.0, c: 3.0 }]));
    kernel.add_buffer::<Vec2>(Buffer::new(
        "pos",
        &[
            Vec2 { x: 1.0, y: 10.0 },
            Vec2 { x: 2.0, y: 20.0 },
            Vec2 { x: 3.0, y: 30.0 },
            Vec2 { x: 4.0, y: 40.0 },
            Vec2 { x: 5.0, y: 50.0 },
        ])
    );
    kernel.add_buffer::<Vec2>(
        Buffer::empty::<Vec2>("output", 5)
    );

    kernel.set_work_buffer("pos");
    kernel.pack().unwrap();

    let mut backend = OpenglBackend::from_kernel(&kernel).unwrap();
    backend.execute();

    let out = backend.read_buffer::<Vec2>("output").unwrap();
    println!("{:#?}", out);
}
