use struum_scheduler::JobScheduler;

use struum_kernel::*;
use struum_macros::gpu_type;
use struum_types::Buffer;
use tokio::sync::mpsc;

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

#[tokio::main]
async fn main() {

    // Need to create a channel for communication between process and job schedular worker
    let (tx, mut rx) = mpsc::channel(100);
    let js = JobScheduler::new(tx, 2);

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

    // Create kernel
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

    // Submit the job
    let job_id = js.add_job(kernel).await;

    println!("Submitted job: {job_id}");

    // Job response listener
    loop {
        // Get response from the job schedular
        let Some(id) = rx.recv().await else {
            break;
        };

        println!("Completed: {id}");

        // Get the kernel as the result of the job
        if let Some(kernel) = js.get_result_of_job(&id).await {

            // Read the output buffer from kernel
            let output = kernel
                .get_buffer("output")
                .unwrap();
            let values = output.as_slice::<Vec2>();
            println!("{values:#?}");
        }
    }
}
