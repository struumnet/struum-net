use struum_coordinator::Coordinator;
use struum_kernel::*;
use struum_macros::gpu_type;
use struum_node::node::Node;
use struum_types::Buffer;
use struum_types::StruumError;
use struum_types::network::*;

#[gpu_type]
#[derive(Debug, PartialEq)]
struct Vec2 {
    x: f32,
    y: f32,
}

#[gpu_type]
#[derive(Debug, PartialEq)]
struct Param {
    m: f32,
    c: f32,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {

    let mut coordinator = Coordinator::<2048>::new(39001, 39002).await?;

    let node1_id = UID::new(1);
    let mut node1 = Node::<2048>::new(node1_id, 39011, 39012, NodeBackend::OpenGL, 2).await?;

    let node2_id = UID::new(2);
    let mut node2 = Node::<2048>::new(node2_id, 39021, 39022, NodeBackend::Cpu, 2).await?;
    let details1 = node1.details();
    let details2 = node2.details();

    coordinator.register_node(details1.clone());
    coordinator.register_node(details2.clone());

    let selected = coordinator.select_node().expect("Should select a node");

    let opengl_node = coordinator
        .select_node_by_backend(NodeBackend::OpenGL)
        .expect("Should find OpenGL node");


    let listen_handle1 = tokio::spawn(async move {
        let sibling = node1.listen_sibling_introduction().await?;
        Ok::<(Node<2048>, NodeDetails), StruumError>((node1, sibling))
    });

    let listen_handle2 = tokio::spawn(async move {
        let sibling = node2.listen_sibling_introduction().await?;
        Ok::<(Node<2048>, NodeDetails), StruumError>((node2, sibling))
    });

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    coordinator.introduce_nodes(&details1.id, &details2.id).await?;

    let (mut node1, sibling_for_node1) = listen_handle1.await??;

    let (_node2, sibling_for_node2) = listen_handle2.await??;

    assert_eq!(sibling_for_node1.id, details2.id);
    assert_eq!(sibling_for_node2.id, details1.id);

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
        ],
    ));
    kernel.add_buffer::<Vec2>(Buffer::empty::<Vec2>("output", 5));
    kernel.set_work_buffer("pos");
    kernel.pack().unwrap();

    // Submit job to Node 1's local JobScheduler
    let job_id = node1.submit_job(kernel).await;

    // Receive completion signal from local worker threads
    let completed_id = node1.recv_completed_job().await.expect("Expected completed job ID");
    assert_eq!(completed_id, job_id);

    // Retrieve finished kernel result
    let result_kernel = node1
        .get_job_result(&completed_id)
        .await
        .expect("Expected result kernel");

    let output_buffer = result_kernel.get_buffer("output").expect("Expected output buffer");
    let values = output_buffer.as_slice::<Vec2>();

    assert_eq!(values.len(), 5);
    assert_eq!(values[0], Vec2 { x: 2.0, y: 30.0 });
    assert_eq!(values[4], Vec2 { x: 10.0, y: 150.0 });

    Ok(())
}
