use struum_relay::RelayServer;
use struum_types::StruumError;

#[tokio::main]
async fn main() -> Result<(), StruumError> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let udp_port = std::env::var("RELAY_UDP_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(39001);

    let tcp_port = std::env::var("RELAY_TCP_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(39002);

    println!("Starting struum_relay server on UDP: {} / TCP: {}", udp_port, tcp_port);
    let mut relay = RelayServer::<2048>::new(udp_port, tcp_port).await?;

    println!("struum_relay is running and waiting for node registrations...");
    relay.run().await
}
