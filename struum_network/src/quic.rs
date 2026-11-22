//! QUIC transport: one UDP socket per node that both accepts and dials.
//!
//! Every node makes a self-signed certificate at startup. Both sides present
//! one and accept any; trust comes from a shared token instead (devices you
//! own, see `STRUUM_TOKEN`). On each new connection both sides send a
//! [`Hello`] on the connection's first stream:
//!
//! ```text
//! mac = HMAC-SHA256(token, "struum/1" ‖ sender_id ‖ sender_cert_fp ‖ receiver_cert_fp)
//! ```
//!
//! The fingerprints are the SHA-256 of the certificates each side saw in this
//! TLS session, and TLS has already checked that each side holds its
//! certificate's key. A man in the middle sees different certificates on its
//! two legs, so it can't forward a valid `Hello` without knowing the token. A
//! peer whose `Hello` doesn't check out is closed before any work is sent.
//!
//! After the `Hello`s, the same stream carries packets exactly like a TCP
//! connection, through a [`PeerConnection`], so the dispatcher and
//! [`PeerConnection`]-based serving work unchanged over QUIC.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use hmac::{Hmac, Mac};
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::crypto::{CryptoProvider, WebPkiSupportedAlgorithms};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer, ServerName, UnixTime};
use rustls::server::danger::{ClientCertVerified, ClientCertVerifier};
use rustls::{DigitallySignedStruct, DistinguishedName, SignatureScheme};
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;

use struum_types::StruumError;

use crate::layer::PeerConnection;

type HmacSha256 = Hmac<Sha256>;

/// ALPN protocol and `Hello` label; bump when the handshake changes.
const PROTOCOL: &[u8] = b"struum/1";
/// Name sent in TLS; certificates are self-signed, so it is never checked.
const SERVER_NAME: &str = "struum";
/// Application close code for a peer that failed authentication.
const CLOSE_AUTH: u32 = 1;
/// How long a new connection may take to finish the `Hello` exchange.
const HELLO_TIMEOUT: Duration = Duration::from_secs(10);

fn err(context: &str, e: impl std::fmt::Display) -> StruumError {
    StruumError::NetworkConnectionError(format!("{context}: {e}"))
}

/// The shared token from `STRUUM_TOKEN`. Without one, every node with this
/// build can join, which is fine on a home LAN but not on the internet.
pub fn token_from_env() -> Vec<u8> {
    match std::env::var("STRUUM_TOKEN") {
        Ok(token) if !token.is_empty() => token.into_bytes(),
        _ => {
            log::warn!("STRUUM_TOKEN is not set: any struum node can connect to this one");
            Vec::new()
        }
    }
}

/// Authentication message each side sends first on a new connection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hello {
    /// Node id of the sender.
    pub id: u64,
    pub mac: [u8; 32],
}

impl Hello {
    const LEN: usize = PROTOCOL.len() + 8 + 32;

    fn mac(token: &[u8], id: u64, sender_fp: &[u8; 32], receiver_fp: &[u8; 32]) -> HmacSha256 {
        let mut mac = HmacSha256::new_from_slice(token).expect("HMAC takes keys of any length");
        mac.update(PROTOCOL);
        mac.update(&id.to_le_bytes());
        mac.update(sender_fp);
        mac.update(receiver_fp);
        mac
    }

    fn new(token: &[u8], id: u64, own_fp: &[u8; 32], peer_fp: &[u8; 32]) -> Self {
        let mac = Self::mac(token, id, own_fp, peer_fp).finalize().into_bytes().into();
        Self { id, mac }
    }

    /// Checks a `Hello` received from the peer, in constant time.
    fn verify(&self, token: &[u8], peer_fp: &[u8; 32], own_fp: &[u8; 32]) -> bool {
        Self::mac(token, self.id, peer_fp, own_fp).verify_slice(&self.mac).is_ok()
    }

    fn to_bytes(&self) -> [u8; Self::LEN] {
        let mut out = [0u8; Self::LEN];
        out[..PROTOCOL.len()].copy_from_slice(PROTOCOL);
        out[PROTOCOL.len()..PROTOCOL.len() + 8].copy_from_slice(&self.id.to_le_bytes());
        out[PROTOCOL.len() + 8..].copy_from_slice(&self.mac);
        out
    }

    fn from_bytes(bytes: &[u8; Self::LEN]) -> Option<Self> {
        if &bytes[..PROTOCOL.len()] != PROTOCOL {
            return None;
        }
        let id = u64::from_le_bytes(bytes[PROTOCOL.len()..PROTOCOL.len() + 8].try_into().ok()?);
        let mac = bytes[PROTOCOL.len() + 8..].try_into().ok()?;
        Some(Self { id, mac })
    }
}

/// An authenticated QUIC peer.
#[derive(Clone)]
pub struct QuicPeer {
    /// Node id the peer proved with its `Hello`.
    pub id: u64,
    /// Packet connection over the peer's first stream.
    pub conn: PeerConnection,
    connection: quinn::Connection,
}

impl QuicPeer {
    /// Address the peer's packets come from (its NAT mapping when it's behind one).
    pub fn remote_addr(&self) -> SocketAddr {
        self.connection.remote_address()
    }

    /// Current round-trip estimate.
    pub fn rtt(&self) -> Duration {
        self.connection.rtt()
    }

    /// Closes the connection.
    pub fn close(&self) {
        self.connection.close(0u32.into(), b"bye");
    }
}

impl std::fmt::Debug for QuicPeer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuicPeer").field("id", &self.id).field("addr", &self.remote_addr()).finish()
    }
}

/// One QUIC endpoint: accepts and dials on the same UDP socket.
///
/// Dropping it closes the endpoint and every connection made through it.
pub struct QuicLayer {
    endpoint: quinn::Endpoint,
    auth: Arc<Auth>,
    /// Peers that dialed in and authenticated, fed by the accept task.
    accepted: tokio::sync::Mutex<tokio::sync::mpsc::Receiver<QuicPeer>>,
}

/// Who we are, for the `Hello` exchange.
struct Auth {
    id: u64,
    token: Arc<[u8]>,
    cert_fp: [u8; 32],
}

impl std::fmt::Debug for QuicLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuicLayer").field("id", &self.auth.id).field("addr", &self.local_addr().ok()).finish()
    }
}

impl Drop for QuicLayer {
    fn drop(&mut self) {
        // Also ends the accept task, which holds a handle to the endpoint.
        self.endpoint.close(0u32.into(), b"shutdown");
    }
}

/// Authenticated peers waiting for [`QuicLayer::accept`].
const ACCEPT_BACKLOG: usize = 64;

impl QuicLayer {
    /// Binds a UDP socket on `addr` (port 0 picks one). Needs a tokio runtime.
    pub fn bind(addr: SocketAddr, id: u64, token: &[u8]) -> Result<Self, StruumError> {
        let socket = std::net::UdpSocket::bind(addr).map_err(|e| err(&format!("bind UDP {addr}"), e))?;
        Self::from_socket(socket, id, token)
    }

    /// Uses an already-bound UDP socket, e.g. the one a relay has seen, so its
    /// NAT mapping is the one peers punch through. Needs a tokio runtime.
    pub fn from_socket(socket: std::net::UdpSocket, id: u64, token: &[u8]) -> Result<Self, StruumError> {
        let cert = rcgen::generate_simple_self_signed(vec![SERVER_NAME.to_string()])
            .map_err(|e| err("generate certificate", e))?;
        let cert_der = CertificateDer::from(cert.cert.der().to_vec());
        let key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(cert.key_pair.serialize_der()));
        let cert_fp = fingerprint(&cert_der);
        let provider = Arc::new(rustls::crypto::ring::default_provider());

        let mut server = rustls::ServerConfig::builder_with_provider(provider.clone())
            .with_protocol_versions(&[&rustls::version::TLS13])
            .map_err(|e| err("TLS config", e))?
            .with_client_cert_verifier(Arc::new(AnyCert::new(&provider)))
            .with_single_cert(vec![cert_der.clone()], key.clone_key())
            .map_err(|e| err("TLS server config", e))?;
        server.alpn_protocols = vec![PROTOCOL.to_vec()];

        let mut client = rustls::ClientConfig::builder_with_provider(provider.clone())
            .with_protocol_versions(&[&rustls::version::TLS13])
            .map_err(|e| err("TLS config", e))?
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(AnyCert::new(&provider)))
            .with_client_auth_cert(vec![cert_der], key)
            .map_err(|e| err("TLS client config", e))?;
        client.alpn_protocols = vec![PROTOCOL.to_vec()];

        let transport = Arc::new(transport_config());
        let mut server_config = quinn::ServerConfig::with_crypto(Arc::new(
            quinn::crypto::rustls::QuicServerConfig::try_from(server).map_err(|e| err("QUIC server config", e))?,
        ));
        server_config.transport_config(transport.clone());
        let mut client_config = quinn::ClientConfig::new(Arc::new(
            quinn::crypto::rustls::QuicClientConfig::try_from(client).map_err(|e| err("QUIC client config", e))?,
        ));
        client_config.transport_config(transport);

        let runtime = quinn::default_runtime().ok_or_else(|| err("QUIC", "no async runtime"))?;
        let mut endpoint = quinn::Endpoint::new(quinn::EndpointConfig::default(), Some(server_config), socket, runtime)
            .map_err(|e| err("QUIC endpoint", e))?;
        endpoint.set_default_client_config(client_config);
        let auth = Arc::new(Auth { id, token: token.into(), cert_fp });

        // Handshakes run in their own tasks, so a slow or hostile dialer can't
        // hold up the peers behind it.
        let (tx, rx) = tokio::sync::mpsc::channel(ACCEPT_BACKLOG);
        let (driver, driver_auth) = (endpoint.clone(), auth.clone());
        tokio::spawn(async move {
            while let Some(incoming) = driver.accept().await {
                let (auth, tx) = (driver_auth.clone(), tx.clone());
                tokio::spawn(async move {
                    let remote = incoming.remote_address();
                    match auth.accept_one(incoming).await {
                        Ok(peer) => {
                            let _ = tx.send(peer).await;
                        }
                        Err(e) => log::warn!("Rejected QUIC peer {remote}: {e}"),
                    }
                });
            }
        });
        Ok(Self { endpoint, auth, accepted: tokio::sync::Mutex::new(rx) })
    }

    /// The bound UDP address.
    pub fn local_addr(&self) -> Result<SocketAddr, StruumError> {
        self.endpoint.local_addr().map_err(|e| err("local address", e))
    }

    /// This node's id, as sent in its `Hello`.
    pub fn id(&self) -> u64 {
        self.auth.id
    }

    /// Dials `addr` and authenticates. Fails if either side's token differs.
    pub async fn connect(&self, addr: SocketAddr) -> Result<QuicPeer, StruumError> {
        let connecting = self
            .endpoint
            .connect(addr, SERVER_NAME)
            .map_err(|e| err(&format!("connect to {addr}"), e))?;
        let connection = connecting.await.map_err(|e| err(&format!("connect to {addr}"), e))?;
        let auth = &self.auth;
        let handshake = async {
            let (mut send, mut recv) = connection.open_bi().await.map_err(|e| err("open stream", e))?;
            auth.hello(&connection, &mut send, &mut recv, true).await.map(|id| (id, send, recv))
        };
        let (id, send, recv) = auth.finish(&connection, handshake).await?;
        Ok(auth.peer(id, connection, send, recv))
    }

    /// Waits for the next peer that dials in and authenticates. Peers that fail
    /// to authenticate are logged and closed. Returns an error once the
    /// endpoint is closed.
    pub async fn accept(&self) -> Result<QuicPeer, StruumError> {
        self.accepted.lock().await.recv().await.ok_or_else(|| err("accept", "endpoint closed"))
    }

    /// Closes every connection and stops accepting.
    pub fn close(&self) {
        self.endpoint.close(0u32.into(), b"shutdown");
    }
}

impl Auth {
    async fn accept_one(&self, incoming: quinn::Incoming) -> Result<QuicPeer, StruumError> {
        let connection = incoming.await.map_err(|e| err("handshake", e))?;
        let handshake = async {
            let (mut send, mut recv) = connection.accept_bi().await.map_err(|e| err("accept stream", e))?;
            self.hello(&connection, &mut send, &mut recv, false).await.map(|id| (id, send, recv))
        };
        let (id, send, recv) = self.finish(&connection, handshake).await?;
        Ok(self.peer(id, connection, send, recv))
    }

    /// Runs a `Hello` exchange with a time limit, closing the connection if it fails.
    async fn finish<T>(
        &self,
        connection: &quinn::Connection,
        handshake: impl Future<Output = Result<T, StruumError>>,
    ) -> Result<T, StruumError> {
        let out = tokio::time::timeout(HELLO_TIMEOUT, handshake)
            .await
            .unwrap_or_else(|_| Err(err("authenticate", "timed out")));
        if out.is_err() {
            connection.close(CLOSE_AUTH.into(), b"authentication failed");
        }
        out
    }

    /// Sends our `Hello` and checks the peer's. The dialer speaks first, since
    /// QUIC only tells the other side about a stream once data is written on it.
    async fn hello(
        &self,
        connection: &quinn::Connection,
        send: &mut quinn::SendStream,
        recv: &mut quinn::RecvStream,
        dialer: bool,
    ) -> Result<u64, StruumError> {
        let peer_fp = peer_fingerprint(connection)?;
        let ours = Hello::new(&self.token, self.id, &self.cert_fp, &peer_fp).to_bytes();
        let read = async {
            let mut buf = [0u8; Hello::LEN];
            recv.read_exact(&mut buf).await.map_err(|e| err("read hello", e))?;
            Hello::from_bytes(&buf).ok_or_else(|| err("authenticate", "peer is not a struum node"))
        };
        let check = |theirs: Hello| {
            if theirs.verify(&self.token, &peer_fp, &self.cert_fp) {
                Ok(theirs.id)
            } else {
                Err(err("authenticate", "wrong token"))
            }
        };
        let id = if dialer {
            send.write_all(&ours).await.map_err(|e| err("send hello", e))?;
            check(read.await?)?
        } else {
            // Check before answering, so a peer without the token learns nothing.
            let id = check(read.await?)?;
            send.write_all(&ours).await.map_err(|e| err("send hello", e))?;
            id
        };
        send.flush().await.map_err(|e| err("send hello", e))?;
        Ok(id)
    }

    fn peer(&self, id: u64, connection: quinn::Connection, send: quinn::SendStream, recv: quinn::RecvStream) -> QuicPeer {
        let keep = Arc::new(connection.clone());
        let conn = PeerConnection::from_halves(connection.remote_address(), recv, send, Some(keep));
        log::info!("QUIC peer {} authenticated at {}", id, connection.remote_address());
        QuicPeer { id, conn, connection }
    }

}

/// Large windows so one stream can keep a fast LAN busy; keep-alives stop NAT
/// mappings expiring while a shard runs.
fn transport_config() -> quinn::TransportConfig {
    let mut t = quinn::TransportConfig::default();
    t.stream_receive_window((16u32 << 20).into());
    t.receive_window((32u32 << 20).into());
    t.send_window(32 << 20);
    t.keep_alive_interval(Some(Duration::from_secs(5)));
    t.max_idle_timeout(Some(Duration::from_secs(30).try_into().expect("30 s fits")));
    t
}

fn fingerprint(cert: &CertificateDer<'_>) -> [u8; 32] {
    Sha256::digest(cert.as_ref()).into()
}

fn peer_fingerprint(connection: &quinn::Connection) -> Result<[u8; 32], StruumError> {
    let identity = connection
        .peer_identity()
        .ok_or_else(|| err("authenticate", "peer sent no certificate"))?;
    let certs = identity
        .downcast::<Vec<CertificateDer<'static>>>()
        .map_err(|_| err("authenticate", "unexpected peer identity"))?;
    certs
        .first()
        .map(fingerprint)
        .ok_or_else(|| err("authenticate", "peer sent no certificate"))
}

/// Accepts any certificate, but still checks the handshake signatures, so the
/// peer must hold the key of the certificate whose fingerprint the `Hello` binds.
#[derive(Debug)]
struct AnyCert(WebPkiSupportedAlgorithms);

impl AnyCert {
    fn new(provider: &CryptoProvider) -> Self {
        Self(provider.signature_verification_algorithms)
    }
}

impl ServerCertVerifier for AnyCert {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(message, cert, dss, &self.0)
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(message, cert, dss, &self.0)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.0.supported_schemes()
    }
}

impl ClientCertVerifier for AnyCert {
    fn root_hint_subjects(&self) -> &[DistinguishedName] {
        &[]
    }

    fn verify_client_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _now: UnixTime,
    ) -> Result<ClientCertVerified, rustls::Error> {
        Ok(ClientCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(message, cert, dss, &self.0)
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(message, cert, dss, &self.0)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.0.supported_schemes()
    }

    fn client_auth_mandatory(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hello_round_trips_and_binds_both_certificates() {
        let (a, b, c) = ([1u8; 32], [2u8; 32], [3u8; 32]);
        let hello = Hello::new(b"secret", 7, &a, &b);
        let parsed = Hello::from_bytes(&hello.to_bytes()).unwrap();
        assert_eq!(parsed, hello);
        assert!(parsed.verify(b"secret", &a, &b));
        assert!(!parsed.verify(b"other", &a, &b), "wrong token");
        assert!(!parsed.verify(b"secret", &c, &b), "different sender certificate");
        assert!(!parsed.verify(b"secret", &a, &c), "different receiver certificate");
        let mut forged = parsed.clone();
        forged.id = 8;
        assert!(!forged.verify(b"secret", &a, &b), "id is covered by the mac");
    }

    #[tokio::test]
    async fn a_silent_dialer_does_not_hold_up_others() {
        let local: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let server = QuicLayer::bind(local, 1, b"t").unwrap();
        let addr = server.local_addr().unwrap();

        // Finishes TLS and opens a stream, but never sends its Hello.
        let silent = QuicLayer::bind(local, 2, b"t").unwrap();
        let stalled = silent.endpoint.connect(addr, SERVER_NAME).unwrap().await.unwrap();
        let (_send, _recv) = stalled.open_bi().await.unwrap();

        let honest = QuicLayer::bind(local, 3, b"t").unwrap();
        let (accepted, connected) = tokio::join!(
            tokio::time::timeout(Duration::from_secs(2), server.accept()),
            honest.connect(addr)
        );
        connected.unwrap();
        assert_eq!(accepted.expect("not stuck behind the silent dialer").unwrap().id, 3);
    }
}
