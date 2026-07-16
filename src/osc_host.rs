//! OSC (Open Sound Control) host provider for PhiFlow.
//!
//! When plugged into an Evaluator, this host broadcasts every PhiFlow construct
//! event as an OSC message over UDP. Any OSC-capable environment — TouchDesigner,
//! Three.js, Unity, SuperCollider, Max/MSP, PureData, vvvv, Notch — can receive
//! the stream and render it in real-time.
//!
//! ## Port allocation
//!
//! Default OSC port: **18032** (sibling to the metrics bridge on 18030).
//! Default WebSocket bridge port: **18528** (528 Hz = Creation frequency).
//! Both follow the PhiFlow 18xxx port scheme (see PORT_REGISTRY.md).
//!
//!
//! ## OSC address scheme
//!
//! | PhiFlow construct | OSC address | Arguments |
//! |-------------------|-------------|-----------|
//! | `intention "x" {}` | `/phi/intention/push` | `s` name, `i` depth |
//! | intention exits | `/phi/intention/pop` | `s` name, `i` depth |
//! | `witness` | `/phi/witness` | `f` coherence, `f` timestamp, `s` intention |
//! | `resonate value` | `/phi/resonate` | `s` intention, `s` value |
//! | `coherence` | `/phi/coherence` | `f` value |
//! | `stream "x" {}` | `/phi/stream/push` | `s` name |
//! | stream breaks | `/phi/stream/break` | `s` name |
//! | `anchor "sensor" {}` | `/phi/anchor/gate` | `s` sensor, `f` actual, `f` threshold, `i` open |
//! | `broadcast ch msg` | `/phi/broadcast` | `s` channel, `s` message |
//! | `listen ch` | `/phi/listen` | `s` channel, `s` message |
//! | program start | `/phi/start` | `s` source |
//! | program end | `/phi/end` | `f` final_coherence |

use crate::host::{PhiHostProvider, WitnessAction, WitnessSnapshot};
use crate::phi_ir::SensorKind;
use rosc::OscPacket;
use std::net::UdpSocket;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Inner state shared between the host provider and the caller.
struct OscInner {
    socket: UdpSocket,
    addr: String,
    start_time: Instant,
    depth: i32,
    /// Delay in ms between OSC events (for visualizer pacing).
    delay_ms: u64,
}

/// An OSC host provider that broadcasts PhiFlow runtime events over UDP.
/// Cloneable via Arc so the caller can retain a handle to emit start/end events.
pub struct OscHostProvider {
    inner: Arc<Mutex<OscInner>>,
}

impl OscHostProvider {
    /// Create a new OSC host that sends to `127.0.0.1:<port>`.
    /// `delay_ms` adds a sleep after each OSC event so visualizers can keep up.
    pub fn new(port: u16) -> std::io::Result<Self> {
        Self::with_delay(port, 0)
    }

    /// Create an OSC host with a delay (in ms) between events.
    pub fn with_delay(port: u16, delay_ms: u64) -> std::io::Result<Self> {
        let socket = UdpSocket::bind("0.0.0.0:0")?;
        let addr = format!("127.0.0.1:{}", port);
        Ok(OscHostProvider {
            inner: Arc::new(Mutex::new(OscInner {
                socket,
                addr,
                start_time: Instant::now(),
                depth: 0,
                delay_ms,
            })),
        })
    }

    /// Emit a program-started event.
    pub fn emit_start(&self, source: &str) {
        self.send(&OscPacket::Message(rosc::OscMessage {
            addr: "/phi/start".to_string(),
            args: vec![rosc::OscType::String(source.to_string())],
        }));
    }

    /// Emit a program-ended event with final coherence.
    pub fn emit_end(&self, final_coherence: f64) {
        self.send(&OscPacket::Message(rosc::OscMessage {
            addr: "/phi/end".to_string(),
            args: vec![rosc::OscType::Float(final_coherence as f32)],
        }));
    }

    fn send(&self, packet: &OscPacket) {
        let delay = {
            let inner = self.inner.lock().unwrap();
            if let Ok(bytes) = rosc::encoder::encode(packet) {
                let _ = inner.socket.send_to(&bytes, &inner.addr);
            }
            inner.delay_ms
        };
        if delay > 0 {
            std::thread::sleep(std::time::Duration::from_millis(delay));
        }
    }

    fn elapsed_secs(&self) -> f32 {
        self.inner.lock().unwrap().start_time.elapsed().as_secs_f32()
    }
}

impl Clone for OscHostProvider {
    fn clone(&self) -> Self {
        OscHostProvider {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl PhiHostProvider for OscHostProvider {
    fn get_coherence(&self, internal_coherence: f64) -> f64 {
        self.send(&OscPacket::Message(rosc::OscMessage {
            addr: "/phi/coherence".to_string(),
            args: vec![rosc::OscType::Float(internal_coherence as f32)],
        }));
        internal_coherence
    }

    fn on_resonate(&self, intention: &str, value: &str) {
        self.send(&OscPacket::Message(rosc::OscMessage {
            addr: "/phi/resonate".to_string(),
            args: vec![
                rosc::OscType::String(intention.to_string()),
                rosc::OscType::String(value.to_string()),
            ],
        }));
    }

    fn on_witness(&self, snapshot: &WitnessSnapshot) -> WitnessAction {
        let intention = snapshot
            .intention_stack
            .last()
            .cloned()
            .unwrap_or_else(|| "global".to_string());

        self.send(&OscPacket::Message(rosc::OscMessage {
            addr: "/phi/witness".to_string(),
            args: vec![
                rosc::OscType::Float(snapshot.coherence as f32),
                rosc::OscType::Float(self.elapsed_secs()),
                rosc::OscType::String(intention),
            ],
        }));

        WitnessAction::Continue
    }

    fn on_witness_sensor(&self, sensor: SensorKind) -> Option<f64> {
        let value = crate::sensors::read_sensor(sensor);
        self.send(&OscPacket::Message(rosc::OscMessage {
            addr: "/phi/sensor".to_string(),
            args: vec![
                rosc::OscType::String(sensor.as_name().to_string()),
                rosc::OscType::Float(value.unwrap_or(0.0) as f32),
            ],
        }));
        value
    }

    fn on_intention_push(&self, intention: &str) {
        let depth = {
            let mut inner = self.inner.lock().unwrap();
            inner.depth += 1;
            inner.depth
        };
        self.send(&OscPacket::Message(rosc::OscMessage {
            addr: "/phi/intention/push".to_string(),
            args: vec![
                rosc::OscType::String(intention.to_string()),
                rosc::OscType::Int(depth),
            ],
        }));
    }

    fn on_intention_pop(&self, intention: &str) {
        let depth = {
            let mut inner = self.inner.lock().unwrap();
            inner.depth = (inner.depth - 1).max(0);
            inner.depth
        };
        self.send(&OscPacket::Message(rosc::OscMessage {
            addr: "/phi/intention/pop".to_string(),
            args: vec![
                rosc::OscType::String(intention.to_string()),
                rosc::OscType::Int(depth),
            ],
        }));
    }

    fn broadcast(&self, channel: &str, message: &str) {
        self.send(&OscPacket::Message(rosc::OscMessage {
            addr: "/phi/broadcast".to_string(),
            args: vec![
                rosc::OscType::String(channel.to_string()),
                rosc::OscType::String(message.to_string()),
            ],
        }));
    }

    fn listen(&self, channel: &str) -> Option<String> {
        self.send(&OscPacket::Message(rosc::OscMessage {
            addr: "/phi/listen".to_string(),
            args: vec![rosc::OscType::String(channel.to_string())],
        }));
        None
    }

    fn emit_signal(&self, frequency: f64, intensity: f64) {
        self.send(&OscPacket::Message(rosc::OscMessage {
            addr: "/phi/signal".to_string(),
            args: vec![
                rosc::OscType::Float(frequency as f32),
                rosc::OscType::Float(intensity as f32),
            ],
        }));
    }

    fn on_entangle(&self, frequency: f64) {
        self.send(&OscPacket::Message(rosc::OscMessage {
            addr: "/phi/entangle".to_string(),
            args: vec![rosc::OscType::Float(frequency as f32)],
        }));
    }
}
