use phiflow::resonance_bus::{ResonanceEvent, MqttConfig};
use rumqttc::{Client, MqttOptions, QoS};
use std::time::Duration;
use serde_json::json;

#[test]
fn test_mqtt_daemon_integration() {
    // This is primarily a smoke test to ensure that rumqttc dependencies link
    // and the MqttOptions instantiate correctly. Actually running a full broker
    // requires a live MQTT instance which we don't assume in pure cargo tests.
    
    let config = MqttConfig::default();
    
    let mut mqttoptions = MqttOptions::new("phiflow-test-client", &config.host, config.port);
    mqttoptions.set_keep_alive(Duration::from_secs(5));

    // For a real test, we would expect a broker. We'll simply construct it 
    // to ensure there are no compilation errors in the API usage.
    let _client_result = Client::new(mqttoptions, 10);
    
    // Construct an evolve payload we would send
    let event = ResonanceEvent {
        event_type: "evolve".to_string(),
        value: json!("let test = 42"),
        intention: "test".to_string(),
        ts: "time".to_string(),
        source: "test".to_string(),
        id: "123".to_string(),
    };
    
    let json_line = serde_json::to_string(&event).unwrap();
    assert!(json_line.contains("evolve"));
}
