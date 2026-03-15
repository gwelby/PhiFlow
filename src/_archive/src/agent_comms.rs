use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub frequency: f64,
    pub content: String,
}

pub struct AgentComms {
    channels: HashMap<String, mpsc::Sender<Message>>,
}

impl AgentComms {
    pub fn new() -> Self {
        AgentComms {
            channels: HashMap::new(),
        }
    }

    pub async fn send_message(&self, target: &str, msg: Message) -> anyhow::Result<()> {
        if let Some(tx) = self.channels.get(target) {
            tx.send(msg).await?;
        }
        Ok(())
    }

    pub fn add_channel(&mut self, id: String, tx: mpsc::Sender<Message>) {
        self.channels.insert(id, tx);
    }
}
