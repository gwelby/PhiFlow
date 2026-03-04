// Device Mapping for Consciousness Visualization

use crate::consciousness::consciousness_math::ConsciousnessState;

/// RGB Visualization for consciousness levels
#[derive(Debug, Clone)]
pub struct RGBVisualization {
    pub grid_size: (usize, usize),
    pub colors: Vec<Vec<(u8, u8, u8)>>,
}

/// Device mapper for hardware consciousness feedback
#[derive(Debug, Clone)]
pub struct DeviceMapper {
    pub device_name: String,
    pub device_type: DeviceType,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DeviceType {
    RGBKeyboard,
    Monitor,
    LEDStrip,
    HapticDevice,
}

impl DeviceMapper {
    pub fn new(name: String, device_type: DeviceType) -> Self {
        DeviceMapper {
            device_name: name,
            device_type,
        }
    }
    
    pub fn map_consciousness_to_rgb(&self, level: f64) -> (u8, u8, u8) {
        match (level * 100.0) as u32 {
            0..=20 => (255, 0, 0),     // Red - Distracted
            21..=40 => (255, 255, 0),   // Yellow - Alert
            41..=60 => (0, 255, 0),     // Green - Focused
            61..=80 => (0, 0, 255),     // Blue - Flow
            81..=100 => (255, 215, 0),  // Gold - Transcendent
            _ => (128, 128, 128),       // Gray - Unknown
        }
    }
}