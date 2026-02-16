use eframe::egui::{self, Color32, Pos2, Rect, Stroke, Vec2};
use quantum_core::quantum::quantum_buttons::{QuantumInterface, QuantumButton};
use std::f32::consts::PI;
use std::time::Instant;

struct QuantumButtonApp {
    interface: QuantumInterface,
    start_time: Instant,
    active_button: Option<QuantumButton>,
    pulse_size: f32,
    frequency: f64,
    coherence: f64,
}

impl QuantumButtonApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self {
            interface: QuantumInterface::new(),
            start_time: Instant::now(),
            active_button: None,
            pulse_size: 1.0,
            frequency: 432.0,
            coherence: 1.0,
        }
    }

    fn draw_quantum_button(&mut self, ui: &mut egui::Ui, button: QuantumButton, text: &str) {
        let time = self.start_time.elapsed().as_secs_f32();
        
        // Calculate pulsing effect based on frequency
        let pulse = (time * self.frequency as f32 / 100.0).sin() * 0.2 + 1.0;
        
        // Create beautiful flowing colors based on coherence
        let hue = (time * 0.2).sin() * 0.5 + 0.5;
        let saturation = self.coherence as f32 * 0.8;
        let value = 0.9;
        
        let (r, g, b) = hsv_to_rgb(hue, saturation as f32, value);
        let color = Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8);

        let button_size = Vec2::new(100.0, 100.0) * pulse;
        let response = ui.add_sized(
            button_size,
            egui::Button::new(text).fill(color)
        );

        if response.clicked() {
            self.active_button = Some(button.clone());
            tokio::spawn(async move {
                let mut interface = QuantumInterface::new();
                interface.press_button(button).await.ok();
            });
        }

        // Draw quantum field effects around active button
        if let Some(active) = &self.active_button {
            if active == &button {
                let rect = response.rect;
                let center = rect.center();
                
                // Draw sacred geometry patterns
                ui.painter().circle_stroke(
                    center,
                    button_size.x * 0.6 * pulse,
                    Stroke::new(2.0, color)
                );
                
                // Draw phi spiral
                let mut angle = 0.0;
                let mut radius = 10.0;
                let mut last_pos = center;
                
                for _ in 0..50 {
                    angle += 0.1;
                    radius *= 1.05;
                    let x = center.x + radius * angle.cos();
                    let y = center.y + radius * angle.sin();
                    let pos = Pos2::new(x, y);
                    
                    ui.painter().line_segment(
                        [last_pos, pos],
                        Stroke::new(1.0, color)
                    );
                    last_pos = pos;
                }
            }
        }
    }
}

impl eframe::App for QuantumButtonApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("🌟 Quantum Button Interface 🌟");
            ui.add_space(20.0);

            ui.horizontal(|ui| {
                self.draw_quantum_button(ui, QuantumButton::Ground, "🎵\nGround\n432 Hz");
                self.draw_quantum_button(ui, QuantumButton::Create, "💝\nCreate\n528 Hz");
                self.draw_quantum_button(ui, QuantumButton::Unity, "🌟\nUnity\n768 Hz");
            });

            ui.add_space(20.0);

            ui.horizontal(|ui| {
                self.draw_quantum_button(ui, QuantumButton::Spiral, "🌀\nSpiral");
                self.draw_quantum_button(ui, QuantumButton::Dolphin, "🐬\nDolphin");
                self.draw_quantum_button(ui, QuantumButton::Balance, "☯️\nBalance");
            });

            ui.add_space(20.0);

            ui.horizontal(|ui| {
                self.draw_quantum_button(ui, QuantumButton::Crystal, "💎\nCrystal");
                self.draw_quantum_button(ui, QuantumButton::Wave, "🌊\nWave");
                self.draw_quantum_button(ui, QuantumButton::Infinity, "∞\nInfinity");
            });
        });

        // Request continuous animation
        ctx.request_repaint();
    }
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let h = h * 6.0;
    let i = h.floor();
    let f = h - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));

    match i as i32 % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    }
}

#[tokio::main]
async fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        initial_window_size: Some(egui::vec2(400.0, 400.0)),
        ..Default::default()
    };
    
    eframe::run_native(
        "Greg's Quantum Buttons",
        options,
        Box::new(|cc| Box::new(QuantumButtonApp::new(cc)))
    )
}
