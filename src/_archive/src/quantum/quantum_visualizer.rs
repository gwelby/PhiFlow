use plotters::prelude::*;
use anyhow::Result;
use std::path::PathBuf;
use super::quantum_constants::{
    GROUND_FREQUENCY,
    CREATE_FREQUENCY,
    HEART_FREQUENCY,
    VOICE_FREQUENCY,
    VISION_FREQUENCY,
    UNITY_FREQUENCY,
    PHI,
    PHI_SQUARED,
    PHI_CUBED,
};

pub struct QuantumFieldVisualizer {
    output_path: PathBuf,
    width: u32,
    height: u32,
}

impl QuantumFieldVisualizer {
    pub fn new(output_path: PathBuf, width: u32, height: u32) -> Self {
        Self {
            output_path,
            width,
            height,
        }
    }

    pub async fn draw_quantum_field(&self) -> Result<()> {
        let root = BitMapBackend::new(&self.output_path, (self.width, self.height))
            .into_drawing_area();
        root.fill(&WHITE)?;

        // Sacred frequencies
        let frequencies = vec![
            GROUND_FREQUENCY,
            CREATE_FREQUENCY,
            HEART_FREQUENCY,
            VOICE_FREQUENCY,
            VISION_FREQUENCY,
            UNITY_FREQUENCY,
        ];

        // Phi harmonics
        let amplitudes = vec![
            1.0,
            PHI,
            PHI_SQUARED,
            PHI_CUBED,
            PHI_SQUARED * PHI,
            PHI_CUBED * PHI,
        ];

        let max_freq = frequencies.iter().copied().fold(0./0., f64::max);
        let max_amp = amplitudes.iter().copied().fold(0./0., f64::max);

        let mut chart = ChartBuilder::on(&root)
            .caption("Sacred Frequency Spectrum (432-768 Hz)", ("sans-serif", 30))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(
                400f64..800f64,
                0f64..max_amp * 1.1,
            )?;

        chart.configure_mesh()
            .x_desc("Frequency (Hz)")
            .y_desc("Amplitude (φ)")
            .draw()?;

        // Draw sacred geometry patterns
        let patterns = [
            ("Ground", GROUND_FREQUENCY, &BLUE),
            ("Create", CREATE_FREQUENCY, &RED),
            ("Heart", HEART_FREQUENCY, &GREEN),
            ("Voice", VOICE_FREQUENCY, &MAGENTA),
            ("Vision", VISION_FREQUENCY, &CYAN),
            ("Unity", UNITY_FREQUENCY, &BLACK),
        ];

        for (name, freq, color) in patterns.iter() {
            let amp = PHI.powf((freq - GROUND_FREQUENCY) / 100.0);
            
            chart.draw_series(std::iter::once(
                Circle::new((*freq, amp), 5, color.clone().filled())
            ))?.label(name)
             .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color.clone()));
        }

        chart.configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;

        Ok(())
    }

    pub fn plot_frequencies(&self, frequencies: &[f64], amplitudes: &[f64]) -> Result<()> {
        let root = BitMapBackend::new(&self.output_path, (self.width, self.height))
            .into_drawing_area();
        root.fill(&WHITE)?;

        let max_freq = frequencies.iter().copied().fold(0./0., f64::max);
        let max_amp = amplitudes.iter().copied().fold(0./0., f64::max);

        let mut chart = ChartBuilder::on(&root)
            .caption("Quantum Frequency Spectrum", ("sans-serif", 30))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(
                0f64..max_freq * 1.1,
                0f64..max_amp * 1.1,
            )?;

        chart.configure_mesh()
            .x_desc("Frequency (Hz)")
            .y_desc("Amplitude")
            .draw()?;

        chart.draw_series(
            frequencies.iter().zip(amplitudes.iter()).map(|(&x, &y)| {
                Circle::new((x, y), 3, &BLUE)
            }),
        )?;

        Ok(())
    }
}
