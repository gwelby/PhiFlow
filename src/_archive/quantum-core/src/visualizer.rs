use plotters::prelude::*;
use plotters_svg::SVGBackend;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Visualizer {
    width: u32,
    height: u32,
}

impl Visualizer {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    pub fn draw_field(&self, path: &Path, field_data: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
        let root = SVGBackend::new(path, (self.width, self.height)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption("Quantum Field Visualization", ("sans-serif", 40))
            .margin(10)
            .set_all_label_area_size(50)
            .build_cartesian_2d(
                0.0..field_data.len() as f64,
                0.0..1.0
            )?;

        chart.configure_mesh()
            .x_desc("Time")
            .y_desc("Field Strength")
            .axis_desc_style(("sans-serif", 20))
            .draw()?;

        // Draw field data
        let points: Vec<(f64, f64)> = field_data.iter()
            .enumerate()
            .map(|(i, &v)| (i as f64, v))
            .collect();

        chart.draw_series(LineSeries::new(
            points,
            &BLUE.mix(0.8),
        ))?;

        // Add sacred frequencies
        let sacred_freqs = [(432.0, "Ground"), (528.0, "Create"), (768.0, "Unity")];
        for (freq, label) in sacred_freqs.iter() {
            let normalized_freq = freq / 1000.0;
            chart.draw_series(PointSeries::of_element(
                vec![(field_data.len() as f64 / 2.0, normalized_freq)],
                5,
                &RED.mix(0.8),
                &|c, s, st| {
                    return EmptyElement::at(c)
                        + Circle::new((0, 0), s, st.filled())
                        + Text::new(label.to_string(), (10, 0), ("sans-serif", 15));
                },
            ))?;
        }

        root.present()?;
        Ok(())
    }
}
