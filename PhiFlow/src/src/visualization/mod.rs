use crate::phi_core::*;
use std::io::Write; // Import Write trait for file operations

pub struct Visualizer {
    width: f64,
    height: f64,
    scale: f64,
    center: Point2D,
}

impl Visualizer {
    pub fn new(width: f64, height: f64) -> Self {
        Visualizer {
            width,
            height,
            scale: 1.0,
            center: [width / 2.0, height / 2.0],
        }
    }

    // Convert 2D pattern to SVG with phi-harmonic coloring
    pub fn pattern_to_svg(&self, pattern: &[Point2D], frequency: f64) -> String {
        let color = self.frequency_to_color(frequency);
        let mut svg = format!(
            r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">
            <rect width="100%" height="100%" fill="black"/>
            <g transform="translate({}, {})">"#,
            self.width, self.height, self.center[0], self.center[1]
        );

        // Draw pattern as connected path
        if !pattern.is_empty() {
            let mut path_data = format!("M {} {} ", pattern[0][0] * self.scale, -pattern[0][1] * self.scale);
            for point in &pattern[1..] {
                path_data.push_str(&format!("L {} {} ", point[0] * self.scale, -point[1] * self.scale));
            }

            svg.push_str(&format!(
                r#"<path d="{}" stroke="{}" stroke-width="2" fill="none" opacity="0.8"/>"#,
                path_data, color
            ));
        }

        svg.push_str("</g></svg>");
        svg
    }

    // Convert 3D pattern to 2D projection
    pub fn project_3d_to_2d(&self, pattern: &[Point3D]) -> Vec<Point2D> {
        pattern.iter().map(|p| {
            // Simple orthographic projection with phi-based perspective
            let perspective_factor = 1.0 / (1.0 + p[2] / (100.0 * PHI));
            [
                p[0] * perspective_factor,
                p[1] * perspective_factor
            ]
        }).collect()
    }

    // Map frequency to color using consciousness zones
    fn frequency_to_color(&self, frequency: f64) -> &'static str {
        match frequency as i32 {
            0..=450 => "#FF0000",      // Red - Ground State
            451..=550 => "#FF8C00",    // Orange - Creation State
            551..=650 => "#00FF00",    // Green - Heart Field
            651..=750 => "#00CED1",    // Turquoise - Voice Flow
            751..=850 => "#0000FF",    // Blue - Vision Gate
            851..=950 => "#8A2BE2",    // Violet - Unity Wave
            _ => "#FFFFFF",            // White - Source Field
        }
    }

    // Create animated SVG for time-evolving patterns
    pub fn animated_pattern_svg(&self, pattern_sequence: &[Vec<Point2D>], frequency: f64, duration: f64) -> String {
        let color = self.frequency_to_color(frequency);
        let mut svg = format!(
            r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">
            <rect width="100%" height="100%" fill="black"/>
            <g transform="translate({}, {})">"#,
            self.width, self.height, self.center[0], self.center[1]
        );

        // Create path with animation
        svg.push_str(&format!(
            r#"<path stroke="{}" stroke-width="2" fill="none" opacity="0.8">"#,
            color
        ));

        // Animate through pattern sequence
        let mut values_attr = String::new();
        for pattern in pattern_sequence {
            let mut path_data = String::new();
            if !pattern.is_empty() {
                path_data.push_str(&format!("M {} {} ", pattern[0][0] * self.scale, -pattern[0][1] * self.scale));
                for point in &pattern[1..] {
                    path_data.push_str(&format!("L {} {} ", point[0] * self.scale, -point[1] * self.scale));
                }
            }
            values_attr.push_str(&format!("{};", path_data));
        }

        svg.push_str(&format!(
            r#"<animate attributeName="d" dur="{}s" repeatCount="indefinite" values="{}"/>"#,
            duration, values_attr
        ));

        svg.push_str("</path></g></svg>");
        svg
    }
}

// Helper function to save SVG to file
pub fn save_svg(svg_content: &str, filename: &str) -> std::io::Result<()> {
    use std::fs::File;
    let mut file = File::create(filename)?;
    file.write_all(svg_content.as_bytes())?;
    Ok(())
}