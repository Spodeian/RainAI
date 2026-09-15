//! Real-time waterfall spectrogram component for frequency and energy visualization.

use egui::{Color32, Rect, Ui, Vec2};
use std::collections::VecDeque;

pub struct SpectrogramHistory {
    pub columns: VecDeque<[f32; 32]>,
    max_cols: usize,
}

impl Default for SpectrogramHistory {
    fn default() -> Self {
        Self {
            columns: VecDeque::with_capacity(256),
            max_cols: 256,
        }
    }
}

impl SpectrogramHistory {
    pub fn push_frame(&mut self, bins: [f32; 32]) {
        if self.columns.len() >= self.max_cols {
            self.columns.pop_front();
        }
        self.columns.push_back(bins);
    }
}

pub fn render_spectrogram_panel(ui: &mut Ui, history: &SpectrogramHistory) {
    egui::TopBottomPanel::bottom("spectrogram_panel")
        .min_height(130.0)
        .show_inside(ui, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Real-Time Neural Spectrogram / Waterfall");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label("Frequency Range: 20Hz – 20kHz | Resolution: 32 Sub-Bands");
                });
            });

            let available_size = ui.available_size();
            let (response, painter) = ui.allocate_painter(available_size, egui::Sense::hover());
            let rect = response.rect;

            // Deep background fill
            painter.rect_filled(rect, 4.0, Color32::from_rgb(15, 18, 24));

            if history.columns.is_empty() {
                painter.text(
                    rect.center(),
                    egui::Align2::CENTER_CENTER,
                    "Awaiting Audio Stream & Neural Inference...",
                    egui::FontId::proportional(14.0),
                    Color32::GRAY,
                );
                return;
            }

            let num_cols = history.columns.len() as f32;
            let col_width = rect.width() / num_cols.max(1.0);

            for (col_idx, col_bins) in history.columns.iter().enumerate() {
                let num_bins = col_bins.len() as f32;
                let row_height = rect.height() / num_bins;
                let x = rect.min.x + (col_idx as f32) * col_width;

                for (bin_idx, &energy) in col_bins.iter().enumerate() {
                    // Invert Y axis so low frequencies sit at the bottom
                    let y = rect.max.y - ((bin_idx as f32 + 1.0) * row_height);
                    let cell_rect = Rect::from_min_size(
                        egui::pos2(x, y),
                        Vec2::new(col_width + 0.5, row_height + 0.5),
                    );

                    let intensity = energy.clamp(0.0, 1.0);
                    let color = energy_to_thermal_color(intensity);
                    painter.rect_filled(cell_rect, 0.0, color);
                }
            }
        });
}

#[inline]
fn energy_to_thermal_color(val: f32) -> Color32 {
    // Thermal / Magma perceptual colormap approximation
    let r = (val * 255.0) as u8;
    let g = ((val * 1.6).clamp(0.0, 1.0) * 255.0) as u8;
    let b = ((1.0 - val) * 140.0) as u8;
    Color32::from_rgb(r, g, b)
}