//! Native Desktop Studio Executable for RainAI: Real-Time Neural & Physical Spatial Soundscape Synthesis.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use app::TemplateApp;
use eframe::NativeOptions;
use eframe::egui;
use spodeian_telemetry::init_default;

fn main() -> eframe::Result<()> {
    // Universal telemetry & logging initialization
    init_default();

    // Native window viewport configurations
    let options = NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("🌧 RainAI · Neural Spatial Soundscape Studio")
            .with_inner_size([1100.0, 750.0])
            .with_min_inner_size([700.0, 500.0])
            .with_active(true)
            .with_resizable(true),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };

    eframe::run_native(
        "🌧 RainAI · Neural Spatial Soundscape Studio",
        options,
        Box::new(|cc| Ok(Box::new(TemplateApp::new(cc)))),
    )
}
