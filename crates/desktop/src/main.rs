//! Native desktop application runner for the Serverless & Desktop Template.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use app::TemplateApp;
use eframe::NativeOptions;
use eframe::egui;
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

fn main() -> eframe::Result<()> {
    // Logging setup: INFO for application logs, WARN for external library modules
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,wgpu=warn,egui=warn")),
        )
        .init();

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
