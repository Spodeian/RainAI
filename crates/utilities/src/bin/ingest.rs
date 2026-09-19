use anyhow::Result;
use std::sync::{atomic::AtomicBool, Arc};
use tracing::info;
use utilities::ingest::run_ingestion_pipeline_async;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    info!("Starting Asynchronous Multi-Source Open Audio Ingest Engine v4.0 (Pure-Rust)...");

    let stop_signal = Arc::new(AtomicBool::new(false));
    let downloaded = run_ingestion_pipeline_async(stop_signal, None).await?;
    info!("Ingest pipeline completed: processed/verified {} audio assets.", downloaded);
    Ok(())
}