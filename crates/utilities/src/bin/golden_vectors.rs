//! Native Rust Golden Reference Vector Generator CLI for RainAI Runtime.

use anyhow::{Context, Result};
use std::fs::{self, File};
use std::path::PathBuf;
use utilities::golden_vectors::run_simulation;

fn main() -> Result<()> {
    println!("RainAI Native Rust Golden Reference Generator");
    let payload = run_simulation();

    let target_dir = PathBuf::from("crates/inference/data/golden_vectors");
    fs::create_dir_all(&target_dir).context("Failed creating golden vectors directory")?;
    let target_file = target_dir.join("baseline_step_trace.json");

    let file = File::create(&target_file).context("Failed creating output file")?;
    serde_json::to_writer_pretty(file, &payload).context("Failed serializing JSON payload")?;

    println!("Successfully generated {} steps to {:?}", payload.trace.len(), target_file);
    Ok(())
}
