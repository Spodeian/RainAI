use anyhow::Result;
use hound::{SampleFormat, WavSpec, WavWriter};
use rayon::prelude::*;
use std::fs;
use std::path::Path;
use tracing::info;
use utilities::synth_rain::{generate_rain_texture, DEFAULT_SAMPLE_RATE};

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    info!("Running Parallel Native Physical Rain Synthesizer...");

    let out_dir = Path::new("Data/rain/Synthetic");
    fs::create_dir_all(out_dir)?;

    let configs = vec![
        ("gentle_drizzle", 1.5, "pavement"),
        ("steady_rain", 12.0, "pavement"),
        ("heavy_downpour", 45.0, "pavement"),
        ("urban_pavement", 15.0, "pavement"),
        ("window_rain", 8.0, "glass"),
        ("roof_rain", 25.0, "tin"),
        ("canvas_tent", 12.0, "canvas"),
        ("wood_deck", 18.0, "wood"),
        ("pine_needles", 8.0, "pine"),
        ("forest_foliage", 10.0, "foliage"),
        ("water_deep", 25.0, "water"),
        ("puddle_shallow", 15.0, "puddle"),
        ("compound_urban_balcony", 20.0, "tin"),
        ("compound_forest_camp", 15.0, "canvas"),
        ("compound_porch_storm", 35.0, "wood"),
        ("thunderstorm", 50.0, "pavement"),
    ];

    configs.into_par_iter().for_each(|(name, rate, surf)| {
        let file_path = out_dir.join(format!("synth_{}.wav", name));
        info!("Synthesizing {} -> {:?}", name, file_path);
        let stereo = generate_rain_texture(15.0, rate, surf, DEFAULT_SAMPLE_RATE);

        let spec = WavSpec {
            channels: 2,
            sample_rate: DEFAULT_SAMPLE_RATE,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };

        if let Ok(mut writer) = WavWriter::create(&file_path, spec) {
            for i in 0..stereo[0].len() {
                let _ = writer.write_sample(stereo[0][i]);
                let _ = writer.write_sample(stereo[1][i]);
            }
            let _ = writer.finalize();
        }
    });

    info!("Synthetic generation complete!");
    Ok(())
}