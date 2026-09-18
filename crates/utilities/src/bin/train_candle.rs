//! Master CLI Runner for Pure Rust Candle Model Training.

use anyhow::Result;
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;
use utilities::candle_train::{run_candle_training_pipeline, CandleTrainConfig, TrainingPhase};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into()))
        .init();

    let mut config = CandleTrainConfig::default();
    let mut is_autopilot = false;

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--autopilot" => {
                is_autopilot = true;
            }
            "--phases" => {
                config.phases.clear();
                while i + 1 < args.len() && !args[i + 1].starts_with("--") {
                    match args[i + 1].to_lowercase().as_str() {
                        "vae" => config.phases.push(TrainingPhase::Vae),
                        "mamba" => config.phases.push(TrainingPhase::Mamba),
                        "export" => config.phases.push(TrainingPhase::Export),
                        "all" => config.phases.push(TrainingPhase::All),
                        _ => {}
                    }
                    i += 1;
                }
                if config.phases.is_empty() {
                    config.phases.push(TrainingPhase::All);
                }
            }
            "--vae-epochs" => {
                if i + 1 < args.len() {
                    config.vae_epochs = args[i + 1].parse().unwrap_or(config.vae_epochs);
                    i += 1;
                }
            }
            "--mamba-epochs" => {
                if i + 1 < args.len() {
                    config.mamba_epochs = args[i + 1].parse().unwrap_or(config.mamba_epochs);
                    i += 1;
                }
            }
            "--epochs" => {
                if i + 1 < args.len() {
                    let eps: usize = args[i + 1].parse().unwrap_or(config.vae_epochs);
                    config.vae_epochs = eps;
                    config.mamba_epochs = eps;
                    i += 1;
                }
            }
            "--batch-size" => {
                if i + 1 < args.len() {
                    config.batch_size = args[i + 1].parse().unwrap_or(config.batch_size);
                    i += 1;
                }
            }
            "--max-batches" => {
                if i + 1 < args.len() {
                    config.max_batches = args[i + 1].parse().unwrap_or(config.max_batches);
                    i += 1;
                }
            }
            "--lr" => {
                if i + 1 < args.len() {
                    config.learning_rate = args[i + 1].parse().unwrap_or(config.learning_rate);
                    i += 1;
                }
            }
            "--accumulation-steps" => {
                if i + 1 < args.len() {
                    config.accumulation_steps = args[i + 1].parse().unwrap_or(config.accumulation_steps);
                    i += 1;
                }
            }
            "--cfg-dropout" => {
                if i + 1 < args.len() {
                    config.cfg_dropout = args[i + 1].parse().unwrap_or(config.cfg_dropout);
                    i += 1;
                }
            }
            "--output-dir" => {
                if i + 1 < args.len() {
                    config.output_dir = PathBuf::from(&args[i + 1]);
                    i += 1;
                }
            }
            "--use-real-data" => {
                config.use_real_data = true;
            }
            "--no-real-data" => {
                config.use_real_data = false;
            }
            "--use-flow-matching" => {
                config.use_flow_matching = true;
            }
            "--tau-cov" => {
                if i + 1 < args.len() {
                    config.tau_cov = args[i + 1].parse().unwrap_or(config.tau_cov);
                    i += 1;
                }
            }
            "--thinking-steps" | "--max-thinking-steps" => {
                if i + 1 < args.len() {
                    config.max_thinking_steps = args[i + 1].parse().unwrap_or(config.max_thinking_steps);
                    i += 1;
                }
            }
            "--eps-thinking-halt" => {
                if i + 1 < args.len() {
                    config.eps_thinking_halt = args[i + 1].parse().unwrap_or(config.eps_thinking_halt);
                    i += 1;
                }
            }
            "--max-grad-norm" => {
                if i + 1 < args.len() {
                    config.max_grad_norm = args[i + 1].parse().unwrap_or(config.max_grad_norm);
                    i += 1;
                }
            }
            "--warmup-steps" => {
                if i + 1 < args.len() {
                    config.warmup_steps = args[i + 1].parse().unwrap_or(config.warmup_steps);
                    i += 1;
                }
            }
            "--val-ratio" => {
                if i + 1 < args.len() {
                    config.val_ratio = args[i + 1].parse().unwrap_or(config.val_ratio);
                    i += 1;
                }
            }
            "--stft-mode" => {
                if i + 1 < args.len() {
                    match args[i + 1].to_lowercase().as_str() {
                        "envelope" | "envelope16" => config.stft_mode = utilities::stft_loss::StftLossMode::Envelope16,
                        "waveform" | "waveformfoa" => config.stft_mode = utilities::stft_loss::StftLossMode::WaveformFoa,
                        "mel" | "melspectral" => config.stft_mode = utilities::stft_loss::StftLossMode::MelSpectral,
                        "combined" => config.stft_mode = utilities::stft_loss::StftLossMode::Combined,
                        _ => {}
                    }
                    i += 1;
                }
            }
            "--stft-weight" => {
                if i + 1 < args.len() {
                    config.stft_weight = args[i + 1].parse().unwrap_or(config.stft_weight);
                    i += 1;
                }
            }
            "--lambda-vel" => {
                if i + 1 < args.len() {
                    config.lambda_vel = args[i + 1].parse().unwrap_or(config.lambda_vel);
                    i += 1;
                }
            }
            "--lambda-acc" => {
                if i + 1 < args.len() {
                    config.lambda_acc = args[i + 1].parse().unwrap_or(config.lambda_acc);
                    i += 1;
                }
            }
            "--lambda-drag" => {
                if i + 1 < args.len() {
                    config.lambda_drag = args[i + 1].parse().unwrap_or(config.lambda_drag);
                    i += 1;
                }
            }
            "--lambda-z" => {
                if i + 1 < args.len() {
                    config.lambda_z = args[i + 1].parse().unwrap_or(config.lambda_z);
                    i += 1;
                }
            }
            "--lambda-doa" => {
                if i + 1 < args.len() {
                    config.lambda_doa = args[i + 1].parse().unwrap_or(config.lambda_doa);
                    i += 1;
                }
            }
            "--lambda-diff" => {
                if i + 1 < args.len() {
                    config.lambda_diff = args[i + 1].parse().unwrap_or(config.lambda_diff);
                    i += 1;
                }
            }
            "--lambda-straight" => {
                if i + 1 < args.len() {
                    config.lambda_straight = args[i + 1].parse().unwrap_or(config.lambda_straight);
                    i += 1;
                }
            }
            "--lambda-div" => {
                if i + 1 < args.len() {
                    config.lambda_div = args[i + 1].parse().unwrap_or(config.lambda_div);
                    i += 1;
                }
            }
            "--gamma-tabu" => {
                if i + 1 < args.len() {
                    config.gamma_tabu = args[i + 1].parse().unwrap_or(config.gamma_tabu);
                    i += 1;
                }
            }
            "--enable-distillation" => {
                config.enable_distillation = true;
            }
            "--no-distillation" => {
                config.enable_distillation = false;
            }
            "--lambda-distill" => {
                if i + 1 < args.len() {
                    config.lambda_distill = args[i + 1].parse().unwrap_or(config.lambda_distill);
                    i += 1;
                }
            }
            "--so3-aug-prob" => {
                if i + 1 < args.len() {
                    config.so3_aug_prob = args[i + 1].parse().unwrap_or(config.so3_aug_prob);
                    i += 1;
                }
            }
            "--surface-mixup-prob" => {
                if i + 1 < args.len() {
                    config.surface_mixup_prob = args[i + 1].parse().unwrap_or(config.surface_mixup_prob);
                    i += 1;
                }
            }
            "--thinking-curriculum" => {
                config.thinking_curriculum = true;
            }
            "--no-thinking-curriculum" => {
                config.thinking_curriculum = false;
            }
            "--stochastic-jitter-sigma" => {
                if i + 1 < args.len() {
                    config.stochastic_jitter_sigma = args[i + 1].parse().unwrap_or(config.stochastic_jitter_sigma);
                    i += 1;
                }
            }
            "--device" => {
                if i + 1 < args.len() {
                    config.device = args[i + 1].clone();
                    i += 1;
                }
            }
            "--help" | "-h" => {
                println!("RainAI Pure Rust Candle Training CLI");
                println!("Usage: rainai_train_candle [OPTIONS]");
                println!();
                println!("Options:");
                println!("  --autopilot                        Run autonomous self-driving training mission");
                println!("  --phases <vae|mamba|export|all...> Training phases to execute");
                println!("  --epochs <N>                       Sets both VAE and Mamba epochs");
                println!("  --batch-size <N>                   Batch size (default: 4)");
                println!("  --max-batches <N>                  Max batches per epoch (default: 10)");
                println!("  --device <cpu|cuda|metal|auto>     Compute accelerator device (default: auto)");
                println!("  --lr <float>                       Base learning rate (default: 1e-3)");
                println!("  --use-real-data / --no-real-data   Use real manifest data vs synthetic");
                println!("  --stft-mode <combined|mel|...>     STFT loss function mode");
                println!("  --output-dir <path>                Directory to save trained weights");
                return Ok(());
            }
            _ => {}
        }
        i += 1;
    }

    if is_autopilot {
        let mut trainer = utilities::autopilot::AutoPilotTrainer::new(None);
        trainer.run_mission()?;
    } else {
        run_candle_training_pipeline(&config)?;
    }

    Ok(())
}
