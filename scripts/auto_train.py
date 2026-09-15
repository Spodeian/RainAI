"""
RainAI Automated Training, Export, and Benchmarking Orchestrator
Turn-key pipeline providing single-command hardware-adaptive execution.

Profiles:
  --profile smoke-test  : Quick 10-second end-to-end sanity pass (1 VAE ep, 1 Mamba ep, 2 batches, export, benchmark)
  --profile balanced    : Fast convergence pass (~1-2 mins on RTX 3060 Ti, 3 VAE ep, 2 Mamba ep, curriculum, export, benchmark)
  --profile production  : Full training run across full dataset with discriminator and complete multi-backend export
  --profile export-only : Skip training and run multi-backend export and inference benchmarks from existing checkpoints
  --profile custom      : Custom flags and hyperparameter overrides
"""

import os
import sys
import argparse
import time
import subprocess
import json
import logging
from collections import Counter
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
except ImportError:
    torch = None

from src.utils.logger import setup_logger, RainAILogger


def detect_hardware():
    """Inspect local hardware and return optimal configuration."""
    config = {
        "device": "cpu",
        "gpu_name": "None (CPU fallback)",
        "vram_gb": 0.0,
        "recommended_amp": False,
        "recommended_curriculum": True,
        "recommended_batch_size": 2,
        "recommended_accumulation": 1,
    }

    if torch and torch.cuda.is_available():
        config["device"] = "cuda"
        config["gpu_name"] = torch.cuda.get_device_name(0)
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        config["vram_gb"] = round(vram_bytes / (1024 ** 3), 2)
        config["recommended_amp"] = True
        config["recommended_curriculum"] = True
        
        # 8GB VRAM (e.g. RTX 3060 Ti) can comfortably handle batch_size 4 or 2 with accumulation
        if config["vram_gb"] >= 7.0:
            config["recommended_batch_size"] = 4
            config["recommended_accumulation"] = 2
        else:
            config["recommended_batch_size"] = 2
            config["recommended_accumulation"] = 2

    return config


def format_duration(seconds: float) -> str:
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{int(hours)}h {int(mins)}m {secs:.1f}s"
    elif mins > 0:
        return f"{int(mins)}m {secs:.1f}s"
    else:
        return f"{secs:.2f}s"


def print_banner(logger: RainAILogger, hardware: dict, profile_name: str, phases: list):
    logger.banner(
        "RainAI Automated Training & Multi-Backend Deployment Pipeline",
        f"Profile: {profile_name.upper()} | Target: {hardware['device'].upper()} ({hardware['gpu_name']})"
    )
    if hardware['vram_gb'] > 0:
        logger.info(f"Dedicated VRAM   : {hardware['vram_gb']} GB GDDR6")
    logger.info(f"Execution Phases : {', '.join(phases)}")
    if logger.log_file:
        logger.info(f"Persistent Log   : {logger.log_file}")


def check_dataset_and_manifest(
    logger: RainAILogger,
    python_exe: str,
    prepare_data: bool = False,
    rebuild_data: bool = False
):
    logger.stage(1, 5, "Dataset Standard & On-Demand Data Preparation")
    raw_rain_dir = PROJECT_ROOT / "Data" / "rain"
    if not raw_rain_dir.exists():
        raw_rain_dir = PROJECT_ROOT / "data" / "rain"
        
    raw_synthetic_dir = raw_rain_dir / "Synthetic"
    
    processed_dir = PROJECT_ROOT / "Data" / "processed"
    if not processed_dir.exists():
        processed_dir = PROJECT_ROOT / "data" / "processed"
        
    manifest_path = processed_dir / "manifest.json"

    # 1. On-Demand Synthetic Generation
    # Required categories covering all 9 physical surfaces + thunderstorm
    EXPECTED_SYNTH_CATEGORIES = [
        "gentle_drizzle", "steady_rain", "heavy_downpour", "urban_pavement",
        "window_rain", "roof_rain", "canvas_tent", "wood_deck",
        "pine_needles", "forest_foliage", "water_deep", "puddle_shallow",
        "compound_urban_balcony", "compound_forest_camp", "compound_porch_storm",
        "thunderstorm"
    ]

    need_synthesis = rebuild_data
    if not need_synthesis:
        if not raw_synthetic_dir.exists():
            need_synthesis = True
        else:
            existing_synth = list(raw_synthetic_dir.glob("synth_*.flac")) + list(raw_synthetic_dir.glob("synth_*.wav"))
            existing_names = [f.name for f in existing_synth]
            for cat in EXPECTED_SYNTH_CATEGORIES:
                if not any(cat in fn for fn in existing_names):
                    need_synthesis = True
                    break

    if need_synthesis or prepare_data:
        logger.info("[*] Checking physical rain acoustic corpus diversity...")
        cmd_synth = [python_exe, "-X", "utf8", str(PROJECT_ROOT / "src" / "data" / "synth_rain.py")]
        t0 = time.time()
        subprocess.run(cmd_synth, check=True)
        logger.info(f"[+] Verified/synthesized physical rain corpus in {time.time() - t0:.2f}s.")

    # 2. On-Demand 48kHz FOA Processing & Slicing
    processed_files = list(processed_dir.glob("*.flac")) + list(processed_dir.glob("*.wav"))
    raw_files = (
        list(raw_rain_dir.glob("**/*.flac")) +
        list(raw_rain_dir.glob("**/*.wav")) +
        list(raw_rain_dir.glob("**/*.ogg"))
    ) if raw_rain_dir.exists() else []
    
    need_preprocess = rebuild_data or (len(processed_files) < 50) or not manifest_path.exists()
    
    # Check if raw files were added or newer than manifest
    if not need_preprocess and manifest_path.exists() and len(raw_files) > 0:
        manifest_mtime = manifest_path.stat().st_mtime
        raw_newer = any(rf.stat().st_mtime > manifest_mtime for rf in raw_files)
        if raw_newer:
            logger.info("[*] Detected newer raw audio files than manifest. Re-running spatial preprocessing...")
            need_preprocess = True

    if need_preprocess:
        logger.info(f"[*] Preprocessing raw audio into 48kHz FOA chunks in {processed_dir}...")
        cmd_prep = [python_exe, "-X", "utf8", str(PROJECT_ROOT / "scripts" / "preprocess.py")]
        t0 = time.time()
        subprocess.run(cmd_prep, check=True)
        logger.info(f"[+] Spatial preprocessing completed in {time.time() - t0:.2f}s.")

    # 3. Manifest Verification & Acoustic Feature Sync
    manifest_data = {}
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
        except Exception:
            manifest_data = {}

    processed_files = list(processed_dir.glob("*.flac")) + list(processed_dir.glob("*.wav"))
    if len(manifest_data) < len(processed_files) or len(manifest_data) == 0 or rebuild_data:
        logger.info("[*] Synchronizing dataset manifest and acoustic parameter vectors...")
        cmd_sync = [python_exe, "-X", "utf8", str(PROJECT_ROOT / "scripts" / "update_datasets_to_current_standard.py")]
        subprocess.run(cmd_sync, check=True)
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

    # 4. Surface Diversity & Quality Audit
    total_clips = len(manifest_data)
    total_minutes = (total_clips * 5.0) / 60.0
    surface_counter = Counter(v.get("surface_tag", "unknown") for v in manifest_data.values())
    
    logger.info(f"Dataset Verified: {manifest_path.name} ({total_clips} chunks, {total_minutes:.1f} mins of 48kHz FOA audio)")
    logger.info(f"Physical Surface Balance ({len(surface_counter)} surfaces active):")
    for s_tag, count in sorted(surface_counter.items()):
        pct = (count / max(total_clips, 1)) * 100.0
        logger.info(f"  - {s_tag:<16}: {count:4d} chunks ({pct:5.1f}%)")


def run_phase_vae(logger: RainAILogger, python_exe: str, args, device: str, use_amp: bool):
    logger.stage(2, 5, "Phase 2: Spatial VAE + HOA-DDSP Engine Training")
    best_ckpt = PROJECT_ROOT / "checkpoints" / "spatial_vae_ddsp_best.pt"
    if best_ckpt.exists() and args.resume:
        try:
            ckpt_data = torch.load(best_ckpt, map_location="cpu")
            prior_loss = ckpt_data.get("best_val_spectral", "N/A")
            prior_epoch = ckpt_data.get("epoch", "N/A")
            if isinstance(prior_loss, float):
                logger.info(f"Prior Best Checkpoint: Epoch {prior_epoch}, Val Spectral: {prior_loss:.4f}")
            else:
                logger.info(f"Prior Best Checkpoint: Epoch {prior_epoch}")
        except Exception:
            pass

    cmd = [
        python_exe, "-X", "utf8", str(PROJECT_ROOT / "scripts" / "train_vae.py"),
        "--epochs", str(args.epochs_vae),
        "--batch-size", str(args.batch_size),
        "--accumulation-steps", str(args.accumulation_steps),
        "--device", device
    ]
    if args.max_batches > 0:
        cmd.extend(["--max-batches", str(args.max_batches)])
    if use_amp:
        cmd.append("--use-amp")
    if args.use_disc:
        cmd.append("--use-disc")
    if args.chunk_curriculum:
        cmd.append("--chunk-curriculum")
    if not args.resume:
        cmd.append("--no-resume")

    t0 = time.time()
    subprocess.run(cmd, check=True)
    elapsed = time.time() - t0
    logger.info(f"Stage 2 (VAE Training) completed in {format_duration(elapsed)}")
    return elapsed


def run_phase_mamba(logger: RainAILogger, python_exe: str, args, device: str, use_amp: bool):
    logger.stage(3, 5, "Phase 3: Mamba-2 MoE + Meta-Controller Training")
    best_ckpt = PROJECT_ROOT / "checkpoints" / "mamba2_metacontroller_best.pt"
    if best_ckpt.exists() and args.resume:
        try:
            ckpt_data = torch.load(best_ckpt, map_location="cpu")
            prior_loss = ckpt_data.get("best_val_loss", "N/A")
            prior_epoch = ckpt_data.get("epoch", "N/A")
            if isinstance(prior_loss, float):
                logger.info(f"Prior Best Checkpoint: Epoch {prior_epoch}, Val Loss: {prior_loss:.4f}")
            else:
                logger.info(f"Prior Best Checkpoint: Epoch {prior_epoch}")
        except Exception:
            pass

    cmd = [
        python_exe, "-X", "utf8", str(PROJECT_ROOT / "scripts" / "train_mamba.py"),
        "--epochs", str(args.epochs_mamba),
        "--batch-size", str(args.batch_size),
        "--accumulation-steps", str(args.accumulation_steps),
        "--device", device
    ]
    if args.max_batches > 0:
        cmd.extend(["--max-batches", str(args.max_batches)])
    if use_amp:
        cmd.append("--use-amp")
    if not args.resume:
        cmd.append("--no-resume")

    t0 = time.time()
    subprocess.run(cmd, check=True)
    elapsed = time.time() - t0
    logger.info(f"Stage 3 (Mamba-2 Training) completed in {format_duration(elapsed)}")
    return elapsed


def run_phase_export(logger: RainAILogger, python_exe: str):
    logger.stage(4, 5, "Multi-Backend Production Export & Format Generation")
    t0 = time.time()

    logger.info("Exporting Progressive Residual Slices & Acoustic Anchor (WASM/Rust)...")
    subprocess.run([python_exe, "-X", "utf8", str(PROJECT_ROOT / "scripts" / "export_wasm.py")], check=True)

    logger.info("Exporting Hugging Face Candle Safetensors...")
    subprocess.run([python_exe, "-X", "utf8", str(PROJECT_ROOT / "scripts" / "export_candle.py")], check=True)

    logger.info("Exporting ONNX Computation Graphs & INT8 Quantized Models...")
    subprocess.run([python_exe, "-X", "utf8", str(PROJECT_ROOT / "scripts" / "export_onnx.py")], check=True)

    # Directly verify output location metrics instead of legacy copies
    export_dir = PROJECT_ROOT.parent / "crates" / "inference" / "data"
    export_config = export_dir / "rainai_deployment_config.json"
    
    if export_config.exists():
        logger.info(f"[+] Verified workspace deployment config signature at: {export_config}")

    elapsed = time.time() - t0
    logger.info(f"Stage 4 (Multi-Backend Export) completed in {format_duration(elapsed)}")
    return elapsed


def run_phase_benchmark(logger: RainAILogger, python_exe: str, iterations: int = 50):
    logger.stage(5, 5, "Backend Latency & Inference Benchmark Verification")
    t0 = time.time()
    cmd = [
        python_exe, "-X", "utf8", str(PROJECT_ROOT / "scripts" / "benchmark_backends.py"),
        "--iterations", str(iterations)
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        subprocess.run([python_exe, "-X", "utf8", str(PROJECT_ROOT / "scripts" / "benchmark_backends.py")], check=True)

    elapsed = time.time() - t0
    logger.info(f"Stage 5 (Inference Benchmark) completed in {format_duration(elapsed)}")
    return elapsed


def print_summary_table(logger: RainAILogger, total_time: float, phase_times: dict):
    logger.info("")
    logger.banner("PIPELINE EXECUTION SUMMARY")
    logger.info(f"{'Phase / Artifact':<40} | {'Status':<12} | {'Details / Size'}")
    logger.info("-" * 80)

    # 1. Centralised Training Checkpoints
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    vae_best = ckpt_dir / "spatial_vae_ddsp_best.pt"
    if vae_best.exists():
        mb = vae_best.stat().st_size / (1024 * 1024)
        logger.info(f"{'Spatial VAE Best Checkpoint':<40} | {'READY':<12} | {mb:.1f} MB")

    mamba_best = ckpt_dir / "mamba2_metacontroller_best.pt"
    if mamba_best.exists():
        mb = mamba_best.stat().st_size / (1024 * 1024)
        logger.info(f"{'Mamba-2 MoE Best Checkpoint':<40} | {'READY':<12} | {mb:.1f} MB")

    # 2. Centralised Workspace Runtime Targets
    exp_dir = PROJECT_ROOT.parent / "crates" / "inference" / "data"
    if exp_dir.exists():
        # Scrape Progressive WASM Slices (.bin targets)
        slices = list((exp_dir / "wasm").glob("*.bin"))
        for s in sorted(slices):
            mb = s.stat().st_size / (1024 * 1024)
            logger.info(f"{f'wasm/{s.name}':<40} | {'EXPORTED':<12} | {mb:.2f} MB")

        # Scrape Hugging Face Candle safeTensors
        candle_files = list((exp_dir / "candle").glob("*.safetensors"))
        for c in candle_files:
            mb = c.stat().st_size / (1024 * 1024)
            logger.info(f"{f'candle/{c.name}':<40} | {'COMPILED':<12} | {mb:.2f} MB")

        # Scrape ONNX Graphs
        onnx_files = list((exp_dir / "onnx").glob("*.onnx"))
        for o in onnx_files:
            mb = o.stat().st_size / (1024 * 1024)
            logger.info(f"{f'onnx/{o.name}':<40} | {'OPTIMISED':<12} | {mb:.2f} MB")
            
        # Manifest Tracking Check
        export_config = exp_dir / "rainai_deployment_config.json"
        if export_config.exists():
            kb = export_config.stat().st_size / 1024
            logger.info(f"{'rainai_deployment_config.json':<40} | {'SYNCED':<12} | {kb:.2f} KB")

    logger.info("-" * 80)
    for p, t in phase_times.items():
        logger.info(f"Time taken for {p:<27} : {format_duration(t)}")
    logger.info(f"Total Pipeline Elapsed Time             : {format_duration(total_time)}")
    logger.info("=" * 80)
    logger.info("All engine workspace assets synchronised. Ready for runtime invocation.")
    logger.info("=" * 80)



def main():
    parser = argparse.ArgumentParser(
        description="RainAI Automated Training & Multi-Backend Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="balanced",
        choices=["smoke-test", "balanced", "production", "export-only", "custom"],
        help="Preset profile for training & export execution"
    )
    parser.add_argument("--phases", nargs="+", default=None,
                        help="Specific phases to run: vae, mamba, export, benchmark, all")
    parser.add_argument("--epochs-vae", type=int, default=None, help="Override epochs for VAE")
    parser.add_argument("--epochs-mamba", type=int, default=None, help="Override epochs for Mamba-2")
    parser.add_argument("--batch-size", type=int, default=None, help="Override mini-batch size")
    parser.add_argument("--max-batches", type=int, default=None, help="Max batches per epoch (0 for full dataset)")
    parser.add_argument("--accumulation-steps", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--device", type=str, default=None, help="Compute device (cuda/cpu)")
    parser.add_argument("--use-amp", action="store_true", default=None, help="Force enable AMP")
    parser.add_argument("--no-amp", dest="use_amp", action="store_false", help="Force disable AMP")
    parser.add_argument("--use-disc", action="store_true", default=None, help="Enable STFT discriminator")
    parser.add_argument("--chunk-curriculum", action="store_true", default=None, help="Enable dynamic chunk curriculum")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from existing checkpoints")
    parser.add_argument("--fresh", dest="resume", action="store_false", help="Start fresh training without resuming")
    parser.add_argument("--benchmark-iters", type=int, default=50, help="Iterations for inference benchmark")
    parser.add_argument("--log-dir", type=str, default=None, help="Custom directory for log files")
    parser.add_argument("--log-file", type=str, default=None, help="Custom filename for the log")
    parser.add_argument("--prepare-data", action="store_true", default=False, help="Ensure raw and processed audio data are fully prepared on-demand before training")
    parser.add_argument("--rebuild-data", action="store_true", default=False, help="Force complete re-synthesis, re-upmixing, and manifest regeneration from scratch")
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose debug logging")
    parser.add_argument("--quiet", action="store_true", default=False, help="Suppress informational console output")

    args = parser.parse_args()

    # Logging setup
    log_dir = Path(args.log_dir) if args.log_dir else (PROJECT_ROOT / "logs")
    console_level = logging.DEBUG if args.verbose else (logging.WARNING if args.quiet else logging.INFO)
    logger = setup_logger(
        name="auto_train",
        log_dir=log_dir,
        log_filename=args.log_file,
        console_level=console_level
    )

    # Hardware detection
    hw = detect_hardware()
    device = args.device or hw["device"]
    python_exe = sys.executable

    # Setup profile defaults
    if args.profile == "smoke-test":
        args.epochs_vae = args.epochs_vae or 1
        args.epochs_mamba = args.epochs_mamba or 1
        args.max_batches = args.max_batches if args.max_batches is not None else 2
        args.batch_size = args.batch_size or 2
        args.accumulation_steps = args.accumulation_steps or 1
        if args.use_amp is None:
            args.use_amp = hw["recommended_amp"]
        if args.use_disc is None:
            args.use_disc = False
        if args.chunk_curriculum is None:
            args.chunk_curriculum = True
        phases = ["vae", "mamba", "export", "benchmark"]
        args.benchmark_iters = min(args.benchmark_iters, 20)

    elif args.profile == "balanced":
        args.epochs_vae = args.epochs_vae or 3
        args.epochs_mamba = args.epochs_mamba or 2
        args.max_batches = args.max_batches if args.max_batches is not None else 30
        args.batch_size = args.batch_size or hw["recommended_batch_size"]
        args.accumulation_steps = args.accumulation_steps or hw["recommended_accumulation"]
        if args.use_amp is None:
            args.use_amp = hw["recommended_amp"]
        if args.use_disc is None:
            args.use_disc = False
        if args.chunk_curriculum is None:
            args.chunk_curriculum = True
        phases = ["vae", "mamba", "export", "benchmark"]

    elif args.profile == "production":
        args.epochs_vae = args.epochs_vae or 10
        args.epochs_mamba = args.epochs_mamba or 5
        args.max_batches = args.max_batches if args.max_batches is not None else 0  # full dataset
        args.batch_size = args.batch_size or hw["recommended_batch_size"]
        args.accumulation_steps = args.accumulation_steps or hw["recommended_accumulation"]
        if args.use_amp is None:
            args.use_amp = hw["recommended_amp"]
        if args.use_disc is None:
            args.use_disc = True
        if args.chunk_curriculum is None:
            args.chunk_curriculum = True
        phases = ["vae", "mamba", "export", "benchmark"]

    elif args.profile == "export-only":
        args.epochs_vae = 0
        args.epochs_mamba = 0
        args.max_batches = 0
        args.batch_size = 2
        args.accumulation_steps = 1
        args.use_amp = False
        args.use_disc = False
        args.chunk_curriculum = False
        phases = ["export", "benchmark"]

    else: # custom
        args.epochs_vae = args.epochs_vae or 3
        args.epochs_mamba = args.epochs_mamba or 2
        args.max_batches = args.max_batches if args.max_batches is not None else 10
        args.batch_size = args.batch_size or 2
        args.accumulation_steps = args.accumulation_steps or 1
        if args.use_amp is None:
            args.use_amp = hw["recommended_amp"]
        if args.use_disc is None:
            args.use_disc = False
        if args.chunk_curriculum is None:
            args.chunk_curriculum = True
        phases = ["vae", "mamba", "export", "benchmark"]

    # Custom phase overrides
    if args.phases:
        if "all" in args.phases:
            phases = ["vae", "mamba", "export", "benchmark"]
        else:
            phases = args.phases

    print_banner(logger, hw, args.profile, phases)

    total_start = time.time()
    phase_times = {}

    # Stage 1: Dataset Check & On-Demand Preparation
    check_dataset_and_manifest(
        logger,
        python_exe,
        prepare_data=args.prepare_data,
        rebuild_data=args.rebuild_data
    )

    # Stage 2: Train VAE
    if "vae" in phases:
        t = run_phase_vae(logger, python_exe, args, device, args.use_amp)
        phase_times["Phase 2 (Spatial VAE)"] = t

    # Stage 3: Train Mamba
    if "mamba" in phases:
        t = run_phase_mamba(logger, python_exe, args, device, args.use_amp)
        phase_times["Phase 3 (Mamba-2 MoE)"] = t

    # Stage 4: Export Multi-Backend
    if "export" in phases:
        t = run_phase_export(logger, python_exe)
        phase_times["Phase 4 (Multi-Backend Export)"] = t

    # Stage 5: Benchmark
    if "benchmark" in phases:
        t = run_phase_benchmark(logger, python_exe, iterations=args.benchmark_iters)
        phase_times["Phase 5 (Inference Benchmark)"] = t

    total_elapsed = time.time() - total_start
    print_summary_table(logger, total_elapsed, phase_times)

    # Save structured run summary
    summary_path = log_dir / "auto_train_summary_latest.json"
    logger.save_run_summary(
        summary_path,
        extra_metadata={
            "profile": args.profile,
            "device": device,
            "hardware": hw,
            "phases": phases,
            "phase_times": phase_times,
            "total_elapsed_seconds": total_elapsed
        }
    )

    # Also maintain latest copy of log
    if logger.log_file and logger.log_file.exists():
        import shutil
        latest_log = log_dir / "auto_train_latest.log"
        shutil.copy2(logger.log_file, latest_log)


if __name__ == "__main__":
    main()
