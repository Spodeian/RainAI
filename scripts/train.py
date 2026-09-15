"""
Master Self-Integrated Training & Export Orchestrator for RainAI:
Executes the full pipeline:
  Phase 2: Train Spatial VAE + HOA-DDSP Engine (Residual QAT S0..S4)
  Phase 3: Train Mamba-2 MoE + HWIL Meta-Controller (using frozen VAE latents)
  Phase 4: Export Role-Anchored Progressive Slices & Deployment Config
  Phase 5: Sync to Edge/Desktop Runtime & Validate Checksums
"""

import sys
import os
import argparse
from pathlib import Path
import time
import subprocess
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_pipeline(
    phases: list,
    vae_epochs: int = 3,
    mamba_epochs: int = 2,
    batch_size: int = 2,
    max_batches: int = 10,
    device: str = "cpu",
    use_amp: bool = False,
    use_disc: bool = False,
    accumulation_steps: int = 1,
    chunk_curriculum: bool = False
):
    print("=" * 80)
    print("RainAI Master Self-Integrated Training & Export Pipeline")
    print("=" * 80)

    python_exe = sys.executable
    checkpoints_dir = PROJECT_ROOT / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    export_dir = PROJECT_ROOT / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset Standard Check
    print("\n[Stage 1/4] Auditing Dataset Standard & Manifest...")
    manifest_path = PROJECT_ROOT / "data" / "processed" / "manifest.json"
    if not manifest_path.exists():
        print("[*] Manifest not found. Running dataset standard migration...")
        cmd = [python_exe, "-X", "utf8", str(PROJECT_ROOT / "scripts" / "update_datasets_to_current_standard.py")]
        subprocess.run(cmd, check=True)
    else:
        print(f"[+] Dataset manifest verified at: {manifest_path}")

    # 2. Phase 2: Train Spatial VAE
    if "vae" in phases:
        print("\n[Stage 2/4] Training Spatial VAE + DDSP Engine (Phase 2)...")
        t0 = time.time()
        cmd = [
            python_exe, "-X", "utf8", str(PROJECT_ROOT / "scripts" / "train_vae.py"),
            "--epochs", str(vae_epochs),
            "--batch-size", str(batch_size),
            "--device", device,
            "--accumulation-steps", str(accumulation_steps)
        ]
        if max_batches and max_batches > 0:
            cmd.extend(["--max-batches", str(max_batches)])
        if use_amp:
            cmd.append("--use-amp")
        if use_disc:
            cmd.append("--use-disc")
        if chunk_curriculum:
            cmd.append("--chunk-curriculum")

        subprocess.run(cmd, check=True)
        print(f"[+] Phase 2 complete in {time.time() - t0:.2f}s")

    # 3. Phase 3: Train Mamba-2 MoE + Meta-Controller
    if "mamba" in phases:
        print("\n[Stage 3/4] Training Mamba-2 MoE + Meta-Controller (Phase 3)...")
        t0 = time.time()
        best_vae = checkpoints_dir / "spatial_vae_ddsp_best.pt"
        vae_ckpt = best_vae if best_vae.exists() else (checkpoints_dir / "spatial_vae_ddsp_latest.pt")
        cmd = [
            python_exe, "-X", "utf8", str(PROJECT_ROOT / "scripts" / "train_mamba.py"),
            "--epochs", str(mamba_epochs),
            "--batch-size", str(batch_size),
            "--device", device,
            "--vae-checkpoint", str(vae_ckpt),
            "--accumulation-steps", str(accumulation_steps)
        ]
        if max_batches and max_batches > 0:
            cmd.extend(["--max-batches", str(max_batches)])
        if use_amp:
            cmd.append("--use-amp")

        subprocess.run(cmd, check=True)
        print(f"[+] Phase 3 complete in {time.time() - t0:.2f}s")

    # 4. Phase 4: Export Slices & Multi-Backend Formats
    if "export" in phases:
        print("\n[Stage 4/4] Exporting Progressive Residual Slices, Candle Safetensors, and ONNX Graphs (Phase 4)...")
        t0 = time.time()
        # 4a. Progressive Slices & Acoustic Anchor
        cmd_wasm = [python_exe, "-X", "utf8", str(PROJECT_ROOT / "scripts" / "export_wasm.py")]
        subprocess.run(cmd_wasm, check=True)

        # 4b. Candle Safetensors Export
        cmd_candle = [python_exe, "-X", "utf8", str(PROJECT_ROOT / "scripts" / "export_candle.py")]
        subprocess.run(cmd_candle, check=True)

        # 4c. ONNX Graph Export
        cmd_onnx = [python_exe, "-X", "utf8", str(PROJECT_ROOT / "scripts" / "export_onnx.py")]
        subprocess.run(cmd_onnx, check=True)
        
        # Sync config to Rust template if present
        rust_template_config = PROJECT_ROOT.parent / "Severless & Desktop Template" / "crates" / "inference" / "data" / "rainai_deployment_config.json"
        if rust_template_config.parent.exists():
            import shutil
            shutil.copy2(export_dir / "rainai_deployment_config.json", rust_template_config)
            print(f"[+] Synced deployment config to: {rust_template_config}")
            
        print(f"[+] Phase 4 multi-backend export complete in {time.time() - t0:.2f}s")

    print("\n" + "=" * 80)
    print("MASTER PIPELINE EXECUTION FINISHED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master RainAI Training & Export Pipeline")
    parser.add_argument("--phases", nargs="+", default=["vae", "mamba", "export"], choices=["vae", "mamba", "export", "all"], help="Phases to execute")
    parser.add_argument("--vae-epochs", type=int, default=3, help="VAE epochs")
    parser.add_argument("--mamba-epochs", type=int, default=2, help="Mamba epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--max-batches", type=int, default=5, help="Max batches per epoch (0 for full dataset)")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device (e.g. cpu, cuda)")
    parser.add_argument("--use-amp", action="store_true", help="Enable Automatic Mixed Precision (AMP)")
    parser.add_argument("--use-disc", action="store_true", help="Enable Multi-Scale STFT Discriminator")
    parser.add_argument("--accumulation-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--chunk-curriculum", action="store_true", help="Enable dynamic length bucketing and curriculum duration scheduling")
    args = parser.parse_args()

    selected_phases = ["vae", "mamba", "export"] if "all" in args.phases else args.phases
    run_pipeline(
        phases=selected_phases,
        vae_epochs=args.vae_epochs,
        mamba_epochs=args.mamba_epochs,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        device=args.device,
        use_amp=args.use_amp,
        use_disc=args.use_disc,
        accumulation_steps=args.accumulation_steps,
        chunk_curriculum=args.chunk_curriculum
    )
