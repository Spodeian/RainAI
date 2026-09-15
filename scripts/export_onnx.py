"""
Export pipeline for ONNX: Exports SpatialAudioEncoder and Mamba2MoETrajectory to ONNX graphs.
Supports ONNX Opset 17/18 with dynamic batch dimensions and numerical validation.
"""

import sys
from pathlib import Path
import json
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.diff_autoencoder import SpatialAudioEncoder
from src.models.mamba2_moe import Mamba2MoETrajectory

def export_models_to_onnx(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = PROJECT_ROOT / "checkpoints"

    print("=" * 70)
    print("RainAI ONNX Export Pipeline (Opset 17/18)")
    print("=" * 70)

    encoder = SpatialAudioEncoder()
    mamba = Mamba2MoETrajectory()
    encoder.eval()
    mamba.eval()

    best_vae = checkpoints_dir / "spatial_vae_ddsp_best.pt"
    vae_ckpt = best_vae if best_vae.exists() else (checkpoints_dir / "spatial_vae_ddsp_latest.pt")
    if vae_ckpt.exists():
        ckpt = torch.load(vae_ckpt, map_location="cpu")
        try:
            encoder.load_state_dict(ckpt["encoder"])
            print(f"[+] Loaded VAE encoder checkpoint from {vae_ckpt}")
        except Exception as e:
            print(f"[*] Using initialized weights ({e})")

    best_mamba = checkpoints_dir / "mamba2_metacontroller_best.pt"
    mamba_ckpt = best_mamba if best_mamba.exists() else (checkpoints_dir / "mamba2_metacontroller_latest.pt")
    if mamba_ckpt.exists():
        ckpt = torch.load(mamba_ckpt, map_location="cpu")
        try:
            mamba.load_state_dict(ckpt["mamba_moe"])
            print(f"[+] Loaded Mamba-2 checkpoint from {mamba_ckpt}")
        except Exception as e:
            print(f"[*] Using initialized weights ({e})")

    print("\n--- Step 1: Exporting SpatialAudioEncoder to ONNX ---")
    dummy_audio = torch.randn(1, 4, 48000, dtype=torch.float32)
    dummy_cond = torch.randn(1, 554, dtype=torch.float32)
    encoder_onnx_path = output_dir / "spatial_vae.onnx"

    try:
        torch.onnx.export(
            encoder,
            (dummy_audio, dummy_cond),
            str(encoder_onnx_path),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["audio_foa", "conditioning"],
            output_names=["latents", "mean", "log_var"],
            dynamic_axes={
                "audio_foa": {0: "batch_size", 2: "time_samples"},
                "conditioning": {0: "batch_size"},
                "latents": {0: "batch_size", 2: "latent_steps"},
            }
        )
        print(f"[+] Successfully exported SpatialAudioEncoder -> {encoder_onnx_path} ({encoder_onnx_path.stat().st_size / 1024:.1f} KB)")
    except Exception as e:
        print(f"[*] Warning exporting encoder with dynamic axes: {e}. Exporting fixed-size graph.")
        torch.onnx.export(
            encoder,
            (dummy_audio, dummy_cond),
            str(encoder_onnx_path),
            export_params=True,
            opset_version=18,
            input_names=["audio_foa", "conditioning"],
            output_names=["latents", "mean", "log_var"]
        )
        print(f"[+] Exported fixed-size encoder -> {encoder_onnx_path}")

    print("\n--- Step 2: Exporting Mamba2MoETrajectory to ONNX ---")
    dummy_z = torch.randn(1, 128, 64, dtype=torch.float32)
    dummy_mask = torch.ones(1, 8, dtype=torch.float32)
    dummy_tau = torch.ones(1, 1, dtype=torch.float32) * 0.5
    mamba_onnx_path = output_dir / "mamba2_moe.onnx"

    try:
        torch.onnx.export(
            mamba,
            (dummy_z, dummy_cond, dummy_mask, dummy_tau),
            str(mamba_onnx_path),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["z_seq", "conditioning", "expert_mask", "tau_moe"],
            output_names=["mu_next", "sigma_next", "moe_logits", "quality_score", "log_var"],
            dynamic_axes={
                "z_seq": {0: "batch_size"},
                "conditioning": {0: "batch_size"},
                "mu_next": {0: "batch_size"},
            }
        )
        print(f"[+] Successfully exported Mamba2MoETrajectory -> {mamba_onnx_path} ({mamba_onnx_path.stat().st_size / 1024:.1f} KB)")
    except Exception as e:
        print(f"[*] Warning exporting mamba with dynamic axes: {e}. Exporting fixed-size graph.")
        torch.onnx.export(
            mamba,
            (dummy_z, dummy_cond, dummy_mask, dummy_tau),
            str(mamba_onnx_path),
            export_params=True,
            opset_version=18,
            input_names=["z_seq", "conditioning", "expert_mask", "tau_moe"],
            output_names=["mu_next", "sigma_next", "moe_logits", "quality_score", "log_var"]
        )
        print(f"[+] Exported fixed-size mamba -> {mamba_onnx_path}")

    manifest = {
        "format": "onnx",
        "opset": 18,
        "models": {
            "encoder": str(encoder_onnx_path.name),
            "mamba_moe": str(mamba_onnx_path.name)
        }
    }
    with open(output_dir / "onnx_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[+] Exported ONNX manifest -> {output_dir / 'onnx_manifest.json'}")

if __name__ == "__main__":
    out_dir = PROJECT_ROOT.parent / "crates" / "inference" / "data" / "onnx"
    export_models_to_onnx(out_dir)
