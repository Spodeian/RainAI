"""
Export pipeline for ONNX: Exports SpatialAudioEncoder and Mamba2MoETrajectory to ONNX graphs.
Includes automatic residual slice fusion for clean runtime graph compilation.
"""

import sys
from pathlib import Path
import json
import torch
import torch.nn as nn
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.diff_autoencoder import SpatialAudioEncoder
from src.models.mamba2_moe import Mamba2MoETrajectory


def get_export_root() -> Path:
    target = PROJECT_ROOT / "crates" / "inference" / "data"
    target.mkdir(parents=True, exist_ok=True)
    return target

class SpatialAudioEncoderONNXWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, audio_foa, conditioning):
        return self.encoder(audio_foa, conditioning, tau=0.1, active_level=None)

class MambaONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, z_seq, conditioning, expert_mask, tau_moe):
        return self.model(
            z_seq, 
            conditioning, 
            expert_mask, 
            tau_moe, 
            return_stationarity=False
        )

class FusedWeightProvider(nn.Module):
    """
    Stub to replace complex multi-slice weight logic during ONNX tracing.
    Returns the static pre-fused tensor to keep the computation graph clean.
    """
    def __init__(self, fused_weight):
        super().__init__()
        self.fused_weight = nn.Parameter(fused_weight)

    def get_effective_weight(self, **kwargs):
        # Discards dynamic arguments like active_level and quantize_base 
        # to ensure tracing locks onto the target precision layer.
        return self.fused_weight

@torch.no_grad()
def collapse_residual_weights(model: nn.Module, target_level: int = 2):
    """
    Fuses multi-slice ResidualWeights into standard FP16/INT8 weights for inference.
    Target Level 2 corresponds to INT8 baseline precision.
    """
    for module_name, module in list(model.named_modules()):
        if hasattr(module, 'weight_res'):
            fused_weight = module.weight_res.get_effective_weight(
                active_level=target_level, 
                quantize_base=True
            ).clone()
            
            module.weight_res = FusedWeightProvider(fused_weight)
    return model

def export_models_to_onnx(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = PROJECT_ROOT / "checkpoints"

    print("=" * 70)
    print("RainAI ONNX Export Pipeline (Collapsed Slices - Opset 18)")
    print("=" * 70)

    encoder = SpatialAudioEncoder()
    mamba = Mamba2MoETrajectory()

    best_vae = checkpoints_dir / "spatial_vae_ddsp_best.pt"
    vae_ckpt = best_vae if best_vae.exists() else (checkpoints_dir / "spatial_vae_ddsp_latest.pt")
    if vae_ckpt.exists():
        ckpt = torch.load(vae_ckpt, map_location="cpu")
        try:
            encoder.load_state_dict(ckpt["encoder"], strict=False)
            print(f"[+] Loaded VAE encoder checkpoint from {vae_ckpt}")
        except Exception as e:
            print(f"[*] Using initialized weights ({e})")

    best_mamba = checkpoints_dir / "mamba2_metacontroller_best.pt"
    mamba_ckpt = best_mamba if best_mamba.exists() else (checkpoints_dir / "mamba2_metacontroller_latest.pt")
    if mamba_ckpt.exists():
        ckpt = torch.load(mamba_ckpt, map_location="cpu")
        try:
            mamba.load_state_dict(ckpt["mamba_moe"], strict=False)
            print(f"[+] Loaded Mamba-2 checkpoint from {mamba_ckpt}")
        except Exception as e:
            print(f"[*] Using initialized weights ({e})")

    # Collapse multi-slice weights for clean ONNX operator mapping
    encoder = collapse_residual_weights(encoder, target_level=2)
    mamba = collapse_residual_weights(mamba, target_level=2)

    encoder.eval()
    mamba.eval()

    print("\n--- Step 1: Exporting SpatialAudioEncoder to ONNX ---")
    dummy_audio = torch.randn(1, 4, 48000, dtype=torch.float32)
    dummy_cond = torch.randn(1, 554, dtype=torch.float32)
    encoder_onnx_path = output_dir / "spatial_vae.onnx"
    
    wrapped_encoder = SpatialAudioEncoderONNXWrapper(encoder)
    wrapped_encoder.eval()

    try:
        torch.onnx.export(
            wrapped_encoder,
            (dummy_audio, dummy_cond),
            str(encoder_onnx_path),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            dynamo=False,
            input_names=["audio_foa", "conditioning"],
            output_names=["latents_quantized", "latents_aligned", "quality_score"],
        )
        print(
            f"[+] Successfully exported SpatialAudioEncoder -> {encoder_onnx_path} ({encoder_onnx_path.stat().st_size / 1024:.1f} KB)"
        )
    except Exception as e:
        print(f"[-] SpatialAudioEncoder export failed: {e}")
        raise e

    print("\n--- Step 2: Exporting Mamba2MoETrajectory to ONNX ---")
    dummy_z = torch.randn(1, 128, 64, dtype=torch.float32)
    dummy_mask = torch.ones(1, 8, dtype=torch.float32)
    dummy_tau = torch.ones(1, 1, dtype=torch.float32) * 0.5
    mamba_onnx_path = output_dir / "mamba2_moe.onnx"
    
    # Wrap the model prior to export
    wrapped_mamba = MambaONNXWrapper(mamba)
    wrapped_mamba.eval()

    try:
        torch.onnx.export(
            wrapped_mamba,  # Use the wrapper here
            (dummy_z, dummy_cond, dummy_mask, dummy_tau),
            str(mamba_onnx_path),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            dynamo=False,
            input_names=["z_seq", "conditioning", "expert_mask", "tau_moe"],
            output_names=["mu_next", "sigma_next", "moe_logits", "quality_score", "log_var"],
        )
        print(
            f"[+] Successfully exported Mamba2MoETrajectory -> {mamba_onnx_path} ({mamba_onnx_path.stat().st_size / 1024:.1f} KB)"
        )
    except Exception as e:
        print(f"[-] Mamba2MoETrajectory export failed: {e}")
        raise e

    manifest = {
        "format": "onnx",
        "opset": 18,
        "models": {
            "encoder": str(encoder_onnx_path.name),
            "mamba_moe": str(mamba_onnx_path.name),
        },
    }
    with open(output_dir / "onnx_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[+] Exported ONNX manifest -> {output_dir / 'onnx_manifest.json'}")


if __name__ == "__main__":
    out_dir = get_export_root() / "onnx"
    export_models_to_onnx(out_dir)