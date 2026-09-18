"""
Export pipeline for Candle (Safetensors): Exports model weights to standard Hugging Face
.safetensors files readable directly by Candle and Rust tensor engines.
Includes automatic residual slice fusion for deployment inference.
"""

import sys
from pathlib import Path
import json
import struct
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
            
            del module.weight_res
            module.register_parameter('weight', nn.Parameter(fused_weight))
    return model


def save_safetensors(tensors_dict: dict, file_path: Path, metadata: dict = None):
    header = {}
    if metadata:
        header["__metadata__"] = metadata

    current_offset = 0
    raw_data = bytearray()

    for name, arr in sorted(tensors_dict.items()):
        if not isinstance(arr, np.ndarray):
            arr = arr.detach().cpu().numpy()

        arr = np.ascontiguousarray(arr)
        data_bytes = arr.tobytes()
        start = current_offset
        end = start + len(data_bytes)

        dtype_map = {
            np.dtype("float32"): "F32",
            np.dtype("float16"): "F16",
            np.dtype("int8"): "I8",
            np.dtype("int16"): "I16",
            np.dtype("int32"): "I32",
        }
        safetensor_dtype = dtype_map.get(arr.dtype, "F32")

        header[name] = {
            "dtype": safetensor_dtype,
            "shape": list(arr.shape),
            "data_offsets": [start, end],
        }

        raw_data.extend(data_bytes)
        current_offset = end

    header_json = json.dumps(header).encode("utf-8")
    padding_length = (8 - len(header_json) % 8) % 8
    header_json += b" " * padding_length
    header_len = len(header_json)

    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(struct.pack("<Q", header_len))
        f.write(header_json)
        f.write(raw_data)

    print(
        f"[+] Exported Safetensors -> {file_path} ({file_path.stat().st_size / 1024:.1f} KB, {len(tensors_dict)} tensors)"
    )


def export_models_to_candle(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = PROJECT_ROOT / "checkpoints"

    print("=" * 70)
    print("RainAI Candle Safetensors Export Pipeline (Collapsed Slices)")
    print("=" * 70)

    encoder = SpatialAudioEncoder()
    mamba = Mamba2MoETrajectory()

    best_vae = checkpoints_dir / "spatial_vae_ddsp_best.pt"
    vae_ckpt = best_vae if best_vae.exists() else (checkpoints_dir / "spatial_vae_ddsp_latest.pt")
    if vae_ckpt.exists():
        ckpt = torch.load(vae_ckpt, map_location="cpu")
        try:
            encoder.load_state_dict(ckpt["encoder"], strict=False)
            print(f"[+] Loaded VAE checkpoint from {vae_ckpt}")
        except Exception as e:
            print(f"[*] Initialized VAE weights ({e})")

    best_mamba = checkpoints_dir / "mamba2_metacontroller_best.pt"
    mamba_ckpt = best_mamba if best_mamba.exists() else (checkpoints_dir / "mamba2_metacontroller_latest.pt")
    if mamba_ckpt.exists():
        ckpt = torch.load(mamba_ckpt, map_location="cpu")
        try:
            mamba.load_state_dict(ckpt["mamba_moe"], strict=False)
            print(f"[+] Loaded Mamba-2 checkpoint from {mamba_ckpt}")
        except Exception as e:
            print(f"[*] Initialized Mamba-2 weights ({e})")

    # Collapse multi-slice ResidualWeights into standard runtime tensors
    encoder = collapse_residual_weights(encoder, target_level=2)
    mamba = collapse_residual_weights(mamba, target_level=2)

    vae_tensors = {f"encoder.{k}": v for k, v in encoder.state_dict().items()}
    save_safetensors(
        vae_tensors,
        output_dir / "spatial_vae.safetensors",
        metadata={"model": "SpatialAudioEncoder", "format": "safetensors"},
    )

    mamba_tensors = {f"mamba.{k}": v for k, v in mamba.state_dict().items()}
    save_safetensors(
        mamba_tensors,
        output_dir / "mamba2_moe.safetensors",
        metadata={"model": "Mamba2MoETrajectory", "format": "safetensors"},
    )


if __name__ == "__main__":
    out_dir = get_export_root() / "candle"
    export_models_to_candle(out_dir)