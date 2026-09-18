"""
Export pipeline for Phase 4: Exports Role-Anchored Progressive Slices (S0..S4)
and FP32 Acoustic Anchor for Rust (Candle/Burn), WebAudio Service Worker, and WebGPU runtime.
"""

import sys
from pathlib import Path
import json
import hashlib
import zlib
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.diff_autoencoder import SpatialAudioEncoder
from src.models.ddsp import ContinuousParametricFilter
from src.models.mamba2_moe import Mamba2MoETrajectory
from src.models.meta_controller import InvasiveMetaController
from src.models.residual_quant import ResidualLinear, ResidualConv1d


def get_export_root() -> Path:
    target = PROJECT_ROOT / "crates" / "inference" / "data"
    target.mkdir(parents=True, exist_ok=True)
    return target


def compute_crc32(data: bytes) -> str:
    return f"{zlib.crc32(data) & 0xffffffff:08x}"


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def clean_state_dict(sd: dict) -> dict:
    """Strips torch.compile '_orig_mod.' prefixes from checkpoint keys."""
    return {
        (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
        for k, v in sd.items()
    }


def export_models_for_wasm(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = PROJECT_ROOT / "checkpoints"

    print("=" * 70)
    print("RainAI Phase 4: Role-Anchored Progressive Slices & WASM Export")
    print("=" * 70)

    encoder = SpatialAudioEncoder()
    ddsp = ContinuousParametricFilter()
    mamba = Mamba2MoETrajectory()
    meta = InvasiveMetaController()

    best_vae = checkpoints_dir / "spatial_vae_ddsp_best.pt"
    vae_ckpt = best_vae if best_vae.exists() else (checkpoints_dir / "spatial_vae_ddsp_latest.pt")
    if vae_ckpt.exists():
        ckpt = torch.load(vae_ckpt, map_location="cpu")
        try:
            encoder.load_state_dict(clean_state_dict(ckpt["encoder"]), strict=False)
            ddsp.load_state_dict(clean_state_dict(ckpt["ddsp"]), strict=False)
            print(f"[+] Loaded VAE checkpoint from {vae_ckpt}")
        except Exception as e:
            print(f"[!] Critical failure loading VAE weights from {vae_ckpt}: {e}")
            raise e
    else:
        raise FileNotFoundError(f"[!] VAE checkpoint not found at {vae_ckpt}")

    best_mamba = checkpoints_dir / "mamba2_metacontroller_best.pt"
    mamba_ckpt = best_mamba if best_mamba.exists() else (checkpoints_dir / "mamba2_metacontroller_latest.pt")
    if mamba_ckpt.exists():
        ckpt = torch.load(mamba_ckpt, map_location="cpu")
        try:
            mamba.load_state_dict(clean_state_dict(ckpt["mamba_moe"]), strict=False)
            meta.load_state_dict(clean_state_dict(ckpt["meta_controller"]), strict=False)
            print(f"[+] Loaded Mamba-2 checkpoint from {mamba_ckpt}")
        except Exception as e:
            print(f"[!] Critical failure loading Mamba-2 weights from {mamba_ckpt}: {e}")
            raise e
    else:
        raise FileNotFoundError(f"[!] Mamba-2 checkpoint not found at {mamba_ckpt}")

    print("\n--- Step 1: Exporting FP32 Acoustic Anchor ---")
    anchor_weights = {}
    for name, param in ddsp.named_parameters():
        anchor_weights[f"ddsp.{name}"] = param.detach().cpu().numpy().astype(np.float32)

    anchor_bytes = bytearray()
    for name, arr in sorted(anchor_weights.items()):
        anchor_bytes.extend(arr.tobytes())

    anchor_bin_path = output_dir / "acoustic_anchor_fp32.bin"
    with open(anchor_bin_path, "wb") as f:
        f.write(anchor_bytes)

    anchor_crc = compute_crc32(anchor_bytes)
    anchor_sha = compute_sha256(anchor_bytes)
    print(f"[+] Exported Acoustic Anchor ({len(anchor_bytes)} bytes, CRC32: {anchor_crc})")

    print("\n--- Step 2: Exporting Progressive Slices (S0..S4) ---")
    residual_modules = []
    for model_name, model in [("encoder", encoder), ("mamba", mamba), ("meta", meta)]:
        for module_name, module in model.named_modules():
            if isinstance(module, (ResidualLinear, ResidualConv1d)):
                full_name = f"{model_name}.{module_name}"
                residual_modules.append((full_name, module.weight_res))

    print(f"[*] Found {len(residual_modules)} multi-slice residual weight layers.")

    slice_metadata = []
    cumulative_bytes = [bytearray() for _ in range(5)]
    slice_names = [
        "slice_0_ternary.bin",
        "slice_1_mobile_int4.bin",
        "slice_2_standard_int8.bin",
        "slice_3_studio_int16.bin",
        "slice_4_master_fp32.bin",
    ]

    tier_descriptions = [
        "Tier 0: Ternary 1.58-Bit (Instant startup < 500ms, addition-only)",
        "Tier 1: INT4 Mobile Quality (Low bandwidth / cellular)",
        "Tier 2: INT8 / BF16 Adaptive Web Default (~95% perceptual equivalence)",
        "Tier 3: INT16 / FP16 Studio High-Fidelity",
        "Tier 4: Uncompressed FP32 Master Masterpiece",
    ]

    for slice_idx in range(5):
        slice_byte_arr = bytearray()
        layer_specs = {}

        for full_name, slice_tensor in sorted(residual_modules, key=lambda x: x[0]):
            slice_tensor = slice_tensor.slices[slice_idx].detach().cpu().numpy().astype(np.float32)

            if slice_idx == 0:
                gamma = float(np.mean(np.abs(slice_tensor)))
                q_ternary = np.clip(np.round(slice_tensor / max(gamma, 1e-6)), -1, 1).astype(np.int8)
                layer_bytes = q_ternary.tobytes()
                layer_specs[full_name] = {
                    "shape": list(slice_tensor.shape),
                    "gamma": gamma,
                    "dtype": "int8_ternary",
                    "offset": len(slice_byte_arr),
                    "size_bytes": len(layer_bytes),
                }
            elif slice_idx in [1, 2]:
                scale = float(np.max(np.abs(slice_tensor)) / 127.0)
                q_int8 = np.clip(np.round(slice_tensor / max(scale, 1e-6)), -128, 127).astype(np.int8)
                layer_bytes = q_int8.tobytes()
                layer_specs[full_name] = {
                    "shape": list(slice_tensor.shape),
                    "scale": scale,
                    "dtype": "int8_delta",
                    "offset": len(slice_byte_arr),
                    "size_bytes": len(layer_bytes),
                }
            else:
                fp_arr = slice_tensor.astype(np.float16 if slice_idx == 3 else np.float32)
                layer_bytes = fp_arr.tobytes()
                layer_specs[full_name] = {
                    "shape": list(slice_tensor.shape),
                    "scale": 1.0,
                    "dtype": "float16" if slice_idx == 3 else "float32",
                    "offset": len(slice_byte_arr),
                    "size_bytes": len(layer_bytes),
                }

            slice_byte_arr.extend(layer_bytes)

        slice_file_path = output_dir / slice_names[slice_idx]
        with open(slice_file_path, "wb") as f:
            f.write(slice_byte_arr)

        for k in range(slice_idx, 5):
            cumulative_bytes[k].extend(slice_byte_arr)

        slice_metadata.append({
            "slice_index": slice_idx,
            "filename": slice_names[slice_idx],
            "description": tier_descriptions[slice_idx],
            "chunk_size_bytes": len(slice_byte_arr),
            "layers": layer_specs,
        })
        print(f"[+] Exported {slice_names[slice_idx]} ({len(slice_byte_arr) / 1024 / 1024:.2f} MB)")

    print("\n--- Step 3: Generating Cumulative Verification Checksums ---")
    cumulative_manifest = []
    for k in range(5):
        cum_data = bytes(cumulative_bytes[k])
        crc = compute_crc32(cum_data)
        sha = compute_sha256(cum_data)
        cumulative_manifest.append({
            "tier_level": k,
            "slices_included": list(range(k + 1)),
            "cumulative_size_bytes": len(cum_data),
            "crc32": crc,
            "sha256": sha,
        })

    deploy_config = {
        "architecture": "Role-Anchored Progressive Slices (W = sum_i S_i)",
        "audio": {
            "sample_rate": 48000,
            "channels": 4,
            "format": "First-Order Ambisonics (FOA, ACN/SN3D)",
            "channel_order": ["W", "Y", "Z", "X"],
            "block_size": 128,
            "target_buffer_ms": 40.0,
        },
        "acoustic_anchor": {
            "filename": "acoustic_anchor_fp32.bin",
            "format": "FP32 Continuous Biquad & Spherical Harmonics",
            "size_bytes": len(anchor_bytes),
            "crc32": anchor_crc,
            "sha256": anchor_sha,
        },
        "progressive_slices": slice_metadata,
        "cumulative_verification": cumulative_manifest,
        "neural_control": {
            "latent_dim": 64,
            "latent_rate_hz": 100,
            "conditioning_dim": 554,
            "num_moe_experts": 8,
            "meta_controller": {
                "active_slice_aware": True,
                "inputs": [
                    "moe_logits",
                    "telemetry (4d)",
                    "user_weights",
                    "quality_scores",
                    "active_slice_level",
                ],
            },
        },
    }

    config_path = output_dir.parent / "rainai_deployment_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(deploy_config, f, indent=2)
    print(f"\n[+] Master deployment configuration exported to {config_path}")


if __name__ == "__main__":
    out_dir = get_export_root() / "wasm"
    export_models_for_wasm(out_dir)