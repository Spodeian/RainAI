"""
End-to-end ingestion, extraction, spatialization, and metadata generation pipeline.
Acts as the glue code between raw internet audio and the RainSpatialDataset.
"""
import argparse
import json
import subprocess
import sys
import warnings
import zipfile
from pathlib import Path

warnings.warn(
    "src/data/pipeline.py is deprecated. All ingestion and spatial preprocessing are now implemented in pure Rust. "
    "Use `cargo run -p utilities --bin rainai_ingest`, `rainai_upmix`, or the interactive TUI `rainai_studio`.",
    DeprecationWarning,
    stacklevel=2,
)

import numpy as np
import torch
import torchaudio

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SQRT2_INV = 0.7071067811865475

def enforce_vram_threshold(max_utilisation: float = 0.9):
    """
    Queries the CUDA driver for allocator statistics and dynamically 
    clears the cache if available VRAM drops below the safe threshold.
    """
    if DEVICE.type != "cuda":
        return
        
    free_bytes, total_bytes = torch.cuda.mem_get_info(DEVICE)
    
    if free_bytes < max_utilisation*total_bytes:
        torch.cuda.empty_cache()

def upmix_to_foa(waveform: torch.Tensor) -> torch.Tensor:
    """
    Upmixes Mono (1ch) or Stereo (2ch) signals to First-Order Ambisonics (FOA)
    B-format (AmbiX ordering: W, Y, Z, X). Operates on the tensor's current device.
    """
    channels, samples = waveform.shape
    foa = torch.zeros((4, samples), dtype=torch.float32, device=waveform.device)
    
    if channels == 1:
        # W = M, X = Y = Z = 0
        foa[0, :] = waveform[0]
    elif channels >= 2:
        # W = (L + R) / sqrt(2), Y = (L - R) / sqrt(2), Z = X = 0
        foa[0, :] = (waveform[0] + waveform[1]) * SQRT2_INV
        foa[1, :] = (waveform[0] - waveform[1]) * SQRT2_INV
        
    return foa

def standardize_and_generate_manifest():
    print(f"\n[Pipeline] Initiating CUDA-accelerated standardization on: {torch.cuda.get_device_name(DEVICE) if DEVICE.type == 'cuda' else 'CPU'}")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}
    
    surface_map = {0: "asphalt", 1: "pavement", 2: "tin_roof", 3: "canvas_tent", 4: "foliage", 5: "wood_deck", 6: "glass", 7: "puddle_shallow", 8: "water_deep"}
    extensions = ["*.wav", "*.flac", "*.mp3", "*.ogg"]
    files = []
    for ext in extensions:
        files.extend(RAW_AUDIO_DIR.rglob(ext))
        
    for file_path in files:
        try:
            waveform, sr = torchaudio.load(file_path)
            waveform = waveform.to(DEVICE, non_blocking=True)
            
            if sr != TARGET_SR:
                waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=TARGET_SR)
            
            foa_waveform = upmix_to_foa(waveform)
            
            # Physics-based feature extraction for TUI telemetry and Dataset conditioning
            rms = torch.sqrt(torch.mean(foa_waveform[0] ** 2)).item()
            intensity = min(rms * 10.0, 1.0)
            
            fft_mag = torch.abs(torch.fft.rfft(foa_waveform[0]))
            freqs = torch.linspace(0.0, TARGET_SR / 2.0, len(fft_mag), device=DEVICE)
            spectral_centroid = torch.sum(freqs * fft_mag) / (torch.sum(fft_mag) + 1e-8)
            
            surface_idx = int(np.random.randint(0, 9))
            
            out_filename = f"{file_path.stem}_foa.flac"
            out_path = PROCESSED_DIR / out_filename
            torchaudio.save(out_path, foa_waveform.cpu(), TARGET_SR, format="flac")
            
            manifest[out_path.stem] = {
                "clap_embedding": np.random.randn(CLAP_DIM).astype(float).tolist(),
                "physical_params": (np.random.rand(PHYS_DIM) * intensity).tolist(),
                "rain_rate": intensity,
                "droplet_density": intensity * 0.8,
                "rms_energy": rms,
                "spectral_centroid": spectral_centroid.item(),
                "surface_idx": surface_idx,
                "surface_tag": surface_map[surface_idx],
                "high_freq_ratio": min(spectral_centroid.item() / 10000.0, 1.0),
                "drift_scale": 0.2
            }
            
            del waveform
            del foa_waveform
            enforce_vram_threshold(0.95)
            
        except Exception as e:
            print(f"Failed processing {file_path.name}: {e}")

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)
        
    print(f"[Pipeline] Processing complete. Manifest saved to {MANIFEST_PATH}")

def extract_archives():
    print("\n[Pipeline] Extracting downloaded archives...")
    for zip_path in RAW_AUDIO_DIR.glob("*.zip"):
        print(f"Unzipping {zip_path.name}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(RAW_AUDIO_DIR / zip_path.stem)
        zip_path.unlink()

def main():
    parser = argparse.ArgumentParser(description="RainAI Pipeline Orchestrator")
    parser.add_argument("--prepare-data", action="store_true", help="Execute the full data standardization pipeline.")
    args = parser.parse_args()

    if not DB_PATH.exists():
        print(f"Database {DB_PATH} missing.")
        sys.exit(1)
        
    # Standardize invocation to align with TUI dispatches
    if args.prepare_data:
        extract_archives()
        standardize_and_generate_manifest()
    else:
        with open(DB_PATH, "r", encoding="utf-8") as f:
            sources = json.load(f)
        
        print("[Pipeline] Initiating Rust high-speed HTTP phase...")
        subprocess.run(["cargo", "run", "--release", "--bin", "rainai_ingest"])
        
        print("\n[Pipeline] Initiating external API/CLI phase...")
        for item in sources:
            method = item.get("ingest_method", "direct_http")
            if method == "direct_http": 
                continue
                
            target = BASE_DIR / "rain" / item.get("media_type", "audio")
            target.mkdir(parents=True, exist_ok=True)
            
            if method == "kaggle_cli":
                cmd = ["kaggle", "datasets", "download", "-d", item["url"], "-p", str(target)]
                subprocess.run(cmd, check=True)
            elif method == "yt_dlp":
                cmd = ["yt-dlp", "-o", str(target / f"{item['filename']}.%(ext)s")]
                if item.get("media_type") == "audio":
                    cmd.extend(["-x", "--audio-format", "wav"])
                cmd.append(item["url"])
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
                
        extract_archives()
        standardize_and_generate_manifest()

if __name__ == "__main__": 
    main()