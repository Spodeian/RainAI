"""
Master preprocessing pipeline for RainAI:
1. Discovers raw audio files across Data/rain and Data/rain-nc
2. Standardizes sample rates to 48kHz
3. Upmixes to 4-channel First-Order Ambisonics (FOA)
4. Slices into 5-second chunks saved in Data/processed/
5. Generates semantic embedding manifest.json
"""

import sys
from pathlib import Path
import json
import time

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.spatial_upmix import process_audio_file
from src.data.semantic_tag import generate_manifest_for_directory


def run_preprocessing(
    raw_data_dirs: list,
    output_dir: Path,
    manifest_path: Path
):
    print("=" * 70)
    print("RainAI Phase 1: Spatial Preprocessing & Semantic Conditioning")
    print("=" * 70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    all_raw_files = []
    
    for raw_dir in raw_data_dirs:
        raw_path = Path(raw_dir)
        if raw_path.exists():
            for ext in ["*.flac", "*.wav", "*.ogg", "*.mp3"]:
                all_raw_files.extend(list(raw_path.glob(f"**/{ext}")))
                
    # Deduplicate files
    all_raw_files = sorted(list(set(all_raw_files)))
    print(f"[*] Found {len(all_raw_files)} raw audio files across source directories.")
    
    if not all_raw_files:
        print("[!] No raw audio files found to process.")
        return

    # Clean output directory of any old chunks to ensure fresh sync
    for old_flac in output_dir.glob("*.flac"):
        try:
            old_flac.unlink()
        except Exception:
            pass
            
    total_chunks = 0
    start_time = time.time()
    
    print("\n--- Step 1: 48kHz Resampling & Ambisonics FOA Upmixing ---")
    for idx, audio_file in enumerate(all_raw_files, 1):
        clean_name = audio_file.name.encode("ascii", "replace").decode("ascii")
        try:
            chunks_meta = process_audio_file(audio_file, output_dir=output_dir)
            total_chunks += len(chunks_meta)
            print(f"[{idx}/{len(all_raw_files)}] {clean_name} -> {len(chunks_meta)} FOA chunks generated.")
        except Exception as e:
            print(f"[!] Error processing {clean_name}: {e}")

    elapsed_upmix = time.time() - start_time
    print(f"\n[+] Generated {total_chunks} total 4-channel 5-second chunks in {elapsed_upmix:.2f}s.")

    print("\n--- Step 2: Semantic Conditioning Extraction & Manifest Generation ---")
    manifest = generate_manifest_for_directory(output_dir, manifest_path)
    print(f"[+] Manifest created at {manifest_path} containing {len(manifest)} tagged chunks.")
    
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print(f"Processed Directory: {output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Total Chunks: {total_chunks} ({total_chunks * 5.0 / 60.0:.2f} minutes of training data)")
    print("=" * 70)


if __name__ == "__main__":
    # Strictly ingest only compatibly-licensed datasets (CC0, CC-BY, Public Domain)
    # Exclude all NonCommercial (NC) data
    raw_dirs = [
        PROJECT_ROOT / "Data" / "rain"
    ]
    out_dir = PROJECT_ROOT / "Data" / "processed"
    manifest_file = out_dir / "manifest.json"
    
    run_preprocessing(raw_dirs, out_dir, manifest_file)
