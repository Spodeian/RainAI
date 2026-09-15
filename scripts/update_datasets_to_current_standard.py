"""
Batch dataset updater: Migrates and enriches all existing datasets in data/processed
and raw directories to the current RainAI standard (48kHz FOA, physical acoustic features,
41-dimensional parameter vectors, and multi-modal manifest).
"""

import sys
from pathlib import Path
import json
import time
import numpy as np
import soundfile as sf
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.audio_processor import AcousticFeatureExtractor
from data.semantic_tag import SemanticAudioTagger


def update_datasets_to_standard(
    processed_dir: Path,
    manifest_paths: list,
    sample_rate: int = 48000
) -> dict:
    print("=" * 70)
    print("RainAI Dataset Standard Migration & Feature Extraction")
    print("=" * 70)

    extractor = AcousticFeatureExtractor(sample_rate=sample_rate)
    tagger = SemanticAudioTagger(use_neural_clap=False)

    # Load existing manifest if available to retain cached CLAP embeddings
    existing_manifest = {}
    primary_manifest = manifest_paths[0]
    if primary_manifest.exists():
        try:
            with open(primary_manifest, "r", encoding="utf-8") as f:
                existing_manifest = json.load(f)
            print(f"[*] Loaded existing manifest with {len(existing_manifest)} entries from {primary_manifest.name}")
        except Exception as e:
            print(f"[!] Could not load existing manifest: {e}")

    audio_files = sorted(list(processed_dir.glob("*.flac")) + list(processed_dir.glob("*.wav")))
    print(f"[*] Found {len(audio_files)} audio chunks in {processed_dir}")

    updated_manifest = {}
    surface_counts = {}
    rain_rates = []
    droplet_densities = []

    start_time = time.time()
    for f in tqdm(audio_files, desc="Migrating Datasets"):
        stem = f.stem
        # Read audio chunk
        try:
            audio, sr = sf.read(str(f), dtype="float32", always_2d=False)
            if audio.ndim == 2 and audio.shape[0] > audio.shape[1]:
                audio = audio.T
        except Exception as e:
            print(f"[!] Error reading {f.name}: {e}")
            continue

        # Extract complete physical acoustic features
        feats = extractor.compute_features(audio, filename=f.name)

        # Retrieve or compute CLAP embedding
        prev_meta = existing_manifest.get(stem, {})
        if "clap_embedding" in prev_meta and len(prev_meta["clap_embedding"]) == 512:
            clap_emb = prev_meta["clap_embedding"]
        else:
            clap_emb = tagger.extract_embedding_from_file(f).tolist()

        drift_scale = prev_meta.get("drift_scale", 0.2)

        # Build standardized item
        item_entry = {
            "path": str(f.as_posix()),
            "filename": f.name,
            "sample_rate": sample_rate,
            "channels": 4 if audio.ndim == 2 else 1,
            "clap_embedding": clap_emb,
            "drift_scale": drift_scale,
            **feats
        }

        updated_manifest[stem] = item_entry

        # Track statistics
        s_tag = feats["surface_tag"]
        surface_counts[s_tag] = surface_counts.get(s_tag, 0) + 1
        rain_rates.append(feats["rain_rate"])
        droplet_densities.append(feats["droplet_density"])

    elapsed = time.time() - start_time

    # Save to all target manifest paths
    for m_path in manifest_paths:
        m_path.parent.mkdir(parents=True, exist_ok=True)
        with open(m_path, "w", encoding="utf-8") as out_f:
            json.dump(updated_manifest, out_f, indent=2)
        print(f"[+] Saved updated standard manifest to: {m_path}")

    print("\n" + "=" * 70)
    print("DATASET MIGRATION SUMMARY")
    print(f"Total Chunks Standardized: {len(updated_manifest)}")
    print(f"Time Elapsed: {elapsed:.2f}s ({len(updated_manifest) / max(elapsed, 0.001):.1f} chunks/sec)")
    print(f"Rain Rate Range: [{min(rain_rates):.3f}, {max(rain_rates):.3f}] (Mean: {np.mean(rain_rates):.3f})")
    print(f"Droplet Density Range: [{min(droplet_densities):.3f}, {max(droplet_densities):.3f}] (Mean: {np.mean(droplet_densities):.3f})")
    print("Surface Distribution:")
    for surf, count in sorted(surface_counts.items()):
        print(f"  - {surf:12s}: {count:4d} chunks ({count / len(updated_manifest) * 100:.1f}%)")
    print("=" * 70)

    return updated_manifest


if __name__ == "__main__":
    proc_cand = PROJECT_ROOT / "Data" / "processed"
    if not proc_cand.exists():
        proc_cand = PROJECT_ROOT / "data" / "processed"

    target_manifests = [
        proc_cand / "manifest.json",
        proc_cand.parent / "rain_corpus_manifest.json"
    ]

    update_datasets_to_standard(proc_cand, target_manifests)
