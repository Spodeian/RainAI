"""
Acoustic feature extraction, silence trimming, 48kHz resampling,
and dataset manifest generation for the RainAI spatial audio corpus.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import math
import numpy as np
import scipy.signal as signal
import soundfile as sf
from tqdm import tqdm


class AcousticFeatureExtractor:
    """
    Extracts physical acoustic parameters from raw rain recordings:
    - Rain rate / intensity proxy (RMS and power scaling)
    - High-frequency energy ratio (> 2 kHz)
    - Droplet impact density (detected transients per second)
    - Spectral centroid, roll-off, and surface material classification
    """
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate

    def compute_features(
        self,
        audio: np.ndarray,
        filename: Optional[str] = None
    ) -> Dict[str, Union[float, str, List[float]]]:
        """
        Processes multi-channel or mono audio [channels, samples] or [samples].
        Returns normalized physical descriptors with semantic and acoustic grounding.
        """
        if audio.ndim == 2:
            mono = np.mean(audio, axis=0)
        else:
            mono = audio

        num_samples = len(mono)
        duration_sec = max(num_samples / self.sample_rate, 1e-4)

        # 1. RMS Energy & Rain Intensity Calibration (in Decibels)
        rms = float(np.sqrt(np.mean(mono ** 2) + 1e-12))
        rms_db = 20.0 * np.log10(rms + 1e-12)
        # Dynamic energy proxy: maps -52 dB (whisper/drizzle) to -12 dB (extreme storm)
        measured_rate = float(np.clip((rms_db - (-52.0)) / (-12.0 - (-52.0)), 0.05, 1.0))

        # 2. Spectral Analysis (FFT)
        n_fft = 2048
        hop_length = 512
        freqs, times, stft_mag = signal.stft(
            mono,
            fs=self.sample_rate,
            nperseg=n_fft,
            noverlap=n_fft - hop_length
        )
        stft_power = np.abs(stft_mag) ** 2
        total_power = np.sum(stft_power) + 1e-12

        # High-frequency power ratio (> 2000 Hz)
        hf_mask = freqs >= 2000.0
        hf_power = np.sum(stft_power[hf_mask, :])
        hf_ratio = float(np.clip(hf_power / total_power, 0.0, 1.0))

        # Spectral Centroid (Hz)
        freq_mesh = freqs[:, np.newaxis]
        centroid = float(np.sum(freq_mesh * stft_power) / total_power)

        # Spectral Roll-off (85% energy frequency)
        cumulative_power = np.cumsum(np.sum(stft_power, axis=1))
        cutoff = 0.85 * cumulative_power[-1]
        rolloff_idx = np.searchsorted(cumulative_power, cutoff)
        rolloff_freq = float(freqs[min(rolloff_idx, len(freqs) - 1)])

        # Spectral Flatness (Wiener entropy)
        power_spectrum = np.mean(stft_power, axis=1) + 1e-12
        geom_mean = np.exp(np.mean(np.log(power_spectrum)))
        arith_mean = np.mean(power_spectrum)
        spectral_flatness = float(np.clip(geom_mean / arith_mean, 0.0, 1.0))

        # 3. Droplet Impact Density (Transient Detection)
        # High-pass filter above 3 kHz to isolate droplet splash clicks
        sos = signal.butter(4, 3000, btype="highpass", fs=self.sample_rate, output="sos")
        filtered_clicks = signal.sosfilt(sos, mono)
        abs_clicks = np.abs(filtered_clicks)

        # Threshold: 3.5x standard deviation
        thresh = 3.5 * np.std(abs_clicks) + 1e-6
        min_distance_samples = int(self.sample_rate * 0.005)  # min 5ms between impacts
        peaks, _ = signal.find_peaks(abs_clicks, height=thresh, distance=min_distance_samples)
        drops_per_sec = len(peaks) / duration_sec

        # Droplet density normalization: gate against thunder rumble / low-frequency drone
        if hf_ratio < 0.12 and centroid < 1500.0:
            droplet_density_norm = float(np.clip((drops_per_sec / 80.0) * (hf_ratio * 4.0), 0.01, 0.20))
        else:
            droplet_density_norm = float(np.clip(drops_per_sec / 80.0, 0.02, 1.0))

        # 4. Semantic Surface Resolution & Rain Rate Anchoring
        fn_lower = filename.lower() if filename else ""
        compound_surfaces = None

        # Priority 0: Compound Multi-Surface Soundscapes
        if "compound_urban_balcony" in fn_lower:
            surface_tag = "compound_urban_balcony"
            surface_idx = 3  # predominant pavement
            # 0: tin, 1: leaves, 2: pine, 3: pavement, 4: water_deep, 5: puddle, 6: canvas, 7: glass, 8: wood
            compound_surfaces = [0.15, 0.0, 0.0, 0.50, 0.0, 0.0, 0.0, 0.35, 0.0]
            intensity_proxy = float(np.clip(0.40 + 0.30 * measured_rate, 0.25, 0.85))

        elif "compound_forest_camp" in fn_lower:
            surface_tag = "compound_forest_camp"
            surface_idx = 1  # predominant foliage
            compound_surfaces = [0.0, 0.45, 0.20, 0.0, 0.0, 0.0, 0.35, 0.0, 0.0]
            intensity_proxy = float(np.clip(0.35 + 0.35 * measured_rate, 0.20, 0.80))

        elif "compound_porch_storm" in fn_lower:
            surface_tag = "compound_porch_storm"
            surface_idx = 8  # predominant wood_deck
            compound_surfaces = [0.25, 0.0, 0.0, 0.0, 0.0, 0.35, 0.0, 0.0, 0.40]
            intensity_proxy = float(np.clip(0.60 + 0.30 * measured_rate, 0.50, 0.95))

        # Priority 1: Direct Glass Indicators
        elif any(k in fn_lower for k in ["glass", "window", "pane", "skylight"]):
            surface_tag = "glass"
            surface_idx = 7  # glass_window
            intensity_proxy = float(np.clip(0.35 + 0.30 * measured_rate, 0.25, 0.85))

        # Priority 2: Direct Roof / Tin Indicators
        elif any(k in fn_lower for k in ["roof", "tin", "metal", "gutter", "corrugated"]):
            surface_tag = "tin"
            surface_idx = 0  # tin
            intensity_proxy = float(np.clip(0.45 + 0.30 * measured_rate, 0.35, 0.85))

        # Priority 3: Direct Canvas / Tent Indicators
        elif any(k in fn_lower for k in ["canvas", "tent", "umbrella", "awning", "tarp"]):
            surface_tag = "canvas"
            surface_idx = 6  # canvas_tent
            intensity_proxy = float(np.clip(0.30 + 0.40 * measured_rate, 0.20, 0.80))

        # Priority 4: Direct Wood Deck Indicators
        elif any(k in fn_lower for k in ["wood_deck", "deck", "timber", "plank", "patio"]) or (
            "wood" in fn_lower and not any(f in fn_lower for f in ["woods", "forest", "tree", "jungle"])
        ):
            surface_tag = "wood_deck"
            surface_idx = 8  # wood_deck
            intensity_proxy = float(np.clip(0.35 + 0.35 * measured_rate, 0.20, 0.80))

        # Priority 5: Direct Pine Needles Indicators
        elif any(k in fn_lower for k in ["pine", "needle", "needles", "conifer"]):
            surface_tag = "pine_needles"
            surface_idx = 2  # pine_needles
            intensity_proxy = float(np.clip(0.25 + 0.35 * measured_rate, 0.15, 0.75))

        # Priority 6: Direct Deep Water Indicators
        elif any(k in fn_lower for k in ["water_deep", "deep_water", "lake", "pond", "ocean", "river", "pool"]):
            surface_tag = "water_deep"
            surface_idx = 4  # water_deep
            intensity_proxy = float(np.clip(0.30 + 0.45 * measured_rate, 0.20, 0.90))

        # Priority 7: Direct Shallow Puddle Indicators
        elif any(k in fn_lower for k in ["puddle", "shallow", "splash", "curb"]):
            surface_tag = "puddle_shallow"
            surface_idx = 5  # puddle_shallow
            intensity_proxy = float(np.clip(0.40 + 0.35 * measured_rate, 0.25, 0.85))

        # Priority 8: Direct Foliage / Vegetation Indicators
        elif any(k in fn_lower for k in ["lea", "leaf", "leaves", "jungle", "forest", "tree", "canopy", "plants", "garden", "foliage", "bush"]):
            surface_tag = "foliage"
            surface_idx = 1  # leaves_broad
            intensity_proxy = float(np.clip(0.25 + 0.40 * measured_rate, 0.15, 0.75))

        # Priority 9: Known Rain Categories on Pavement / Open Ground
        elif any(k in fn_lower for k in ["drizzle", "downpour", "steady", "thunder", "storm", "pavement", "street", "suburb", "london", "barish", "rainfall"]):
            surface_tag = "pavement"
            surface_idx = 3  # pavement
            if "drizzle" in fn_lower:
                intensity_proxy = float(np.clip(0.10 + 0.10 * measured_rate, 0.05, 0.25))
            elif "downpour" in fn_lower or "heavy" in fn_lower:
                intensity_proxy = float(np.clip(0.75 + 0.25 * measured_rate, 0.70, 1.00))
            elif "steady" in fn_lower:
                intensity_proxy = float(np.clip(0.40 + 0.20 * measured_rate, 0.30, 0.65))
            elif "thunder" in fn_lower or "storm" in fn_lower:
                intensity_proxy = float(np.clip(0.60 + 0.35 * measured_rate, 0.50, 1.00))
            else:
                intensity_proxy = measured_rate

        # Fallback: Calibrated Acoustic Heuristics when filename has no clues
        else:
            if centroid > 5500 and droplet_density_norm > 0.4 and hf_ratio > 0.65:
                surface_tag = "glass"
                surface_idx = 7
            elif centroid > 4000 and hf_ratio > 0.60:
                surface_tag = "tin"
                surface_idx = 0
            elif 1500 <= centroid <= 3200 and hf_ratio < 0.45:
                surface_tag = "foliage"
                surface_idx = 1
            elif spectral_flatness > 0.10:
                surface_tag = "pavement"
                surface_idx = 3
            elif centroid < 1500 and hf_ratio < 0.25:
                surface_tag = "canvas"
                surface_idx = 6
            else:
                surface_tag = "pavement"
                surface_idx = 3
            intensity_proxy = measured_rate

        # 5. Build standard 41-dim physical parameter vector
        # [base_sliders (10), surfaces (9), wind (4), side_sounds (18)]
        base_sliders = [
            intensity_proxy,
            0.2,   # default wind speed
            0.5,   # wind azimuth
            surface_idx / 8.0,
            intensity_proxy * 0.6,  # runoff proxy
            0.5,   # temp
            0.8,   # humidity
            0.2,   # pitch
            0.3,   # distance
            0.1    # enclosure
        ]
        if compound_surfaces is not None:
            surfaces = compound_surfaces
        else:
            surfaces = [0.0] * 9
            surfaces[surface_idx] = 1.0
        wind = [0.2, 0.1, 0.1, 0.05]
        side_sounds = [0.0] * 18

        physical_params = base_sliders + surfaces + wind + side_sounds

        return {
            "duration_sec": round(duration_sec, 3),
            "rms_energy": round(rms, 6),
            "rain_rate": round(intensity_proxy, 4),
            "droplet_density": round(droplet_density_norm, 4),
            "drops_per_second": round(drops_per_sec, 2),
            "high_freq_ratio": round(hf_ratio, 4),
            "spectral_centroid": round(centroid, 1),
            "spectral_rolloff": round(rolloff_freq, 1),
            "spectral_flatness": round(spectral_flatness, 4),
            "surface_tag": surface_tag,
            "surface_idx": surface_idx,
            "physical_params": physical_params
        }


class AudioProcessor:
    """
    Standardizes, slices, extracts acoustic features, and builds manifests
    for raw downloaded audio.
    """
    def __init__(
        self,
        target_sample_rate: int = 48000,
        chunk_duration_sec: float = 5.0,
        overlap_sec: float = 1.0,
        silence_thresh_db: float = -60.0
    ):
        self.target_sr = target_sample_rate
        self.chunk_duration = chunk_duration_sec
        self.chunk_samples = int(target_sample_rate * chunk_duration_sec)
        self.overlap_samples = int(target_sample_rate * overlap_sec)
        self.step_samples = self.chunk_samples - self.overlap_samples
        self.silence_thresh_db = silence_thresh_db
        self.extractor = AcousticFeatureExtractor(sample_rate=target_sample_rate)

    def trim_silence(self, audio: np.ndarray) -> np.ndarray:
        """Trims leading and trailing silence below silence_thresh_db."""
        if audio.ndim == 2:
            mono = np.mean(audio, axis=0)
        else:
            mono = audio

        db = 20 * np.log10(np.abs(mono) + 1e-12)
        non_silent = np.where(db > self.silence_thresh_db)[0]
        if len(non_silent) == 0:
            return audio

        start = non_silent[0]
        end = non_silent[-1] + 1
        return audio[:, start:end] if audio.ndim == 2 else audio[start:end]

    def resample_and_standardize(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        """Resamples to target_sr and ensures 4-channel FOA format."""
        # Ensure 2D shape [channels, samples]
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]

        # Resample if needed
        if orig_sr != self.target_sr:
            gcd = math.gcd(self.target_sr, orig_sr)
            up = self.target_sr // gcd
            down = orig_sr // gcd
            audio = signal.resample_poly(audio, up, down, axis=-1)

        # Standardize channel layout to 4-channel FOA
        # Channels: W (omni), Y (left-right), Z (up-down), X (front-back)
        num_ch = audio.shape[0]
        if num_ch == 1:
            # Mono to FOA: W = mono, X, Y, Z = simulated mild diffuse
            w = audio[0]
            y = np.zeros_like(w)
            z = np.zeros_like(w)
            x = np.zeros_like(w)
            audio = np.stack([w, y, z, x], axis=0)
        elif num_ch == 2:
            # Stereo to FOA: W = (L+R)/sqrt(2), Y = (L-R)/sqrt(2), Z = 0, X = 0
            l = audio[0]
            r = audio[1]
            w = (l + r) * 0.7071
            y = (l - r) * 0.7071
            z = np.zeros_like(w)
            x = np.zeros_like(w)
            audio = np.stack([w, y, z, x], axis=0)
        elif num_ch > 4:
            audio = audio[:4, :]

        return audio.astype(np.float32)

    def process_file(
        self,
        input_path: Path,
        output_dir: Path
    ) -> List[Tuple[Path, Dict]]:
        """
        Loads, resamples, slices into chunks, extracts acoustic features,
        and saves standardized 4-channel 48kHz WAV slices.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []

        try:
            raw_audio, orig_sr = sf.read(str(input_path), dtype="float32", always_2d=False)
        except Exception as e:
            print(f"[!] Could not read audio file {input_path.name}: {e}")
            return []

        # If soundfile returned [samples, channels], transpose to [channels, samples]
        if raw_audio.ndim == 2 and raw_audio.shape[0] > raw_audio.shape[1]:
            raw_audio = raw_audio.T

        # Standardize sample rate and channels
        audio = self.resample_and_standardize(raw_audio, orig_sr)
        audio = self.trim_silence(audio)

        total_samples = audio.shape[-1]
        if total_samples < self.chunk_samples:
            # Pad short audio with reflection
            pad_len = self.chunk_samples - total_samples
            audio = np.pad(audio, ((0, 0), (0, pad_len)), mode="reflect")
            total_samples = audio.shape[-1]

        # Slice into chunks
        chunk_idx = 0
        stem = input_path.stem
        start = 0

        while start + self.chunk_samples <= total_samples:
            chunk = audio[:, start:start + self.chunk_samples]
            features = self.extractor.compute_features(chunk, filename=input_path.name)
            features["source_file"] = input_path.name
            features["chunk_index"] = chunk_idx

            chunk_filename = f"{stem}_slice{chunk_idx:03d}.wav"
            chunk_path = output_dir / chunk_filename

            # Save as 48kHz 24-bit PCM WAV (or float32)
            sf.write(str(chunk_path), chunk.T, self.target_sr, subtype="FLOAT")
            results.append((chunk_path, features))

            chunk_idx += 1
            start += self.step_samples

        return results

    def build_dataset_manifest(
        self,
        raw_dir: Path,
        processed_dir: Path,
        manifest_output_path: Path
    ) -> Dict[str, dict]:
        """
        Scans raw_dir for audio, processes each into processed_dir,
        and writes rain_corpus_manifest.json.
        """
        raw_dir = Path(raw_dir)
        processed_dir = Path(processed_dir)
        processed_dir.mkdir(parents=True, exist_ok=True)

        audio_extensions = {".wav", ".flac", ".mp3", ".ogg"}
        raw_files = [p for p in raw_dir.glob("**/*") if p.suffix.lower() in audio_extensions]

        manifest = {}
        print(f"[*] Processing {len(raw_files)} raw audio files into {processed_dir}...")
        for file_path in tqdm(raw_files, desc="Processing Audio"):
            # Skip RIR files which are used for convolution, not direct audio chunks
            if "rir" in file_path.name.lower():
                continue

            processed_chunks = self.process_file(file_path, processed_dir)
            for chunk_path, features in processed_chunks:
                manifest[chunk_path.stem] = features

        # Save manifest JSON
        with open(manifest_output_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print(f"[+] Generated manifest with {len(manifest)} chunks at: {manifest_output_path}")
        return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RainAI Acoustic Audio Processor")
    parser.add_argument("--raw-dir", type=str, default="./data/raw_audio", help="Raw input directory")
    parser.add_argument("--processed-dir", type=str, default="./data/processed_audio", help="Processed slices output directory")
    parser.add_argument("--manifest-path", type=str, default="./data/rain_corpus_manifest.json", help="Manifest output file")
    args = parser.parse_args()

    processor = AudioProcessor()
    processor.build_dataset_manifest(
        raw_dir=Path(args.raw_dir),
        processed_dir=Path(args.processed_dir),
        manifest_output_path=Path(args.manifest_path)
    )
