"""
PyTorch Dataset and DataLoader for 4-channel FOA Rain Audio with Multi-Modal Conditioning.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.spatial_upmix import apply_spatial_rir_convolution

TARGET_SAMPLE_RATE = 48000
CHUNK_DURATION_SEC = 5.0
CHUNK_SAMPLES = int(TARGET_SAMPLE_RATE * CHUNK_DURATION_SEC)
CLAP_EMBED_DIM = 512
NUM_BASE_SLIDERS = 10  # [intensity, wind_speed, wind_azimuth, surface_material, runoff, temp, humidity, pitch, distance, enclosure]
NUM_SURFACES = 9       # [tin, leaves_broad, pine_needles, pavement, water_deep, puddle_shallow, canvas_tent, glass_window, wood_deck]
NUM_WIND_PARAMS = 4    # [wind_speed, wind_gustiness, wind_turbulence, wind_howl]
NUM_SIDE_SOUNDS = 18   # [insects(3), birds(3), fireplace(4), thunder(4), traffic(4)]
NUM_PHYSICAL_PARAMS = NUM_BASE_SLIDERS + NUM_SURFACES + NUM_WIND_PARAMS + NUM_SIDE_SOUNDS  # 41 params
TOTAL_CONDITION_DIM = CLAP_EMBED_DIM + NUM_PHYSICAL_PARAMS + 1  # 512 + 41 + 1 = 554

_PINK_NOISE_CACHE: Dict[int, torch.Tensor] = {}

def generate_pink_noise(samples: int) -> torch.Tensor:
    """Generates normalized pink noise (1/f spectral decay) via PyTorch FFT."""
    decay = _PINK_NOISE_CACHE.get(samples)
    if decay is None:
        freqs = torch.fft.rfftfreq(samples)
        freqs[0] = 1.0
        decay = (1.0 / torch.sqrt(freqs)).float()
        _PINK_NOISE_CACHE[samples] = decay
    white = torch.randn(1, samples, dtype=torch.float32)
    pink = torch.fft.irfft(torch.fft.rfft(white) * decay, n=samples)
    peak = torch.max(torch.abs(pink)) + 1e-8
    return pink / peak


def select_active_crop(
    audio: np.ndarray,
    chunk_samples: int,
    min_active_ratio: float = 0.25,
    max_attempts: int = 5
) -> np.ndarray:
    """
    Selects a temporal window containing active acoustic signal via VAD / energy thresholding.
    Optimized with safe edge padding and fallback handling.
    """
    curr_samples = audio.shape[-1]
    if curr_samples <= chunk_samples:
        if curr_samples < chunk_samples:
            return np.pad(audio, ((0, 0), (0, chunk_samples - curr_samples)), mode="reflect")
        return audio

    max_start = curr_samples - chunk_samples
    w_ch = audio[0]
    rms_full = np.sqrt(np.mean(w_ch ** 2) + 1e-12)
    silence_thresh = max(1e-4, 0.10 * rms_full)

    best_start = 0
    best_score = -1.0

    for _ in range(max_attempts):
        start = int(np.random.randint(0, max_start + 1))
        cand = w_ch[start : start + chunk_samples]

        block_size = min(2400, chunk_samples)
        num_blocks = len(cand) // block_size
        if num_blocks > 1:
            blocks = cand[: num_blocks * block_size].reshape(num_blocks, block_size)
            block_rms = np.sqrt(np.mean(blocks ** 2, axis=1) + 1e-12)
            active_ratio = np.mean(block_rms > silence_thresh)
            score = float(np.sum(block_rms))
        else:
            active_ratio = 1.0
            score = 1.0

        if active_ratio >= min_active_ratio:
            return audio[:, start : start + chunk_samples]

        if score > best_score:
            best_score = score
            best_start = start

    return audio[:, best_start : best_start + chunk_samples]


class RainSpatialDataset(Dataset):
    """
    Dataset yielding:
        - audio_foa: Tensor of shape (4, CHUNK_SAMPLES)
        - conditioning_u: Tensor of shape (CLAP_EMBED_DIM + NUM_PHYSICAL_PARAMS + 1,) -> 554 dimensions
            [clap_embedding (512), physical_sliders (41), drift_scale (1)]
        - noise_ref: Tensor of shape (1, CHUNK_SAMPLES) (tunable colored noise input)
    """
    def __init__(
        self,
        data_dir: Union[str, Path],
        manifest_path: Optional[Union[str, Path]] = None,
        sample_rate: int = TARGET_SAMPLE_RATE,
        chunk_samples: int = CHUNK_SAMPLES,
        transform: Optional[callable] = None,
        split: Optional[str] = None,
        val_ratio: float = 0.15,
        seed: int = 42,
        augment_yaw: bool = False,
        cache_in_memory: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.chunk_samples = chunk_samples
        self.transform = transform
        self.split = split
        self.augment_yaw = augment_yaw
        self.cache_in_memory = cache_in_memory
        self._audio_cache: Dict[Path, np.ndarray] = {}
        
        # Discover all .flac or .wav files
        all_files = sorted(list(self.data_dir.glob("**/*.flac")) + list(self.data_dir.glob("**/*.wav")))
        
        # Deterministic Train/Val/Test Split
        if split in ["train", "val"]:
            rng = np.random.RandomState(seed)
            indices = rng.permutation(len(all_files))
            num_val = max(1, int(len(all_files) * val_ratio))
            if split == "val":
                selected_indices = indices[:num_val]
            else: # train
                selected_indices = indices[num_val:]
            self.audio_files = [all_files[i] for i in sorted(selected_indices)]
        else:
            self.audio_files = all_files
        
        # Load or initialize manifest metadata
        self.metadata: Dict[str, dict] = {}
        if manifest_path and Path(manifest_path).exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            # Check default candidate paths for manifest.json and rain_corpus_manifest.json
            cand_paths = [
                self.data_dir / "manifest.json",
                self.data_dir / "rain_corpus_manifest.json",
                self.data_dir.parent / "rain_corpus_manifest.json",
                self.data_dir.parent / "Data" / "rain_corpus_manifest.json",
                Path("Data/processed/manifest.json"),
                Path("Data/rain_corpus_manifest.json"),
                Path("data/processed/manifest.json"),
                Path("data/rain_corpus_manifest.json"),
            ]
            for c in cand_paths:
                if c.exists():
                    try:
                        with open(c, "r", encoding="utf-8") as f:
                            self.metadata = json.load(f)
                            break
                    except Exception:
                        pass

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_path = self.audio_files[idx]
        
        # Read audio file from cache or disk
        if self.cache_in_memory and file_path in self._audio_cache:
            audio = self._audio_cache[file_path].copy()
        else:
            audio, sr = sf.read(str(file_path), dtype="float32", always_2d=True)
            audio = audio.T  # Shape: (channels, samples)
            if self.cache_in_memory:
                self._audio_cache[file_path] = audio.copy()
        
        # Ensure 4-channel FOA
        if audio.shape[0] < 4:
            pad_ch = np.zeros((4 - audio.shape[0], audio.shape[1]), dtype=np.float32)
            audio = np.concatenate([audio, pad_ch], axis=0)
        elif audio.shape[0] > 4:
            audio = audio[:4, :]
            
        # Select active acoustic window via VAD / energy thresholding
        audio = select_active_crop(audio, self.chunk_samples)

        # Retrieve or synthesize multi-modal conditioning u
        stem = file_path.stem
        item_meta = self.metadata.get(stem, {})
        
        # 1. CLAP Embedding (512-dim)
        if "clap_embedding" in item_meta:
            clap_embed = np.array(item_meta["clap_embedding"], dtype=np.float32)
            if len(clap_embed) < CLAP_EMBED_DIM:
                clap_embed = np.pad(clap_embed, (0, CLAP_EMBED_DIM - len(clap_embed)))
            elif len(clap_embed) > CLAP_EMBED_DIM:
                clap_embed = clap_embed[:CLAP_EMBED_DIM]
        else:
            # Default zero-centered normalized embedding
            clap_embed = np.random.randn(CLAP_EMBED_DIM).astype(np.float32)
            clap_embed /= (np.linalg.norm(clap_embed) + 1e-8)
            
        # 2. Physical Parameters (41-dim)
        if "physical_params" in item_meta and len(item_meta["physical_params"]) == NUM_PHYSICAL_PARAMS:
            physical_params = np.array(item_meta["physical_params"], dtype=np.float32)
        else:
            # Derive plausible physical sliders from metadata or audio characteristics
            rms = np.sqrt(np.mean(audio[0] ** 2) + 1e-12)
            intensity = np.clip((rms * 10.0), 0.05, 1.0)
            wind_speed = np.clip(np.random.uniform(0.05, 0.7), 0.0, 1.0)
            wind_azimuth = np.clip(np.random.uniform(0.0, 1.0), 0.0, 1.0) # normalized [0, 1] for [-pi, pi]
            surface_material = float(np.random.choice([0.0, 0.33, 0.66, 1.0])) # legacy material
            runoff = np.clip(intensity * np.random.uniform(0.4, 1.0), 0.0, 1.0)
            temperature = np.clip(np.random.uniform(0.3, 0.8), 0.0, 1.0) # -10C to 40C
            humidity = np.clip(np.random.uniform(0.6, 0.99), 0.0, 1.0)
            pitch_angle = np.clip(np.random.uniform(0.0, 0.5), 0.0, 1.0) # 0 to 90 deg
            distance = np.clip(np.random.uniform(0.1, 0.8), 0.0, 1.0)
            enclosure = float(np.random.choice([0.0, 0.2, 0.8, 1.0]))
            
            base_sliders = [
                intensity, wind_speed, wind_azimuth, surface_material, runoff,
                temperature, humidity, pitch_angle, distance, enclosure
            ]
            
            # Continuous mixture over 9 surfaces (Dirichlet distribution for partition of unity)
            surface_weights = np.random.dirichlet(np.ones(NUM_SURFACES)).astype(np.float32).tolist()
            
            # Wind dynamics (4)
            wind_dynamics = [
                float(wind_speed),
                float(np.random.uniform(0.0, 0.8)),  # gustiness
                float(np.random.uniform(0.0, 0.6)),  # turbulence
                float(np.random.uniform(0.0, 0.5)),  # howl resonance
            ]
            
            # Spatialized side sounds (18)
            # Insects (3): [density, proximity, azimuth]
            insects = [float(np.random.uniform(0.0, 0.3)), float(np.random.uniform(0.2, 1.0)), float(np.random.uniform(0.0, 1.0))]
            # Birds (3): [activity, proximity, elevation]
            birds = [float(np.random.uniform(0.0, 0.2)), float(np.random.uniform(0.3, 1.0)), float(np.random.uniform(0.2, 0.8))]
            # Fireplace (4): [intensity, crackle_rate, azimuth, elevation]
            fireplace = [float(np.random.choice([0.0, 0.0, 0.5])), float(np.random.uniform(0.1, 0.9)), float(np.random.uniform(0.0, 1.0)), 0.0]
            # Thunder (4): [proximity, rumble_length, azimuth, elevation]
            thunder = [float(np.random.choice([0.0, 0.0, 0.8])), float(np.random.uniform(0.2, 1.0)), float(np.random.uniform(0.0, 1.0)), float(np.random.uniform(0.5, 1.0))]
            # Traffic (4): [distance, wetness, azimuth_start, azimuth_end]
            traffic = [float(np.random.uniform(0.4, 1.0)), float(np.random.uniform(0.5, 1.0)), float(np.random.uniform(0.0, 0.5)), float(np.random.uniform(0.5, 1.0))]
            
            physical_params = np.array(
                base_sliders + surface_weights + wind_dynamics + insects + birds + fireplace + thunder + traffic,
                dtype=np.float32
            )

        # 3. Apply FOA Yaw Rotation Augmentation if enabled
        if self.augment_yaw:
            yaw_angle = np.random.uniform(-np.pi, np.pi)
            cos_y = np.cos(yaw_angle)
            sin_y = np.sin(yaw_angle)
            x_ch = audio[3, :].copy()
            y_ch = audio[1, :].copy()
            # Standard B-format yaw rotation:
            # X' = X*cos(phi) - Y*sin(phi)
            # Y' = X*sin(phi) + Y*cos(phi)
            audio[3, :] = x_ch * cos_y - y_ch * sin_y
            audio[1, :] = x_ch * sin_y + y_ch * cos_y
            
            # Rotate wind azimuth accordingly (index 2 in physical_params)
            yaw_norm = yaw_angle / (2.0 * np.pi)
            physical_params[2] = float((physical_params[2] + yaw_norm) % 1.0)

        # Apply Room Impulse Response (RIR) spatial convolution using enclosure & distance
        enc_val = float(physical_params[9])
        dist_val = float(physical_params[8])
        audio = apply_spatial_rir_convolution(audio, enclosure=enc_val, distance=dist_val)
        audio_tensor = torch.from_numpy(audio.astype(np.float32))

        # 4. Drift tolerance scalar in [0, 1]
        drift_scale = np.array([item_meta.get("drift_scale", 0.2)], dtype=np.float32)
        
        # Combined conditioning vector u: 512 + 41 + 1 = 554
        u = np.concatenate([clap_embed, physical_params, drift_scale], axis=0)
        u_tensor = torch.from_numpy(u)

        # 5. Tunable colored noise input (pink noise) for subtractive DDSP
        noise_tensor = generate_pink_noise(self.chunk_samples)

        sample = {
            "audio": audio_tensor,      # (4, 240000)
            "conditioning": u_tensor,   # (554,)
            "noise": noise_tensor,      # (1, 240000)
            "path": str(file_path),
            "sample_idx": idx,
            "rain_rate": torch.tensor(item_meta.get("rain_rate", float(physical_params[0])), dtype=torch.float32),
            "droplet_density": torch.tensor(item_meta.get("droplet_density", 0.5), dtype=torch.float32),
            "surface_idx": torch.tensor(item_meta.get("surface_idx", 0), dtype=torch.long),
            "high_freq_ratio": torch.tensor(item_meta.get("high_freq_ratio", 0.5), dtype=torch.float32),
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class ActiveLearningSampler(torch.utils.data.Sampler):
    """
    Active Learning / Hard Negative Mining Sampler:
    Dynamically tracks sample losses and prioritizes difficult acoustic scenarios.
    Samples with higher historical losses receive proportionally higher sampling probability.
    """
    def __init__(self, dataset_size: int, temperature: float = 1.5, min_weight: float = 0.1):
        self.dataset_size = dataset_size
        self.temperature = temperature
        self.min_weight = min_weight
        self.sample_losses = np.ones(dataset_size, dtype=np.float32)
        self.counts = np.zeros(dataset_size, dtype=np.int32)

    def update_losses(self, indices: List[int], losses: List[float]):
        for idx, loss in zip(indices, losses):
            if 0 <= idx < self.dataset_size:
                prev = self.sample_losses[idx]
                self.sample_losses[idx] = 0.7 * prev + 0.3 * float(loss)
                self.counts[idx] += 1

    def __iter__(self):
        weights = (self.sample_losses ** self.temperature) + self.min_weight
        probs = weights / np.sum(weights)
        sampled_indices = np.random.choice(
            self.dataset_size,
            size=self.dataset_size,
            replace=True,
            p=probs
        )
        return iter(sampled_indices.tolist())

    def __len__(self) -> int:
        return self.dataset_size



class StratifiedImportanceSampler(torch.utils.data.Sampler):
    """
    Decoupled Stratified & Importance Weighted Audio Sampler.
    Assigns importance weights w_i to decouple selection probability from duration,
    preventing over-sampling of long background rain and ensuring rare transient acoustic events
    (e.g. thunder strikes, heavy tin plate bursts) are sampled fairly.
    """
    def __init__(
        self,
        dataset: RainSpatialDataset,
        importance_weights: Optional[Dict[str, float]] = None,
        temperature: float = 1.0
    ):
        self.dataset = dataset
        self.num_samples = len(dataset)
        self.weights = np.ones(self.num_samples, dtype=np.float32)
        
        if importance_weights is not None:
            for i, fpath in enumerate(dataset.audio_files):
                stem = fpath.stem
                meta = dataset.metadata.get(stem, {})
                tag = meta.get("surface_tag", "default")
                self.weights[i] = importance_weights.get(tag, 1.0)
                
        self.probs = (self.weights ** temperature) / np.sum(self.weights ** temperature)

    def __iter__(self):
        indices = np.random.choice(self.num_samples, size=self.num_samples, replace=True, p=self.probs)
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples


class CurriculumCollateFn:
    """
    Dynamic Length Bucketing & Curriculum Duration Collation.
    Samples a single chunk duration per batch so that all batch elements share identical length
    (zero internal batch padding), while dynamically varying lengths across batches and scheduling
    longer chunks as training progresses.
    """
    def __init__(
        self,
        chunk_sizes: Tuple[int, ...] = (48000, 96000, 240000),
        chunk_weights: Tuple[float, ...] = (0.6, 0.3, 0.1),
        use_curriculum: bool = False,
    ):
        self.chunk_sizes = list(chunk_sizes)
        self.chunk_weights = list(chunk_weights)
        self.use_curriculum = use_curriculum
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        if self.use_curriculum:
            p_48k = max(0.20, 0.70 - 0.08 * epoch)
            p_96k = min(0.45, 0.25 + 0.03 * epoch)
            p_240k = max(0.05, 1.0 - p_48k - p_96k)
            tot = p_48k + p_96k + p_240k
            self.chunk_weights = [p_48k / tot, p_96k / tot, p_240k / tot]

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if not self.use_curriculum:
            return torch.utils.data.dataloader.default_collate(batch)

        target_chunk = int(np.random.choice(self.chunk_sizes, p=self.chunk_weights))

        audios = []
        noises = []
        for item in batch:
            raw_audio = item["audio"].numpy()
            cropped = select_active_crop(raw_audio, target_chunk)
            audios.append(torch.from_numpy(cropped))
            noises.append(generate_pink_noise(target_chunk).squeeze(0))

        return {
            "audio": torch.stack(audios, dim=0),
            "conditioning": torch.stack([b["conditioning"] for b in batch], dim=0),
            "noise": torch.stack(noises, dim=0).unsqueeze(1),
            "path": [b["path"] for b in batch],
            "sample_idx": torch.tensor([b["sample_idx"] for b in batch], dtype=torch.long),
            "rain_rate": torch.stack([b["rain_rate"] for b in batch], dim=0),
            "droplet_density": torch.stack([b["droplet_density"] for b in batch], dim=0),
            "surface_idx": torch.stack([b["surface_idx"] for b in batch], dim=0),
            "high_freq_ratio": torch.stack([b["high_freq_ratio"] for b in batch], dim=0),
        }


def create_dataloader(
    data_dir: Union[str, Path],
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    manifest_path: Optional[Union[str, Path]] = None,
    split: Optional[str] = None,
    val_ratio: float = 0.15,
    seed: int = 42,
    augment_yaw: bool = False,
    use_active_learning: bool = False,
    return_sampler: bool = False,
    chunk_curriculum: bool = False,
    chunk_sizes: Tuple[int, ...] = (48000, 96000, 240000),
    chunk_weights: Tuple[float, ...] = (0.6, 0.3, 0.1),
    importance_weights: Optional[Dict[str, float]] = None,
    pin_memory: bool = False,
) -> Union[DataLoader, Tuple[DataLoader, Optional[torch.utils.data.Sampler]]]:
    """Helper creating DataLoader with standard collation, splits, active learning, and chunk curriculum support."""
    dataset = RainSpatialDataset(
        data_dir=data_dir,
        manifest_path=manifest_path,
        split=split,
        val_ratio=val_ratio,
        seed=seed,
        augment_yaw=augment_yaw
    )
    
    # Validation split is always evaluated on pristine full 5.0s audio chunks
    enable_curriculum = chunk_curriculum and (split == "train" or split is None)
    collate_fn = CurriculumCollateFn(
        chunk_sizes=chunk_sizes,
        chunk_weights=chunk_weights,
        use_curriculum=enable_curriculum
    )

    sampler = None
    if use_active_learning and (split == "train" or split is None):
        sampler = ActiveLearningSampler(len(dataset))
    elif importance_weights is not None and (split == "train" or split is None):
        sampler = StratifiedImportanceSampler(dataset, importance_weights=importance_weights)

    if sampler is not None:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            collate_fn=collate_fn
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            collate_fn=collate_fn
        )
        
    # Attach collate_fn to dataloader for epoch scheduling
    loader.curriculum_collate = collate_fn

    if return_sampler:
        return loader, sampler
    return loader
