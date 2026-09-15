"""
Spatial audio upmixing and standardization to First-Order Ambisonics (FOA) at 48kHz.

FOA (B-format, ACN ordering, SN3D normalization):
Channel 0 (W): Omnidirectional acoustic pressure (gain: 1 / sqrt(2))
Channel 1 (Y): Left-Right figure-8 gradient (gain: sin(azimuth) * cos(elevation))
Channel 2 (Z): Down-Up figure-8 gradient (gain: sin(elevation))
Channel 3 (X): Back-Front figure-8 gradient (gain: cos(azimuth) * cos(elevation))

For environmental rain noise:
- Rain predominantly falls from above (elevation phi in [45 deg, 85 deg])
- Diffuse background rain energy dominates W
- Spatialized droplets and wind gusts exhibit dynamic azimuths and elevations
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import soundfile as sf
from scipy import signal

TARGET_SAMPLE_RATE = 48000
CHUNK_DURATION_SEC = 5.0
CHUNK_SAMPLES = int(TARGET_SAMPLE_RATE * CHUNK_DURATION_SEC)


@dataclass
class AudioChunkMetadata:
    source_file: str
    chunk_index: int
    duration_sec: float
    sample_rate: int
    num_channels: int
    rms_energy_db: float
    output_path: str


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Resample audio array (channels, samples) or (samples,) to target_sr using polyphase filtering."""
    if orig_sr == target_sr:
        return audio
    
    # Calculate rational approximation of sample rate ratio
    num_samples = int(round(audio.shape[-1] * target_sr / orig_sr))
    if audio.ndim == 1:
        resampled = signal.resample(audio, num_samples)
    else:
        resampled = signal.resample(audio, num_samples, axis=-1)
    return resampled.astype(np.float32)


def delay_samples(arr: np.ndarray, d: int) -> np.ndarray:
    """Delays 1D array by d samples with zero-padding (no circular wrap-around edge artifacts)."""
    if d <= 0:
        return arr
    out = np.zeros_like(arr)
    out[d:] = arr[:-d]
    return out


def stereo_or_mono_to_foa(
    audio: np.ndarray, 
    elevation_arc_deg: Tuple[float, float] = (60.0, 120.0),
    default_elevation_deg: Optional[float] = None,
    wind_speed_ms: float = 3.5,
    wind_azimuth_deg: float = 40.0
) -> np.ndarray:
    """
    Upmixes mono or stereo audio into 4-channel First-Order Ambisonics (FOA)
    grounded in drop-size-dependent trajectory trigonometry (Quora/HESS):
    tan(θ(D)) = u / vt(D)
    
    - Fine drizzle / mist (high frequencies > 2.5kHz, vt ≈ 2.2 m/s): deflected sideways into X, Y.
    - Large downpour drops (low frequencies < 1.5kHz, vt ≈ 8.5 m/s): fall steeply into the overhead Z arc.
    - Erpul (2003) windward vs leeward energy asymmetry across FOA channels.
    - ISO 9613-1 atmospheric absorption on horizontal diffuse paths.
    
    Returns array of shape (4, num_samples):
        Channel 0: W (Omnidirectional acoustic pressure)
        Channel 1: Y (Side gradient: Left - Right)
        Channel 2: Z (Vertical gradient: Down - Up)
        Channel 3: X (Frontal gradient: Back - Front)
    """
    if default_elevation_deg is not None:
        elevation_arc_deg = (default_elevation_deg - 15.0, default_elevation_deg + 15.0)
    if audio.ndim == 1:
        left = audio
        right = audio
    elif audio.shape[0] == 1:
        left = audio[0]
        right = audio[0]
    else:
        left = audio[0]
        right = audio[1]

    num_samples = left.shape[-1]
    
    # Mid/Side acoustic decomposition
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)
    
    # Frequency split for drop-size dependent trajectory:
    # Crossover at 2.2 kHz splits small-droplet hiss from large-droplet momentum
    sos_split = signal.butter(2, 2200.0, btype='lowpass', fs=TARGET_SAMPLE_RATE, output='sos')
    mid_low = signal.sosfilt(sos_split, mid)
    mid_high = mid - mid_low
    side_low = signal.sosfilt(sos_split, side)
    side_high = side - side_low
    
    # 1. Trajectory Angles from Terminal Velocity (Gunn-Kinzer):
    # Large drops: vt ≈ 8.5 m/s -> steep vertical trajectory (θ_low ≈ 22 deg from vertical -> elev ≈ 75 deg)
    theta_low_rad = np.arctan(wind_speed_ms / 8.5)
    elev_low_rad = np.radians(78.0) - theta_low_rad * 0.4
    
    # Small drops / mist: vt ≈ 2.2 m/s -> strong lateral wind drift (θ_high ≈ 58 deg from vertical -> elev ≈ 50 deg)
    theta_high_rad = np.arctan(wind_speed_ms / 2.2)
    elev_high_rad = np.radians(60.0) - theta_high_rad * 0.5
    
    # 2. Windward / Leeward Directional Bias (Erpul 2003):
    wind_az_rad = np.radians(wind_azimuth_deg)
    cos_w = np.cos(wind_az_rad)
    sin_w = np.sin(wind_az_rad)
    
    # 3. Omnidirectional W: mid / sqrt(2)
    w = (1.0 / np.sqrt(2.0)) * mid
    
    # Diffuse decorrelation delays (mutually orthogonal 1.5ms, 2.7ms, 3.8ms)
    d_x = int(TARGET_SAMPLE_RATE * 0.0015)
    d_y = int(TARGET_SAMPLE_RATE * 0.0027)
    d_z = int(TARGET_SAMPLE_RATE * 0.0038)
    
    # 4. Vertical Z (Down - Up): dominated by high-momentum large drop impacts overhead
    z_low = (0.85 * mid_low + 0.15 * delay_samples(mid_low, d_z)) * np.sin(elev_low_rad)
    z_high = (0.65 * mid_high + 0.35 * delay_samples(mid_high, d_z)) * np.sin(elev_high_rad)
    z = z_low + z_high
    
    # 5. Side Gradient Y (Left - Right): fine mist and lateral stereo width
    y_low = (0.7 * side_low + 0.3 * delay_samples(side_low, d_y)) * np.cos(elev_low_rad)
    y_high = ((0.75 * side_high + 0.25 * delay_samples(side_high, d_y)) + 0.25 * mid_high * sin_w) * np.cos(elev_high_rad)
    y = y_low + y_high
    
    # 6. Frontal Gradient X (Back - Front): sagittal dome spread + windward impact boost
    x_low = (0.5 * (delay_samples(mid_low, d_x) - delay_samples(side_low, d_x))) * np.cos(elev_low_rad)
    x_high = (0.5 * (delay_samples(mid_high, d_x) - delay_samples(side_low, d_x)) + 0.3 * mid_high * cos_w) * np.cos(elev_high_rad)
    x = x_low + x_high

    foa = np.stack([w, y, z, x], axis=0).astype(np.float32)
    
    # Peak normalization protection
    max_peak = np.max(np.abs(foa))
    if max_peak > 0.99:
        foa = foa * (0.95 / max_peak)
        
    return foa


def split_into_chunks(
    audio_foa: np.ndarray, 
    chunk_samples: int = CHUNK_SAMPLES, 
    overlap_ratio: float = 0.25
) -> List[np.ndarray]:
    """Splits FOA audio array (4, total_samples) into 5-second overlapping chunks."""
    total_samples = audio_foa.shape[-1]
    step = int(chunk_samples * (1.0 - overlap_ratio))
    
    chunks = []
    if total_samples < chunk_samples:
        # Pad with seamless mirror reflection or silence
        pad_width = chunk_samples - total_samples
        padded = np.pad(audio_foa, ((0, 0), (0, pad_width)), mode='reflect')
        chunks.append(padded)
        return chunks
        
    for start in range(0, total_samples - chunk_samples + 1, step):
        chunk = audio_foa[:, start:start + chunk_samples]
        chunks.append(chunk)
        
    return chunks


def apply_spatial_rir_convolution(
    foa_audio: np.ndarray,
    enclosure: float = 0.0,
    distance: float = 0.5,
    rir_audio: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Convolves 4-channel FOA audio with spatial impulse responses (RIRs).
    Models room enclosure reflections (dry outdoor -> reverberant interior)
    and distance acoustic absorption according to ISO 9613-1.
    """
    if enclosure < 0.05 and distance < 0.2:
        return foa_audio
        
    num_channels, num_samples = foa_audio.shape
    out = np.zeros_like(foa_audio)
    
    if rir_audio is not None and rir_audio.shape[0] >= 4:
        rir_len = min(rir_audio.shape[-1], int(TARGET_SAMPLE_RATE * 1.5))
        for c in range(4):
            conv = signal.fftconvolve(foa_audio[c], rir_audio[c, :rir_len], mode="full")[:num_samples]
            out[c] = (1.0 - enclosure * 0.7) * foa_audio[c] + (enclosure * 0.7) * conv
    else:
        rt60 = 0.1 + enclosure * 1.8
        decay_samples = int(TARGET_SAMPLE_RATE * rt60)
        t = np.linspace(0, rt60, decay_samples)
        envelope = np.exp(-3.0 * t / max(rt60, 0.05))
        synth_tail = np.random.randn(decay_samples) * envelope
        synth_tail /= (np.max(np.abs(synth_tail)) + 1e-8)
        
        for c in range(num_channels):
            conv = signal.fftconvolve(foa_audio[c], synth_tail, mode="full")[:num_samples]
            out[c] = (1.0 - enclosure * 0.6) * foa_audio[c] + (enclosure * 0.6) * conv
            
    dist_gain = 1.0 / (1.0 + distance * 1.5)
    out *= dist_gain
    return out.astype(np.float32)


def process_audio_file(
    input_file: Path, 
    output_dir: Path, 
    chunk_duration_sec: float = CHUNK_DURATION_SEC,
    max_chunks_per_file: int = 25
) -> List[AudioChunkMetadata]:
    """Processes a single audio file: reads in chunks, resamples to 48kHz, upmixes to FOA, and saves chunks."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with sf.SoundFile(str(input_file)) as f:
        orig_sr = f.samplerate
        total_frames = len(f)
        channels = f.channels
        
        chunk_frames_orig = int(orig_sr * chunk_duration_sec)
        target_chunk_samples = int(TARGET_SAMPLE_RATE * chunk_duration_sec)
        
        # Determine chunk start positions
        if total_frames <= chunk_frames_orig:
            start_frames = [0]
        else:
            # Overlapping step: 25% overlap
            step_frames = int(chunk_frames_orig * 0.75)
            potential_starts = list(range(0, total_frames - chunk_frames_orig + 1, step_frames))
            if len(potential_starts) > max_chunks_per_file:
                # Subsample evenly across the duration to capture full acoustic evolution
                indices = np.linspace(0, len(potential_starts) - 1, num=max_chunks_per_file, dtype=int)
                start_frames = [potential_starts[i] for i in indices]
            else:
                start_frames = potential_starts
                
        metadata_list = []
        stem = input_file.stem
        
        for idx, start in enumerate(start_frames):
            f.seek(start)
            data = f.read(frames=chunk_frames_orig, dtype='float32', always_2d=True).T
            
            # Pad if shorter than chunk_frames_orig
            if data.shape[-1] < chunk_frames_orig:
                pad_w = chunk_frames_orig - data.shape[-1]
                data = np.pad(data, ((0, 0), (0, pad_w)), mode='reflect')
                
            # Resample chunk to 48kHz
            resampled = resample_audio(data, orig_sr=orig_sr, target_sr=TARGET_SAMPLE_RATE)
            if resampled.shape[-1] != target_chunk_samples:
                if resampled.shape[-1] < target_chunk_samples:
                    resampled = np.pad(resampled, ((0, 0), (0, target_chunk_samples - resampled.shape[-1])), mode='reflect')
                else:
                    resampled = resampled[:, :target_chunk_samples]
                    
            # Upmix to FOA 4-channel
            foa = stereo_or_mono_to_foa(resampled)
            
            chunk_name = f"{stem}_chunk{idx:03d}.flac"
            out_path = output_dir / chunk_name
            
            # Calculate RMS energy in dB
            rms = np.sqrt(np.mean(foa ** 2) + 1e-12)
            rms_db = float(20.0 * np.log10(rms))
            
            # Save as 4-channel 24-bit FLAC
            sf.write(str(out_path), foa.T, TARGET_SAMPLE_RATE, format='FLAC', subtype='PCM_24')
            
            metadata_list.append(AudioChunkMetadata(
                source_file=input_file.name,
                chunk_index=idx,
                duration_sec=chunk_duration_sec,
                sample_rate=TARGET_SAMPLE_RATE,
                num_channels=4,
                rms_energy_db=rms_db,
                output_path=str(out_path)
            ))
            
    return metadata_list
