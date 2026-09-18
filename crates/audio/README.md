# 🔊 RainAI Real-Time Audio Engine (`audio`)

The `audio` crate provides zero-allocation real-time spatial soundscape synthesis, procedural DDSP acoustic filterbanks, First-Order Ambisonic (FOA) decoders, continuous hardware-in-the-loop (HWIL) governors, and lossless audio streaming export.

## 🎧 Architecture & Core Modules

- **`engine`**: Sample-accurate audio stream callback driving circular ring buffers, soft clipping limiters, and seamless cross-fades between neural model inference and procedural fallbacks.
- **`decoder`**: Ambisonic B-format ($W, Y, Z, X$) decoding matrices:
  - **Binaural HRTF**: Virtual 3D headphone listening with pinna spectral filtering.
  - **Stereo Speakers**: Blumlein and cross-feed panned monitor projection.
  - **Quadraphonic & 5.1 Surround**: Discrete loudspeaker matrix decoding.
- **`meta_governor`**: Dynamic hardware-in-the-loop governor monitoring buffer underruns, CPU/GPU headroom, and thermal stress to dynamically scale quantization tiers and active MoE experts.
- **`procedural`**: High-efficiency procedural rain synthesizer utilizing a 16-band subtractive filterbank with learned parametric resonant drift profiles for ultra-low-power execution.
- **`physical`**: Fluid mechanics and aeroacoustics simulation modules (Ulbrich Gamma DSD, Gunn-Kinzer terminal velocity aerodynamics, Pumphrey-Crum bubble cavitation, and ISO 140-18 structural plate modes).
- **`corruptions`**: Acoustic channel stress testing (codec artifacts, packet dropouts, decimation, and thermal noise).
- **`export`**: Chunk-streamed lossless 24-bit/32-bit WAV and Ambisonic B-format file rendering.
