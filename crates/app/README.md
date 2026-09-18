# 🎛️ RainAI Studio UI & View Controllers (`app`)

The `app` crate provides the cross-platform immediate-mode user interface powered by **`egui`** and **`eframe`**.

## 🎨 Studio Features

- **3D Ambisonic Radar**: Interactive polar soundfield visualization displaying active acoustic intensity vectors, particle velocities, and elevation angles.
- **Real-Time Spectrogram**: 16-band interactive FFT visualizer displaying instantaneous spectral flux and frequency-stratified energy envelopes.
- **Surface Material Matrix**: Sliders for fluid/solid acoustic mixture blending across all 9 physical surfaces.
- **Governor Telemetry Monitor**: Visual indicators for buffer health, CPU/GPU utilization, active MoE expert counts, and dynamic quantization tiers.
- **Preset Management**: Instant switching between acoustic scenes with smooth cross-fading.
- **State Persistence**: Automatic configuration synchronization to browser `localStorage` or native filesystem storage.
