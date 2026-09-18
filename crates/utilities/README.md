# 🛠️ RainAI Utilities & Training Engine (`utilities`)

The `utilities` crate provides offline acoustic processing, dataset engineering, procedural rain synthesis, and the master native **Hugging Face Candle** deep learning training engine.

## 🚀 Binaries Included

| Binary | Source Path | Description |
| :--- | :--- | :--- |
| **`rainai_train_candle`** | `src/bin/train_candle.rs` | Master native Candle training engine (pure Rust; zero Python/CUDA dependency) |
| **`rainai_studio`** | `src/bin/tui.rs` | Interactive terminal user interface (TUI) studio powered by `ratatui` & `crossterm` |
| **`rainai_synth`** | `src/bin/synth_rain.rs` | Standalone physical acoustics synthesizer (Pumphrey-Crum bubbles & modal plates) |
| **`rainai_features`** | `src/bin/features.rs` | 16-band spectral energy & Ambisonic directional feature extractor |
| **`rainai_ingest`** | `src/bin/ingest.rs` | Automated dataset downloader validating CC0/CC-BY audio sources, acoustic quality screening, and 9-surface diversity quotas |
| **`rainai_upmix`** | `src/bin/spatial_upmix.rs` | High-fidelity stereo-to-FOA (B-format) spatial audio upmixer |
| **`rainai_golden_vectors`** | `src/bin/golden_vectors.rs` | Numerical drift and regression golden vector test generator |

## 🌊 Ingestion Diversity & Acoustic Screening Engine (`rainai_ingest`)

`rainai_ingest` downloads, audits, and curates multi-source audio across **130 ethically verified open-access environmental recordings** spanning 14 platforms (Hugging Face, Freesound, BigSoundBank, Internet Archive, ESC-50, Wikimedia Commons, NPS, Smithsonian, Kaggle, etc.):

### Key Ingestion Capabilities
- **Ethical License Verification**: Rejects `NC` (NonCommercial) and `ND` (NoDerivatives) clauses; approves and logs `CC0`, `Public Domain`, `CC-BY 4.0`, and `CC-BY-SA 4.0`. Verified via `LicenseVerifier`.
- **Acoustic Quality Screening (`AcousticQualityMetrics`)**:
  - RMS energy ($RMS \ge 10^{-4}$) to eliminate silent dead zones
  - Peak amplitude & digital clipping detection ($\text{clipping\_ratio} \le 5\%$)
  - Normalized Shannon spectral entropy ($H \ge 0.35$) to reject single-tone hums / test tones and ensure stochastic broadband precipitation texture
  - Wiener spectral flatness (geometric mean / arithmetic mean)
  - High-frequency transient energy ratio ($> 4\text{ kHz}$) validating droplet impact shockwaves
- **Canonical 9-Surface Quota Balancing (`SurfaceBalanceQuota`)**:
  - Automatically maps heterogeneous tags to 9 physical surfaces: `Asphalt` (11), `Pavement` (31), `TinRoof` (17), `CanvasTent` (11), `Foliage` (15), `WoodDeck` (12), `Glass` (11), `PuddleShallow` (11), and `WaterDeep` (11).
  - Computes Shannon diversity index $H = -\sum p_i \ln p_i$ (normalized against theoretical maximum $\ln(9) \approx 2.197$).
  - Alerts on underrepresented surfaces to maintain stratified training balance.
- **Cryptographic Provenance Manifest (`ProvenanceManifest`)**:
  - Computes SHA-256 digests for every ingested asset.
  - Emits structured `data/rain/manifest_provenance.json` recording license tier, file size, surface classification, and acoustic quality telemetry.

## 🎛️ Fully In-Process Terminal Studio (`rainai_studio`)

The `rainai_studio` TUI binary runs **100% in-process** on background threads without launching child processes or python scripts:
- **Resilient Atomic Checkpointing**: Checkpoints (`.safetensors.tmp` / `.json.tmp`) are written, synced, and atomically renamed. Clean exit (`[q]`) saves session state; startup rehydrates previous progress immediately.
- **Dynamic Host Resource Governor**: Senses terminal focus (`EnableFocusChange`). Allocates ~80% host compute when focused (`throttle = 0µs`), automatically throttling to ~50% when unfocused (`throttle = 15,000µs/batch`).
- **15 GB Rolling Quota & Balancer**: Enforces strict 15 GB ceiling (`MAX_DATASET_BYTES`), rotating through all 130 sources while maintaining high surface diversity ($H \ge 0.90$).
- **Centralized Model Deployment**: Keypress `[d]` deploys weights directly to `crates/inference/data/candle/` as the single centralized source of truth.

## 🧠 Master Candle Training Engine

### Key Features
- **Spatial VAE + HOA-DDSP**: Sub-band spectral reconstruction trained with Multi-Resolution STFT and Direction-of-Arrival (DOA) active intensity loss.
- **Mamba-2 SSD Mixture-of-Experts**:
  - Threshold-coverage routing ($\tau_{\text{cov}} = 0.75$)
  - Temporal tabu anti-repetition penalty ($\gamma_{\text{tabu}} = 1.0$)
  - Iterative latent deliberation with step embeddings and contractive latent jittering
  - Pairwise cosine orthogonality expert diversity loss ($\mathcal{L}_{\text{div}}$)
  - 1-Step Consistency Distillation Jump Head emitting edge-ready `mamba2_moe_fast.safetensors`
- **Physics-Preserving Augmentations**: 3D SO(3) Ambisonic rotation matrix augmentation ($\mathbf{R} \in \text{SO}(3)$) conserving 100% of omnidirectional acoustic pressure $W$, paired with convex surface wetness mixup.
- **Decoupled Optimization**: Parameter-group gradient dampening ($0.1\times$) for Mamba SSD state decay ($A_{\log}$), paired with cosine annealing and global Frobenius gradient clipping.

### CLI Usage Example
```bash
cargo run --bin rainai_train_candle -- \
  --epochs 5 \
  --batch-size 8 \
  --use-flow-matching \
  --tau-cov 0.75 \
  --thinking-steps 3 \
  --gamma-tabu 1.0 \
  --enable-distillation \
  --so3-aug-prob 0.3 \
  --stft-mode combined
```

