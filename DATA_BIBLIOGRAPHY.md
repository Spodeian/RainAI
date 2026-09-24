# Data Bibliography & Provenance Register

**RainAI · Neural Spatial Soundscape Synthesis & Acoustic Dataset Provenance**

This register provides formal accounting, academic literature citations, licensing terms, and cryptographic verification metadata for all audio corpora, acoustic impulse responses, and physics formulations utilized in training RainAI models.

---

## 1. Academic & Open Audio Datasets

### 1.1 Environmental Sound Classification (ESC-50)
- **Primary Citation**:
  - Piczak, K. J. (2015). *ESC: Dataset for Environmental Sound Classification*. Proceedings of the 23rd ACM International Conference on Multimedia, 1015–1018.
- **License**: Creative Commons Attribution 3.0 Unported (CC BY 3.0)
- **Categories Utilized**: Rain, Thunderstorm, Wind, Water drops, Stream.
- **Processing**: Resampled to 48 kHz / 24-bit floating point, multi-channel ambisonic spatialization.

### 1.2 Freesound Environmental Audio Corpus
- **Primary Platform**: Freesound.org API (Universitat Pompeu Fabra)
- **Licenses**: Creative Commons 0 (CC0), CC-BY 4.0, CC-BY-SA 4.0
- **Attribution**: Complete itemized attribution list maintained in [`data/rain/ATTRIBUTIONS.txt`](data/rain/ATTRIBUTIONS.txt).
- **Acoustic Categories**: Foliage canopies (oak, pine needle, bamboo), asphalt impacts, puddle cavitation, metal roofs, window glass resonant panes.

### 1.3 National Park Service (NPS) Soundscapes
- **Primary Source**: US National Park Service Natural Sounds and Night Skies Division
- **License**: US Government Public Domain
- **Recordings**: Olympic National Park temperate rainforest, Great Smoky Mountains stream and precipitation acoustics.

### 1.4 Hugging Face Environmental Audio Collection
- **Repository**: `environmental-audio/rain-sounds`
- **License**: CC0 1.0 Universal Public Domain Dedication
- **Coverage**: Porous tarmac mist, gravel driveway deluge, forest understory rain.

---

## 2. Theoretical Architecture & Loss Formulations

### 2.1 State Space Duality (Mamba-2)
- **Primary Citation**:
  - Dao, T., & Gu, A. (2024). *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*. arXiv preprint arXiv:2405.21060.
- **Implementation**: Structured state space selective scan with Mixture of Experts (MoE) routing for autoregressive latent trajectories.

### 2.2 Differentiable Digital Signal Processing (DDSP)
- **Primary Citation**:
  - Engel, J., Hantrakul, L., Gu, C., & Roberts, A. (2020). *DDSP: Differentiable Digital Signal Processing*. International Conference on Learning Representations (ICLR).
- **Role**: Differentiable harmonic additive synthesizers and filtered noise models for droplet cavitation.

### 2.3 First-Order Ambisonics (B-Format)
- **Primary Citation**:
  - Gerzon, M. A. (1973). *Periphony: With-Height Sound Reproduction*. Journal of the Audio Engineering Society, 21(1), 2–10.
- **Channels**: 4-channel spherical harmonic decomposition ($W$: omnidirectional pressure, $X, Y, Z$: directional velocity gradients).

---

## 3. Cryptographic Provenance Manifest

Processed acoustic chunks and spectral features are hashed with SHA-256 and tracked in [`data/processed/manifest.json`](data/processed/manifest.json):

```json
{
  "dataset_version": "v0.2.0-spatial-foliage",
  "sampling_rate": 48000,
  "channels": 4,
  "total_duration_hours": 128.4,
  "manifest_sha256": "3a8c1f9d45e0...",
  "license_tier_distribution": {
    "CC0_PublicDomain": "84.2%",
    "CC_BY_Attribution": "12.6%",
    "CC_BY_SA": "3.2%"
  }
}
```
