# RainAI Development Roadmap

---

## Phase 1: Core Neural DSP & Multi-Tier Runtime [COMPLETE]
- [x] Mamba-2 MoE trajectory generator in PyTorch and Candle.
- [x] WebGPU hardware-accelerated WGSL compute shader backend.
- [x] Multi-tier quantization (1.58-bit Ternary, INT8, FP16, FP32).
- [x] Ambisonic First-Order Ambisonics (FOA) real-time spatializer.

---

## Phase 2: Interoperability & Parity Verification [CURRENT]
- [x] PyTorch canonical source of truth with automated Safetensors export.
- [x] Automated Cross-Framework Parity Test Suite (`parity_candle_pytorch.rs`).
- [x] Runge-Kutta 4th Order (RK4) continuous trajectory flow integrator.
- [x] Formal Data Provenance & Bibliography Register (`DATA_BIBLIOGRAPHY.md`).
- [x] Universal `cargo-ndk` Android mobile deployment pipeline (`scripts/build-android.ps1`, `scripts/build-android.sh`).

---

## Phase 3: Advanced Training & Continual Adaptation
- [ ] On-the-job Test-Time Adaptation (TTA) of spatial reflection heads.
- [ ] Dynamic Engram Bank retrieval-augmented acoustic memory.
- [ ] Sharpness-Aware Minimization (SAM) training for cross-space generalization.
