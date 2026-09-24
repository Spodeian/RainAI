# RainAI Neural Acoustic Documentation Standards

**Mamba-2 Architecture, WebGPU Compute Shaders, Candle Inference & DSP Conventions**

---

## 1. Overview & Core Philosophy

RainAI is a neural spatial soundscape synthesis studio combining continuous mixed-precision models (Ternary 1.58-bit to FP32), WebGPU compute shaders, and Candle SafeTensors runtimes. All mathematical DSP algorithms, neural layers, and ambisonic formulas must be documented with formal rigor.

### The Five Pillars:
1. **Mathematical Grounding**: Every neural block (Mamba-2 SSM selective scan, DDSP harmonic synth, MoE router) and spatial ambisonic formulation ($W, X, Y, Z$) must state its governing differential equations and KaTeX notation.
2. **Deterministic Parity Contract**: PyTorch serves as the single source of truth for research and training. Candle (Rust) implementations must maintain bit-accurate numerical equivalence verified by automated parity integration tests.
3. **Zero-Warning Hygiene**: `cargo doc --workspace` and `cargo clippy --workspace --all-targets -- -D warnings` must compile with zero warnings.
4. **Strict Test Isolation**: All unit and integration tests must reside in dedicated test files under `tests/`; no inline tests inside production source files.
5. **Dynamic Parameter Discovery**: Physical kinematics (velocity, acceleration, jerk) and acoustic impedance coefficients must be learnable or dynamically inferred rather than statically hardcoded.

---

## 2. KaTeX Mathematical Notation

- **First-Order Ambisonics (B-Format) Encoding**:
  $$\begin{aligned}
  W &= \frac{1}{\sqrt{2}} S \\
  X &= S \cos(\theta) \cos(\phi) \\
  Y &= S \sin(\theta) \cos(\phi) \\
  Z &= S \sin(\phi)
  \end{aligned}$$
- **Mamba-2 Continuous State Space Model**:
  $$\begin{aligned}
  h'(t) &= A h(t) + B x(t) \\
  y(t) &= C h(t) + D x(t)
  \end{aligned}$$

---

## 3. Verification Checklist

Before opening PRs to `main`:
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` passes with 0 warnings.
- [ ] `cargo test --workspace` passes 100% of integration tests.
- [ ] PyTorch <-> Candle numerical parity test passes within $10^{-4}$ tolerance.
- [ ] Android mobile build (`scripts/build-android.ps1` / `scripts/build-android.sh`) verifies with `cargo-ndk`.
- [ ] No `DOCUMENTATION_STANDARDS.md` or internal roadmap files are included on `main`.
