# ⚡ RainAI Neural Inference Runtime (`inference`)

The `inference` crate houses the multi-backend execution engine for continuous mixed-precision models.

## 🚀 Execution Backends

1. **Hugging Face Candle Engine**: Pure-Rust native tensor evaluation loading SafeTensors weights directly into memory with zero Python runtime overhead.
2. **SIMD Micro-Kernels**: Hand-tuned vector baselines for fast matrix-vector products, 1.58-bit ternary weight unpacking, and continuous Box-Cox inverse activations.
3. **WebGPU Compute Shaders (WGSL)**: Hardware-accelerated GPU pipelines for WebAssembly browser execution with async buffer staging and pipeline caching.
4. **ONNX Runtime Engine**: Direct hardware graph execution via the ONNX Runtime provider.

## 📊 Deployment Quantization Tiers

| Tier | Slice Name | Precision | Quantization Format | Target Environment |
| :---: | :--- | :--- | :--- | :--- |
| **0** | `slice_0_ternary.bin` | 1.58-bit | Ternary $\\{-1, 0, +1\\}$ | Embedded IoT / Microcontrollers |
| **1** | `slice_1_mobile_int4.bin` | 4-bit | INT4 Symmetric | Battery-constrained mobile devices |
| **2** | `slice_2_standard_int8.bin` | 8-bit | INT8 Asymmetric | WebAssembly (WASM) & Browser |
| **3** | `slice_3_studio_int16.bin` | 16-bit | INT16 / FP16 | Desktop DAWs & VST3 plugins |
| **4** | `slice_4_master_fp32.bin` | 32-bit | Uncompressed FP32 | Master Studio Workstations |

## 📁 Centralized Model & Shader Repository (`data/`)

The `crates/inference/data` directory serves as the **single source of truth** across all platforms and frontends:
- **`candle/`**: Hugging Face Candle SafeTensors (`spatial_vae.safetensors`, `mamba2_moe.safetensors`, `candle_manifest.json`)
- **`onnx/`**: ONNX computational graphs (`spatial_vae.onnx`, `mamba2_moe.onnx`, `onnx_manifest.json`)
- **`wasm/`**: Progressive binary quantization slices (`slice_0_ternary.bin` through `slice_4_master_fp32.bin`) & acoustic anchor
- **`src/shaders/`**: All 12 production WGSL compute shaders:
  1. `mamba2_ssd.wgsl` (State Space Duality chunk recurrence)
  2. `dense_soup_dispatch.wgsl` (Stochastic weight interpolation)
  3. `mla_attention.wgsl` (Multi-Head Latent Attention)
  4. `consistency_jump.wgsl` (Euler fast-forward step)
  5. `foa_projection.wgsl` (Ambisonic B-format spatial projection)
  6. `layer_forward.wgsl` (General layer compute)
  7. `mamba2.wgsl` (Single-head SSD core)
  8. `mamba2_deliberation.wgsl` (Iterative latent deliberation)
  9. `binaural_convolver.wgsl` (SO3 HRTF spatial audio rendering)
  10. `dequant.wgsl` (2-bit ternary & Box-Cox dequantizer)
  11. `moe_dispatch.wgsl` (Continuous smooth softmax & Hermite smoothstep expert modulation)
  12. `droplet_panning.wgsl` (Procedural droplet spatialization)

## 🧪 Testing & Verification
```bash
# Run all inference tests (including WGSL Naga shader validation and SafeTensors loading)
cargo test -p inference -j 2

# Run kernel and multi-backend benchmarks
cargo bench -p inference --bench kernel_benchmarks
cargo bench -p inference --bench backend_comparison
```

