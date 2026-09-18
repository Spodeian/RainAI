# 🌐 RainAI WebAssembly & WebGPU Studio (`web`)

The `web` crate provides client-side WebAssembly (WASM) entry points and PWA service workers for serverless web deployment.

## 🚀 Local Development

Ensure you have [Trunk](https://trunkrs.dev/) installed:

```bash
cargo install trunk
trunk serve crates/web/index.html
```

Open `http://localhost:8080` in a WebGPU-enabled browser (Chrome, Edge, Firefox Nightly).

## 📦 Production Deployment & Asset Architecture

```bash
# Full unified deployment script
bash deploy.sh

# Or direct release build with Trunk
trunk build --release crates/web/index.html
```

Build artifacts will be emitted to `crates/web/dist`, ready for instant deployment to Cloudflare Pages, GitHub Pages, or Netlify.

### 🏛️ Declarative Zero-Duplication Asset Pipeline
All heavy model weights, ONNX graphs, progressive quantization slices, and WGSL compute shaders reside exclusively in `crates/inference/data/` and `crates/inference/src/shaders/` (single source of truth). Trunk copies them directly into the output bundle at build time via `index.html` directives:
- `models/`: Hugging Face Candle SafeTensors (`spatial_vae.safetensors`, `mamba2_moe.safetensors`)
- `models/onnx/`: ONNX computational graphs (`spatial_vae.onnx`, `mamba2_moe.onnx`, `onnx_manifest.json`)
- `data/`: Progressive binary quantization slices (`slice_0_ternary.bin` through `slice_4_master_fp32.bin`) & deployment config
- `shaders/`: All 12 WebGPU WGSL compute shaders

