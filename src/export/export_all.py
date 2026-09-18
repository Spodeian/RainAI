"""
RainAI Unified Multi-Backend Model Exporter.
Orchestrates the complete export of:
1. Role-Anchored Progressive Quantization Slices (S0..S4) & WebGPU WGSL assets
2. ONNX computational graphs (Opset 17/18) with static shapes
3. Candle-compatible SafeTensors binary bundles
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.export.export_wasm import export_models_for_wasm
from src.export.export_onnx import export_models_to_onnx
from src.export.export_candle import export_models_to_candle


def get_export_root() -> Path:
    """Resolves strictly to crates/inference/data relative to PROJECT_ROOT."""
    target = PROJECT_ROOT / "crates" / "inference" / "data"
    target.mkdir(parents=True, exist_ok=True)
    return target


def main():
    print("=" * 70)
    print("RainAI Master Multi-Backend AI Engine Exporter")
    print("=" * 70)
    start_time = time.time()

    export_root = get_export_root()

    print("\n[Step 1/3] Exporting Progressive Quantization Slices & Config...")
    t0 = time.time()
    wasm_dir = export_root / "wasm"
    export_models_for_wasm(wasm_dir)
    print(f"[+] Progressive slices exported in {time.time() - t0:.2f}s")

    print("\n[Step 2/3] Exporting ONNX Runtime Models...")
    t0 = time.time()
    onnx_dir = export_root / "onnx"
    export_models_to_onnx(onnx_dir)
    print(f"[+] ONNX graphs exported in {time.time() - t0:.2f}s")

    print("\n[Step 3/3] Exporting Candle SafeTensors...")
    t0 = time.time()
    candle_dir = export_root / "candle"
    export_models_to_candle(candle_dir)
    print(f"[+] Candle SafeTensors exported in {time.time() - t0:.2f}s")

    print("\n" + "=" * 70)
    print(f"ALL BACKEND EXPORTS COMPLETED IN {time.time() - start_time:.2f}s")
    print(f"Target Artifacts Directory: {export_root}")
    print("=" * 70)


if __name__ == "__main__":
    main()
