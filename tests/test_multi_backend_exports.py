"""
Multi-backend integration tests matching the centralized repository workspace structure.
"""

import sys
import struct
import json
from pathlib import Path
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.export_candle import export_models_to_candle
from scripts.export_onnx import export_models_to_onnx
from scripts.benchmark_backends import run_backend_benchmarks

def test_safetensors_export(tmp_path, monkeypatch):
    # Route target path away from development production data into test staging
    out_dir = tmp_path / "crates" / "inference" / "data" / "candle"
    
    # Overwrite stand-alone executable defaults via monkeypatching parent definitions
    monkeypatch.setattr(Path, "parent", tmp_path)
    export_models_to_candle(out_dir)

    vae_st = out_dir / "spatial_vae.safetensors"
    mamba_st = out_dir / "mamba2_moe.safetensors"

    assert vae_st.exists(), f"SafeTensors target missing at {vae_st}"
    assert mamba_st.exists()
    assert vae_st.stat().st_size > 1000
    assert mamba_st.stat().st_size > 1000

    # Parse compliant header specifications
    with open(vae_st, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))

    assert "__metadata__" in header
    assert any("encoder" in k for k in header.keys())

def test_onnx_export(tmp_path, monkeypatch):
    import onnx
    out_dir = tmp_path / "crates" / "inference" / "data" / "onnx"
    
    monkeypatch.setattr(Path, "parent", tmp_path)
    export_models_to_onnx(out_dir)

    vae_onnx = out_dir / "spatial_vae.onnx"
    mamba_onnx = out_dir / "mamba2_moe.onnx"

    assert vae_onnx.exists()
    assert mamba_onnx.exists()

    # Validate computational graph architecture
    model_mamba = onnx.load(str(mamba_onnx))
    onnx.checker.check_model(model_mamba)

def test_backend_benchmark_execution(tmp_path, monkeypatch):
    # Instruct benchmarks to use the mocked workspace structure
    monkeypatch.setattr(Path, "parent", tmp_path)
    
    results = run_backend_benchmarks(num_iterations=10)
    assert "pytorch_eager" in results
    assert "candle_safetensors" in results
    assert "onnx_graph" in results

    assert results["candle_safetensors"]["mean_us"] > 0
    assert results["onnx_graph"]["mean_us"] > 0
    assert results["candle_safetensors"]["rtf_128"] < 0.1
