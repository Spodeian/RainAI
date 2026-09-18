# 🔬 RainAI PyTorch Deep Learning & Research Suite (`src`)

The `src` directory houses the PyTorch deep learning codebase, physical loss formulation development, dataset preparation pipelines, and multi-backend export tooling.

## 📁 Package Structure

- **`models/`**: Neural network architectures:
  - `mamba2_moe.py`: Mamba-2 State Space Duality (SSD) with Mixture-of-Experts.
  - `diff_autoencoder.py`: Differentiable spatial autoencoder with learned Box-Cox quantizers.
  - `ddsp.py`: Continuous HOA-DDSP parametric filterbank and FDN synthesis.
  - `meta_controller.py`: Hardware-adaptive meta-controller.
  - `physics_losses.py`: PyTorch implementations of acoustic active intensity, trajectory kinematics, and spectral flux.
- **`training/`**: Training coordinators:
  - `auto_train.py`: Automated profile-based hardware-adaptive trainer.
  - `train_mamba.py`: Dedicated Mamba-2 MoE trajectory trainer.
  - `train_vae.py`: Spatial VAE and HOA-DDSP trainer.
- **`export/`**: Exporters:
  - `export_all.py`: Master multi-backend export script.
  - `export_candle.py`: SafeTensors exporter.
  - `export_onnx.py`: ONNX Runtime graph compiler.
  - `export_wasm.py`: Custom binary quantization slice generator.
- **`data/`**: Data pipelines:
  - `dataset.py`: Multi-source dataset streamer.
  - `corruptions.py`: Acoustic stress and corruption augmentation suite.
  - `pipeline.py`: End-to-end dataset preprocessing.
- **`tests/`**: Comprehensive pytest validation suite.

## 🧪 Testing
```bash
pytest src/tests -v
```
