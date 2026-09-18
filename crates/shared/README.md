# 📦 RainAI Shared Domain Models (`shared`)

The `shared` crate defines the common domain data types, physical parameters, surface material categories, Ambisonic decoding modes, hardware telemetry metrics, and audio export settings shared across native desktop, web browser, and audio synthesis runtimes.

## 🌧️ Core Domain Specifications

- **`RainState`**: Master parameter state holding intensity, wind speed/direction, surface material mixture, spatial orientation, audio master volume, and playback state.
- **`SurfaceType`**: 9 physical surface categories:
  - `TinRoof`, `Foliage`, `PineNeedles`, `UrbanPavement`, `DeepWater`, `ShallowPuddle`, `CanvasTent`, `GlassWindow`, `WoodDecking`.
- **`EngineTelemetry`**: Real-time hardware health metrics (buffer health ms, CPU headroom, GPU headroom, panic factor).
- **`ThemeMode`**: Studio UI color themes (`Dark`, `Light`, `HighContrastDark`, `HighContrastLight`).
- **`ExportSettings`**: Multi-format audio rendering configuration (WAV, FLAC, B-format Ambisonics).
