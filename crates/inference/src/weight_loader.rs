//! Binary weight slice parsing, offset resolution, and memory caching for multi-tier inference.

use crate::model::{BoxCoxDequantizer, LayerMetadata, PrecisionFormat, QuantizedModelManifest};
use shared::rain::QualityTier;
use std::collections::HashMap;
use std::sync::Arc;

/// Raw embedded slice for default tier 0 (ternary 1.58-bit)
pub const EMBEDDED_SLICE_0_TERNARY: &[u8] = include_bytes!("../data/slice_0_ternary.bin");

/// Memory-resident decoded tensor layer
#[derive(Clone, Debug)]
pub struct LoadedLayer {
    pub name: String,
    pub shape: Vec<usize>,
    pub scale: f32,
    pub format: PrecisionFormat,
    /// Fully expanded float weights (used for FP32, FP16, and INT8 fallback tiers)
    pub weights: Vec<f32>,
    /// Raw 2-bit packed weights (used for Ternary158 zero-allocation SIMD kernels)
    pub packed_weights: Vec<u8>,
}

impl LoadedLayer {
    #[inline]
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }
}

/// In-memory cache holding deserialized and unpacked layer weights for active execution
#[derive(Clone, Debug, Default)]
pub struct WeightCache {
    layers: HashMap<String, Arc<LoadedLayer>>,
    active_tier: Option<QualityTier>,
}

impl WeightCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, layer: LoadedLayer) {
        self.layers.insert(layer.name.clone(), Arc::new(layer));
    }

    pub fn get(&self, layer_name: &str) -> Option<Arc<LoadedLayer>> {
        self.layers.get(layer_name).cloned()
    }

    pub fn contains(&self, layer_name: &str) -> bool {
        self.layers.contains_key(layer_name)
    }

    pub fn active_tier(&self) -> Option<QualityTier> {
        self.active_tier
    }

    pub fn clear(&mut self) {
        self.layers.clear();
        self.active_tier = None;
    }
}

/// Binary weight slice loader and precision unpacker
pub struct WeightLoader;

impl WeightLoader {
    /// Unpacks 2-bit packed ternary weights into discrete `i8` values in $\{-1, 0, 1\}$.
    ///
    /// Bit mapping per 2-bit pair:
    /// - `0b00` =>  0
    /// - `0b01` => +1
    /// - `0b11` => -1
    /// - `0b10` =>  0 (reserved padding)
    pub fn unpack_ternary_2bit(packed: &[u8], num_elements: usize) -> Vec<i8> {
        let mut out = Vec::with_capacity(num_elements);
        for &byte in packed {
            for shift in (0..8).step_by(2) {
                if out.len() == num_elements {
                    break;
                }
                let code = (byte >> shift) & 0x03;
                let val = match code {
                    0x01 => 1i8,
                    0x03 => -1i8,
                    _ => 0i8,
                };
                out.push(val);
            }
            if out.len() == num_elements {
                break;
            }
        }
        out
    }

    /// Packs discrete `i8` values in $\{-1, 0, 1\}$ into 2-bit packed bytes (4 weights per byte)
    pub fn pack_ternary_2bit(values: &[i8]) -> Vec<u8> {
        let mut packed = Vec::with_capacity((values.len() + 3) / 4);
        for chunk in values.chunks(4) {
            let mut byte = 0u8;
            for (i, &v) in chunk.iter().enumerate() {
                let code = match v {
                    1 => 0x01,
                    -1 => 0x03,
                    _ => 0x00,
                };
                byte |= code << (i * 2);
            }
            packed.push(byte);
        }
        packed
    }

    /// Loads and unpacks weights from an embedded or memory-mapped binary slice using a manifest
    pub fn load_from_manifest(
        manifest: &QuantizedModelManifest,
        raw_slice: &[u8],
        tier: QualityTier,
    ) -> Result<WeightCache, String> {
        let mut cache = WeightCache::new();
        let mut byte_offset = 0usize;

        // Iterate through deterministic sorted manifest layers
        let mut sorted_layers: Vec<(&String, &LayerMetadata)> = manifest.layers.iter().collect();
        sorted_layers.sort_by_key(|(name, _)| *name);

        for (name, meta) in sorted_layers {
            let num_elements: usize = meta.shape.iter().product();
            let format = PrecisionFormat::from_continuous_bit_width(
                manifest
                    .activation_qat
                    .per_channel_bit_widths
                    .first()
                    .copied()
                    .unwrap_or(1.58),
            );

            let mut weights = Vec::new();
            let mut packed_weights = Vec::new();

            match format {
                PrecisionFormat::Ternary158 => {
                    // 2-bit packed ternary: 4 values per byte
                    let bytes_needed = (num_elements + 3) / 4;
                    if byte_offset + bytes_needed > raw_slice.len() {
                        return Err(format!(
                            "Slice underflow reading layer {name}: needed {bytes_needed} bytes at offset {byte_offset}, total available {}",
                            raw_slice.len()
                        ));
                    }
                    // Load raw bytes directly; bypass f32 expansion to minimize memory footprint
                    packed_weights = raw_slice[byte_offset..byte_offset + bytes_needed].to_vec();
                    byte_offset += bytes_needed;
                }
                PrecisionFormat::Fp32 => {
                    let bytes_needed = num_elements * 4;
                    if byte_offset + bytes_needed > raw_slice.len() {
                        return Err(format!(
                            "Slice underflow reading FP32 layer {name}: needed {bytes_needed} bytes at offset {byte_offset}, total available {}",
                            raw_slice.len()
                        ));
                    }
                    let chunk = &raw_slice[byte_offset..byte_offset + bytes_needed];
                    byte_offset += bytes_needed;
                    
                    weights.resize(num_elements, 0.0);
                    for (i, w) in weights.iter_mut().enumerate() {
                        let idx = i * 4;
                        let bits = u32::from_le_bytes([chunk[idx], chunk[idx + 1], chunk[idx + 2], chunk[idx + 3]]);
                        *w = f32::from_bits(bits) * meta.scale;
                    }
                }
                PrecisionFormat::Fp16 | PrecisionFormat::Bf16 => {
                    let bytes_needed = num_elements * 2;
                    if byte_offset + bytes_needed > raw_slice.len() {
                        return Err(format!(
                            "Slice underflow reading 16-bit layer {name}: needed {bytes_needed} bytes at offset {byte_offset}, total available {}",
                            raw_slice.len()
                        ));
                    }
                    let chunk = &raw_slice[byte_offset..byte_offset + bytes_needed];
                    byte_offset += bytes_needed;

                    weights.resize(num_elements, 0.0);
                    for (i, w) in weights.iter_mut().enumerate() {
                        let idx = i * 2;
                        let bits = u16::from_le_bytes([chunk[idx], chunk[idx + 1]]);
                        *w = if format == PrecisionFormat::Fp16 {
                            BoxCoxDequantizer::decode_fp16(bits)
                        } else {
                            BoxCoxDequantizer::decode_bf16(bits)
                        } * meta.scale;
                    }
                }
                _ => {
                    // Standard single-byte fallback (INT8)
                    let bytes_needed = num_elements;
                    if byte_offset + bytes_needed > raw_slice.len() {
                        return Err(format!(
                            "Slice underflow reading byte-quantized layer {name}: needed {bytes_needed} bytes at offset {byte_offset}, total available {}",
                            raw_slice.len()
                        ));
                    }
                    let chunk = &raw_slice[byte_offset..byte_offset + bytes_needed];
                    byte_offset += bytes_needed;

                    weights.resize(num_elements, 0.0);
                    for (i, w) in weights.iter_mut().enumerate() {
                        let signed = chunk[i] as i8 as f32 / 127.0;
                        *w = BoxCoxDequantizer::dequantize_scalar(signed, meta.shape[0] as f32, 0.0) * meta.scale;
                    }
                }
            }

            cache.insert(LoadedLayer {
                name: name.clone(),
                shape: meta.shape.clone(),
                scale: meta.scale,
                format,
                weights,
                packed_weights,
            });
        }

        cache.active_tier = Some(tier);
        Ok(cache)
    }

    /// Convenience loader to unpack the default embedded Ternary 1.58-bit tier
    pub fn load_embedded_ternary() -> Result<WeightCache, String> {
        let manifest = QuantizedModelManifest::load_default_ternary()
            .map_err(|e| format!("Failed to parse embedded ternary manifest: {e}"))?;
        Self::load_from_manifest(&manifest, EMBEDDED_SLICE_0_TERNARY, QualityTier::Ternary158)
    }
}
