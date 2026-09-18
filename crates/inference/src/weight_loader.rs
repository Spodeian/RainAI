//! Binary weight slice parsing, offset resolution, and memory caching for multi-tier inference.

use crate::model::{BoxCoxDequantizer, LayerMetadata, PrecisionFormat, QuantizedModelManifest};
use shared::rain::QualityTier;
use std::collections::HashMap;
use std::sync::Arc;

/// Raw embedded slice for default tier 0 (ternary 1.58-bit)
pub const EMBEDDED_SLICE_0_TERNARY: &[u8] = include_bytes!("../data/slice_0_ternary.bin");

/// Pre-packed, cache-aligned weight buffer for zero-overhead runtime SIMD execution
#[derive(Clone, Debug)]
pub enum WeightBuffer {
    Ternary2Bit {
        packed: Vec<u8>,
        gamma: f32,
    },
    Int8 {
        weights: Vec<i8>,
        scale: f32,
    },
    Posit8 {
        raw: Vec<u8>,
        scale: f32,
    },
    Bf16 {
        raw: Vec<u16>,
        scale: f32,
    },
    Fp16 {
        raw: Vec<u16>,
        scale: f32,
    },
    Fp32 {
        weights: Vec<f32>,
    },
}

impl Default for WeightBuffer {
    fn default() -> Self {
        Self::Fp32 { weights: Vec::new() }
    }
}

impl WeightBuffer {
    pub fn format(&self) -> PrecisionFormat {
        match self {
            Self::Ternary2Bit { .. } => PrecisionFormat::Ternary158,
            Self::Int8 { .. } => PrecisionFormat::Int8,
            Self::Posit8 { .. } => PrecisionFormat::Posit8,
            Self::Bf16 { .. } => PrecisionFormat::Bf16,
            Self::Fp16 { .. } => PrecisionFormat::Fp16,
            Self::Fp32 { .. } => PrecisionFormat::Fp32,
        }
    }
}

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
    /// Zero-overhead tagged enum buffer for direct SIMD execution without runtime dequantization
    pub buffer: WeightBuffer,
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
        let mut packed = Vec::with_capacity(values.len().div_ceil(4));
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
                    let bytes_needed = num_elements.div_ceil(4);
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

            let buffer = match format {
                PrecisionFormat::Ternary158 => WeightBuffer::Ternary2Bit {
                    packed: packed_weights.clone(),
                    gamma: meta.scale,
                },
                PrecisionFormat::Bf16 => {
                    let raw_bf16: Vec<u16> = weights.iter().map(|&w| BoxCoxDequantizer::encode_bf16(w)).collect();
                    WeightBuffer::Bf16 { raw: raw_bf16, scale: meta.scale }
                }
                PrecisionFormat::Fp16 => {
                    let raw_fp16: Vec<u16> = weights.iter().map(|&w| BoxCoxDequantizer::encode_fp16(w)).collect();
                    WeightBuffer::Fp16 { raw: raw_fp16, scale: meta.scale }
                }
                PrecisionFormat::Posit8 => {
                    let raw_p8: Vec<u8> = weights.iter().map(|&w| (BoxCoxDequantizer::encode_posit16(w) >> 8) as u8).collect();
                    WeightBuffer::Posit8 { raw: raw_p8, scale: meta.scale }
                }
                PrecisionFormat::Int8 => {
                    let raw_i8: Vec<i8> = weights.iter().map(|&w| (w / meta.scale.max(1e-6) * 127.0).round().clamp(-128.0, 127.0) as i8).collect();
                    WeightBuffer::Int8 { weights: raw_i8, scale: meta.scale }
                }
                _ => WeightBuffer::Fp32 { weights: weights.clone() },
            };

            cache.insert(LoadedLayer {
                name: name.clone(),
                shape: meta.shape.clone(),
                scale: meta.scale,
                format,
                weights,
                packed_weights,
                buffer,
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

    /// Deserializes model weights directly from standard Hugging Face SafeTensors byte buffer
    pub fn load_safetensors_bytes(bytes: &[u8], tier: QualityTier) -> Result<WeightCache, String> {
        use safetensors::{Dtype, SafeTensors};

        let tensors = SafeTensors::deserialize(bytes)
            .map_err(|e| format!("SafeTensors deserialization error: {e}"))?;
        let mut cache = WeightCache::new();

        for (name, view) in tensors.tensors() {
            let shape = view.shape().to_vec();
            let dtype = view.dtype();
            let data = view.data();

            let (format, weights, packed_weights, buffer) = match dtype {
                Dtype::F32 => {
                    let f32_slice: &[f32] = bytemuck::cast_slice(data);
                    let w_vec = f32_slice.to_vec();
                    (PrecisionFormat::Fp32, w_vec.clone(), Vec::new(), WeightBuffer::Fp32 { weights: w_vec })
                }
                Dtype::F16 => {
                    let u16_slice: &[u16] = bytemuck::cast_slice(data);
                    let decoded: Vec<f32> = u16_slice
                        .iter()
                        .map(|&bits| BoxCoxDequantizer::decode_fp16(bits))
                        .collect();
                    (PrecisionFormat::Fp16, decoded, Vec::new(), WeightBuffer::Fp16 { raw: u16_slice.to_vec(), scale: 1.0 })
                }
                Dtype::BF16 => {
                    let u16_slice: &[u16] = bytemuck::cast_slice(data);
                    let decoded: Vec<f32> = u16_slice
                        .iter()
                        .map(|&bits| BoxCoxDequantizer::decode_bf16(bits))
                        .collect();
                    (PrecisionFormat::Bf16, decoded, Vec::new(), WeightBuffer::Bf16 { raw: u16_slice.to_vec(), scale: 1.0 })
                }
                Dtype::I8 => {
                    let i8_slice: &[i8] = bytemuck::cast_slice(data);
                    let is_ternary = i8_slice.iter().all(|&v| v == -1 || v == 0 || v == 1);
                    if is_ternary {
                        let packed = Self::pack_ternary_2bit(i8_slice);
                        let f32_weights: Vec<f32> = i8_slice.iter().map(|&v| v as f32).collect();
                        (PrecisionFormat::Ternary158, f32_weights, packed.clone(), WeightBuffer::Ternary2Bit { packed, gamma: 1.0 })
                    } else {
                        let decoded: Vec<f32> = i8_slice.iter().map(|&v| v as f32 / 127.0).collect();
                        (PrecisionFormat::Int8, decoded, Vec::new(), WeightBuffer::Int8 { weights: i8_slice.to_vec(), scale: 1.0 })
                    }
                }
                Dtype::U8 => {
                    let num_elements = shape.iter().product();
                    let unpacked = Self::unpack_ternary_2bit(data, num_elements);
                    let f32_weights: Vec<f32> = unpacked.iter().map(|&v| v as f32).collect();
                    (PrecisionFormat::Ternary158, f32_weights, data.to_vec(), WeightBuffer::Ternary2Bit { packed: data.to_vec(), gamma: 1.0 })
                }
                _ => {
                    return Err(format!(
                        "Unsupported SafeTensors dtype {dtype:?} for tensor {name}"
                    ));
                }
            };

            cache.insert(LoadedLayer {
                name,
                shape,
                scale: 1.0,
                format,
                weights,
                packed_weights,
                buffer,
            });
        }

        cache.active_tier = Some(tier);
        Ok(cache)
    }

    /// Loads model weights directly from a SafeTensors file on disk
    pub fn load_safetensors_file(
        path: &std::path::Path,
        tier: QualityTier,
    ) -> Result<WeightCache, String> {
        let bytes = std::fs::read(path)
            .map_err(|e| format!("Failed to read SafeTensors file at {path:?}: {e}"))?;
        Self::load_safetensors_bytes(&bytes, tier)
    }
}

