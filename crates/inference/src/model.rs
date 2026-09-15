//! Continuous Mixed-Precision Box-Cox dequantization, multi-precision formats, and model structures.
//!
//! Mirrors the numerical kernels designed in `src/models/diff_autoencoder.py`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Universal arithmetic precision formats supported across the inference and audio pipelines
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PrecisionFormat {
    Pruned,
    Ternary158,
    Int2,
    Int4,
    Int6,
    #[default]
    Int8,
    Int16,
    Int32,
    Fp8E4M3,
    Fp8E5M2,
    Fp16,
    Fp32,
    Bf16,
    Tf32,
    Posit8,
    Posit16,
    BoxCoxCompanded,
}

impl PrecisionFormat {
    pub fn label(self) -> &'static str {
        match self {
            Self::Pruned => "Pruned (0-Bit / Zero-Skip)",
            Self::Ternary158 => "Ternary 1.58-Bit ({-1, 0, +1})",
            Self::Int2 => "INT2 (2-Bit, 4 Levels)",
            Self::Int4 => "INT4 (4-Bit, 16 Levels)",
            Self::Int6 => "INT6 (6-Bit, 64 Levels)",
            Self::Int8 => "INT8 (8-Bit, 256 Levels)",
            Self::Int16 => "INT16 (16-Bit Fixed-Point)",
            Self::Int32 => "INT32 (32-Bit Fixed-Point)",
            Self::Fp8E4M3 => "FP8 E4M3 (Weights)",
            Self::Fp8E5M2 => "FP8 E5M2 (Dynamic Range)",
            Self::Fp16 => "FP16 (Half-Precision / WebGPU)",
            Self::Fp32 => "FP32 (Single-Precision / Studio Master)",
            Self::Bf16 => "BF16 (Brain Float 16 / Recurrent Memory)",
            Self::Tf32 => "TF32 (TensorFloat-32 / 19-Bit DL)",
            Self::Posit8 => "Posit8 <8, 1> (Tapered Unum)",
            Self::Posit16 => "Posit16 <16, 1> (Tapered Unum Studio)",
            Self::BoxCoxCompanded => "Box-Cox Companded Manifold",
        }
    }

    pub fn nominal_bits(self) -> f32 {
        match self {
            Self::Pruned => 0.0,
            Self::Ternary158 => 1.58,
            Self::Int2 => 2.0,
            Self::Int4 => 4.0,
            Self::Int6 => 6.0,
            Self::Int8 => 8.0,
            Self::Int16 => 16.0,
            Self::Int32 => 32.0,
            Self::Fp8E4M3 | Self::Fp8E5M2 | Self::Posit8 => 8.0,
            Self::Fp16 | Self::Bf16 | Self::Posit16 => 16.0,
            Self::Tf32 => 19.0,
            Self::Fp32 => 32.0,
            Self::BoxCoxCompanded => 8.0,
        }
    }

    pub fn is_integer(self) -> bool {
        matches!(
            self,
            Self::Pruned
                | Self::Ternary158
                | Self::Int2
                | Self::Int4
                | Self::Int6
                | Self::Int8
                | Self::Int16
                | Self::Int32
        )
    }

    pub fn is_floating_point(self) -> bool {
        matches!(
            self,
            Self::Fp8E4M3 | Self::Fp8E5M2 | Self::Fp16 | Self::Fp32 | Self::Bf16 | Self::Tf32
        )
    }

    pub fn is_posit(self) -> bool {
        matches!(self, Self::Posit8 | Self::Posit16)
    }

    /// Discrete level snapping via L = round(2^b)
    pub fn from_continuous_bit_width(b: f32) -> Self {
        if b <= 0.5 {
            Self::Pruned
        } else if b <= 1.8 {
            Self::Ternary158
        } else if b <= 3.0 {
            Self::Int2
        } else if b <= 5.0 {
            Self::Int4
        } else if b <= 7.0 {
            Self::Int6
        } else if b <= 10.0 {
            Self::Int8
        } else if b <= 16.5 {
            Self::Int16
        } else {
            Self::Fp32
        }
    }
}

/// Structural role of a neural/acoustic layer, dictating its optimal unquantized representation
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerRole {
    MambaStateSpaceRecurrence,
    LatentBottleneck,
    DenseProjection,
    FilterCoefficients,
    AmbisonicRotation,
    MacroConditioning,
}

impl LayerRole {
    /// Optimal non-integer format when not down-quantized into integer/ternary/pruned
    pub fn optimal_unquantized_format(self) -> PrecisionFormat {
        match self {
            Self::MambaStateSpaceRecurrence => PrecisionFormat::Bf16,
            Self::LatentBottleneck => PrecisionFormat::Posit16,
            Self::DenseProjection => PrecisionFormat::Fp16,
            Self::FilterCoefficients => PrecisionFormat::Fp32,
            Self::AmbisonicRotation => PrecisionFormat::Fp32,
            Self::MacroConditioning => PrecisionFormat::Posit8,
        }
    }

    pub fn down_quantization_fallback(self, target_bits: f32) -> PrecisionFormat {
        if target_bits >= 16.5 {
            self.optimal_unquantized_format()
        } else {
            PrecisionFormat::from_continuous_bit_width(target_bits)
        }
    }
}

/// Equal-power Hann crossfade between two audio buffers across T samples
pub fn apply_equal_power_crossfade(source: &[f32], target: &[f32], output: &mut [f32]) {
    let n = source.len().min(target.len()).min(output.len());
    if n == 0 {
        return;
    }
    let half_pi = std::f32::consts::FRAC_PI_2;
    for i in 0..n {
        let t = i as f32 / (n - 1).max(1) as f32;
        let theta = t * half_pi;
        output[i] = source[i] * theta.cos() + target[i] * theta.sin();
    }
}

/// Continuous Box-Cox homotopy dequantization kernel
pub struct BoxCoxDequantizer;

impl BoxCoxDequantizer {
    /// Encode and decode Brain Float 16 (BF16)
    #[inline]
    pub fn encode_bf16(val: f32) -> u16 {
        (val.to_bits() >> 16) as u16
    }

    #[inline]
    pub fn decode_bf16(bits: u16) -> f32 {
        f32::from_bits((bits as u32) << 16)
    }

    /// Encode and decode IEEE-754 Half-Precision Float (FP16)
    pub fn encode_fp16(val: f32) -> u16 {
        let bits = val.to_bits();
        let sign = ((bits >> 31) & 1) as u16;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let mant = (bits & 0x7FFFFF) as u32;

        if exp == 255 {
            // Inf or NaN
            let m = if mant != 0 { 0x200 } else { 0 };
            return (sign << 15) | 0x7C00 | m;
        }

        let new_exp = exp - 127 + 15;
        if new_exp >= 31 {
            // Overflow to Inf
            (sign << 15) | 0x7C00
        } else if new_exp <= 0 {
            // Subnormal or zero
            if 14 - new_exp > 24 {
                sign << 15
            } else {
                let m = (mant | 0x800000) >> (14 - new_exp + 1);
                (sign << 15) | (m as u16)
            }
        } else {
            let new_mant = (mant >> 13) as u16;
            (sign << 15) | ((new_exp as u16) << 10) | new_mant
        }
    }

    pub fn decode_fp16(bits: u16) -> f32 {
        let sign = ((bits >> 15) & 1) as u32;
        let exp = ((bits >> 10) & 0x1F) as u32;
        let mant = (bits & 0x3FF) as u32;

        if exp == 0x1F {
            if mant != 0 {
                f32::NAN
            } else if sign == 1 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else if exp == 0 {
            if mant == 0 {
                if sign == 1 { -0.0 } else { 0.0 }
            } else {
                // Subnormal
                let val = (mant as f32) / 1024.0 * 2.0f32.powi(-14);
                if sign == 1 { -val } else { val }
            }
        } else {
            let new_exp = (exp as i32 - 15 + 127) as u32;
            let new_mant = mant << 13;
            f32::from_bits((sign << 31) | (new_exp << 23) | new_mant)
        }
    }

    /// Encode and decode Posit16 <16, 1>
    pub fn encode_posit16(val: f32) -> u16 {
        if val == 0.0 {
            return 0;
        }
        if val.is_nan() || val.is_infinite() {
            return 0x8000;
        }

        let is_neg = val < 0.0;
        let abs_val = val.abs();

        let log2_val = abs_val.log2();
        let floor_log2 = log2_val.floor() as i32;
        let k = floor_log2.div_euclid(2);
        let e = floor_log2.rem_euclid(2) as u16;

        let scale_denom = 2.0f32.powi(2 * k + e as i32);
        let frac = (abs_val / scale_denom - 1.0).clamp(0.0, 1.0);

        let mut bits: u32 = 0;
        let mut bit_pos = 14;

        if k >= 0 {
            for _ in 0..=(k as usize) {
                if bit_pos < 0 {
                    break;
                }
                bits |= 1 << bit_pos;
                bit_pos -= 1;
            }
            if bit_pos >= 0 {
                // Terminating 0
                bit_pos -= 1;
            }
        } else {
            let num_zeros = (-k) as usize;
            for _ in 0..num_zeros {
                if bit_pos < 0 {
                    break;
                }
                bit_pos -= 1;
            }
            if bit_pos >= 0 {
                // Terminating 1
                bits |= 1 << bit_pos;
                bit_pos -= 1;
            }
        }

        if bit_pos >= 0 {
            if e == 1 {
                bits |= 1 << bit_pos;
            }
            bit_pos -= 1;
        }

        if bit_pos >= 0 {
            let frac_bits = ((frac * ((1 << (bit_pos + 1)) as f32)).round() as u32) & ((1 << (bit_pos + 1)) - 1);
            bits |= frac_bits;
        }

        let mut out = (bits & 0xFFFF) as u16;
        if is_neg {
            out = (!out).wrapping_add(1);
        }
        out
    }

    pub fn decode_posit16(bits: u16) -> f32 {
        if bits == 0 {
            return 0.0;
        }
        if bits == 0x8000 {
            return f32::NAN;
        }

        let is_neg = (bits >> 15) == 1;
        let u = if is_neg { (!bits).wrapping_add(1) } else { bits };

        let r = (u >> 14) & 1;
        let mut bit_pos = 13i32;
        let mut run_len = 1;

        while bit_pos >= 0 && ((u >> bit_pos) & 1) == r {
            run_len += 1;
            bit_pos -= 1;
        }
        bit_pos -= 1; // Skip terminating bit

        let k = if r == 1 { run_len as i32 - 1 } else { -(run_len as i32) };

        let e = if bit_pos >= 0 {
            let bit = (u >> bit_pos) & 1;
            bit_pos -= 1;
            bit as i32
        } else {
            0
        };

        let frac = if bit_pos >= 0 {
            let num_frac_bits = (bit_pos + 1) as u32;
            let mask = (1 << num_frac_bits) - 1;
            let frac_raw = (u as u32) & mask;
            frac_raw as f32 / (1 << num_frac_bits) as f32
        } else {
            0.0
        };

        let val = (1.0 + frac) * 2.0f32.powi(2 * k + e);
        if is_neg { -val } else { val }
    }

    /// Dequantize a normalized value from the companded manifold back to linear domain
    #[inline]
    pub fn dequantize_scalar(val: f32, bit_width: f32, lambda: f32) -> f32 {
        if bit_width <= 0.05 {
            return 0.0;
        }

        // Capacity staircase: n intervals
        let n = (2.0f32.powf(bit_width) - 1.0).max(1.0);
        let quantized = (val * n).round() / n;

        // Linear manifold
        if lambda <= 1e-4 {
            return quantized;
        }

        // Inverse Box-Cox companding
        let sign = quantized.signum();
        let abs_y = quantized.abs();

        if (1.0 - lambda).abs() < 1e-3 {
            // Clamped Taylor expansion: exp(|y|) - 1
            sign * (abs_y.exp() - 1.0)
        } else {
            let p = 1.0 - lambda;
			let inner = 1.0 + abs_y * p;
			if p < 0.0 && inner <= 0.0 {
				sign * f32::INFINITY
			} else {
				sign * (inner.max(0.0).powf(1.0 / p) - 1.0)
			}
        }
    }

    /// Dequantize with explicit target PrecisionFormat
    #[inline]
    pub fn dequantize_with_format(val: f32, format: PrecisionFormat, lambda: f32) -> f32 {
        match format {
            PrecisionFormat::Pruned => 0.0,
            PrecisionFormat::Ternary158 => val.round().clamp(-1.0, 1.0),
            PrecisionFormat::Fp32 => val,
            PrecisionFormat::Fp16 => Self::decode_fp16(Self::encode_fp16(val)),
            PrecisionFormat::Bf16 => Self::decode_bf16(Self::encode_bf16(val)),
            PrecisionFormat::Posit16 => Self::decode_posit16(Self::encode_posit16(val)),
            PrecisionFormat::Posit8 => {
                let p16 = Self::encode_posit16(val);
                Self::decode_posit16(p16 & 0xFF00)
            }
            _ => {
                let bits = format.nominal_bits();
                Self::dequantize_scalar(val, bits, lambda)
            }
        }
    }

    /// Dequantize a slice of weights in-place
    pub fn dequantize_slice(weights: &mut [f32], bit_width: f32, lambda: f32, scale: f32) {
        for w in weights.iter_mut() {
            *w = Self::dequantize_scalar(*w, bit_width, lambda) * scale;
        }
    }

    /// Dequantize a ternary 1.58-bit weight matrix (values in {-1, 0, 1} * gamma)
    pub fn dequantize_ternary(ternary_weights: &[i8], gamma: f32, output: &mut [f32]) {
        for (i, &t) in ternary_weights.iter().enumerate() {
            output[i] = (t as f32) * gamma;
        }
    }
}

/// Quantized layer weight representation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuantizedLayer {
    pub name: String,
    pub bit_width: f32,
    pub lambda: f32,
    pub scale: f32,
    pub weights: Vec<f32>,
}

/// Metadata for an individual exported layer
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerMetadata {
    pub shape: Vec<usize>,
    pub scale: f32,
    #[serde(default)]
    pub zero_skip_sparsity: f32,
    #[serde(default)]
    pub dtype: String,
}

/// Activation QAT schedule metadata
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActivationQatMetadata {
    pub per_channel_bit_widths: Vec<f32>,
    #[serde(default)]
    pub box_cox_lambdas: Vec<f32>,
    #[serde(default)]
    pub pruning_deltas: Vec<f32>,
    #[serde(default)]
    pub pruned_channels: Vec<usize>,
    #[serde(default)]
    pub ternary_channels: Vec<usize>,
    #[serde(default)]
    pub affine_alignment_applied: bool,
}

/// Complete exported quantized model manifest
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuantizedModelManifest {
    pub tier: String,
    pub description: String,
    pub activation_qat: ActivationQatMetadata,
    #[serde(rename = "weights", default)]
    pub layers: HashMap<String, LayerMetadata>,
}

impl QuantizedModelManifest {
    /// Loads the embedded default ternary 1.58-bit manifest
    pub fn load_default_ternary() -> Result<Self, serde_json::Error> {
        let manifest_str = include_str!("../data/quantized_model_manifest_ternary.json");
        serde_json::from_str(manifest_str)
    }

    /// Deserializes a manifest from a JSON string
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}
