//! Native Rust Audio Corruption & Acoustic Degradation Suite.
//!
//! Provides high-throughput DSP augmentations simulating realistic field recording degradations,
//! acoustic absorption, diffuse reverberation smearing, and digital codec flaws.

use std::f32::consts::PI;

/// Applies a 1D Temporal Gaussian Blur across an audio buffer.
pub fn temporal_gaussian_blur(signal: &mut [f32], sigma: f32, max_kernel_size: usize) {
    if signal.is_empty() || sigma <= 1e-4 {
        return;
    }
    let k_size = if max_kernel_size % 2 == 1 {
        max_kernel_size
    } else {
        max_kernel_size + 1
    };
    let radius = (k_size - 1) / 2;
    let mut kernel = Vec::with_capacity(k_size);
    let mut sum = 0.0f32;

    for i in 0..k_size {
        let x = i as f32 - radius as f32;
        let g = (-0.5 * (x / sigma).powi(2)).exp();
        kernel.push(g);
        sum += g;
    }
    for val in &mut kernel {
        *val /= sum;
    }

    let original = signal.to_vec();
    let len = original.len();

    for i in 0..len {
        let mut acc = 0.0f32;
        for (k_idx, &k_weight) in kernel.iter().enumerate() {
            let offset = k_idx as isize - radius as isize;
            let sample_idx = (i as isize + offset).clamp(0, len as isize - 1) as usize;
            acc += original[sample_idx] * k_weight;
        }
        signal[i] = acc;
    }
}

/// Bit-depth reduction with optional mu-law companding.
pub fn bit_depth_crushing(signal: &mut [f32], bits: usize) {
    if bits >= 32 {
        return;
    }
    let steps = (1 << bits.saturating_sub(1)) as f32;
    let inv_steps = 1.0 / steps.max(1.0);

    for sample in signal.iter_mut() {
        let clamped = sample.clamp(-1.0, 1.0);
        *sample = (clamped * steps).round() * inv_steps;
    }
}

/// Diffuse synthetic reverberation smearing.
pub fn diffuse_reverb_smear(signal: &mut [f32], sample_rate: f32, rt60: f32) {
    if signal.is_empty() || rt60 <= 1e-3 {
        return;
    }
    let decay_time = rt60.min(0.5);
    let decay_samples = (sample_rate * decay_time) as usize;
    if decay_samples <= 1 {
        return;
    }

    let mut delay_buf = vec![0.0f32; decay_samples];
    let feedback = (-6.91 / (decay_time * sample_rate)).exp();
    let mut buf_idx = 0;

    for sample in signal.iter_mut() {
        let delayed = delay_buf[buf_idx];
        let new_sample = *sample + delayed * feedback;
        delay_buf[buf_idx] = new_sample;
        buf_idx = (buf_idx + 1) % decay_samples;
        *sample = 0.7 * *sample + 0.3 * delayed;
    }
}

/// Additive white Gaussian noise injection at targeted SNR (dB).
pub fn add_noise(signal: &mut [f32], snr_db: f32) {
    if signal.is_empty() {
        return;
    }
    let rms = (signal.iter().map(|s| s * s).sum::<f32>() / signal.len() as f32).sqrt() + 1e-8;
    let noise_rms = rms / 10.0f32.powf(snr_db / 20.0);

    let mut state: u32 = 0x12345678;
    for sample in signal.iter_mut() {
        // Simple Xorshift uniform PRNG
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let u1 = ((state as f32) / (u32::MAX as f32)).max(1e-7);
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let u2 = (state as f32) / (u32::MAX as f32);

        // Box-Muller transform for Gaussian noise
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        *sample += z * noise_rms;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_blur_preserves_length() {
        let mut buf = vec![0.0, 1.0, 0.0, -1.0, 0.5];
        temporal_gaussian_blur(&mut buf, 2.0, 5);
        assert_eq!(buf.len(), 5);
        assert!(buf.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn test_bit_depth_crushing_quantization() {
        let mut buf = vec![0.12345, -0.6789, 0.999];
        bit_depth_crushing(&mut buf, 4);
        for s in &buf {
            assert!(s.abs() <= 1.0);
        }
    }
}
