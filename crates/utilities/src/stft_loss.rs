//! Multi-Resolution Short-Time Fourier Transform (MR-STFT) and Spectral Acoustic Loss Module.
//!
//! Provides comprehensive frequency-domain loss formulations comparing:
//! 1. `WaveformFoa`: Multi-scale STFT across raw acoustic channels (FFT sizes 512, 1024, 2048)
//! 2. `Envelope16`: Multi-band filterbank spectral envelope convergence and log-energy distance
//! 3. `MelSpectral`: Perceptual filterbank spectral convergence across frequency tiers
//! 4. `Combined`: Joint multi-tier acoustic loss combining waveform fine-structure and spectral envelopes

use anyhow::Result;
use candle_core::{Device, Tensor};
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

/// Operational mode for spectral acoustic loss evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum StftLossMode {
    /// 16-band DDSP energy envelope convergence and log-spectral difference.
    Envelope16,
    /// Multi-resolution raw waveform STFT over acoustic frames.
    WaveformFoa,
    /// Perceptual spectral convergence across frequency sub-bands.
    MelSpectral,
    /// Combined multi-scale waveform and envelope loss.
    #[default]
    Combined,
}

/// Single STFT configuration tier.
#[derive(Debug, Clone)]
pub struct StftResolution {
    pub fft_size: usize,
    pub hop_size: usize,
    pub window_size: usize,
}

impl StftResolution {
    pub fn new(fft_size: usize, hop_size: usize, window_size: usize) -> Self {
        Self {
            fft_size,
            hop_size,
            window_size,
        }
    }
}

/// Multi-Resolution STFT Loss Calculator in pure Rust.
pub struct MultiResolutionStftLoss {
    pub resolutions: Vec<StftResolution>,
    pub eps: f32,
    pub log_weight: f32,
    pub conv_weight: f32,
}

impl Default for MultiResolutionStftLoss {
    fn default() -> Self {
        Self {
            resolutions: vec![
                StftResolution::new(512, 128, 512),
                StftResolution::new(1024, 256, 1024),
                StftResolution::new(2048, 512, 2048),
            ],
            eps: 1e-6,
            log_weight: 1.0,
            conv_weight: 1.0,
        }
    }
}

impl MultiResolutionStftLoss {
    pub fn new(resolutions: Vec<StftResolution>, log_weight: f32, conv_weight: f32) -> Self {
        Self {
            resolutions,
            eps: 1e-6,
            log_weight,
            conv_weight,
        }
    }

    /// Computes Hanning window of given length.
    pub fn hanning_window(size: usize) -> Vec<f32> {
        let mut w = Vec::with_capacity(size);
        let factor = 2.0 * std::f32::consts::PI / (size as f32 - 1.0);
        for i in 0..size {
            w.push(0.5 - 0.5 * (factor * i as f32).cos());
        }
        w
    }

    /// Computes STFT magnitude spectrogram for a 1D audio frame using `rustfft`.
    pub fn compute_magnitude_spectrogram(
        signal: &[f32],
        fft_size: usize,
        hop_size: usize,
        window: &[f32],
    ) -> Vec<Vec<f32>> {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);

        let num_frames = if signal.len() >= fft_size {
            (signal.len() - fft_size) / hop_size + 1
        } else {
            0
        };

        let num_bins = fft_size / 2 + 1;
        let mut spectrogram = Vec::with_capacity(num_frames);
        let mut buffer = vec![Complex::new(0.0f32, 0.0f32); fft_size];

        for frame_idx in 0..num_frames {
            let start = frame_idx * hop_size;
            for i in 0..fft_size {
                let s = signal[start + i] * window[i];
                buffer[i] = Complex::new(s, 0.0);
            }

            fft.process(&mut buffer);

            let mut frame_mag = Vec::with_capacity(num_bins);
            for bin in 0..num_bins {
                let mag = (buffer[bin].re * buffer[bin].re + buffer[bin].im * buffer[bin].im).sqrt();
                frame_mag.push(mag);
            }
            spectrogram.push(frame_mag);
        }

        spectrogram
    }

    /// Computes spectral convergence between target and predicted magnitude spectrograms:
    /// L_sc = || |Y| - |\hat{Y}| ||_F / || |Y| ||_F
    pub fn spectral_convergence(pred_mag: &[Vec<f32>], target_mag: &[Vec<f32>]) -> f32 {
        let num_frames = pred_mag.len().min(target_mag.len());
        if num_frames == 0 {
            return 0.0;
        }

        let mut diff_f_sq = 0.0f32;
        let mut target_f_sq = 0.0f32;

        for f in 0..num_frames {
            let bins = pred_mag[f].len().min(target_mag[f].len());
            for b in 0..bins {
                let diff = target_mag[f][b] - pred_mag[f][b];
                diff_f_sq += diff * diff;
                target_f_sq += target_mag[f][b] * target_mag[f][b];
            }
        }

        diff_f_sq.sqrt() / (target_f_sq.sqrt() + 1e-6)
    }

    /// Computes log-magnitude L1 spectral distance:
    /// L_mag = 1/N || log(|Y| + eps) - log(|\hat{Y}| + eps) ||_1
    pub fn log_magnitude_distance(pred_mag: &[Vec<f32>], target_mag: &[Vec<f32>], eps: f32) -> f32 {
        let num_frames = pred_mag.len().min(target_mag.len());
        if num_frames == 0 {
            return 0.0;
        }

        let mut total_l1 = 0.0f32;
        let mut count = 0usize;

        for f in 0..num_frames {
            let bins = pred_mag[f].len().min(target_mag[f].len());
            for b in 0..bins {
                let log_t = (target_mag[f][b] + eps).ln();
                let log_p = (pred_mag[f][b] + eps).ln();
                total_l1 += (log_t - log_p).abs();
                count += 1;
            }
        }

        total_l1 / count.max(1) as f32
    }

    /// Evaluates raw 1D audio frame waveforms across all multi-resolution tiers.
    pub fn evaluate_waveform_loss(&self, pred_audio: &[f32], target_audio: &[f32]) -> (f32, f32, f32) {
        let mut total_sc = 0.0f32;
        let mut total_mag = 0.0f32;

        for res in &self.resolutions {
            let window = Self::hanning_window(res.window_size);
            let pred_spec = Self::compute_magnitude_spectrogram(pred_audio, res.fft_size, res.hop_size, &window);
            let target_spec = Self::compute_magnitude_spectrogram(target_audio, res.fft_size, res.hop_size, &window);

            let sc = Self::spectral_convergence(&pred_spec, &target_spec);
            let mag = Self::log_magnitude_distance(&pred_spec, &target_spec, self.eps);

            total_sc += sc;
            total_mag += mag;
        }

        let n = self.resolutions.len().max(1) as f32;
        let avg_sc = total_sc / n;
        let avg_mag = total_mag / n;
        let total = self.conv_weight * avg_sc + self.log_weight * avg_mag;

        (total, avg_sc, avg_mag)
    }

    /// Evaluates 16-band energy envelopes against spectral target distributions.
    /// Computes envelope spectral convergence and logarithmic band distance.
    pub fn evaluate_envelope_loss(
        pred_bands: &Tensor,
        target_bands: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        // pred_bands, target_bands: [B, 16]
        let diff = (pred_bands - target_bands)?;
        let diff_sq = diff.sqr()?.sum_all()?;
        let target_sq = target_bands.sqr()?.sum_all()?;

        // Spectral envelope convergence: sqrt(diff_sq) / sqrt(target_sq + eps)
        let eps_f64 = eps.max(1e-6) as f64;
        let num = diff_sq.sqrt()?;
        let den = (target_sq + eps_f64)?.sqrt()?;
        let l_sc = (num / den)?;

        // Log-energy band distance: mean | log(|b| + eps) - log(|\hat{b}| + eps) |
        let pred_safe = (pred_bands.abs()? + eps_f64)?;
        let target_safe = (target_bands.abs()? + eps_f64)?;
        let log_pred = pred_safe.log()?;
        let log_target = target_safe.log()?;
        let l_log = (log_pred - log_target)?.abs()?.mean_all()?;

        let total = ((&l_sc + &l_log)? * 0.5)?;
        Ok((total, l_sc, l_log))
    }

    /// Evaluates spectral loss according to the configured `StftLossMode`.
    pub fn evaluate_loss(
        &self,
        mode: StftLossMode,
        pred_bands: &Tensor,
        target_bands: &Tensor,
        pred_audio: Option<&[f32]>,
        target_audio: Option<&[f32]>,
        device: &Device,
    ) -> Result<Tensor> {
        match mode {
            StftLossMode::Envelope16 => {
                let (total, _, _) = Self::evaluate_envelope_loss(pred_bands, target_bands, self.eps)?;
                Ok(total)
            }
            StftLossMode::WaveformFoa => {
                if let (Some(pred), Some(target)) = (pred_audio, target_audio) {
                    let (total, _, _) = self.evaluate_waveform_loss(pred, target);
                    Ok(Tensor::from_slice(&[total], (), device)?)
                } else {
                    // Fallback to envelope if raw audio is not supplied
                    let (total, _, _) = Self::evaluate_envelope_loss(pred_bands, target_bands, self.eps)?;
                    Ok(total)
                }
            }
            StftLossMode::MelSpectral => {
                // Perceptually weighted sub-band envelope comparison using Bark critical band weighting
                let (total, _, _) = Self::evaluate_envelope_loss(pred_bands, target_bands, self.eps)?;
                let weights = Tensor::from_slice(&BARK_CRITICAL_WEIGHTS, (1, 16), device)?;
                let weighted_diff = (pred_bands - target_bands)?.sqr()?.broadcast_mul(&weights)?.mean_all()?;
                let combined = ((&total + &weighted_diff)? * 0.5)?;
                Ok(combined)
            }
            StftLossMode::Combined => {
                let (env_loss, _, _) = Self::evaluate_envelope_loss(pred_bands, target_bands, self.eps)?;
                if let (Some(pred), Some(target)) = (pred_audio, target_audio) {
                    let (wf_loss, _, _) = self.evaluate_waveform_loss(pred, target);
                    let wf_tensor = Tensor::from_slice(&[wf_loss], (), device)?;
                    let joint = ((&env_loss + (&wf_tensor * 0.5)?)? * 0.667)?;
                    Ok(joint)
                } else {
                    Ok(env_loss)
                }
            }
        }
    }
}

// ============================================================================
// Spatial Ambisonic Acoustics & Soundfield Loss Formulations
// ============================================================================

/// Psychoacoustic Bark-scale equal-loudness sensitivity weights across 16 DDSP bands.
/// Emphasizes the ear-canal resonant frequencies (1kHz - 4kHz) while gracefully
/// tapering at extreme low sub-bass and ultra-high air absorption bands.
pub const BARK_CRITICAL_WEIGHTS: [f32; 16] = [
    0.55, 0.65, 0.75, 0.85, 1.00, 1.15, 1.30, 1.45,
    1.50, 1.45, 1.35, 1.20, 1.05, 0.90, 0.75, 0.60,
];

/// Numerically stable Smooth-L1 / Huber loss.
pub fn huber_loss(diff: &Tensor, delta: f64) -> Result<Tensor> {
    let abs_diff = diff.abs()?;
    let quadratic = (diff.sqr()? * 0.5)?;
    let linear = ((&abs_diff * delta)? - (0.5 * delta * delta))?;
    let delta_t = Tensor::full(delta as f32, diff.shape(), diff.device())?;
    let is_small = abs_diff.le(&delta_t)?;
    let loss = is_small.where_cond(&quadratic, &linear)?;
    Ok(loss.mean_all()?)
}

/// Evaluates 3D acoustic active intensity vector I = W * [X, Y, Z]^T
/// and computes Safe-Norm Direction-of-Arrival (DOA) angular error.
/// L_DOA = Huber(1.0 - (I_pred . I_target) / (||I_pred||_eps * ||I_target||_eps))
pub fn compute_acoustic_intensity_and_doa_loss(
    pred_foa: &Tensor,   // [B, 4] -> [W, X, Y, Z]
    target_foa: &Tensor, // [B, 4]
    delta: f64,
) -> Result<(Tensor, Tensor)> {
    let eps = 1e-5f64;

    // Split FOA components: W = pressure [B, 1], U = velocity gradient [B, 3]
    let w_pred = pred_foa.narrow(1, 0, 1)?;
    let u_pred = pred_foa.narrow(1, 1, 3)?;

    let w_target = target_foa.narrow(1, 0, 1)?;
    let u_target = target_foa.narrow(1, 1, 3)?;

    // Acoustic active intensity vector I = W * U: [B, 3]
    let i_pred = u_pred.broadcast_mul(&w_pred)?;
    let i_target = u_target.broadcast_mul(&w_target)?;

    // Safe-norm hypotenuse: sqrt(||I||^2 + eps^2)
    let norm_pred = (i_pred.sqr()?.sum_keepdim(1)? + (eps * eps))?.sqrt()?;
    let norm_target = (i_target.sqr()?.sum_keepdim(1)? + (eps * eps))?.sqrt()?;

    let n_pred = i_pred.broadcast_div(&norm_pred)?;
    let n_target = i_target.broadcast_div(&norm_target)?;

    // Cosine alignment: dot product [B, 1]
    let dot = (&n_pred * &n_target)?.sum_keepdim(1)?;
    let angular_err = (1.0 - dot)?;

    let l_doa = huber_loss(&angular_err, delta)?;

    // Mean squared error on acoustic intensity magnitude
    let mag_diff = (&norm_pred - &norm_target)?;
    let l_mag = huber_loss(&mag_diff, delta)?;

    let total = (&l_doa + (&l_mag * 0.5)?)?;
    Ok((total, l_doa))
}

/// Evaluates soundfield diffuseness ratio psi = 1.0 - ||I||_eps / (E_total + eps)
/// and computes Smooth-L1 diffuseness consistency loss.
pub fn compute_soundfield_diffuseness_loss(
    pred_foa: &Tensor,
    target_foa: &Tensor,
    delta: f64,
) -> Result<Tensor> {
    let eps = 1e-5f64;

    let compute_psi = |foa: &Tensor| -> Result<Tensor> {
        let w = foa.narrow(1, 0, 1)?;
        let u = foa.narrow(1, 1, 3)?;
        let i_vec = u.broadcast_mul(&w)?;
        let i_norm = (i_vec.sqr()?.sum_keepdim(1)? + (eps * eps))?.sqrt()?;

        // Total energy density E = W^2 + 1/3 * (X^2 + Y^2 + Z^2)
        let e_w = w.sqr()?;
        let e_u = (u.sqr()?.sum_keepdim(1)? * (1.0 / 3.0))?;
        let e_tot = (&e_w + &e_u)?;

        let ratio = i_norm.broadcast_div(&(e_tot + eps)?)?;
        let psi = (1.0 - ratio)?.clamp(0.0, 1.0)?;
        Ok(psi)
    };

    let psi_pred = compute_psi(pred_foa)?;
    let psi_target = compute_psi(target_foa)?;

    let diff = (psi_pred - psi_target)?;
    huber_loss(&diff, delta)
}

/// Computes positive onset half-wave spectral flux between consecutive temporal frames:
/// Delta S[k, f] = |S[k, f]| - |S[k-1, f]|
/// L_flux = Huber(ReLU(Delta S_pred) - ReLU(Delta S_target))
pub fn compute_spectral_flux_loss(
    pred_mag: &[Vec<f32>],
    target_mag: &[Vec<f32>],
    delta: f32,
) -> f32 {
    let num_frames = pred_mag.len().min(target_mag.len());
    if num_frames < 2 {
        return 0.0;
    }

    let mut total_flux_loss = 0.0f32;
    let mut count = 0usize;

    for f in 1..num_frames {
        let bins = pred_mag[f].len().min(target_mag[f].len());
        for b in 0..bins {
            let flux_p = (pred_mag[f][b] - pred_mag[f - 1][b]).max(0.0);
            let flux_t = (target_mag[f][b] - target_mag[f - 1][b]).max(0.0);
            let diff = (flux_p - flux_t).abs();

            let loss = if diff <= delta {
                0.5 * diff * diff
            } else {
                delta * (diff - 0.5 * delta)
            };
            total_flux_loss += loss;
            count += 1;
        }
    }

    total_flux_loss / count.max(1) as f32
}

/// Applies a 3D SO(3) Euler angle spatial rotation matrix to First-Order Ambisonic (FOA) soundfields.
/// Preserves 100% of omnidirectional pressure energy W while rotating directional acoustic particle
/// velocity vector components [X, Y, Z]^T across the spatial sphere.
pub fn apply_so3_foa_rotation(
    foa: &Tensor,
    angles: (f32, f32, f32), // (yaw_z, pitch_y, roll_x)
) -> Result<Tensor> {
    let (yaw, pitch, roll) = angles;
    let (cy, sy) = (yaw.cos(), yaw.sin());
    let (cp, sp) = (pitch.cos(), pitch.sin());
    let (cr, sr) = (roll.cos(), roll.sin());

    // 3x3 Orthogonal Rotation Matrix R = R_z(yaw) * R_y(pitch) * R_x(roll)
    // Row 0:
    let r00 = cy * cp;
    let r01 = cy * sp * sr - sy * cr;
    let r02 = cy * sp * cr + sy * sr;
    // Row 1:
    let r10 = sy * cp;
    let r11 = sy * sp * sr + cy * cr;
    let r12 = sy * sp * cr - cy * sr;
    // Row 2:
    let r20 = -sp;
    let r21 = cp * sr;
    let r22 = cp * cr;

    // R^T for right-matrix multiplication: U_rot = U * R^T
    let rot_t_data = [
        r00, r10, r20,
        r01, r11, r21,
        r02, r12, r22,
    ];

    let rot_matrix_t = Tensor::from_slice(&rot_t_data, (3, 3), foa.device())?;

    let w = foa.narrow(1, 0, 1)?;
    let u = foa.narrow(1, 1, 3)?;
    let u_rot = u.matmul(&rot_matrix_t)?;

    Ok(Tensor::cat(&[&w, &u_rot], 1)?)
}

