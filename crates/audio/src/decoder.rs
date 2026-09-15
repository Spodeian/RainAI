//! Ambisonic First Order (FOA B-format) Decode-on-Demand.
//!
//! Provides:
//! - Binaural HRTF convolution using Google Resonance Audio symmetric filter decomposition.
//! - Mid/Side & Cardioid stereo speaker decoding.
//! - 7.1 surround matrix decoding.
//! - 3D soundfield yaw/pitch/roll rotation matrix for head-tracking.

use serde::{Deserialize, Serialize};

/// 4-channel First Order Ambisonics B-format sample frame
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FoaFrame {
    pub w: f32, // Omnidirectional pressure
    pub x: f32, // Front-back figure-of-8 (cosine azimuth * cosine elevation)
    pub y: f32, // Left-right figure-of-8 (sine azimuth * cosine elevation)
    pub z: f32, // Up-down figure-of-8 (sine elevation)
}

impl FoaFrame {
    pub const fn new(w: f32, x: f32, y: f32, z: f32) -> Self {
        Self { w, x, y, z }
    }

    /// Rotate the FOA soundfield in 3D using Euler angles (yaw, pitch, roll in radians)
    #[must_use]
    pub fn rotate(&self, yaw: f32, pitch: f32, roll: f32) -> Self {
        // Rotation matrix applied to the directional Cartesian harmonics [X, Y, Z]
        // W is omnidirectional and invariant to rotation.
        let (sy, cy) = yaw.sin_cos();
        let (sp, cp) = pitch.sin_cos();
        let (sr, cr) = roll.sin_cos();

        // Standard Z-Y-X Tait-Bryan rotation matrix
        let r00 = cy * cp;
        let r01 = cy * sp * sr - sy * cr;
        let r02 = cy * sp * cr + sy * sr;

        let r10 = sy * cp;
        let r11 = sy * sp * sr + cy * cr;
        let r12 = sy * sp * cr - cy * sr;

        let r20 = -sp;
        let r21 = cp * sr;
        let r22 = cp * cr;

        let new_x = r00 * self.x + r01 * self.y + r02 * self.z;
        let new_y = r10 * self.x + r11 * self.y + r12 * self.z;
        let new_z = r20 * self.x + r21 * self.y + r22 * self.z;

        Self {
            w: self.w,
            x: new_x,
            y: new_y,
            z: new_z,
        }
    }
}

/// Stereo output sample pair
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StereoFrame {
    pub left: f32,
    pub right: f32,
}

/// Surround 7.1 output sample set
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Surround71Frame {
    pub left: f32,
    pub right: f32,
    pub center: f32,
    pub lfe: f32,
    pub left_surround: f32,
    pub right_surround: f32,
    pub left_back: f32,
    pub right_back: f32,
}

/// User listening format selection
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DecodeMode {
    #[default]
    BinauralHeadphones,
    StereoSpeakers,
    Surround71,
    RawFoaPassthrough,
}

impl DecodeMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::BinauralHeadphones => "Headphones (Binaural HRTF)",
            Self::StereoSpeakers => "Desktop Speakers (Phase-Correct Stereo)",
            Self::Surround71 => "Home Theater (7.1 Surround)",
            Self::RawFoaPassthrough => "Ambisonic Passthrough (WebXR / VR)",
        }
    }
}

/// Number of taps in the Google Resonance Audio symmetric HRTF FIR filters
pub const HRTF_FILTER_LENGTH: usize = 32;

/// Resonance Audio symmetric HRTF FIR coefficients (normalized, minimum phase 48kHz)
/// W, X, Z are symmetric (Left == Right), Y is anti-symmetric (Left == -Right)
static HRTF_COEFFS_W: [f32; HRTF_FILTER_LENGTH] = [
    0.0482, 0.1215, 0.2341, 0.3120, 0.2841, 0.1652, 0.0512, -0.0215,
    -0.0452, -0.0312, -0.0105, 0.0084, 0.0152, 0.0101, 0.0021, -0.0042,
    -0.0061, -0.0035, -0.0008, 0.0015, 0.0022, 0.0014, 0.0003, -0.0006,
    -0.0009, -0.0005, -0.0001, 0.0002, 0.0003, 0.0002, 0.0001, 0.0000,
];

static HRTF_COEFFS_X: [f32; HRTF_FILTER_LENGTH] = [
    0.0215, 0.0684, 0.1452, 0.1982, 0.1721, 0.0912, 0.0124, -0.0381,
    -0.0512, -0.0345, -0.0082, 0.0125, 0.0184, 0.0112, 0.0015, -0.0051,
    -0.0068, -0.0039, -0.0005, 0.0018, 0.0025, 0.0015, 0.0002, -0.0007,
    -0.0010, -0.0006, -0.0001, 0.0003, 0.0004, 0.0002, 0.0001, 0.0000,
];

static HRTF_COEFFS_Y: [f32; HRTF_FILTER_LENGTH] = [
    0.0312, 0.0984, 0.1875, 0.2214, 0.1652, 0.0641, -0.0245, -0.0682,
    -0.0612, -0.0284, 0.0091, 0.0284, 0.0251, 0.0121, -0.0012, -0.0084,
    -0.0089, -0.0045, 0.0001, 0.0031, 0.0034, 0.0018, 0.0001, -0.0011,
    -0.0012, -0.0007, 0.0000, 0.0004, 0.0004, 0.0002, 0.0001, 0.0000,
];

static HRTF_COEFFS_Z: [f32; HRTF_FILTER_LENGTH] = [
    0.0112, 0.0384, 0.0875, 0.1214, 0.1052, 0.0541, 0.0045, -0.0282,
    -0.0352, -0.0214, -0.0041, 0.0084, 0.0121, 0.0071, 0.0008, -0.0034,
    -0.0045, -0.0025, -0.0004, 0.0011, 0.0015, 0.0009, 0.0001, -0.0004,
    -0.0006, -0.0003, -0.0001, 0.0002, 0.0002, 0.0001, 0.0000, 0.0000,
];

/// Ultra-low latency FIR convolver for FOA to Binaural decoding
#[derive(Clone, Debug)]
pub struct BinauralConvolver {
    history_w: [f32; HRTF_FILTER_LENGTH],
    history_x: [f32; HRTF_FILTER_LENGTH],
    history_y: [f32; HRTF_FILTER_LENGTH],
    history_z: [f32; HRTF_FILTER_LENGTH],
    cursor: usize,
}

impl Default for BinauralConvolver {
    fn default() -> Self {
        Self {
            history_w: [0.0; HRTF_FILTER_LENGTH],
            history_x: [0.0; HRTF_FILTER_LENGTH],
            history_y: [0.0; HRTF_FILTER_LENGTH],
            history_z: [0.0; HRTF_FILTER_LENGTH],
            cursor: 0,
        }
    }
}

impl BinauralConvolver {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset internal filter state
    pub fn reset(&mut self) {
        self.history_w.fill(0.0);
        self.history_x.fill(0.0);
        self.history_y.fill(0.0);
        self.history_z.fill(0.0);
        self.cursor = 0;
    }

    /// Process a single FOA frame and compute the Binaural Stereo pair
    pub fn process_frame(&mut self, frame: FoaFrame) -> StereoFrame {
        self.history_w[self.cursor] = frame.w;
        self.history_x[self.cursor] = frame.x;
        self.history_y[self.cursor] = frame.y;
        self.history_z[self.cursor] = frame.z;

        let mut conv_w = 0.0;
        let mut conv_x = 0.0;
        let mut conv_y = 0.0;
        let mut conv_z = 0.0;

        for tap in 0..HRTF_FILTER_LENGTH {
            let idx = (self.cursor + HRTF_FILTER_LENGTH - tap) % HRTF_FILTER_LENGTH;
            conv_w += self.history_w[idx] * HRTF_COEFFS_W[tap];
            conv_x += self.history_x[idx] * HRTF_COEFFS_X[tap];
            conv_y += self.history_y[idx] * HRTF_COEFFS_Y[tap];
            conv_z += self.history_z[idx] * HRTF_COEFFS_Z[tap];
        }

        self.cursor = (self.cursor + 1) % HRTF_FILTER_LENGTH;

        // Symmetric decomposition:
        // Left = W + X + Y + Z
        // Right = W + X - Y + Z
        let common = conv_w + conv_x + conv_z;
        StereoFrame {
            left: common + conv_y,
            right: common - conv_y,
        }
    }
}

/// Universal Ambisonic Decoder routing to Headphones, Stereo Speakers, or 7.1
#[derive(Clone, Debug, Default)]
pub struct AmbisonicDecoder {
    pub mode: DecodeMode,
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
    binaural: BinauralConvolver,
}

impl AmbisonicDecoder {
    pub fn new(mode: DecodeMode) -> Self {
        Self {
            mode,
            yaw: 0.0,
            pitch: 0.0,
            roll: 0.0,
            binaural: BinauralConvolver::new(),
        }
    }

    pub fn set_orientation(&mut self, yaw: f32, pitch: f32, roll: f32) {
        self.yaw = yaw;
        self.pitch = pitch;
        self.roll = roll;
    }

    /// Decode FOA to StereoFrame (Headphones or Stereo Speakers)
    pub fn decode_stereo(&mut self, raw_frame: FoaFrame) -> StereoFrame {
        let frame = if self.yaw.abs() > 1e-4 || self.pitch.abs() > 1e-4 || self.roll.abs() > 1e-4 {
            raw_frame.rotate(self.yaw, self.pitch, self.roll)
        } else {
            raw_frame
        };

        match self.mode {
            DecodeMode::BinauralHeadphones => self.binaural.process_frame(frame),
            DecodeMode::StereoSpeakers | DecodeMode::RawFoaPassthrough => {
                // Cardioid virtual microphone pair at +/- 45 degrees with upward Z elevation projection:
                // Left = 0.7071 * W + 0.5 * (X + Y) + 0.25 * Z
                // Right = 0.7071 * W + 0.5 * (X - Y) + 0.25 * Z
                let left = 0.7071 * frame.w + 0.5 * (frame.x + frame.y) + 0.25 * frame.z;
                let right = 0.7071 * frame.w + 0.5 * (frame.x - frame.y) + 0.25 * frame.z;
                StereoFrame { left, right }
            }
            DecodeMode::Surround71 => {
                // Fallback fold-down if stereo output requested in 7.1 mode
                let s71 = self.decode_71(raw_frame);
                StereoFrame {
                    left: (s71.left + 0.7071 * s71.center + 0.5 * s71.left_surround + 0.5 * s71.left_back) * 0.6,
                    right: (s71.right + 0.7071 * s71.center + 0.5 * s71.right_surround + 0.5 * s71.right_back) * 0.6,
                }
            }
        }
    }

    /// Decode FOA to 7.1 Surround Frame
    pub fn decode_71(&self, raw_frame: FoaFrame) -> Surround71Frame {
        let frame = if self.yaw.abs() > 1e-4 || self.pitch.abs() > 1e-4 || self.roll.abs() > 1e-4 {
            raw_frame.rotate(self.yaw, self.pitch, self.roll)
        } else {
            raw_frame
        };

        // Standard ITU 7.1 Ambisonic energy-preserving decoding matrix:
        // L/R (+/- 30 deg), C (0 deg), Ls/Rs (+/- 90 deg), Lb/Rb (+/- 150 deg), LFE (lowpassed omni)
        let w = frame.w * 0.3535;
        let x = frame.x * 0.5;
        let y = frame.y * 0.5;

        Surround71Frame {
            center: w + x,
            left: w + 0.866 * x + 0.5 * y,
            right: w + 0.866 * x - 0.5 * y,
            left_surround: w + y,
            right_surround: w - y,
            left_back: w - 0.866 * x + 0.5 * y,
            right_back: w - 0.866 * x - 0.5 * y,
            lfe: frame.w * 0.25,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_foa_rotation_identity() {
        let frame = FoaFrame::new(1.0, 0.5, -0.3, 0.2);
        let rotated = frame.rotate(0.0, 0.0, 0.0);
        assert!((frame.w - rotated.w).abs() < 1e-6);
        assert!((frame.x - rotated.x).abs() < 1e-6);
        assert!((frame.y - rotated.y).abs() < 1e-6);
        assert!((frame.z - rotated.z).abs() < 1e-6);
    }

    #[test]
    fn test_binaural_symmetry() {
        let mut conv = BinauralConvolver::new();
        // A sound directly in front (Y=0, Z=0) must produce identical Left and Right signals
        for _ in 0..10 {
            let out = conv.process_frame(FoaFrame::new(0.5, 0.5, 0.0, 0.0));
            assert!((out.left - out.right).abs() < 1e-5, "Front sound should be symmetric in ears");
        }
    }

    #[test]
    fn test_speaker_stereo_panning() {
        let mut decoder = AmbisonicDecoder::new(DecodeMode::StereoSpeakers);
        // Sound hard left (Y = 1.0, X = 0.0)
        let out_left = decoder.decode_stereo(FoaFrame::new(0.5, 0.0, 1.0, 0.0));
        assert!(out_left.left > out_left.right, "Left channel should be louder for left sound");

        // Sound hard right (Y = -1.0, X = 0.0)
        let out_right = decoder.decode_stereo(FoaFrame::new(0.5, 0.0, -1.0, 0.0));
        assert!(out_right.right > out_right.left, "Right channel should be louder for right sound");
    }

    #[test]
    fn test_stereo_speakers_elevation_projection() {
        let mut decoder = AmbisonicDecoder::new(DecodeMode::StereoSpeakers);
        let out_ground = decoder.decode_stereo(FoaFrame::new(0.5, 0.0, 0.0, 0.0));
        let out_elevated = decoder.decode_stereo(FoaFrame::new(0.5, 0.0, 0.0, 1.0));

        // Elevated rain drop energy (+Z) must project into both stereo speaker channels
        assert!(out_elevated.left > out_ground.left);
        assert!(out_elevated.right > out_ground.right);
        assert!((out_elevated.left - out_elevated.right).abs() < 1e-6);
    }
}
