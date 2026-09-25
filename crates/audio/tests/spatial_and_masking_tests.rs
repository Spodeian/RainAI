//! Integration tests for SOFA HRTF 3D binaural convolution and adaptive noise masking.

use audio::{AmbientNoiseMasker, NoiseSpectrum, SofaSpatializer, SphericalPosition};

#[test]
fn test_sofa_hrtf_binaural_spatialization() {
    let mut spatializer = SofaSpatializer::new(48000);

    // Front source: azimuth = 0, elevation = 0
    spatializer.add_hrir(
        SphericalPosition::new(0.0, 0.0, 1.0),
        vec![1.0, 0.1, 0.0],
        vec![1.0, 0.1, 0.0],
    );

    // Right source: azimuth = 90 (closer to right ear, right ear has louder immediate onset)
    spatializer.add_hrir(
        SphericalPosition::new(90.0, 0.0, 1.0),
        vec![0.3, 0.1, 0.0],
        vec![1.2, 0.2, 0.0],
    );

    let input_signal = vec![0.5, -0.5, 0.2];

    // Spatialize at 90 deg azimuth
    let (left, right) = spatializer.spatialize_mono(&input_signal, SphericalPosition::new(90.0, 0.0, 1.0));

    assert_eq!(left.len(), input_signal.len() + 2);
    assert_eq!(right.len(), input_signal.len() + 2);

    // Right ear must receive more immediate energy than left ear
    let right_energy: f32 = right.iter().map(|s| s * s).sum();
    let left_energy: f32 = left.iter().map(|s| s * s).sum();
    assert!(right_energy > left_energy, "Right ear energy ({}) must exceed left ({}) for sound at 90 deg", right_energy, left_energy);
}

#[test]
fn test_ambient_noise_masking_adaptation() {
    let mut masker = AmbientNoiseMasker::new(10.0);

    // Quiet room baseline: -60 dBFS
    let quiet_spectrum = NoiseSpectrum {
        rms_db: -60.0,
        low_energy: 0.001,
        mid_energy: 0.001,
        high_energy: 0.0005,
    };
    let rec_quiet = masker.update(&quiet_spectrum);
    assert!(rec_quiet.rain_density_scale <= 1.05);
    assert!(rec_quiet.gain_boost_db <= 0.1);

    // Loud room: -35 dBFS (speech and street noise)
    let loud_spectrum = NoiseSpectrum {
        rms_db: -35.0,
        low_energy: 0.08,
        mid_energy: 0.15,
        high_energy: 0.05,
    };

    // Adapt over multiple buffer ticks
    let mut rec_loud = rec_quiet;
    for _ in 0..10 {
        rec_loud = masker.update(&loud_spectrum);
    }

    // Masker should adaptively increase rainfall density and master gain
    assert!(rec_loud.rain_density_scale > rec_quiet.rain_density_scale);
    assert!(rec_loud.gain_boost_db > rec_quiet.gain_boost_db);
    assert!(rec_loud.low_cut_hz >= 80.0);
}
