use audio::decoder::*;

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
    for _ in 0..10 {
        let out = conv.process_frame(FoaFrame::new(0.5, 0.5, 0.0, 0.0));
        assert!((out.left - out.right).abs() < 1e-5, "Front sound should be symmetric in ears");
    }
}

#[test]
fn test_speaker_stereo_panning() {
    let mut decoder = AmbisonicDecoder::new(DecodeMode::StereoSpeakers);
    let out_left = decoder.decode_stereo(FoaFrame::new(0.5, 0.0, 1.0, 0.0));
    assert!(out_left.left > out_left.right, "Left channel should be louder for left sound");

    let out_right = decoder.decode_stereo(FoaFrame::new(0.5, 0.0, -1.0, 0.0));
    assert!(out_right.right > out_right.left, "Right channel should be louder for right sound");
}

#[test]
fn test_stereo_speakers_elevation_projection() {
    let mut decoder = AmbisonicDecoder::new(DecodeMode::StereoSpeakers);
    let out_ground = decoder.decode_stereo(FoaFrame::new(0.5, 0.0, 0.0, 0.0));
    let out_elevated = decoder.decode_stereo(FoaFrame::new(0.5, 0.0, 0.0, 1.0));

    assert!(out_elevated.left > out_ground.left);
    assert!(out_elevated.right > out_ground.right);
    assert!((out_elevated.left - out_elevated.right).abs() < 1e-6);
}
