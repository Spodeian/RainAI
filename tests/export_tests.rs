use audio::export::*;
use audio::DecodeMode;
use shared::RainState;

#[test]
fn test_wav_export_streaming_size() {
    let mut state = RainState::default();
    state.is_playing = true;

    let mut output = Vec::new();
    let bytes = render_wav_stream(
        &state,
        0.1,
        48000,
        DecodeMode::StereoSpeakers,
        &mut output,
        |_| {},
    )
    .expect("Render should succeed");

    assert_eq!(bytes, 38_444);
    assert_eq!(output.len(), 38_444);
    assert_eq!(&output[0..4], b"RIFF");
    assert_eq!(&output[8..12], b"WAVE");
}