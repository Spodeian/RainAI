use audio::{ProceduralSynthesizer, SubtractiveFilterbank16, NOMINAL_BAND_FREQS, NOMINAL_BAND_Q, LEARNED_DRIFT};
use audio::decoder::FoaFrame;
use shared::RainState;

#[test]
fn test_procedural_silence_when_stopped() {
    let mut synth = ProceduralSynthesizer::new(48000.0);
    let mut state = RainState::default();
    state.is_playing = false;

    let frame = synth.process_frame(&state);
    assert_eq!(frame, FoaFrame::default());
}

#[test]
fn test_procedural_audio_generation() {
    let mut synth = ProceduralSynthesizer::new(48000.0);
    let mut state = RainState::default();
    state.is_playing = true;
    state.master_volume = 1.0;

    let mut nonzero_count = 0;
    for _ in 0..100 {
        let frame = synth.process_frame(&state);
        if frame.w.abs() > 1e-4 {
            nonzero_count += 1;
        }
    }
    assert!(nonzero_count > 90, "Procedural synthesizer should generate continuous audio");
}

#[test]
fn test_subtractive_filterbank_16_bands() {
    let filterbank = SubtractiveFilterbank16::new(48000.0);
    assert_eq!(filterbank.filters.len(), 16);
    assert_eq!(NOMINAL_BAND_FREQS.len(), 16);
    assert_eq!(NOMINAL_BAND_Q.len(), 16);
    assert_eq!(LEARNED_DRIFT.len(), 16);

    assert!(LEARNED_DRIFT[0] > 0.0);
    assert!(LEARNED_DRIFT[3] < 0.0);
}
