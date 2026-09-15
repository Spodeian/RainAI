//! Offline faster-than-realtime streaming WAV exporter.
//!
//! Generates WAV audio in discrete, low-memory chunks and streams them directly
//! to a writer without buffering the entire uncompressed audio file in memory.

use crate::decoder::{AmbisonicDecoder, DecodeMode, FoaFrame};
use crate::procedural::ProceduralSynthesizer;
use shared::rain::RainState;
use std::io::{self, Write};

pub const CHUNK_FRAMES: usize = 2048;

/// Writes a standard 32-bit IEEE float WAV file header
pub fn write_wav_header<W: Write>(
    writer: &mut W,
    num_channels: u16,
    sample_rate: u32,
    num_frames: u32,
) -> io::Result<()> {
    let bits_per_sample = 32u16;
    let byte_rate = sample_rate * u32::from(num_channels) * u32::from(bits_per_sample / 8);
    let block_align = num_channels * (bits_per_sample / 8);
    let data_size = num_frames * u32::from(block_align);
    let riff_size = 36 + data_size;

    // RIFF header
    writer.write_all(b"RIFF")?;
    writer.write_all(&riff_size.to_le_bytes())?;
    writer.write_all(b"WAVE")?;

    // fmt subchunk (IEEE float = format 3)
    writer.write_all(b"fmt ")?;
    writer.write_all(&16u32.to_le_bytes())?; // Subchunk1Size (16 for PCM/float)
    writer.write_all(&3u16.to_le_bytes())?;  // AudioFormat: 3 = IEEE Float
    writer.write_all(&num_channels.to_le_bytes())?;
    writer.write_all(&sample_rate.to_le_bytes())?;
    writer.write_all(&byte_rate.to_le_bytes())?;
    writer.write_all(&block_align.to_le_bytes())?;
    writer.write_all(&bits_per_sample.to_le_bytes())?;

    // data subchunk
    writer.write_all(b"data")?;
    writer.write_all(&data_size.to_le_bytes())?;

    Ok(())
}

/// Streams faster-than-realtime synthesized audio into any `Write` stream in small chunks
pub fn render_wav_stream<W: Write>(
    state: &RainState,
    duration_secs: f32,
    sample_rate: u32,
    mode: DecodeMode,
    writer: &mut W,
    mut progress_cb: impl FnMut(f32),
) -> io::Result<u64> {
    let num_channels = match mode {
        DecodeMode::RawFoaPassthrough => 4u16,
        DecodeMode::Surround71 => 8u16,
        DecodeMode::BinauralHeadphones | DecodeMode::StereoSpeakers => 2u16,
    };

    let total_frames = (duration_secs * sample_rate as f32).round() as u32;
    write_wav_header(writer, num_channels, sample_rate, total_frames)?;

    let mut synth = ProceduralSynthesizer::new(sample_rate as f32);
    
    // Load inference weights and setup engine
    let weight_cache = inference::weight_loader::WeightLoader::load_embedded_ternary().unwrap_or_default();
    let mut runner = inference::runner::InferenceRunner::new(state.quality_tier, weight_cache);
    let mut decoder = AmbisonicDecoder::new(mode);

    let mut frames_remaining = total_frames;
    let mut foa_buf = [FoaFrame::default(); CHUNK_FRAMES];
    let mut bytes_written = 44u64; // WAV header size
    
    // Force active synth state for offline export even if UI is paused
    let mut active_state = state.clone();
    active_state.is_playing = true;
    let blend = active_state.telemetry.synthesis_blend;

    while frames_remaining > 0 {
        let frames_to_process = (frames_remaining as usize).min(CHUNK_FRAMES);
        let slice = &mut foa_buf[..frames_to_process];

        // Synthesize mixed Procedural/Neural frame buffer
        for frame in slice.iter_mut() {
            let foa_proc = synth.process_frame(&active_state);
            
            *frame = if blend >= 0.999 {
                foa_proc
            } else {
                let cond = active_state.to_conditioning_array();
                let (nw, nx, ny, nz) = runner.step(&cond);
                let foa_neural = crate::decoder::FoaFrame::new(nw, nx, ny, nz);

                crate::decoder::FoaFrame::new(
                    foa_proc.w * blend + foa_neural.w * (1.0 - blend),
                    foa_proc.x * blend + foa_neural.x * (1.0 - blend),
                    foa_proc.y * blend + foa_neural.y * (1.0 - blend),
                    foa_proc.z * blend + foa_neural.z * (1.0 - blend),
                )
            };
        }

        match mode {
            DecodeMode::BinauralHeadphones | DecodeMode::StereoSpeakers => {
                for frame in slice.iter() {
                    let stereo = decoder.decode_stereo(*frame);
                    let left = crate::engine::soft_limit(stereo.left);
                    let right = crate::engine::soft_limit(stereo.right);
                    writer.write_all(&left.to_le_bytes())?;
                    writer.write_all(&right.to_le_bytes())?;
                    bytes_written += 8;
                }
            }
            DecodeMode::RawFoaPassthrough => {
                for frame in slice.iter() {
                    writer.write_all(&frame.w.to_le_bytes())?;
                    writer.write_all(&frame.x.to_le_bytes())?;
                    writer.write_all(&frame.y.to_le_bytes())?;
                    writer.write_all(&frame.z.to_le_bytes())?;
                    bytes_written += 16;
                }
            }
            DecodeMode::Surround71 => {
                for frame in slice.iter() {
                    let s71 = decoder.decode_71(*frame);
                    writer.write_all(&s71.left.to_le_bytes())?;
                    writer.write_all(&s71.right.to_le_bytes())?;
                    writer.write_all(&s71.center.to_le_bytes())?;
                    writer.write_all(&s71.lfe.to_le_bytes())?;
                    writer.write_all(&s71.left_surround.to_le_bytes())?;
                    writer.write_all(&s71.right_surround.to_le_bytes())?;
                    writer.write_all(&s71.left_back.to_le_bytes())?;
                    writer.write_all(&s71.right_back.to_le_bytes())?;
                    bytes_written += 32;
                }
            }
        }

        frames_remaining -= frames_to_process as u32;
        let progress = 1.0 - (frames_remaining as f32 / total_frames as f32);
        progress_cb(progress);
    }

    Ok(bytes_written)
}
