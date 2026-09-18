#![allow(unsafe_code)]
//! Neural Inference Runtime for Continuous Mixed-Precision RainAI Models.
//!
//! Exposes unified execution engines across CPU SIMD vector baselines, Hugging Face Candle
//! SafeTensors, ONNX Runtime graphs, and WebGPU WGSL compute shaders. Supports multi-tier
//! quantization slices ranging from 1.58-bit Ternary ($\{-1, 0, +1\}$) to Master-Class FP32.
//!
//! # Architecture & Modules
//!
//! - [`runner`]: High-level inference coordinator executing Mamba-2 MoE trajectory step recurrence
//!   and spatial autoencoder forward passes with zero runtime allocations.
//! - [`webgpu_backend`]: Hardware-accelerated GPU inference engine compiling native WGSL shaders
//!   with asynchronous buffer staging, pipeline caching, and WebGPU/WASM browser bindings.
//! - [`kernels`]: Hand-tuned CPU vector kernels implementing fast matrix multiplication, 2-bit
//!   ternary weight packing/unpacking, and continuous Box-Cox inverse activation mappings.
//! - [`asset_manager`]: Multi-tier model artifact loader fetching weights from local cache,
//!   embedded static slices, or remote distribution endpoints.
//! - [`weight_loader`]: Safe parser for custom binary quantization slices, SafeTensors, and ONNX files.
//! - [`weight_cache_manager`]: In-memory weight cache with instant LRU eviction and memory-mapped buffers.
//! - [`model`]: Data models representing layer weights, precision formats, and model manifests.

pub mod asset_manager;
pub mod engram;
pub mod kernels;
pub mod model;
pub mod runner;
pub mod webgpu_backend;
pub mod weight_cache_manager;
pub mod weight_loader;

pub use asset_manager::*;
pub use engram::*;
pub use kernels::*;
pub use model::*;
pub use runner::*;
pub use weight_cache_manager::*;
pub use weight_loader::*;


use ringbuf::HeapRb;
use std::sync::Arc;
use webgpu_backend::GpuInferenceRunner;

#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum InferenceBackend {
    CpuNeural(InferenceRunner),
    WebGpuNeural(GpuInferenceRunner),
}

pub struct InferenceEngine {
    pub primary: InferenceBackend,
    pub fallback: InferenceRunner, // Always maintain a CPU runner loaded with weights
    pub foa_consumer: Option<ringbuf::Consumer<f32, Arc<HeapRb<f32>>>>,
}

impl std::fmt::Debug for InferenceEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InferenceEngine")
            .field("primary", &self.primary)
            .field("fallback", &self.fallback)
            .field("has_foa_consumer", &self.foa_consumer.is_some())
            .finish()
    }
}

impl InferenceEngine {
    pub fn new(primary: InferenceBackend, fallback: InferenceRunner) -> Self {
        Self {
            primary,
            fallback,
            foa_consumer: None,
        }
    }

    pub fn with_consumer(
        primary: InferenceBackend,
        fallback: InferenceRunner,
        foa_consumer: Option<ringbuf::Consumer<f32, Arc<HeapRb<f32>>>>,
    ) -> Self {
        Self {
            primary,
            fallback,
            foa_consumer,
        }
    }

    pub fn attach_foa_consumer(&mut self, consumer: ringbuf::Consumer<f32, Arc<HeapRb<f32>>>) {
        self.foa_consumer = Some(consumer);
    }

    pub fn infer(&mut self, conditioning: &[f32; shared::rain::CONDITION_DIM]) -> Result<(f32, f32, f32, f32), String> {
        // Fast path: if async WebGPU producer has pushed synthesized Ambisonic FOA frames,
        // pop 4 samples directly with lock-free wait-free semantics!
        if let Some(consumer) = &mut self.foa_consumer {
            if consumer.len() >= 4 {
                let mut w = 0.0f32;
                let mut x = 0.0f32;
                let mut y = 0.0f32;
                let mut z = 0.0f32;
                if let Some(v) = consumer.pop() { w = v; }
                if let Some(v) = consumer.pop() { x = v; }
                if let Some(v) = consumer.pop() { y = v; }
                if let Some(v) = consumer.pop() { z = v; }
                return Ok((w, x, y, z));
            }
        }

        match &mut self.primary {
            InferenceBackend::CpuNeural(cpu_runner) => {
                if cpu_runner.use_consistency_jump && cpu_runner.has_consistency_jump_head() {
                    Ok(cpu_runner.fast_consistency_step(conditioning))
                } else {
                    Ok(cpu_runner.step(conditioning))
                }
            }
            InferenceBackend::WebGpuNeural(gpu_runner) => {
                // Check if WebGPU context was lost due to browser backgrounding or resource exhaustion
                if gpu_runner.is_context_lost() {
                    tracing::warn!("WebGPU context loss detected during inference. Hot-swapping to CPU SIMD fallback.");

                    // Fallback execution
                    let result = if self.fallback.use_consistency_jump && self.fallback.has_consistency_jump_head() {
                        self.fallback.fast_consistency_step(conditioning)
                    } else {
                        self.fallback.step(conditioning)
                    };

                    // Mutate primary state to CPU to prevent repeated checks
                    self.primary = InferenceBackend::CpuNeural(self.fallback.clone());

                    return Ok(result);
                }

                // Normal execution path using fallback/cpu runner or consistency jump
                if self.fallback.use_consistency_jump && self.fallback.has_consistency_jump_head() {
                    Ok(self.fallback.fast_consistency_step(conditioning))
                } else {
                    Ok(self.fallback.step(conditioning))
                }
            }
        }
    }
}
