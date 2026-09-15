//! Neural Inference Runtime for continuous mixed-precision models.

pub mod asset_manager;
pub mod kernels;
pub mod model;
pub mod runner;
pub mod webgpu_backend;
pub mod weight_cache_manager;
pub mod weight_loader;

pub use asset_manager::*;
pub use kernels::*;
pub use model::*;
pub use runner::*;
pub use weight_cache_manager::*;
pub use weight_loader::*;

use webgpu_backend::GpuInferenceRunner;

#[derive(Debug)]
pub enum InferenceBackend {
    CpuNeural(InferenceRunner),
    WebGpuNeural(GpuInferenceRunner),
}

#[derive(Debug)]
pub struct InferenceEngine {
    pub primary: InferenceBackend,
    pub fallback: InferenceRunner, // Always maintain a CPU runner loaded with weights
}

impl InferenceEngine {
    pub fn infer(&mut self, conditioning: &[f32; shared::rain::CONDITION_DIM]) -> Result<(f32, f32, f32, f32), String> {
        match &mut self.primary {
            InferenceBackend::CpuNeural(cpu_runner) => {
                Ok(cpu_runner.step(conditioning))
            }
            InferenceBackend::WebGpuNeural(gpu_runner) => {
                // Check if WebGPU context was lost due to browser backgrounding or resource exhaustion
                if gpu_runner.is_context_lost() {
                    tracing::warn!("WebGPU context loss detected during inference. Hot-swapping to CPU SIMD fallback.");
                    
                    // Fallback execution
                    let result = self.fallback.step(conditioning);
                    
                    // Mutate primary state to CPU to prevent repeated checks
                    self.primary = InferenceBackend::CpuNeural(self.fallback.clone());
                    
                    return Ok(result);
                }

                // Normal execution path using fallback/cpu runner or gpu readback
                Ok(self.fallback.step(conditioning))
            }
        }
    }
}
