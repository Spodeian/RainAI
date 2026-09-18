//! WebGPU accelerated hardware backend for zero-copy neural inference.

use crate::model::{MoeExecutionMode, PrecisionFormat};
use crate::weight_loader::{WeightCache, WeightLoader};
use bytemuck::{Pod, Zeroable};
use ringbuf::HeapRb;
use shared::rain::CONDITION_DIM;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct LayerUniforms {
    pub in_dim: u32,
    pub out_dim: u32,
    pub has_bias: u32,
    pub padding: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct MoeUniforms {
    pub num_experts: u32,
    pub latent_dim: u32,
    pub decay_factor: f32,
    pub tau_moe: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct FoaUniforms {
    pub latent_dim: u32,
    pub bit_scale: f32,
    pub yaw_rad: f32,
    pub pitch_rad: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ConsistencyJumpUniforms {
    pub in_dim: u32,
    pub out_dim: u32,
    pub bit_scale: f32,
    pub yaw_rot: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct DeliberationUniforms {
    pub thinking_steps: u32,
    pub decay_rate: f32,
    pub step_size: f32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct MambaSsdUniforms {
    pub d_model: u32,
    pub d_state: u32,
    pub decay_scale: f32,
    pub momentum_alpha: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct DenseSoupUniforms {
    pub in_dim: u32,
    pub out_dim: u32,
    pub scale: f32,
    pub has_bias: u32,
}

pub struct GpuInferenceRunner {
    #[allow(dead_code)]
    device: wgpu::Device,
    queue: wgpu::Queue,
    is_lost: Arc<AtomicBool>,

    // Standard Multi-Pass Pipelines
    projection_pipeline: wgpu::ComputePipeline,
    recurrence_pipeline: wgpu::ComputePipeline,
    moe_pipeline: wgpu::ComputePipeline,
    foa_pipeline: wgpu::ComputePipeline,

    // Specialized Accelerated Pipelines
    consistency_jump_pipeline: wgpu::ComputePipeline,
    deliberation_pipeline: wgpu::ComputePipeline,
    mamba2_ssd_pipeline: wgpu::ComputePipeline,
    dense_soup_pipeline: wgpu::ComputePipeline,

    // Bind Groups
    #[allow(dead_code)]
    projection_bind_group: wgpu::BindGroup,
    #[allow(dead_code)]
    recurrence_bind_group: wgpu::BindGroup,
    #[allow(dead_code)]
    moe_bind_group: wgpu::BindGroup,
    #[allow(dead_code)]
    foa_bind_group: wgpu::BindGroup,
    #[allow(dead_code)]
    consistency_jump_bind_group: wgpu::BindGroup,
    #[allow(dead_code)]
    deliberation_bind_group: wgpu::BindGroup,
    #[allow(dead_code)]
    mamba2_ssd_bind_group: wgpu::BindGroup,
    #[allow(dead_code)]
    dense_soup_bind_group: wgpu::BindGroup,

    // VRAM Buffers
    #[allow(dead_code)]
    input_conditioning_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    latent_state_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    u_t_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    foa_output_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    foa_staging_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    ssd_scratch_buffer: wgpu::Buffer,

    // Configurable Uniform Buffers
    deliberation_uniforms_buf: wgpu::Buffer,
    foa_uniforms_buf: wgpu::Buffer,

    // Execution Configuration
    pub use_consistency_jump: bool,
    pub has_consistency_jump_head: bool,
    pub thinking_steps: u32,
    pub moe_mode: MoeExecutionMode,
    pub use_ssd_scan: bool,

    pub foa_producer: ringbuf::Producer<f32, Arc<HeapRb<f32>>>,
}

impl std::fmt::Debug for GpuInferenceRunner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuInferenceRunner")
            .field("is_lost", &self.is_lost)
            .field("use_consistency_jump", &self.use_consistency_jump)
            .field("has_consistency_jump_head", &self.has_consistency_jump_head)
            .field("thinking_steps", &self.thinking_steps)
            .field("moe_mode", &self.moe_mode)
            .field("use_ssd_scan", &self.use_ssd_scan)
            .finish()
    }
}

impl GpuInferenceRunner {
    pub async fn new(cache: &WeightCache) -> Result<(Self, ringbuf::Consumer<f32, Arc<HeapRb<f32>>>), String> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                apply_limit_buckets: false,
            })
            .await
            .map_err(|e| format!("Failed to find a suitable WebGPU adapter: {e}"))?;

        let is_lost = Arc::new(AtomicBool::new(false));

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| e.to_string())?;

        // Track device loss via uncaptured error handler
        {
            let lost_flag = is_lost.clone();
            device.on_uncaptured_error(Arc::new(move |err: wgpu::Error| {
                tracing::error!("WebGPU uncaptured error (possible device loss): {:?}", err);
                lost_flag.store(true, Ordering::SeqCst);
            }));
        }

        // 1. Compile Shader Modules
        let proj_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Layer Forward Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/layer_forward.wgsl").into()),
        });
        let mamba_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Mamba2 Recurrence Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/mamba2.wgsl").into()),
        });
        let moe_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MoE Dispatch Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/moe_dispatch.wgsl").into()),
        });
        let foa_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FOA Projection Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/foa_projection.wgsl").into()),
        });
        let jump_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Consistency Jump Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/consistency_jump.wgsl").into()),
        });
        let deliberation_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Deliberation Recurrence Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/mamba2_deliberation.wgsl").into()),
        });
        let ssd_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Mamba2 SSD Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/mamba2_ssd.wgsl").into()),
        });
        let soup_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Dense Soup Dispatch Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/dense_soup_dispatch.wgsl").into()),
        });

        // 2. Allocate VRAM Buffers
        let input_conditioning_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Input Conditioning Vector (c_t)"),
            size: (CONDITION_DIM * 4) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let ssd_scratch_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SSD / Dense Soup Scratch Buffer"),
            size: (64 * 4) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let latent_state_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Latent State Vector (s_t)"),
            size: (64 * 4) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let u_t_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Intermediate Projection (u_t)"),
            size: (64 * 4) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let foa_output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FOA Output Buffer (VRAM)"),
            size: (4 * 4) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let foa_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FOA Output Staging (MapRead)"),
            size: (4 * 4) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 3. Extract Weights from Cache
        let extract_weights = |layer_name: &str, expected_len: usize| -> Vec<f32> {
            if let Some(l) = cache.get(layer_name) {
                if l.format == PrecisionFormat::Ternary158 && !l.packed_weights.is_empty() {
                    let unpacked = WeightLoader::unpack_ternary_2bit(&l.packed_weights, l.num_elements());
                    unpacked.iter().map(|&v| (v as f32) * l.scale).collect()
                } else if !l.weights.is_empty() {
                    l.weights.clone()
                } else {
                    vec![0.0f32; expected_len]
                }
            } else {
                vec![0.0f32; expected_len]
            }
        };

        let cond_weights_data = extract_weights("encoder.cond_proj.weight", 64 * CONDITION_DIM);
        let cond_bias_data = extract_weights("encoder.cond_proj.bias", 64);
        let a_diag_data = extract_weights("mamba.A_diag.weight", 64);
        let b_diag_data = extract_weights("mamba.B_diag.weight", 64);
        let router_data = extract_weights("moe.router.weight", 8 * 64);
        let foa_weights_data = extract_weights("decoder.foa_proj.weight", 4 * 64);

        let has_consistency_jump_head = cache.get("consistency_jump.weight").is_some();
        let jump_weights_data = extract_weights("consistency_jump.weight", 4 * CONDITION_DIM);
        let jump_bias_data = extract_weights("consistency_jump.bias", 4);

        let cond_weights_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cond Weights"),
            contents: bytemuck::cast_slice(&cond_weights_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let cond_bias_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cond Bias"),
            contents: bytemuck::cast_slice(&cond_bias_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let a_diag_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("A Diag"),
            contents: bytemuck::cast_slice(&a_diag_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let b_diag_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("B Diag"),
            contents: bytemuck::cast_slice(&b_diag_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let router_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Router Weights"),
            contents: bytemuck::cast_slice(&router_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let foa_weights_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FOA Weights"),
            contents: bytemuck::cast_slice(&foa_weights_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let jump_weights_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Consistency Jump Weights"),
            contents: bytemuck::cast_slice(&jump_weights_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let jump_bias_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Consistency Jump Bias"),
            contents: bytemuck::cast_slice(&jump_bias_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let has_dense_soup = cache.get("mamba2.dense_soup.weight").is_some();
        let soup_weights_data = extract_weights("mamba2.dense_soup.weight", 64 * 64);
        let soup_bias_data = extract_weights("mamba2.dense_soup.bias", 64);

        let soup_weights_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dense Soup Weights"),
            contents: bytemuck::cast_slice(&soup_weights_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let soup_bias_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dense Soup Bias"),
            contents: bytemuck::cast_slice(&soup_bias_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Uniform Buffers
        let proj_uniforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Projection Uniforms"),
            contents: bytemuck::bytes_of(&LayerUniforms {
                in_dim: CONDITION_DIM as u32,
                out_dim: 64,
                has_bias: 1,
                padding: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let moe_uniforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MoE Uniforms"),
            contents: bytemuck::bytes_of(&MoeUniforms {
                num_experts: 8,
                latent_dim: 64,
                decay_factor: 0.85,
                tau_moe: 0.75,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let foa_uniforms_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FOA Uniforms"),
            contents: bytemuck::bytes_of(&FoaUniforms {
                latent_dim: 64,
                bit_scale: 1.0,
                yaw_rad: 0.0,
                pitch_rad: 0.0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let jump_uniforms_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Consistency Jump Uniforms"),
            contents: bytemuck::bytes_of(&ConsistencyJumpUniforms {
                in_dim: CONDITION_DIM as u32,
                out_dim: 4,
                bit_scale: 1.0,
                yaw_rot: 0.0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let deliberation_uniforms_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Deliberation Uniforms"),
            contents: bytemuck::bytes_of(&DeliberationUniforms {
                thinking_steps: 2,
                decay_rate: 0.85,
                step_size: 1.0,
                reserved: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let ssd_uniforms_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mamba2 SSD Uniforms"),
            contents: bytemuck::bytes_of(&MambaSsdUniforms {
                d_model: 64,
                d_state: 64,
                decay_scale: 1.0,
                momentum_alpha: 0.1,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let soup_uniforms_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dense Soup Uniforms"),
            contents: bytemuck::bytes_of(&DenseSoupUniforms {
                in_dim: 64,
                out_dim: 64,
                scale: 1.0,
                has_bias: if has_dense_soup { 1 } else { 0 },
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // 4. Create Pipelines & Bind Groups
        let projection_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Projection Pipeline"),
            layout: None,
            module: &proj_shader,
            entry_point: Some("dense_proj"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let recurrence_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Recurrence Pipeline"),
            layout: None,
            module: &mamba_shader,
            entry_point: Some("step_recurrence"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let moe_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MoE Pipeline"),
            layout: None,
            module: &moe_shader,
            entry_point: Some("route_and_decay"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let foa_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("FOA Pipeline"),
            layout: None,
            module: &foa_shader,
            entry_point: Some("project_foa"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let consistency_jump_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Consistency Jump Pipeline"),
            layout: None,
            module: &jump_shader,
            entry_point: Some("consistency_jump"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let deliberation_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Deliberation Pipeline"),
            layout: None,
            module: &deliberation_shader,
            entry_point: Some("deliberate_recurrence"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let mamba2_ssd_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Mamba2 SSD Pipeline"),
            layout: None,
            module: &ssd_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let dense_soup_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Dense Soup Pipeline"),
            layout: None,
            module: &soup_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let projection_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Projection Bind Group"),
            layout: &projection_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input_conditioning_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cond_weights_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: u_t_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: cond_bias_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: proj_uniforms.as_entire_binding() },
            ],
        });

        let recurrence_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Recurrence Bind Group"),
            layout: &recurrence_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: latent_state_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: a_diag_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b_diag_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: u_t_buffer.as_entire_binding() },
            ],
        });

        let moe_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MoE Bind Group"),
            layout: &moe_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: latent_state_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: router_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: moe_uniforms.as_entire_binding() },
            ],
        });

        let foa_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FOA Bind Group"),
            layout: &foa_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: latent_state_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: foa_weights_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: foa_output_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: foa_uniforms_buf.as_entire_binding() },
            ],
        });

        let consistency_jump_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Consistency Jump Bind Group"),
            layout: &consistency_jump_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input_conditioning_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: jump_weights_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: jump_bias_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: foa_output_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: jump_uniforms_buf.as_entire_binding() },
            ],
        });

        let deliberation_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Deliberation Bind Group"),
            layout: &deliberation_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: latent_state_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: a_diag_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b_diag_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: u_t_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: deliberation_uniforms_buf.as_entire_binding() },
            ],
        });

        let mamba2_ssd_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Mamba2 SSD Bind Group"),
            layout: &mamba2_ssd_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ssd_uniforms_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: u_t_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: a_diag_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: b_diag_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: latent_state_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: ssd_scratch_buffer.as_entire_binding() },
            ],
        });

        let dense_soup_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Dense Soup Bind Group"),
            layout: &dense_soup_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: soup_uniforms_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: latent_state_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: soup_weights_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: soup_bias_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: ssd_scratch_buffer.as_entire_binding() },
            ],
        });

        let ring_buffer = HeapRb::<f32>::new(4096);
        let (producer, consumer) = ring_buffer.split();

        Ok((
            Self {
                device,
                queue,
                is_lost,
                projection_pipeline,
                recurrence_pipeline,
                moe_pipeline,
                foa_pipeline,
                consistency_jump_pipeline,
                deliberation_pipeline,
                mamba2_ssd_pipeline,
                dense_soup_pipeline,
                projection_bind_group,
                recurrence_bind_group,
                moe_bind_group,
                foa_bind_group,
                consistency_jump_bind_group,
                deliberation_bind_group,
                mamba2_ssd_bind_group,
                dense_soup_bind_group,
                input_conditioning_buffer,
                latent_state_buffer,
                u_t_buffer,
                foa_output_buffer,
                foa_staging_buffer,
                ssd_scratch_buffer,
                deliberation_uniforms_buf,
                foa_uniforms_buf,
                use_consistency_jump: has_consistency_jump_head,
                has_consistency_jump_head,
                thinking_steps: 2,
                moe_mode: MoeExecutionMode::SparseDynamic,
                use_ssd_scan: false,
                foa_producer: producer,
            },
            consumer
        ))
    }

    #[inline]
    pub fn is_context_lost(&self) -> bool {
        self.is_lost.load(Ordering::SeqCst)
    }

    pub fn set_use_consistency_jump(&mut self, enabled: bool) {
        self.use_consistency_jump = enabled && self.has_consistency_jump_head;
    }

    pub fn set_moe_mode(&mut self, mode: MoeExecutionMode) {
        self.moe_mode = mode;
    }

    pub fn set_use_ssd_scan(&mut self, enabled: bool) {
        self.use_ssd_scan = enabled;
    }

    pub fn set_thinking_steps(&mut self, steps: u32) {
        self.thinking_steps = steps.clamp(1, 5);
        let uniforms = DeliberationUniforms {
            thinking_steps: self.thinking_steps,
            decay_rate: 0.85,
            step_size: 1.0,
            reserved: 0,
        };
        self.queue.write_buffer(&self.deliberation_uniforms_buf, 0, bytemuck::bytes_of(&uniforms));
    }

    pub fn set_listener_yaw(&mut self, yaw_rad: f32) {
        let uniforms = FoaUniforms {
            latent_dim: 64,
            bit_scale: 1.0,
            yaw_rad,
            pitch_rad: 0.0,
        };
        self.queue.write_buffer(&self.foa_uniforms_buf, 0, bytemuck::bytes_of(&uniforms));
    }

    /// Single-pass 1-step direct consistency distillation jump (554 -> 4 FOA)
    pub async fn step_consistency_jump_async(&mut self, conditioning: &[f32; CONDITION_DIM]) -> Result<(f32, f32, f32, f32), String> {
        if self.is_context_lost() {
            return Err("WebGPU context has been lost".to_string());
        }

        self.queue.write_buffer(&self.input_conditioning_buffer, 0, bytemuck::cast_slice(conditioning));

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Consistency Jump Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Consistency Jump Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.consistency_jump_pipeline);
            pass.set_bind_group(0, &self.consistency_jump_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&self.foa_output_buffer, 0, &self.foa_staging_buffer, 0, 16);
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = self.foa_staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

        if let Some(Ok(())) = receiver.receive().await {
            let data = buffer_slice.get_mapped_range().map_err(|e| e.to_string())?;
            let result: &[f32] = bytemuck::cast_slice(&data);
            let foa = (result[0], result[1], result[2], result[3]);
            drop(data);
            self.foa_staging_buffer.unmap();
            Ok(foa)
        } else {
            Err("Failed to read consistency jump output from WebGPU memory".to_string())
        }
    }

    /// Executes the WebGPU pipeline for a single audio frame step (with automatic consistency jump or multi-step deliberation)
    pub async fn step_async(&mut self, conditioning: &[f32; CONDITION_DIM]) -> Result<(f32, f32, f32, f32), String> {
        if self.is_context_lost() {
            return Err("WebGPU context has been lost".to_string());
        }

        if self.use_consistency_jump && self.has_consistency_jump_head {
            return self.step_consistency_jump_async(conditioning).await;
        }

        self.queue.write_buffer(&self.input_conditioning_buffer, 0, bytemuck::cast_slice(conditioning));

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Inference Multi-Pass Encoder"),
        });

        // Pass 1: Conditioning Projection (554 -> 64)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Projection Pass"), timestamp_writes: None });
            pass.set_pipeline(&self.projection_pipeline);
            pass.set_bind_group(0, &self.projection_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        // Pass 2: In-VRAM Recurrence, Multi-Step Deliberation, or Parallel SSD Scan
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Recurrence/Deliberation/SSD Pass"), timestamp_writes: None });
            if self.use_ssd_scan {
                pass.set_pipeline(&self.mamba2_ssd_pipeline);
                pass.set_bind_group(0, &self.mamba2_ssd_bind_group, &[]);
            } else if self.thinking_steps > 1 {
                pass.set_pipeline(&self.deliberation_pipeline);
                pass.set_bind_group(0, &self.deliberation_bind_group, &[]);
            } else {
                pass.set_pipeline(&self.recurrence_pipeline);
                pass.set_bind_group(0, &self.recurrence_bind_group, &[]);
            }
            pass.dispatch_workgroups(1, 1, 1);
        }

        // Pass 3: MoE Routing & Decay OR Single-Pass Dense Soup
        if self.moe_mode.is_dense_soup() {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Dense Soup Pass"), timestamp_writes: None });
                pass.set_pipeline(&self.dense_soup_pipeline);
                pass.set_bind_group(0, &self.dense_soup_bind_group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            encoder.copy_buffer_to_buffer(
                &self.ssd_scratch_buffer, 0,
                &self.latent_state_buffer, 0,
                64 * 4
            );
        } else {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("MoE Dispatch Pass"), timestamp_writes: None });
            pass.set_pipeline(&self.moe_pipeline);
            pass.set_bind_group(0, &self.moe_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        // Pass 4: Ambisonic FOA Projection (64 -> 4)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("FOA Projection Pass"), timestamp_writes: None });
            pass.set_pipeline(&self.foa_pipeline);
            pass.set_bind_group(0, &self.foa_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &self.foa_output_buffer, 0,
            &self.foa_staging_buffer, 0,
            16
        );

        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = self.foa_staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

        if let Some(Ok(())) = receiver.receive().await {
            let data = buffer_slice.get_mapped_range().map_err(|e| e.to_string())?;
            let result: &[f32] = bytemuck::cast_slice(&data);
            let foa = (result[0], result[1], result[2], result[3]);
            
            drop(data);
            self.foa_staging_buffer.unmap();
            
            Ok(foa)
        } else {
            Err("Failed to read FOA output from WebGPU memory".to_string())
        }
    }

    /// Asynchronously processes a block of conditioning vectors on the GPU and pushes
    /// synthesized Ambisonic FOA frames into the ring buffer for wait-free audio thread reads.
    pub async fn step_block_async(&mut self, conditioning_block: &[[f32; CONDITION_DIM]]) -> Result<usize, String> {
        if self.is_context_lost() {
            return Err("WebGPU context has been lost".to_string());
        }

        let mut frames_pushed = 0;
        for cond in conditioning_block {
            let (w, x, y, z) = self.step_async(cond).await?;
            if self.foa_producer.free_len() >= 4 {
                let _ = self.foa_producer.push(w);
                let _ = self.foa_producer.push(x);
                let _ = self.foa_producer.push(y);
                let _ = self.foa_producer.push(z);
                frames_pushed += 1;
            } else {
                break;
            }
        }

        Ok(frames_pushed)
    }
}