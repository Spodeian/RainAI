//! WebGPU accelerated hardware backend for zero-copy neural inference.

use crate::model::PrecisionFormat;
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
struct LayerUniforms {
    in_dim: u32,
    out_dim: u32,
    has_bias: u32,
    padding: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct MoeUniforms {
    num_experts: u32,
    latent_dim: u32,
    decay_factor: f32,
    padding: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct FoaUniforms {
    latent_dim: u32,
    bit_scale: f32,
    padding: [u32; 2],
}

pub struct GpuInferenceRunner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    is_lost: Arc<AtomicBool>,
    
    // Pipelines
    projection_pipeline: wgpu::ComputePipeline,
    recurrence_pipeline: wgpu::ComputePipeline,
    moe_pipeline: wgpu::ComputePipeline,
    foa_pipeline: wgpu::ComputePipeline,
    
    // Bind Groups
    projection_bind_group: wgpu::BindGroup,
    recurrence_bind_group: wgpu::BindGroup,
    moe_bind_group: wgpu::BindGroup,
    foa_bind_group: wgpu::BindGroup,
    
    // VRAM Buffers
    input_conditioning_buffer: wgpu::Buffer,
    latent_state_buffer: wgpu::Buffer,
    u_t_buffer: wgpu::Buffer,
    foa_output_buffer: wgpu::Buffer,
    foa_staging_buffer: wgpu::Buffer,
    
    pub foa_producer: ringbuf::Producer<f32, Arc<HeapRb<f32>>>,
}

impl std::fmt::Debug for GpuInferenceRunner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuInferenceRunner")
            .field("is_lost", &self.is_lost)
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
            })
            .await
            .ok_or("Failed to find a suitable WebGPU adapter")?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .map_err(|e| e.to_string())?;

        let is_lost = Arc::new(AtomicBool::new(false));
        let lost_flag = is_lost.clone();

        device.set_device_lost_callback(move |reason, message| {
            tracing::error!("WebGPU device lost! Reason: {:?}, Message: {}", reason, message);
            lost_flag.store(true, Ordering::SeqCst);
        });

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

        // 2. Allocate VRAM Buffers
        let input_conditioning_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Input Conditioning Vector (c_t)"),
            size: (CONDITION_DIM * 4) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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

        // 3. Extract Weights from Cache (handling packed ternary and uncompressed fallbacks)
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
                padding: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let foa_uniforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FOA Uniforms"),
            contents: bytemuck::bytes_of(&FoaUniforms {
                latent_dim: 64,
                bit_scale: 1.0,
                padding: [0, 0],
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // 4. Create Pipelines & Bind Groups
        let projection_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Projection Pipeline"),
            layout: None,
            module: &proj_shader,
            entry_point: "dense_proj",
        });
        let recurrence_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Recurrence Pipeline"),
            layout: None,
            module: &mamba_shader,
            entry_point: "step_recurrence",
        });
        let moe_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MoE Pipeline"),
            layout: None,
            module: &moe_shader,
            entry_point: "route_and_decay",
        });
        let foa_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("FOA Pipeline"),
            layout: None,
            module: &foa_shader,
            entry_point: "project_foa",
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
                wgpu::BindGroupEntry { binding: 3, resource: foa_uniforms.as_entire_binding() },
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
                projection_bind_group,
                recurrence_bind_group,
                moe_bind_group,
                foa_bind_group,
                input_conditioning_buffer,
                latent_state_buffer,
                u_t_buffer,
                foa_output_buffer,
                foa_staging_buffer,
                foa_producer: producer,
            },
            consumer
        ))
    }

    #[inline]
    pub fn is_context_lost(&self) -> bool {
        self.is_lost.load(Ordering::SeqCst)
    }

    /// Executes the full multi-pass WebGPU pipeline for a single audio frame step
    pub async fn step_async(&mut self, conditioning: &[f32; CONDITION_DIM]) -> Result<(f32, f32, f32, f32), String> {
        if self.is_context_lost() {
            return Err("WebGPU context has been lost".to_string());
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

        // Pass 2: Mamba2 Recurrence
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Recurrence Pass"), timestamp_writes: None });
            pass.set_pipeline(&self.recurrence_pipeline);
            pass.set_bind_group(0, &self.recurrence_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        // Pass 3: MoE Routing & Decay
        {
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

        self.device.poll(wgpu::Maintain::Wait);

        if let Some(Ok(())) = receiver.receive().await {
            let data = buffer_slice.get_mapped_range();
            let result: &[f32] = bytemuck::cast_slice(&data);
            let foa = (result[0], result[1], result[2], result[3]);
            
            drop(data);
            self.foa_staging_buffer.unmap();
            
            Ok(foa)
        } else {
            Err("Failed to read FOA output from WebGPU memory".to_string())
        }
    }
}
