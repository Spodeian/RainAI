// droplet_panning.wgsl - WebGPU Compute Shader: Massive Parallel Ambisonic Droplet Panning Engine
// Evaluates Gunn-Kinzer terminal velocity, trajectory angle deflection, kinetic energy,
// and Spherical Harmonics Y_lm(theta, phi) for up to 100,000 concurrent rain droplets directly on GPU.

struct DropletInstance {
    base_azimuth: f32,     // Base azimuth in radians [-pi, pi]
    base_elevation: f32,   // Base elevation in radians [-pi/2, pi/2]
    diameter_mm: f32,      // Droplet diameter D in mm (0.2 to 5.5 mm)
    wind_speed_ms: f32,    // Local wind speed in m/s
    wind_azimuth: f32,     // Wind azimuth in radians [-pi, pi]
    amplitude: f32,        // Base acoustic amplitude
    padding: vec2<f32>,    // 8-byte padding for strict 32-byte VRAM alignment
};

struct AmbisonicFrame {
    ch_w: f32,          // Omnidirectional (Pressure)
    ch_y: f32,          // Side (Left - Right)
    ch_z: f32,          // Vertical (Down - Up)
    ch_x: f32,          // Front (Back - Front)
};

@group(0) @binding(0) var<storage, read> droplets: array<DropletInstance>;
@group(0) @binding(1) var<storage, read_write> output_audio: array<AmbisonicFrame>;

// Workgroup size: 256 threads per workgroup
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let droplet_idx = global_id.x;
    let num_droplets = arrayLength(&droplets);
    
    if (droplet_idx >= num_droplets) {
        return;
    }
    
    let d = droplets[droplet_idx];
    
    // 1. Gunn-Kinzer (1949) Terminal Velocity on GPU
    // vt(D) = 9.65 - 10.3 * exp(-0.6 * D)
    let d_clamped = clamp(d.diameter_mm, 0.2, 5.5);
    let vt = max(0.8, 9.65 - 10.3 * exp(-0.6 * d_clamped));
    
    // 2. Resultant Impact Velocity & Kinetic Energy Flux (Erpul 2003)
    let v_res = sqrt(vt * vt + d.wind_speed_ms * d.wind_speed_ms);
    let volume = 0.523598 * (d_clamped * d_clamped * d_clamped); // (pi/6) * D^3
    let kinetic_energy = 0.5 * volume * (v_res * v_res);
    let energy_scale = clamp(kinetic_energy / 20.0, 0.1, 2.5);
    
    // 3. Drop-size Dependent Incident Trajectory (Quora / HESS)
    // Small drops (low vt) blow sideways at steep angles; large drops cut vertically
    let theta_traj = atan2(d.wind_speed_ms, vt);
    
    // Modulate elevation and azimuth based on wind vector
    let eff_elevation = clamp(d.base_elevation - theta_traj * 0.45, 0.15, 1.5707963);
    let eff_azimuth = d.base_azimuth + 0.3 * sin(d.wind_azimuth - d.base_azimuth) * sin(theta_traj);
    
    // 4. First-Order Ambisonics (FOA) Spherical Harmonics (SN3D / ACN)
    let cos_elev = cos(eff_elevation);
    let sin_elev = sin(eff_elevation);
    let cos_azim = cos(eff_azimuth);
    let sin_azim = sin(eff_azimuth);
    
    let gain_w = 1.0;                       // SN3D normalized zeroth-order
    let gain_y = sin_azim * cos_elev;       // Left - Right
    let gain_z = sin_elev;                  // Vertical rain impact
    let gain_x = cos_azim * cos_elev;       // Front - Back
    
    let sample_val = d.amplitude * energy_scale;
    
    output_audio[droplet_idx].ch_w = sample_val * gain_w;
    output_audio[droplet_idx].ch_y = sample_val * gain_y;
    output_audio[droplet_idx].ch_z = sample_val * gain_z;
    output_audio[droplet_idx].ch_x = sample_val * gain_x;
}
