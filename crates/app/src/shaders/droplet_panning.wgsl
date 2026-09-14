// droplet_panning.wgsl - WebGPU interactive Ambisonic radar & droplet particle visualization

struct Uniforms {
    resolution: vec2<f32>,
    time: f32,
    rain_intensity: f32,
    wind_speed: f32,
    wind_azimuth: f32,
    fireplace_pos: vec2<f32>, // (x, y) normalized [-1, 1]
    fireplace_intensity: f32,
    thunder_pos: vec2<f32>,
    thunder_intensity: f32,
    insect_pos: vec2<f32>,
    insect_density: f32,
    listener_yaw: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    // Fullscreen quad from 3 vertices
    let x = f32((in_vertex_index << 1u) & 2u);
    let y = f32(in_vertex_index & 2u);
    out.position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, 1.0 - y);
    return out;
}

// Pseudo-random noise
fn hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453123);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Center coordinates to [-1.0, 1.0]
    let p = (in.uv - 0.5) * 2.0;
    let dist = length(p);
    let angle = atan2(p.y, p.x) - u.listener_yaw;

    // Background dark acoustic chamber
    var col = vec3<f32>(0.04, 0.06, 0.09);

    // Concentric acoustic distance rings
    let ring1 = abs(dist - 0.33);
    let ring2 = abs(dist - 0.66);
    let ring3 = abs(dist - 1.0);
    let rings = min(min(ring1, ring2), ring3);
    if rings < 0.006 {
        col += vec3<f32>(0.08, 0.15, 0.22);
    }

    // Crosshairs (Azimuth 0, 90, 180, 270 deg)
    if (abs(p.x) < 0.003 || abs(p.y) < 0.003) && dist < 1.05 {
        col += vec3<f32>(0.06, 0.12, 0.18);
    }

    // Procedural rain droplet ripples
    if dist < 1.0 {
        let grid = floor(p * 12.0);
        let cell_rand = hash(grid);
        let drop_t = fract(u.time * (1.0 + u.rain_intensity * 2.0) + cell_rand);
        let ripple_r = drop_t * 0.15;
        let cell_uv = fract(p * 12.0) - 0.5;
        let d_ripple = abs(length(cell_uv) - ripple_r);
        if d_ripple < 0.02 && cell_rand < u.rain_intensity {
            let fade = (1.0 - drop_t) * 0.4;
            col += vec3<f32>(0.2, 0.5, 0.8) * fade;
        }
    }

    // Point Sources
    // 1. Fireplace (Warm Amber glow)
    if u.fireplace_intensity > 0.01 {
        let d_fire = length(p - u.fireplace_pos);
        let fire_glow = exp(-d_fire * 8.0) * u.fireplace_intensity;
        col += vec3<f32>(1.0, 0.5, 0.1) * fire_glow;
        if d_fire < 0.035 {
            col += vec3<f32>(1.0, 0.9, 0.5);
        }
    }

    // 2. Thunder Strike (Cyan flash)
    if u.thunder_intensity > 0.05 {
        let d_thunder = length(p - u.thunder_pos);
        let thunder_glow = exp(-d_thunder * 6.0) * u.thunder_intensity;
        col += vec3<f32>(0.3, 0.8, 1.0) * thunder_glow;
        if d_thunder < 0.04 {
            col += vec3<f32>(0.9, 0.95, 1.0);
        }
    }

    // 3. Insect Colony (Emerald Green swarm)
    if u.insect_density > 0.05 {
        let d_insect = length(p - u.insect_pos);
        let insect_glow = exp(-d_insect * 10.0) * u.insect_density;
        col += vec3<f32>(0.1, 0.9, 0.4) * insect_glow;
    }

    // Center listener orientation arrow
    if dist < 0.05 {
        col = vec3<f32>(0.9, 0.9, 1.0);
    }

    // Soft circular radar boundary vignette
    let vignette = smoothstep(1.02, 0.98, dist);
    col *= vignette;

    return vec4<f32>(col, 1.0);
}
