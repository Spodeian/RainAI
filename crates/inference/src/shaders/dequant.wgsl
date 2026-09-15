@group(0) @binding(0) var<storage, read> packed_weights: array<u32>;
@group(0) @binding(1) var<storage, read_write> expanded_f32: array<f32>;

struct PushConstants {
    total_elements: u32,
    gamma: f32,
}
var<push_constant> params: PushConstants;

@compute @workgroup_size(256)
fn dequantize_ternary_2bit(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let u32_idx = global_id.x;
    
    // Each u32 contains 4 bytes, yielding 16 x 2-bit weights
    let start_f32_idx = u32_idx * 16u;
    
    if start_f32_idx < params.total_elements {
        let chunk = packed_weights[u32_idx];
        
        for (var i = 0u; i < 16u; i = i + 1u) {
            let element_idx = start_f32_idx + i;
            
            if element_idx < params.total_elements {
                let shift = i * 2u;
                let code = (chunk >> shift) & 0x03u;
                
                var w = 0.0;
                if code == 0x01u {
                    w = 1.0;
                } else if code == 0x03u {
                    w = -1.0;
                }
                
                expanded_f32[element_idx] = w * params.gamma;
            }
        }
    }
}
