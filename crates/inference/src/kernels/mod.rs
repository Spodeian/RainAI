pub mod bf16_simd;
pub mod int8_simd;
pub mod layer_forward;
pub mod mamba2_simd;
pub mod moe_dispatch;
pub mod posit_simd;
pub mod quant_activations;
pub mod rk4_flow_solver;
pub mod ternary_simd;

pub use bf16_simd::bf16_matmul_simd_f32;
pub use int8_simd::int8_matmul_simd_f32;
pub use layer_forward::dense_projection;
pub use mamba2_simd::step_recurrence_f32;
pub use moe_dispatch::{route_and_decay, route_and_decay_with_history};
pub use posit_simd::posit8_matmul_simd_f32;
pub use quant_activations::simd_silu_in_place;
pub use rk4_flow_solver::Rk4FlowSolver;
pub use ternary_simd::ternary_matmul_simd_f32;

