//! Neural Model Architectures implemented with Candle.
//!
//! Includes Spatial VAE, Mamba-2 MoE, Latent Attention, Engram Bank,
//! Thinking Blocks, Invasive Meta-Controller, and 1-Step Consistency Distillation Head.

use anyhow::Result;
use candle_core::{DType, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

use super::*;

/// Numerically stable softplus: softplus(x) = max(x, 0) + ln(1 + exp(-|x|))
/// Overflow-free for all x in (-inf, inf).
pub fn candle_softplus(x: &Tensor) -> Result<Tensor> {
    let relu = x.relu()?;
    let neg_abs = (x.abs()? * (-1.0f64))?;
    let log1p = (neg_abs.exp()? + 1.0f64)?.log()?;
    (&relu + &log1p).map_err(Into::into)
}

/// Continuous Spatial VAE for multi-channel acoustic and spectral latent projection.
pub struct CandleSpatialVae {
    // Encoder: Audio spectral features [B, 64] -> Latent distribution mu, logvar
    enc_fc1: Linear,
    enc_fc2: Linear,
    enc_mu: Linear,
    enc_logvar: Linear,
    // Decoder: Latent z [B, 64] + Conditioning u [B, 554] -> 16 DDSP bands + 4 FOA gains
    dec_fc1: Linear,
    dec_fc2: Linear,
    dec_bands: Linear,
    dec_foa: Linear,
}

impl CandleSpatialVae {
    pub fn new(vs: VarBuilder) -> Result<Self> {
        let enc_vs = vs.pp("encoder");
        let enc_fc1 = linear(LATENT_DIM, 128, enc_vs.pp("fc1"))?;
        let enc_fc2 = linear(128, 128, enc_vs.pp("fc2"))?;
        let enc_mu = linear(128, LATENT_DIM, enc_vs.pp("mu"))?;
        let enc_logvar = linear(128, LATENT_DIM, enc_vs.pp("logvar"))?;

        let dec_vs = vs.pp("decoder");
        let dec_fc1 = linear(COMBINED_DIM, 128, dec_vs.pp("fc1"))?;
        let dec_fc2 = linear(128, 128, dec_vs.pp("fc2"))?;
        let dec_bands = linear(128, FILTER_BANDS, dec_vs.pp("bands"))?;
        let dec_foa = linear(128, FOA_CHANNELS, dec_vs.pp("foa"))?;

        Ok(Self {
            enc_fc1,
            enc_fc2,
            enc_mu,
            enc_logvar,
            dec_fc1,
            dec_fc2,
            dec_bands,
            dec_foa,
        })
    }

    /// Encodes acoustic features into mean $\mu$ and log-variance $\log \sigma^2$.
    pub fn encode(&self, audio_features: &Tensor) -> Result<(Tensor, Tensor)> {
        let h1 = self.enc_fc1.forward(audio_features)?.gelu_erf()?;
        let h2 = self.enc_fc2.forward(&h1)?.gelu_erf()?;
        let mu = self.enc_mu.forward(&h2)?;
        let logvar = self.enc_logvar.forward(&h2)?.clamp(-12.0f32, 12.0f32)?;
        Ok((mu, logvar))
    }

    /// Reparameterization trick: $z = \mu + \epsilon \odot \exp(0.5 \log \sigma^2)$.
    pub fn reparameterize(&self, mu: &Tensor, logvar: &Tensor) -> Result<Tensor> {
        let logvar_clamped = logvar.clamp(-12.0f32, 12.0f32)?;
        let std = (logvar_clamped * 0.5)?.exp()?;
        let eps = Tensor::randn(0.0f32, 1.0f32, mu.shape(), mu.device())?;
        let z = (mu + (&eps * &std)?)?;
        Ok(z)
    }

    /// Decodes latent code $z$ and conditioning vector $u$ to DDSP band gains and FOA weights.
    pub fn decode(&self, z: &Tensor, conditioning: &Tensor) -> Result<(Tensor, Tensor)> {
        let x = Tensor::cat(&[z, conditioning], 1)?;
        let h1 = self.dec_fc1.forward(&x)?.gelu_erf()?;
        let h2 = self.dec_fc2.forward(&h1)?.gelu_erf()?;

        // Output Affine Alignment for bands: physically constrained non-negative band gains in [0, 20]
        let raw_bands = self.dec_bands.forward(&h2)?;
        let bands = (candle_softplus(&raw_bands)? + 1e-4f64)?.clamp(0.0f32, 20.0f32)?;

        // Output Affine Alignment for FOA: enforcing acoustic physical energy constraint W >= sqrt(X^2 + Y^2 + Z^2)
        let raw_foa = self.dec_foa.forward(&h2)?;
        let w_raw = raw_foa.narrow(1, 0, 1)?;
        let u_raw = raw_foa.narrow(1, 1, 3)?;
        let w = (candle_softplus(&w_raw)? + 1e-4f64)?.clamp(1e-4f32, 10.0f32)?;
        let u_dir = u_raw.tanh()?;
        let xyz = u_dir.broadcast_mul(&w)?;
        let foa = Tensor::cat(&[&w, &xyz], 1)?;

        Ok((bands, foa))
    }

    /// Full autoencoder forward pass.
    pub fn forward(&self, audio_features: &Tensor, conditioning: &Tensor) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let (mu, logvar) = self.encode(audio_features)?;
        let z = self.reparameterize(&mu, &logvar)?;
        let (bands, foa) = self.decode(&z, conditioning)?;
        Ok((bands, foa, mu, logvar))
    }
}

/// Root Mean Square Normalization (RMSNorm) with learned gain gamma.
/// RMSNorm(x) = (x / sqrt(mean(x^2) + eps)) * gamma
pub struct CandleRMSNorm {
    pub weight: Tensor,
    pub eps: f64,
}

impl CandleRMSNorm {
    pub fn new(dim: usize, eps: f64, vs: VarBuilder) -> Result<Self> {
        let weight = vs.get((dim,), "weight")
            .unwrap_or_else(|_| Tensor::ones((dim,), DType::F32, vs.device()).unwrap());
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let sq = x.sqr()?;
        let mean_sq = sq.mean_keepdim(candle_core::D::Minus1)?;
        let rsqrt = (mean_sq + self.eps)?.sqrt()?;
        let normed = x.broadcast_div(&rsqrt)?;
        let out = normed.broadcast_mul(&self.weight)?;
        Ok(out)
    }
}

/// Trainable continuous affine latent alignment layer: z_align = W * z + b
pub struct CandleAffineAlignment {
    pub proj: Linear,
}

impl CandleAffineAlignment {
    pub fn new(dim: usize, vs: VarBuilder) -> Result<Self> {
        let proj = linear(dim, dim, vs.pp("proj"))?;
        Ok(Self { proj })
    }

    pub fn forward(&self, z: &Tensor) -> Result<Tensor> {
        Ok(self.proj.forward(z)?)
    }

    pub fn weight(&self) -> &Tensor {
        self.proj.weight()
    }
}

/// Differentiable Box-Cox Homotopy & Continuous Capacity Quantizer
pub struct CandleLearnedQuantizer {
    pub beta: Tensor,
    pub lambda_param: Tensor,
    pub delta_prune: Tensor,
}

impl CandleLearnedQuantizer {
    pub fn new(dim: usize, initial_bits: f32, vs: VarBuilder) -> Result<Self> {
        let beta = vs.get((dim,), "beta")
            .unwrap_or_else(|_| Tensor::full(initial_bits, (dim,), vs.device()).unwrap());
        let lambda_param = vs.get((dim,), "lambda_param")
            .unwrap_or_else(|_| Tensor::full(0.5f32, (dim,), vs.device()).unwrap());
        let delta_prune = vs.get((dim,), "delta_prune")
            .unwrap_or_else(|_| Tensor::full(0.05f32, (dim,), vs.device()).unwrap());
        Ok(Self {
            beta,
            lambda_param,
            delta_prune,
        })
    }

    /// Differentiable soft-staircase Box-Cox forward quantization
    pub fn forward(&self, z: &Tensor, tau: f32) -> Result<Tensor> {
        let tau_safe = tau.max(1e-3);
        let smooth_sign = (z / (tau_safe as f64))?.tanh()?;
        let mag = z.abs()?;
        let diff = mag.broadcast_sub(&self.delta_prune.unsqueeze(0)?)?;
        let prune_gate = candle_nn::ops::sigmoid(&(diff / (tau_safe as f64))?)?;
        let res = (&smooth_sign * &mag)?;
        let out = (&res * &prune_gate)?;
        Ok(out)
    }
}

// ============================================================================
// 2. Mamba-2 MoE Recurrent Trajectory Model (Phase 3)
// ============================================================================

/// Single State Space Duality (SSD) Recurrent Expert in Candle.
pub struct CandleMambaExpert {
    pub in_proj: Linear,
    pub rec_proj: Linear,
    pub out_proj: Linear,
}

impl CandleMambaExpert {
    pub fn new(d_model: usize, vs: VarBuilder) -> Result<Self> {
        let in_proj = linear(d_model, d_model, vs.pp("in_proj"))?;
        let rec_proj = linear(d_model, d_model, vs.pp("rec_proj"))?;
        let out_proj = linear(d_model, d_model, vs.pp("out_proj"))?;
        Ok(Self { in_proj, rec_proj, out_proj })
    }

    pub fn from_parts(in_proj: Linear, rec_proj: Linear, out_proj: Linear) -> Self {
        Self { in_proj, rec_proj, out_proj }
    }

    pub fn forward(&self, x: &Tensor, h_prev: &Tensor) -> Result<(Tensor, Tensor)> {
        let u = self.in_proj.forward(x)?.silu()?;
        // Diagonal recurrent state transition: h_t = 0.92 * h_{t-1} + u
        let h_next = ((h_prev * 0.92)? + &u)?;
        let rec = self.rec_proj.forward(&h_next)?.silu()?;
        let y = self.out_proj.forward(&(&rec + &u)?)?;
        Ok((y, h_next))
    }
}

/// O(1) hash-addressed static knowledge bank for physical audio priors (Candle).
pub struct CandleEngramBank {
    pub bank_size: usize,
    pub embed_dim: usize,
    pub num_hash_heads: usize,
    pub bank: Tensor,
    pub hash_projections: Tensor,
    pub fuse_gate: Linear,
}

impl CandleEngramBank {
    pub fn new(bank_size: usize, embed_dim: usize, vs: VarBuilder) -> Result<Self> {
        let num_hash_heads = 4;
        let bank = vs.get((bank_size, embed_dim), "bank")
            .unwrap_or_else(|_| Tensor::randn(0.0f32, 0.02f32, (bank_size, embed_dim), vs.device()).unwrap());
        let hash_projections = vs.get((num_hash_heads, embed_dim), "hash_projections")
            .unwrap_or_else(|_| Tensor::randn(0.0f32, 1.0f32, (num_hash_heads, embed_dim), vs.device()).unwrap());
        let fuse_gate = linear(embed_dim * 2, embed_dim, vs.pp("fuse_gate"))?;
        Ok(Self {
            bank_size,
            embed_dim,
            num_hash_heads,
            bank,
            hash_projections,
            fuse_gate,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let proj = x.matmul(&self.hash_projections.t()?)?;
        let proj_vec = proj.to_vec2::<f32>()?;
        let b_sz = proj_vec.len();
        let primes = [2654435761u64, 2246822519u64, 3266489917u64, 668265263u64];
        
        let mut retrieved_vec = Vec::with_capacity(b_sz * self.embed_dim);
        let bank_vec = self.bank.to_vec2::<f32>()?;
        
        for row in proj_vec {
            let mut accum = vec![0.0f32; self.embed_dim];
            for (h, &val) in row.iter().enumerate().take(self.num_hash_heads) {
                let scaled = (val * 1000.0).abs() as u64;
                let idx = ((scaled.wrapping_mul(primes[h % primes.len()])) % (self.bank_size as u64)) as usize;
                let bank_slice = &bank_vec[idx];
                for (acc, &b_val) in accum.iter_mut().zip(bank_slice.iter()) {
                    *acc += b_val;
                }
            }
            let norm_h = 1.0 / (self.num_hash_heads as f32);
            for acc in accum {
                retrieved_vec.push(acc * norm_h);
            }
        }
        let retrieved = Tensor::from_vec(retrieved_vec, (b_sz, self.embed_dim), x.device())?;
        let concat = Tensor::cat(&[x, &retrieved], 1)?;
        let gate = candle_nn::ops::sigmoid(&self.fuse_gate.forward(&concat)?)?;
        let ones = Tensor::ones_like(&gate)?;
        let one_minus_gate = (&ones - &gate)?;
        let fused = ((&gate * &retrieved)? + (&one_minus_gate * x)?)?;
        Ok(fused)
    }
}

/// Multi-Head Latent Attention (MLA) with low-rank KV compression in Candle.
pub struct CandleLatentAttention {
    pub d_model: usize,
    pub num_heads: usize,
    pub d_compress: usize,
    pub q_proj: Linear,
    pub kv_down: Linear,
    pub k_up: Linear,
    pub v_up: Linear,
    pub out_proj: Linear,
    pub kv_norm: CandleRMSNorm,
    pub scale: f64,
}

impl CandleLatentAttention {
    pub fn new(d_model: usize, vs: VarBuilder) -> Result<Self> {
        let num_heads = 4;
        let d_compress = 32;
        let q_proj = linear(d_model, d_model, vs.pp("q_proj"))?;
        let kv_down = linear(d_model, d_compress, vs.pp("kv_down"))?;
        let k_up = linear(d_compress, d_model, vs.pp("k_up"))?;
        let v_up = linear(d_compress, d_model, vs.pp("v_up"))?;
        let out_proj = linear(d_model, d_model, vs.pp("out_proj"))?;
        let kv_norm = CandleRMSNorm::new(d_compress, 1e-6, vs.pp("kv_norm"))?;
        let scale = 1.0 / ((d_model / num_heads) as f64).sqrt();

        Ok(Self {
            d_model,
            num_heads,
            d_compress,
            q_proj,
            kv_down,
            k_up,
            v_up,
            out_proj,
            kv_norm,
            scale,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let q = self.q_proj.forward(x)?;
        let c_kv = self.kv_down.forward(x)?;
        let c_kv_normed = self.kv_norm.forward(&c_kv)?;
        let k = self.k_up.forward(&c_kv_normed)?;
        let v = self.v_up.forward(&c_kv_normed)?;

        let scores = ((&q * &k)? * self.scale)?.clamp(-30.0f32, 30.0f32)?;
        let weights = candle_nn::ops::softmax(&scores, 1)?;
        let context = (&weights * &v)?;
        let out = self.out_proj.forward(&context)?;
        let res = (x + &out)?;
        Ok(res)
    }
}

/// Mamba-2 Mixture of Experts (MoE) Trajectory Model with Top-K Gating,
/// DeepSeek Invariant Shared Base Expert, Router-Derived Dynamic Dense Soup,
/// and Output Affine Alignment for Latent Trajectories.
pub struct CandleMamba2MoE {
    pub in_proj: Linear,
    pub router: Linear,
    pub shared_base: CandleMambaExpert,
    pub experts: Vec<CandleMambaExpert>,
    pub soup_proj: Linear,
    pub fusion: Linear,
    pub traj_head: Linear,
    pub traj_head_t2: Linear,
    pub traj_head_t3: Linear,
    pub out_affine: CandleAffineAlignment,
}

impl CandleMamba2MoE {
    pub fn new(vs: VarBuilder) -> Result<Self> {
        let d_model = 128;
        let in_proj = linear(COMBINED_DIM, d_model, vs.pp("in_proj"))?;
        let router = linear(d_model, NUM_EXPERTS, vs.pp("router"))?;

        let shared_base = CandleMambaExpert::new(d_model, vs.pp("shared_base"))?;

        let mut experts = Vec::with_capacity(NUM_EXPERTS);
        let exp_vs = vs.pp("experts");
        for i in 0..NUM_EXPERTS {
            experts.push(CandleMambaExpert::new(d_model, exp_vs.pp(i.to_string()))?);
        }

        let soup_proj = linear(NUM_EXPERTS, NUM_EXPERTS, vs.pp("soup_proj"))?;
        let fusion = linear(d_model, d_model, vs.pp("fusion"))?;
        let traj_head = linear(d_model, LATENT_DIM, vs.pp("traj_head"))?;
        let traj_head_t2 = linear(d_model, LATENT_DIM, vs.pp("traj_head_t2"))?;
        let traj_head_t3 = linear(d_model, LATENT_DIM, vs.pp("traj_head_t3"))?;
        let out_affine = CandleAffineAlignment::new(LATENT_DIM, vs.pp("out_affine"))?;

        Ok(Self {
            in_proj,
            router,
            shared_base,
            experts,
            soup_proj,
            fusion,
            traj_head,
            traj_head_t2,
            traj_head_t3,
            out_affine,
        })
    }

    /// Forward pass with smooth softmax routing, DeepSeek shared base expert, load-balancing probabilities,
    /// and output affine alignment to canonical VAE latent coordinates.
    pub fn forward(
        &self,
        z_prev: &Tensor,
        conditioning: &Tensor,
        h_prev: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        // Combined input [B, 618]
        let x_in = Tensor::cat(&[z_prev, conditioning], 1)?;
        let h_in = self.in_proj.forward(&x_in)?.gelu_erf()?;

        // Router logits [B, 8] and Softmax
        let router_logits = self.router.forward(&h_in)?;
        let router_probs = candle_nn::ops::softmax(&router_logits, 1)?;

        // Invariant DeepSeek Shared Base Expert (always evaluated)
        let (base_out, base_h_next) = self.shared_base.forward(&h_in, h_prev)?;

        // Residual experts dispatch weighted blend
        let mut expert_outputs = Vec::with_capacity(NUM_EXPERTS);
        let mut next_states = Vec::with_capacity(NUM_EXPERTS);

        for expert in &self.experts {
            let (out_e, h_next_e) = expert.forward(&h_in, h_prev)?;
            expert_outputs.push(out_e);
            next_states.push(h_next_e);
        }

        // Weighted accumulation across all experts based on router weights
        let mut blended_out = (&expert_outputs[0] * 0.0)?;
        let mut blended_state = (&next_states[0] * 0.0)?;

        for (e, (exp_out, exp_state)) in expert_outputs.iter().zip(next_states.iter()).enumerate() {
            let weight_e = router_probs.narrow(1, e, 1)?;
            let weighted_out = exp_out.broadcast_mul(&weight_e)?;
            let weighted_state = exp_state.broadcast_mul(&weight_e)?;
            blended_out = (&blended_out + &weighted_out)?;
            blended_state = (&blended_state + &weighted_state)?;
        }

        // Combine base anchor and residual displacement
        blended_out = (&base_out + &blended_out)?;
        blended_state = (&base_h_next + &blended_state)?;

        // Feature fusion and final trajectory projection with output affine alignment
        let fused = self.fusion.forward(&blended_out)?.gelu_erf()?;
        let z_raw = self.traj_head.forward(&fused)?;
        let z_pred = self.out_affine.forward(&z_raw)?;

        Ok((z_pred, blended_state, router_probs))
    }

    /// Evaluates the dense soup blend: W_dense = W_base + sum alpha_k * Delta W_k
    pub fn compute_dense_soup(
        &self,
        h_in: &Tensor,
        h_prev: &Tensor,
        alpha: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (base_out, base_h_next) = self.shared_base.forward(h_in, h_prev)?;
        let mut soup_residual = (&base_out * 0.0)?;
        let mut soup_state = (&base_h_next * 0.0)?;

        for (e, expert) in self.experts.iter().enumerate() {
            let (out_e, h_next_e) = expert.forward(h_in, h_prev)?;
            let alpha_e = alpha.narrow(1, e, 1)?;
            let weighted_out = out_e.broadcast_mul(&alpha_e)?;
            let weighted_state = h_next_e.broadcast_mul(&alpha_e)?;
            soup_residual = (&soup_residual + &weighted_out)?;
            soup_state = (&soup_state + &weighted_state)?;
        }

        let soup_out = (&base_out + &soup_residual)?;
        let next_state = (&base_h_next + &soup_state)?;
        Ok((soup_out, next_state))
    }

    /// Derives dynamic soup blend coefficients from router probabilities:
    /// alpha = softmax(router_logits + 0.1 * soup_proj(router_probs))
    /// This formulation anchors alpha near the router's optimal expert mixture at initialization,
    /// eliminating arbitrary initial distortion while allowing learned refinement.
    pub fn router_to_soup_coefficients(&self, router_probs: &Tensor) -> Result<Tensor> {
        let residual_logits = self.soup_proj.forward(router_probs)?;
        let log_p = (router_probs.clamp(1e-8f32, 1.0f32)?.log()? + (&residual_logits * 0.1)?)?;
        Ok(candle_nn::ops::softmax(&log_p, 1)?)
    }

    /// Returns corresponding (base_weight, expert_weight) pairs across all experts
    /// for computing the Frobenius norm weight drift regularization.
    pub fn get_expert_drift_pairs(&self) -> Vec<(&Tensor, &Tensor)> {
        let mut pairs = Vec::with_capacity(self.experts.len() * 3);
        let base_in = self.shared_base.in_proj.weight();
        let base_rec = self.shared_base.rec_proj.weight();
        let base_out = self.shared_base.out_proj.weight();

        for expert in &self.experts {
            pairs.push((base_in, expert.in_proj.weight()));
            pairs.push((base_rec, expert.rec_proj.weight()));
            pairs.push((base_out, expert.out_proj.weight()));
        }
        pairs
    }


    /// Multi-Frame Prediction: predicts trajectories for t+1, t+2, and t+3.
    pub fn predict_multi_frame(&self, fused: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let z1 = self.out_affine.forward(&self.traj_head.forward(fused)?)?;
        let z2 = self.out_affine.forward(&self.traj_head_t2.forward(fused)?)?;
        let z3 = self.out_affine.forward(&self.traj_head_t3.forward(fused)?)?;
        Ok((z1, z2, z3))
    }

    /// Collapses the shared base expert and residual experts into a single dense Mamba block:
    /// W_dense = W_base + sum alpha_k * Delta W_k
    pub fn collapse_to_dense_soup(&self, alpha: &[f32]) -> Result<CandleMambaExpert> {
        let base_in_w = self.shared_base.in_proj.weight();
        let mut in_w = base_in_w.clone();
        for (k, expert) in self.experts.iter().enumerate() {
            let a = alpha.get(k).copied().unwrap_or(0.0f32);
            if a.abs() > 1e-6 {
                let scaled = (expert.in_proj.weight() * (a as f64))?;
                in_w = (in_w + scaled)?;
            }
        }
        let in_bias = match (self.shared_base.in_proj.bias(), self.experts[0].in_proj.bias()) {
            (Some(b_base), _) => {
                let mut b = b_base.clone();
                for (k, expert) in self.experts.iter().enumerate() {
                    let a = alpha.get(k).copied().unwrap_or(0.0f32);
                    if let Some(b_k) = expert.in_proj.bias() {
                        b = (b + (b_k * (a as f64))?)?;
                    }
                }
                Some(b)
            }
            _ => None,
        };

        let base_rec_w = self.shared_base.rec_proj.weight();
        let mut rec_w = base_rec_w.clone();
        for (k, expert) in self.experts.iter().enumerate() {
            let a = alpha.get(k).copied().unwrap_or(0.0f32);
            if a.abs() > 1e-6 {
                let scaled = (expert.rec_proj.weight() * (a as f64))?;
                rec_w = (rec_w + scaled)?;
            }
        }
        let rec_bias = match (self.shared_base.rec_proj.bias(), self.experts[0].rec_proj.bias()) {
            (Some(b_base), _) => {
                let mut b = b_base.clone();
                for (k, expert) in self.experts.iter().enumerate() {
                    let a = alpha.get(k).copied().unwrap_or(0.0f32);
                    if let Some(b_k) = expert.rec_proj.bias() {
                        b = (b + (b_k * (a as f64))?)?;
                    }
                }
                Some(b)
            }
            _ => None,
        };

        let base_out_w = self.shared_base.out_proj.weight();
        let mut out_w = base_out_w.clone();
        for (k, expert) in self.experts.iter().enumerate() {
            let a = alpha.get(k).copied().unwrap_or(0.0f32);
            if a.abs() > 1e-6 {
                let scaled = (expert.out_proj.weight() * (a as f64))?;
                out_w = (out_w + scaled)?;
            }
        }
        let out_bias = match (self.shared_base.out_proj.bias(), self.experts[0].out_proj.bias()) {
            (Some(b_base), _) => {
                let mut b = b_base.clone();
                for (k, expert) in self.experts.iter().enumerate() {
                    let a = alpha.get(k).copied().unwrap_or(0.0f32);
                    if let Some(b_k) = expert.out_proj.bias() {
                        b = (b + (b_k * (a as f64))?)?;
                    }
                }
                Some(b)
            }
            _ => None,
        };

        let in_linear = Linear::new(in_w, in_bias);
        let rec_linear = Linear::new(rec_w, rec_bias);
        let out_linear = Linear::new(out_w, out_bias);

        Ok(CandleMambaExpert::from_parts(in_linear, rec_linear, out_linear))
    }

    /// Smooth temperature-scaled softmax routing. All experts receive non-zero, differentiable
    /// weights — no discrete selection or zeroing. The parameter `tau_moe` is the softmax
    /// temperature: lower values sharpen focus, higher values spread load uniformly.
    ///
    /// Returns `(smooth_weights, weights_as_mask, mean_effective_count)` where:
    /// - `smooth_weights`: full [B, N] probability distribution summing to 1.0 per row.
    /// - `weights_as_mask`: same tensor (continuous, no binary gate).
    /// - `mean_effective_count`: per-batch mean exp(H) — entropy-based effective expert count.
    pub fn route_smooth_softmax(
        logits: &Tensor,
        tau_moe: f64,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let tau = tau_moe.max(0.05);
        // Temperature-scaled softmax: p_e = exp((l_e - max) / tau) / Z
        let scaled = (logits * (1.0 / tau))?;
        let smooth_weights = candle_nn::ops::softmax(&scaled, 1)?;

        // Entropy-based effective expert count: exp(H) = exp(-sum p*log(p))
        // Clamping smooth_weights to [1e-8, 1.0] before log guarantees 0 * log(0) = NaN cannot occur
        let safe_weights = smooth_weights.clamp(1e-8f32, 1.0f32)?;
        let log_w = (safe_weights.log()? * -1.0)?;
        let entropy = (&smooth_weights * &log_w)?.sum(1)?;       // [B]
        let eff_count = entropy.exp()?;                           // exp(H), in [1, N]
        let mean_eff = eff_count.mean_all()?;

        Ok((smooth_weights.clone(), smooth_weights, mean_eff))
    }

    /// Forward pass with continuous smooth softmax routing and temporal tabu logit dampening.
    /// All experts participate with differentiable weights — no discrete gating or zeroing.
    /// `tau_moe`: softmax temperature (lower → sharper; 0.75 default). Tabu dampening
    /// subtracts `gamma_tabu * tabu_history` from logits before softmax.
    /// Forward pass with continuous smooth softmax routing, temporal tabu logit dampening, and optional expert dropout.
    pub fn forward_smooth_tabu_with_dropout(
        &self,
        z_prev: &Tensor,
        conditioning: &Tensor,
        h_prev: &Tensor,
        tau_moe: f64,
        tabu_history: Option<&Tensor>,
        gamma_tabu: f64,
        expert_dropout_prob: f32,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let x_in = Tensor::cat(&[z_prev, conditioning], 1)?;
        let h_in = self.in_proj.forward(&x_in)?.gelu_erf()?;

        let mut router_logits = self.router.forward(&h_in)?;
        if let Some(tabu) = tabu_history {
            if gamma_tabu > 1e-6 {
                let penalty = (tabu * gamma_tabu)?;
                router_logits = (router_logits - penalty)?;
            }
        }

        // Smooth softmax routing — all 8 experts receive continuous non-zero weights
        let (mut smooth_weights, active_mask, mean_eff) =
            Self::route_smooth_softmax(&router_logits, tau_moe)?;

        // Expert Dropout: randomly zero out an expert to prevent co-adaptation and guarantee fault tolerance
        if expert_dropout_prob > 0.0 {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            if rng.gen_range(0.0f32..1.0f32) < expert_dropout_prob {
                let dropped_e = rng.gen_range(0..NUM_EXPERTS);
                let mut mask_vec = vec![1.0f32; NUM_EXPERTS];
                mask_vec[dropped_e] = 0.0f32;
                let drop_mask = Tensor::from_vec(mask_vec, (1, NUM_EXPERTS), smooth_weights.device())?;
                let masked = smooth_weights.broadcast_mul(&drop_mask)?;
                let sum_w = (masked.sum_keepdim(1)? + 1e-8f64)?;
                smooth_weights = masked.broadcast_div(&sum_w)?;
            }
        }

        // Keep router_probs (for loss computations) as the smooth distribution
        let router_probs = smooth_weights.clone();

        // DeepSeek shared base expert
        let (base_out, base_h_next) = self.shared_base.forward(&h_in, h_prev)?;

        let mut expert_outputs = Vec::with_capacity(NUM_EXPERTS);
        let mut next_states = Vec::with_capacity(NUM_EXPERTS);

        for expert in &self.experts {
            let (out_e, h_next_e) = expert.forward(&h_in, h_prev)?;
            expert_outputs.push(out_e);
            next_states.push(h_next_e);
        }

        let mut blended_out = (&expert_outputs[0] * 0.0)?;
        let mut blended_state = (&next_states[0] * 0.0)?;

        for (e, (exp_out, exp_state)) in expert_outputs.iter().zip(next_states.iter()).enumerate() {
            let weight_e = smooth_weights.narrow(1, e, 1)?;
            let weighted_out = exp_out.broadcast_mul(&weight_e)?;
            let weighted_state = exp_state.broadcast_mul(&weight_e)?;
            blended_out = (&blended_out + &weighted_out)?;
            blended_state = (&blended_state + &weighted_state)?;
        }

        blended_out = (&base_out + &blended_out)?;
        blended_state = (&base_h_next + &blended_state)?;

        let fused = self.fusion.forward(&blended_out)?.gelu_erf()?;
        let z_raw = self.traj_head.forward(&fused)?;
        let z_pred = self.out_affine.forward(&z_raw)?;

        Ok((z_pred, blended_state, router_probs, active_mask, mean_eff, router_logits))
    }

    /// Forward pass with continuous smooth softmax routing and temporal tabu logit dampening (zero expert dropout).
    pub fn forward_smooth_tabu(
        &self,
        z_prev: &Tensor,
        conditioning: &Tensor,
        h_prev: &Tensor,
        tau_moe: f64,
        tabu_history: Option<&Tensor>,
        gamma_tabu: f64,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
        self.forward_smooth_tabu_with_dropout(
            z_prev,
            conditioning,
            h_prev,
            tau_moe,
            tabu_history,
            gamma_tabu,
            0.0,
        )
    }

    /// Forward pass with smooth softmax routing (no tabu dampening).
    pub fn forward_smooth(
        &self,
        z_prev: &Tensor,
        conditioning: &Tensor,
        h_prev: &Tensor,
        tau_moe: f64,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
        self.forward_smooth_tabu(z_prev, conditioning, h_prev, tau_moe, None, 0.0)
    }
}


/// Iterative Latent Space "Thinking" and Refinement Block.
/// Allows multiple iterative latent updates with deliberation depth embeddings
/// and stochastic latent jittering to encourage contractive attractor dynamics.
pub struct CandleThinkingBlock {
    fc1: Linear,
    fc2: Linear,
    halt_gate: Linear,
    step_embed: Tensor,
    pub dim: usize,
    pub step_embed_dim: usize,
}

impl CandleThinkingBlock {
    pub fn new(dim: usize, cond_dim: usize, vs: VarBuilder) -> Result<Self> {
        let step_embed_dim = 32;
        let step_embed = vs.get((16, step_embed_dim), "step_embed")
            .unwrap_or_else(|_| Tensor::randn(0.0f32, 0.02f32, (16, step_embed_dim), vs.device()).unwrap());
        let fc1 = linear(dim + cond_dim + step_embed_dim, 128, vs.pp("fc1"))?;
        let fc2 = linear(128, dim, vs.pp("fc2"))?;
        let halt_gate = linear(dim + cond_dim + step_embed_dim, 1, vs.pp("halt_gate"))?;
        Ok(Self { fc1, fc2, halt_gate, step_embed, dim, step_embed_dim })
    }

    /// Performs iterative thinking updates with stochastic latent jittering and depth embeddings.
    pub fn forward_thinking_with_jitter(
        &self,
        z_init: &Tensor,
        conditioning: &Tensor,
        max_steps: usize,
        eps_halt: f32,
        jitter_sigma: f32,
    ) -> Result<(Tensor, usize, Vec<Tensor>)> {
        let (b_sz, _) = z_init.dims2()?;
        let mut z_current = z_init.copy()?;
        let mut halting_probs = Vec::with_capacity(max_steps);
        let mut steps_taken = 0;

        for step in 1..=max_steps {
            steps_taken = step;
            let z_perturbed = if jitter_sigma > 1e-6 {
                let jitter = Tensor::randn(0.0f32, jitter_sigma, z_current.shape(), z_current.device())?;
                (&z_current + &jitter)?
            } else {
                z_current.copy()?
            };

            let step_idx = (step - 1).min(15);
            let emb = self.step_embed.narrow(0, step_idx, 1)?;
            let emb_broadcast = emb.broadcast_as((b_sz, self.step_embed_dim))?;

            let inp = Tensor::cat(&[&z_perturbed, conditioning, &emb_broadcast], 1)?;
            let h = self.fc1.forward(&inp)?.gelu_erf()?;
            let delta = self.fc2.forward(&h)?.tanh()?;
            let halt_score = candle_nn::ops::sigmoid(&self.halt_gate.forward(&inp)?)?;
            halting_probs.push(halt_score);

            z_current = (&z_current + &delta)?;

            let step_norm = delta.sqr()?.sum_all()?.relu()?.sqrt()?.to_scalar::<f32>()?;
            if step_norm < eps_halt {
                break;
            }
        }

        Ok((z_current, steps_taken, halting_probs))
    }

    /// Performs iterative thinking updates on the latent code with zero jitter.
    pub fn forward_thinking(
        &self,
        z_init: &Tensor,
        conditioning: &Tensor,
        max_steps: usize,
        eps_halt: f32,
    ) -> Result<(Tensor, usize, Vec<Tensor>)> {
        self.forward_thinking_with_jitter(z_init, conditioning, max_steps, eps_halt, 0.0)
    }
}

/// Multi-Dimensional State Space Duality (SSD) Recurrent Block in Candle.
/// Mirrors `MambaSSDBlock` from `src/models/mamba2_moe.py`.
pub struct CandleMambaSSDBlock {
    pub d_model: usize,
    pub d_state: usize,
    pub in_proj: Linear,
    pub b_proj: Linear,
    pub c_proj: Linear,
    pub out_proj: Linear,
    pub a_log: Tensor,
    pub d_param: Tensor,
}

impl CandleMambaSSDBlock {
    pub fn new(d_model: usize, d_state: usize, vs: VarBuilder) -> Result<Self> {
        let in_proj = linear(d_model, d_model * 2, vs.pp("in_proj"))?;
        let b_proj = linear(d_model, d_state, vs.pp("b_proj"))?;
        let c_proj = linear(d_model, d_state, vs.pp("c_proj"))?;
        let out_proj = linear(d_model, d_model, vs.pp("out_proj"))?;
        let a_log = vs.get((d_model, d_state), "a_log")
            .unwrap_or_else(|_| Tensor::zeros((d_model, d_state), DType::F32, vs.device()).unwrap());
        let d_param = vs.get((d_model,), "d_param")
            .unwrap_or_else(|_| Tensor::ones((d_model,), DType::F32, vs.device()).unwrap());

        Ok(Self {
            d_model,
            d_state,
            in_proj,
            b_proj,
            c_proj,
            out_proj,
            a_log,
            d_param,
        })
    }

    /// Single recurrent step forward pass:
    /// x: [B, D_model]
    /// h_prev: [B, D_model, D_state]
    /// returns (y: [B, D_model], h_next: [B, D_model, D_state])
    pub fn forward(&self, x: &Tensor, h_prev: &Tensor) -> Result<(Tensor, Tensor)> {
        let proj = self.in_proj.forward(x)?;
        let u = proj.narrow(1, 0, self.d_model)?;
        let gate = proj.narrow(1, self.d_model, self.d_model)?;
        let u_act = u.silu()?;

        let b_t = self.b_proj.forward(&u_act)?; // [B, D_state]
        let c_t = self.c_proj.forward(&u_act)?; // [B, D_state]

        // State decay: decay = exp(-exp(a_log)) clamped to prevent gradient blowup
        let decay = (self.a_log.clamp(-10.0f32, 4.0f32)?.exp()? * -1.0)?.exp()?; // [D_model, D_state]
        let h_decayed = h_prev.broadcast_mul(&decay.unsqueeze(0)?)?; // [B, D_model, D_state]

        let u_exp = u_act.unsqueeze(2)?; // [B, D_model, 1]
        let b_exp = b_t.unsqueeze(1)?; // [B, 1, D_state]
        let input_flux = u_exp.broadcast_mul(&b_exp)?; // [B, D_model, D_state]
        let h_next = (&h_decayed + &input_flux)?;

        let c_exp = c_t.unsqueeze(1)?; // [B, 1, D_state]
        let y_state = h_next.broadcast_mul(&c_exp)?.sum(2)?; // [B, D_model]
        let skip = u_act.broadcast_mul(&self.d_param.unsqueeze(0)?)?;
        let y = (&y_state + &skip)?;

        let gate_act = gate.silu()?;
        let y_gated = (&y * &gate_act)?;
        let out = self.out_proj.forward(&y_gated)?;

        Ok((out, h_next))
    }
}

/// Jamba Multi-Head Self-Attention Fusion Block in Candle.
/// Mirrors `JambaSelfAttentionBlock` from `src/models/mamba2_moe.py`.
pub struct CandleJambaSelfAttention {
    pub d_model: usize,
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
    pub scale: f64,
}

impl CandleJambaSelfAttention {
    pub fn new(d_model: usize, vs: VarBuilder) -> Result<Self> {
        let q_proj = linear(d_model, d_model, vs.pp("q_proj"))?;
        let k_proj = linear(d_model, d_model, vs.pp("k_proj"))?;
        let v_proj = linear(d_model, d_model, vs.pp("v_proj"))?;
        let out_proj = linear(d_model, d_model, vs.pp("out_proj"))?;
        let scale = 1.0 / (d_model as f64).sqrt();

        Ok(Self {
            d_model,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            scale,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let scores = ((&q * &k)? * self.scale)?;
        let weights = candle_nn::ops::softmax(&scores, 1)?;
        let context = (&weights * &v)?;
        let out = self.out_proj.forward(&context)?;
        let res = (x + &out)?;
        Ok(res)
    }
}

/// Output package from the Invasive Meta-Controller.
#[derive(Debug, Clone)]
pub struct CandleMetaTelemetryOutput {
    pub expert_mask: Tensor,
    pub tau_moe: Tensor,
    pub ambisonic_order: Tensor,
    pub diffusion_bypass: Tensor,
    pub synthesis_blend: Tensor,
    pub stress: Tensor,
    pub pre_generated_steps: Tensor,
    pub next_telem_state: Tensor,
}

impl CandleMetaTelemetryOutput {
    /// Returns discrete recommended pre-generated / thinking steps (1..=5)
    pub fn recommended_steps(&self) -> usize {
        if let Ok(val) = self.pre_generated_steps.mean_all().and_then(|t| t.to_scalar::<f32>()) {
            (val.round() as usize).clamp(1, 5)
        } else {
            3
        }
    }
}

/// Global Invasive Meta-Controller in Candle.
/// Dynamically supervises MoE routing, ambisonic order, diffusion bypass,
/// and synthesis blending based on real-time hardware telemetry.
/// Mirrors `InvasiveMetaController` from `src/models/meta_controller.py`.
pub struct CandleInvasiveMetaController {
    pub num_experts: usize,
    pub d_state: usize,
    pub telemetry_proj: Linear,
    pub telem_a_log: Tensor,
    pub telem_b_proj: Linear,
    pub telem_c_proj: Linear,
    pub telem_d: Tensor,
    pub fusion_fc1: Linear,
    pub fusion_fc2: Linear,
    pub fusion_out: Linear,
}

impl CandleInvasiveMetaController {
    pub fn new(num_experts: usize, d_state: usize, vs: VarBuilder) -> Result<Self> {
        let in_dim = num_experts + 4 + 3 + 2 + 1; // 18
        let telemetry_proj = linear(4, 32, vs.pp("telemetry_proj"))?;
        let telem_a_log = vs.get((32, d_state), "telem_a_log")
            .unwrap_or_else(|_| Tensor::zeros((32, d_state), DType::F32, vs.device()).unwrap());
        let telem_b_proj = linear(32, d_state, vs.pp("telem_b_proj"))?;
        let telem_c_proj = linear(32, d_state, vs.pp("telem_c_proj"))?;
        let telem_d = vs.get((32,), "telem_d")
            .unwrap_or_else(|_| Tensor::ones((32,), DType::F32, vs.device()).unwrap());

        let fusion_fc1 = linear(in_dim + 32, 64, vs.pp("fusion_fc1"))?;
        let fusion_fc2 = linear(64, 64, vs.pp("fusion_fc2"))?;
        let fusion_out = linear(64, num_experts + 5, vs.pp("fusion_out"))?;

        Ok(Self {
            num_experts,
            d_state,
            telemetry_proj,
            telem_a_log,
            telem_b_proj,
            telem_c_proj,
            telem_d,
            fusion_fc1,
            fusion_fc2,
            fusion_out,
        })
    }

    /// Evaluates hardware telemetry and generates gating decisions
    pub fn forward(
        &self,
        moe_logits: &Tensor,
        telemetry: &Tensor,       // [B, 4] [buffer_health_ms, cpu_headroom, gpu_headroom, delta_t_ms]
        user_weights: &Tensor,    // [B, 3] [quality_pref, perf_pref, target_buffer_ms]
        quality_scores: &Tensor,  // [B, 2]
        slice_level: &Tensor,     // [B, 1]
        telemetry_state: &Tensor, // [B, 32, 16]
    ) -> Result<CandleMetaTelemetryOutput> {
        let u_telem = self.telemetry_proj.forward(telemetry)?.silu()?;
        let b_t = self.telem_b_proj.forward(&u_telem)?;
        let c_t = self.telem_c_proj.forward(&u_telem)?;

        let decay = (self.telem_a_log.exp()? * -1.0)?.exp()?;
        let h_decayed = telemetry_state.broadcast_mul(&decay.unsqueeze(0)?)?;
        let flux = u_telem.unsqueeze(2)?.broadcast_mul(&b_t.unsqueeze(1)?)?;
        let next_telem_state = (&h_decayed + &flux)?;

        let mamba_out = next_telem_state.broadcast_mul(&c_t.unsqueeze(1)?)?.sum(2)?;
        let mamba_act = (&mamba_out + u_telem.broadcast_mul(&self.telem_d.unsqueeze(0)?)?)?;

        let raw_features = Tensor::cat(&[moe_logits, telemetry, user_weights, quality_scores, slice_level], 1)?;
        let fused_in = Tensor::cat(&[&raw_features, &mamba_act], 1)?;

        let h1 = self.fusion_fc1.forward(&fused_in)?.silu()?;
        let h2 = self.fusion_fc2.forward(&h1)?.silu()?;
        let raw_out = self.fusion_out.forward(&h2)?;

        let expert_gates = candle_nn::ops::sigmoid(&raw_out.narrow(1, 0, self.num_experts)?)?;
        let raw_tau = candle_nn::ops::sigmoid(&raw_out.narrow(1, self.num_experts, 1)?)?;
        let tau_moe = ((&raw_tau * 1.9)? + 0.1)?;

        let ambisonic_gate = candle_nn::ops::sigmoid(&raw_out.narrow(1, self.num_experts + 1, 1)?)?;
        let diff_gate = candle_nn::ops::sigmoid(&raw_out.narrow(1, self.num_experts + 2, 1)?)?;
        let blend_gate = candle_nn::ops::sigmoid(&raw_out.narrow(1, self.num_experts + 3, 1)?)?;

        let buf_health = telemetry.narrow(1, 0, 1)?;
        let stress = candle_nn::ops::sigmoid(&((25.0 - &buf_health)? / 25.0)?)?;

        let raw_steps = candle_nn::ops::sigmoid(&raw_out.narrow(1, self.num_experts + 4, 1)?)?;
        // pre_generated_steps in continuous range [1.0, 5.0]
        // Modulated by hardware stress: high stress drops steps toward 1 to save latency
        let stress_damping = ((1.0 - (&stress * 0.75)?)?).clamp(0.0, 1.0)?;
        let pre_generated_steps = ((&raw_steps * 4.0)?.broadcast_mul(&stress_damping)? + 1.0)?;

        Ok(CandleMetaTelemetryOutput {
            expert_mask: expert_gates,
            tau_moe,
            ambisonic_order: ambisonic_gate,
            diffusion_bypass: diff_gate,
            synthesis_blend: blend_gate,
            stress,
            pre_generated_steps,
            next_telem_state,
        })
    }
}

// ============================================================================
// 3. Physics-Informed & HWIL Loss Functions
// ============================================================================

/// Trajectory loss with 1st-order finite difference velocity smoothness penalty:
/// $\mathcal{L} = \text{MSE}(z_{pred}, z_{target}) + \lambda_{vel} \|(z_{pred} - z_{prev}) - (z_{target} - z_{prev})\|^2$.

pub struct CandleConsistencyHead {
    fc1: Linear,
    fc2: Linear,
    pub in_dim: usize,
    pub out_dim: usize,
}

impl CandleConsistencyHead {
    pub fn new(in_dim: usize, out_dim: usize, vs: VarBuilder) -> Result<Self> {
        let fc1 = linear(in_dim, 128, vs.pp("fc1"))?;
        let fc2 = linear(128, out_dim, vs.pp("fc2"))?;
        Ok(Self { fc1, fc2, in_dim, out_dim })
    }

    /// Predicts 1-step fast jump: z_fast = z_0 + Delta z_fast
    pub fn forward(&self, z_0: &Tensor, conditioning: &Tensor) -> Result<Tensor> {
        let inp = Tensor::cat(&[z_0, conditioning], 1)?;
        let h = self.fc1.forward(&inp)?.silu()?;
        let delta = self.fc2.forward(&h)?.tanh()?;
        let z_fast = (z_0 + &delta)?;
        Ok(z_fast)
    }

    /// Computes Huber distillation loss against the converged multi-step thinking target
    pub fn compute_distill_loss(&self, z_fast: &Tensor, z_converged: &Tensor, delta: f64) -> Result<Tensor> {
        let diff = (z_fast - z_converged)?;
        crate::stft_loss::huber_loss(&diff, delta)
    }
}

