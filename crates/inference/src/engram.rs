//! High-performance O(1) hash-addressed Engram knowledge bank for physical audio priors.
//!
//! Stores material acoustic physics, surface splash profiles, and HRTF priors.
//! Evaluated with allocation-free fixed-stack lookups.

pub const ENGRAM_BANK_SIZE: usize = 32768;
pub const ENGRAM_EMBED_DIM: usize = 64;
pub const ENGRAM_HASH_HEADS: usize = 4;

const HASH_PRIMES: [u64; 4] = [2654435761, 2246822519, 3266489917, 668265263];

/// Real-time Engram static knowledge bank.
#[derive(Clone, Debug)]
pub struct EngramBank {

    pub bank: Vec<[f32; ENGRAM_EMBED_DIM]>,
    pub hash_projections: [[f32; ENGRAM_EMBED_DIM]; ENGRAM_HASH_HEADS],
}

impl Default for EngramBank {
    fn default() -> Self {
        Self::new()
    }
}

impl EngramBank {
    pub fn new() -> Self {
        // Deterministic pseudo-random orthogonal hash heads matching Python model seed
        let mut hash_projections = [[0.0f32; ENGRAM_EMBED_DIM]; ENGRAM_HASH_HEADS];
        for (i, head) in hash_projections.iter_mut().enumerate() {
            let mut seed = (42 + i * 1337) as u64;
            let mut norm = 0.0f32;
            for val in head.iter_mut() {
                // Linear congruential generator for deterministic weights
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let rand_f32 = ((seed >> 33) as f32) / (u32::MAX as f32) - 0.5;
                *val = rand_f32;
                norm += rand_f32 * rand_f32;
            }
            let inv_norm = 1.0 / norm.sqrt().max(1e-6);
            for val in head.iter_mut() {
                *val *= inv_norm;
            }
        }

        Self {
            bank: vec![[0.0f32; ENGRAM_EMBED_DIM]; ENGRAM_BANK_SIZE],
            hash_projections,
        }
    }

    /// Computes deterministic multi-head hash indices from query feature vector.
    pub fn compute_hash_indices(&self, query: &[f32; ENGRAM_EMBED_DIM]) -> [usize; ENGRAM_HASH_HEADS] {
        let mut indices = [0usize; ENGRAM_HASH_HEADS];
        for (h, idx) in indices.iter_mut().enumerate() {
            let proj: f32 = self.hash_projections[h]
                .iter()
                .zip(query.iter())
                .map(|(&w, &q)| w * q)
                .sum();
            let scaled = (proj * 1000.0).abs() as u64;
            *idx = ((scaled.wrapping_mul(HASH_PRIMES[h])) % (ENGRAM_BANK_SIZE as u64)) as usize;
        }
        indices
    }

    /// O(1) allocation-free retrieval across multi-head hash buckets.
    pub fn lookup(&self, query: &[f32; ENGRAM_EMBED_DIM], out: &mut [f32; ENGRAM_EMBED_DIM]) {
        let indices = self.compute_hash_indices(query);
        out.fill(0.0);
        let inv_heads = 1.0 / (ENGRAM_HASH_HEADS as f32);

        for &idx in &indices {
            let entry = &self.bank[idx.min(ENGRAM_BANK_SIZE - 1)];
            for (acc, &val) in out.iter_mut().zip(entry.iter()) {
                *acc += val * inv_heads;
            }
        }
    }

    /// Fused gated lookup: h_fused = gate * prior + (1 - gate) * feature.
    pub fn fuse_in_place(&self, features: &mut [f32; ENGRAM_EMBED_DIM], gate_alpha: f32) {
        let mut prior = [0.0f32; ENGRAM_EMBED_DIM];
        self.lookup(features, &mut prior);
        let alpha = gate_alpha.clamp(0.0, 1.0);
        for (f, &p) in features.iter_mut().zip(prior.iter()) {
            *f = alpha * p + (1.0 - alpha) * *f;
        }
    }
}
