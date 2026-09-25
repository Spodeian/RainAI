//! Runge-Kutta 4th Order (RK4) continuous trajectory ODE solver for flow matching.
//!
//! Provides mathematically exact 4th-order ODE integration for neural probability flow
//! trajectories dx_t/dt = v_theta(x_t, t, c) with O(h^4) local truncation error.

/// Continuous probability flow ODE solver using Runge-Kutta 4th Order (RK4).
pub struct Rk4FlowSolver;

impl Rk4FlowSolver {
    /// Integrates a single step from $t$ to $t + h$ using RK4:
    /// $$k_1 = v(x_t, t)$$
    /// $$k_2 = v\left(x_t + \frac{h}{2} k_1, t + \frac{h}{2}\right)$$
    /// $$k_3 = v\left(x_t + \frac{h}{2} k_2, t + \frac{h}{2}\right)$$
    /// $$k_4 = v(x_t + h k_3, t + h)$$
    /// $$x_{t+h} = x_t + \frac{h}{6} (k_1 + 2k_2 + 2k_3 + k_4)$$
    pub fn step<F>(x: &[f32], t: f32, h: f32, mut velocity_fn: F) -> Vec<f32>
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
    {
        let n = x.len();

        // 1. Compute k1
        let k1 = velocity_fn(x, t);

        // 2. Compute k2 = v(x + h/2 * k1, t + h/2)
        let mut x_half_k1 = vec![0.0f32; n];
        for i in 0..n {
            x_half_k1[i] = x[i] + 0.5 * h * k1[i];
        }
        let k2 = velocity_fn(&x_half_k1, t + 0.5 * h);

        // 3. Compute k3 = v(x + h/2 * k2, t + h/2)
        let mut x_half_k2 = vec![0.0f32; n];
        for i in 0..n {
            x_half_k2[i] = x[i] + 0.5 * h * k2[i];
        }
        let k3 = velocity_fn(&x_half_k2, t + 0.5 * h);

        // 4. Compute k4 = v(x + h * k3, t + h)
        let mut x_full_k3 = vec![0.0f32; n];
        for i in 0..n {
            x_full_k3[i] = x[i] + h * k3[i];
        }
        let k4 = velocity_fn(&x_full_k3, t + h);

        // 5. Final integration
        let mut x_next = vec![0.0f32; n];
        let factor = h / 6.0;
        for i in 0..n {
            x_next[i] = x[i] + factor * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }

        x_next
    }

    /// Full trajectory integration from $t = 0.0$ to $t = 1.0$ over $N$ discrete steps.
    pub fn solve_trajectory<F>(x0: &[f32], num_steps: usize, mut velocity_fn: F) -> Vec<f32>
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
    {
        assert!(num_steps > 0, "Number of steps must be positive");
        let dt = 1.0 / num_steps as f32;
        let mut current_x = x0.to_vec();

        for step_idx in 0..num_steps {
            let t = step_idx as f32 * dt;
            current_x = Self::step(&current_x, t, dt, &mut velocity_fn);
        }

        current_x
    }
}
