#!/usr/bin/env python3
"""
High-Precision Golden Reference Vector Generator for RainAI Inference Runtime.

Computes exact symbolic Box-Cox inversions and deterministic recurrent
baselines for state-space updates, MoE gating, and Ambisonic projection.
"""

from pathlib import Path
import json
import numpy as np
import sympy as sp

def find_project_root() -> Path:
    curr = Path(__file__).resolve().parent
    for _ in range(4):
        if (curr / "Cargo.toml").exists() or (curr / "crates").exists():
            return curr
        curr = curr.parent
    return Path(__file__).resolve().parent.parent

PROJECT_ROOT = find_project_root()

CONDITION_DIM = 554
LATENT_DIM = 64
NUM_EXPERTS = 8
EXPERTS_TOP_K = 2
NUM_STEPS = 32


def generate_symbolic_box_cox_table() -> list[dict]:
    """Uses Sympy to compute high-precision Box-Cox inverse values across test coordinates."""
    y_sym = sp.Symbol('y', real=True)
    lambda_sym = sp.Symbol('lambda', real=True)

    # Box-Cox inverse: x = sign(y) * [ (1 + |y| * (1 - lambda))^(1 / (1 - lambda)) - 1 ]
    p_sym = 1 - lambda_sym
    inv_box_cox_expr = sp.sign(y_sym) * ((1 + sp.Abs(y_sym) * p_sym) ** (1 / p_sym) - 1)
    inv_box_cox_lambda_1 = sp.sign(y_sym) * (sp.exp(sp.Abs(y_sym)) - 1)

    eval_points = [
        (0.0, 0.0),
        (0.5, 0.0),
        (-0.5, 0.0),
        (0.25, 0.5),
        (-0.75, 0.5),
        (0.1, 1.0),
        (-0.5, 1.0),
        (0.4, 0.8),
    ]

    results = []
    for y_val, l_val in eval_points:
        if abs(l_val - 1.0) < 1e-6:
            val = inv_box_cox_lambda_1.subs({y_sym: y_val}).evalf(32)
        else:
            val = inv_box_cox_expr.subs({y_sym: y_val, lambda_sym: l_val}).evalf(32)
        results.append({
            "y": float(y_val),
            "lambda": float(l_val),
            "exact_x": float(val),
        })

    return results


def build_deterministic_matrices():
    """Generates synthetic deterministic parameter tensors for pipeline testing."""
    np.random.seed(42)

    # Diagonal State-Space recurrence parameters (A in (0.85, 0.98), B in (-0.2, 0.2))
    a_diag = (0.90 + 0.05 * np.cos(np.linspace(0, 4 * np.pi, LATENT_DIM))).astype(np.float32)
    b_diag = (0.10 * np.sin(np.linspace(0, 2 * np.pi, LATENT_DIM))).astype(np.float32)

    # Linear projection: 554 conditioning -> 64 latent
    w_in = (0.01 * np.sin(np.outer(np.arange(LATENT_DIM), np.arange(CONDITION_DIM)))).astype(np.float32)
    b_in = (0.005 * np.ones(LATENT_DIM)).astype(np.float32)

    # MoE Router weights: 64 latent -> 8 experts
    w_gate = (0.02 * np.cos(np.outer(np.arange(NUM_EXPERTS), np.arange(LATENT_DIM)))).astype(np.float32)

    # Ambisonic 4-channel projection matrix: 64 latent -> 4 channels (W, X, Y, Z)
    w_foa = np.zeros((4, LATENT_DIM), dtype=np.float32)
    for i in range(LATENT_DIM):
        ch = i % 4
        scale = 1.0 / (np.sqrt(LATENT_DIM / 4.0))
        w_foa[ch, i] = scale * (1.0 if ch == 0 else np.sin((i + 1) * 0.5))

    return a_diag, b_diag, w_in, b_in, w_gate, w_foa


def simulate_recurrence_steps(a_diag, b_diag, w_in, b_in, w_gate, w_foa):
    """Calculates step-by-step mathematical ground truths across multiple time frames."""
    latent_state = np.zeros(LATENT_DIM, dtype=np.float32)
    trace = []

    for t in range(NUM_STEPS):
        t_f = float(t)
        conditioning = (
            0.5 + 0.3 * np.sin(np.arange(CONDITION_DIM) * 0.05 + t_f * 0.1)
        ).astype(np.float32)

        # 1. Conditioning Projection u_t = W_in @ c_t + b_in
        u_t = (w_in @ conditioning) + b_in

        # 2. Diagonal Mamba2 Recurrence: s_t = A * s_{t-1} + B * u_t
        latent_state = (a_diag * latent_state) + (b_diag * u_t)

        # 3. MoE Routing: Top-2 selection out of 8
        router_logits = w_gate @ latent_state
        exp_logits = np.exp(router_logits - np.max(router_logits))
        router_probs = exp_logits / np.sum(exp_logits)

        top_2_indices = np.argsort(router_logits)[-EXPERTS_TOP_K:][::-1].tolist()
        top_2_weights = [float(router_probs[idx]) for idx in top_2_indices]

        # Expert activation modulation
        modulated_latent = latent_state.copy()
        for exp_idx in range(NUM_EXPERTS):
            start = exp_idx * 8
            end = start + 8
            if exp_idx not in top_2_indices:
                modulated_latent[start:end] *= 0.85

        # 4. Ambisonic FOA Projection: y_t = W_foa @ modulated_latent
        foa_out = w_foa @ modulated_latent
        w, x, y, z = [float(val) for val in foa_out]

        trace.append({
            "step": t,
            "conditioning_sample_sum": float(np.sum(conditioning)),
            "latent_state_norm": float(np.linalg.norm(latent_state)),
            "top_2_experts": top_2_indices,
            "top_2_weights": top_2_weights,
            "foa": {"w": w, "x": x, "y": y, "z": z},
        })

    return trace


def main():
    output_dir = PROJECT_ROOT / "crates" / "inference" / "data" / "golden_vectors"
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / "baseline_step_trace.json"

    print("Generating SymPy Box-Cox validation references...")
    box_cox_table = generate_symbolic_box_cox_table()

    print("Synthesizing parameter matrices and stepping Mamba2-MoE recurrence...")
    a_diag, b_diag, w_in, b_in, w_gate, w_foa = build_deterministic_matrices()
    trace = simulate_recurrence_steps(a_diag, b_diag, w_in, b_in, w_gate, w_foa)

    payload = {
        "description": "Deterministic SymPy & NumPy Golden Reference Trace for RainAI Inference Runtime",
        "latent_dim": LATENT_DIM,
        "condition_dim": CONDITION_DIM,
        "num_experts": NUM_EXPERTS,
        "box_cox_reference": box_cox_table,
        "trace": trace,
    }

    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Successfully wrote {len(trace)} golden reference steps to: {target_path.resolve()}")


if __name__ == "__main__":
    main()
