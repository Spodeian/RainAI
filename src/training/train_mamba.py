"""
Optimized Training script for Phase 3: Mamba-2 MoE & Meta-Controller.
Streamlined for high-throughput latent sequence processing.
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Optional, Dict
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import torch._dynamo

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import create_dataloader
from src.models.diff_autoencoder import SpatialAudioEncoder
from src.models.mamba2_moe import Mamba2MoETrajectory
from src.models.meta_controller import InvasiveMetaController
from src.models.physics_losses import (
    PhysicsTrajectoryLoss,
    compute_slice_aware_hwil_penalty,
    MoELoadBalancingLoss
)

torch._dynamo.config.recompile_limit = 32
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.force_parameter_static_shapes = False

def verify_checkpoint_integrity(state_dict: Dict[str, torch.Tensor], name: str = "Checkpoint"):
    for k, v in state_dict.items():
        if torch.isnan(v).any() or torch.isinf(v).any():
            raise ValueError(f"[!] {name} parameter '{k}' contains NaN or Inf!")


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            clean_k = k[len("_orig_mod.")] if k.startswith("_orig_mod.") else k
            if clean_k in self.shadow:
                self.shadow[clean_k].copy_(self.decay * self.shadow[clean_k] + (1.0 - self.decay) * v)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        verify_checkpoint_integrity(state_dict, "Mamba EMA Checkpoint")
        for k, v in state_dict.items():
            clean_k = k[len("_orig_mod.")] if k.startswith("_orig_mod.") else k
            if clean_k in self.shadow:
                self.shadow[clean_k].copy_(v)
            else:
                self.shadow[clean_k] = v.clone().detach()


def train_mamba_epoch(
    encoder: Optional[nn.Module],
    mamba_moe: nn.Module,
    meta_controller: nn.Module,
    dataloader,
    optimizer,
    scaler: GradScaler,
    traj_loss_fn: PhysicsTrajectoryLoss,
    moe_balancer: MoELoadBalancingLoss,
    accumulation_steps: int = 1,
    use_amp: bool = False,
    device: str = "cpu",
    max_batches: Optional[int] = None,
    ema: Optional[ModelEMA] = None,
    cfg_dropout_prob: float = 0.15,
) -> Dict[str, float]:
    if encoder is not None:
        encoder.eval()
    mamba_moe.train()
    meta_controller.train()
    
    total_loss, total_nll, total_vel, total_flow, total_penalty, total_moe_aux, active_experts_count, num_batches = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
    consecutive_nans = 0
    MAX_CONSECUTIVE_NANS = 3
    
    # Pre-allocate static base tensors to prevent per-batch CPU-GPU serialization
    base_telemetry = torch.tensor([[40.0, 1.0, 1.0, 10.0]], device=device)
    base_user_weights = torch.tensor([[0.7, 0.3, 40.0]], device=device)
    num_experts = getattr(mamba_moe, "num_experts", 8)
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(dataloader, 1):
        if max_batches and batch_idx > max_batches:
            break
            
        audio = batch["audio"].to(device)
        u = batch["conditioning"].to(device)
        batch_size = audio.shape[0]
        
        u_eff = torch.zeros_like(u) if torch.rand(1).item() < cfg_dropout_prob else u
        # Keep slice_level on-device to avoid torch.compile guard failures
        slice_level = torch.randint(0, 5, (1,), device=device).squeeze()
            
        with torch.no_grad():
            if encoder is not None:
                z_q, _, enc_quality = encoder(audio, u_eff, tau=0.05, active_level=4)
            else:
                z_q = torch.randn(batch_size, 64, 128, device=device)
                enc_quality = torch.ones(batch_size, 1, device=device)
            
        z_seq = z_q.transpose(1, 2)
        z_input = z_seq[:, :-1, :]
        z_target = z_seq[:, 1:, :]
        
        # Expand pre-allocated tensors via pointers instead of reconstructing
        telemetry = base_telemetry.expand(batch_size, -1)
        user_behavior_weights = base_user_weights.expand(batch_size, -1)
        init_logits = torch.empty(batch_size, num_experts, device=device).normal_()
        dummy_qual = torch.cat([enc_quality, torch.ones_like(enc_quality)], dim=-1)
        
        with torch.amp.autocast(device_type, enabled=use_amp):
            meta_decision = meta_controller(init_logits, telemetry, user_behavior_weights, quality_scores=dummy_qual, active_slice_level=slice_level)
            expert_mask = meta_decision["expert_mask"]
            tau_moe = meta_decision["tau_moe"]
            
            mu, sigma, moe_logits, mamba_quality, log_var = mamba_moe(z_input, u_eff, expert_mask=expert_mask, tau_moe=tau_moe, active_level=slice_level)
            
            traj_losses = traj_loss_fn(mu, log_var, z_target)
            traj_loss = traj_losses["total_traj_loss"]
            
            flow_dict = mamba_moe.compute_flow_matching_loss(z_target, u_eff, expert_mask, tau_moe)
            flow_loss = flow_dict["flow_matching_loss"]
            
            moe_loss_dict = moe_balancer(expert_mask)
            moe_aux_loss = moe_loss_dict["moe_aux_loss"]
            sparsity_loss = mamba_moe.get_activation_sparsity_loss()
            
            hwil_penalty = compute_slice_aware_hwil_penalty(expert_mask, torch.tensor(40.0, device=device), torch.tensor(1.0, device=device), active_slice_level=slice_level)
            
            loss = (traj_loss + 0.3 * flow_loss + hwil_penalty + moe_aux_loss + sparsity_loss) / accumulation_steps
            
            if torch.isnan(loss) or torch.isinf(loss):
                consecutive_nans += 1
                print(f"\n[!] DIAGNOSTIC WARNING: NaN/Inf detected in Mamba-2 Loss at Batch {batch_idx}", flush=True)
                if consecutive_nans >= MAX_CONSECUTIVE_NANS:
                    raise RuntimeError("Aborting Mamba-2 training: Persistent NaN/Inf numerical instability.")
                optimizer.zero_grad()
                continue
            else:
                consecutive_nans = 0
            
        scaler.scale(loss).backward()
        
        is_last = (batch_idx == len(dataloader)) or (max_batches is not None and batch_idx == max_batches)
        if batch_idx % accumulation_steps == 0 or is_last:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(mamba_moe.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(meta_controller.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if ema:
                ema.update(mamba_moe)
                
        num_active = torch.sum(expert_mask, dim=-1, keepdim=True)
        total_loss += loss.item() * accumulation_steps
        total_nll += traj_losses["nll_loss"].item()
        total_vel += traj_losses["velocity_loss"].item()
        total_flow += flow_loss.item()
        total_penalty += hwil_penalty.item()
        total_moe_aux += moe_aux_loss.item()
        active_experts_count += float(torch.mean(num_active).item())
        num_batches += 1
            
    return {
        "loss": total_loss / max(num_batches, 1),
        "nll": total_nll / max(num_batches, 1),
        "vel": total_vel / max(num_batches, 1),
        "turb": 0.0,
        "flow": total_flow / max(num_batches, 1),
        "hwil_penalty": total_penalty / max(num_batches, 1),
        "moe_aux": total_moe_aux / max(num_batches, 1),
        "avg_active_experts": active_experts_count / max(num_batches, 1)
    }

@torch.no_grad()
def validate_mamba_epoch(encoder, mamba_moe, meta_controller, dataloader, traj_loss_fn, device="cpu", max_batches=None):
    if encoder is not None:
        encoder.eval()
    mamba_moe.eval()
    meta_controller.eval()
    
    total_loss, total_nll, total_flow, num_batches = 0.0, 0.0, 0.0, 0
    
    # Pre-allocate
    base_telemetry = torch.tensor([[40.0, 1.0, 1.0, 10.0]], device=device)
    base_user_weights = torch.tensor([[0.7, 0.3, 40.0]], device=device)
    num_experts = getattr(mamba_moe, "num_experts", 8)
    
    for batch_idx, batch in enumerate(dataloader, 1):
        if max_batches and batch_idx > max_batches:
            break
        audio = batch["audio"].to(device)
        u = batch["conditioning"].to(device)
        batch_size = audio.shape[0]
        
        if encoder is not None:
            z_q, _, enc_quality = encoder(audio, u, tau=0.05, active_level=4)
        else:
            z_q = torch.randn(batch_size, 64, 128, device=device)
            enc_quality = torch.ones(batch_size, 1, device=device)
            
        z_seq = z_q.transpose(1, 2)
        z_input = z_seq[:, :-1, :]
        z_target = z_seq[:, 1:, :]
        
        # Expand
        telemetry = base_telemetry.expand(batch_size, -1)
        user_weights = base_user_weights.expand(batch_size, -1)
        init_logits = torch.zeros(batch_size, num_experts, device=device)
        dummy_qual = torch.cat([enc_quality, torch.ones_like(enc_quality)], dim=-1)
        
        meta_decision = meta_controller(init_logits, telemetry, user_weights, dummy_qual, active_slice_level=4)
        mu, _, _, _, log_var = mamba_moe(z_input, u, expert_mask=meta_decision["expert_mask"], tau_moe=meta_decision["tau_moe"], active_level=4)
        
        traj_losses = traj_loss_fn(mu, log_var, z_target)
        flow_dict = mamba_moe.compute_flow_matching_loss(z_target, u, meta_decision["expert_mask"], meta_decision["tau_moe"])
        
        total_loss += traj_losses["total_traj_loss"].item()
        total_nll += traj_losses["nll_loss"].item()
        total_flow += flow_dict["flow_matching_loss"].item()
        num_batches += 1
        
    return {"val_loss": total_loss / max(num_batches, 1), "val_nll": total_nll / max(num_batches, 1), "val_flow": total_flow / max(num_batches, 1)}


def main():
    parser = argparse.ArgumentParser(description="Optimized RainAI Phase 3: Mamba-2 MoE Training")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--use-amp", action="store_true", default=True)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--manifest-path", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--vae-checkpoint", type=str, default=None)
    parser.add_argument("--freeze-encoder", action="store_true", default=True, help="Keep VAE encoder in memory or bypass during training loops")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()

    print("=" * 75)
    print("RainAI Phase 3: Streamlined Mamba-2 SSD MoE & Meta-Controller")
    print("=" * 75)
    
    proc_cand = PROJECT_ROOT / "data" / "processed"
    if not proc_cand.exists():
        proc_cand = PROJECT_ROOT / "Data" / "processed"
    processed_dir = Path(args.data_dir) if args.data_dir else proc_cand
    manifest_path = Path(args.manifest_path) if args.manifest_path else (processed_dir / "manifest.json")
    save_dir = Path(args.save_dir) if args.save_dir else (PROJECT_ROOT / "checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    is_cuda = device.startswith("cuda")

    train_loader = create_dataloader(
        data_dir=processed_dir, manifest_path=manifest_path, batch_size=args.batch_size, 
        split="train", val_ratio=0.15, augment_yaw=True, shuffle=True, 
        pin_memory=is_cuda, num_workers=4
    )
    val_loader = create_dataloader(
        data_dir=processed_dir, manifest_path=manifest_path, batch_size=args.batch_size, 
        split="val", val_ratio=0.15, augment_yaw=False, shuffle=False, 
        pin_memory=is_cuda, num_workers=4
    )

    encoder = SpatialAudioEncoder().to(device) if not args.freeze_encoder else None
    mamba_moe = Mamba2MoETrajectory().to(device)
    meta_controller = InvasiveMetaController(num_experts=8).to(device)

    vae_ckpt_path = Path(args.vae_checkpoint) if args.vae_checkpoint else (save_dir / "spatial_vae_ddsp_latest.pt")
    if encoder is not None and vae_ckpt_path.exists():
        print(f"[*] Loading pretrained SpatialAudioEncoder from {vae_ckpt_path.name}...")
        ckpt = torch.load(vae_ckpt_path, map_location=device)
        verify_checkpoint_integrity(ckpt["encoder"], "Pretrained VAE")
        vae_state_dict = {
            (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v 
            for k, v in ckpt["encoder"].items()
        }
        encoder.load_state_dict(vae_state_dict, strict=False)
        print("[+] Pretrained VAE weights verified and loaded safely.")
    elif encoder is not None:
        print("[!] VAE checkpoint not found. Initializing encoder randomly.")

    if is_cuda and hasattr(torch, "compile"):
        print("[*] Applying torch.compile(mode='default') to Mamba-2 & Meta-Controller...")
        mamba_moe = torch.compile(mamba_moe, mode="default")
        meta_controller = torch.compile(meta_controller, mode="default")

    traj_loss_fn = PhysicsTrajectoryLoss().to(device)
    moe_balancer = MoELoadBalancingLoss(num_experts=8).to(device)

    optimizer = torch.optim.AdamW(list(mamba_moe.parameters()) + list(meta_controller.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda" if is_cuda else "cpu", enabled=args.use_amp)
    ema = ModelEMA(mamba_moe, decay=0.995)

    best_val_loss = float("inf")
    start_epoch = 1
    mamba_best_path = save_dir / "mamba2_metacontroller_best.pt"
    if args.resume and mamba_best_path.exists():
        try:
            mamba_ckpt = torch.load(mamba_best_path, map_location=device)
            if "mamba_moe" in mamba_ckpt:
                verify_checkpoint_integrity(mamba_ckpt["mamba_moe"], "Mamba MoE")
                mamba_sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in mamba_ckpt["mamba_moe"].items()}
                mamba_moe.load_state_dict(mamba_sd, strict=False)
            if "meta_controller" in mamba_ckpt:
                verify_checkpoint_integrity(mamba_ckpt["meta_controller"], "MetaController")
                meta_sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in mamba_ckpt["meta_controller"].items()}
                meta_controller.load_state_dict(meta_sd, strict=False)
            best_val_loss = mamba_ckpt.get("best_val_loss", float("inf"))
            start_epoch = mamba_ckpt.get("epoch", 0) + 1
            print(f"[+] Resumed Mamba-2 checkpoint successfully.")
        except Exception as e:
            print(f"[!] Warning: Mamba checkpoint corrupted ({e}). Starting fresh.")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        train_metrics = train_mamba_epoch(
            encoder=encoder,
            mamba_moe=mamba_moe,
            meta_controller=meta_controller,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            traj_loss_fn=traj_loss_fn,
            moe_balancer=moe_balancer,
            accumulation_steps=args.accumulation_steps,
            use_amp=args.use_amp,
            device=device,
            max_batches=args.max_batches,
            ema=ema
        )
        
        epoch_loss = train_metrics.get("loss", 0.0)
        if math.isfinite(epoch_loss) and epoch_loss > 0.0:
            scheduler.step()
        else:
            print(f"[!] Notice: Skipping scheduler step (Loss: {epoch_loss})", flush=True)
        
        val_metrics = validate_mamba_epoch(
            encoder=encoder,
            mamba_moe=mamba_moe,
            meta_controller=meta_controller,
            dataloader=val_loader,
            traj_loss_fn=traj_loss_fn,
            device=device,
            max_batches=args.max_batches
        )

        print(f"Epoch {epoch:02d} | Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['val_loss']:.4f}", flush=True)

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            torch.save({
                "mamba_moe": mamba_moe.state_dict(),
                "mamba_moe_ema": ema.state_dict(),
                "meta_controller": meta_controller.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss
            }, mamba_best_path)

if __name__ == "__main__":
    main()