"""
Training script for Phase 3: Mamba-2 State Space Duality (SSD) MoE & Hardware-in-the-Loop Meta-Controller.
Includes Jamba Hybrid Attention, Flow Matching Vector Fields, CFG Conditioning Dropout,
MoE Load Balancing & Entropy Regularization, Structured Sparsity, EMA Tracking,
Automatic Mixed Precision (AMP), Gradient Accumulation, and Validation Monitoring.
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Optional, Dict
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
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


class ModelEMA:
    """Maintains an exponential moving average of model weights for high inference stability."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].copy_(self.decay * self.shadow[k] + (1.0 - self.decay) * v)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        for k, v in state_dict.items():
            if k in self.shadow:
                self.shadow[k].copy_(v)
            else:
                self.shadow[k] = v.clone().detach()


def train_mamba_epoch(
    encoder: nn.Module,
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
    encoder.eval()
    mamba_moe.train()
    meta_controller.train()
    
    total_loss = 0.0
    total_nll = 0.0
    total_vel = 0.0
    total_turb = 0.0
    total_flow = 0.0
    total_penalty = 0.0
    total_moe_aux = 0.0
    active_experts_count = 0.0
    num_batches = 0
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(dataloader, 1):
        if max_batches and batch_idx > max_batches:
            break
            
        audio = batch["audio"].to(device)         # (B, 4, 240000)
        u = batch["conditioning"].to(device)     # (B, 554)
        batch_size = audio.shape[0]
        
        if torch.rand(1).item() < cfg_dropout_prob:
            u_eff = torch.zeros_like(u)
        else:
            u_eff = u
            
        if torch.rand(1).item() < 0.5:
            slice_level = 4
        else:
            slice_level = int(torch.randint(0, 4, (1,)).item())
            
        with torch.no_grad():
            z_q, _, enc_quality = encoder(audio, u_eff, tau=0.05, active_level=4)
            
        z_seq = z_q.transpose(1, 2)
        z_input = z_seq[:, :-1, :]
        z_target = z_seq[:, 1:, :]
        
        profile_choice = torch.randint(0, 8, (batch_size, 1), device=device)
        buf_ms = torch.empty(batch_size, 1, device=device).uniform_(30.0, 50.0)
        cpu_h = torch.empty(batch_size, 1, device=device).uniform_(0.8, 1.0)
        gpu_h = torch.empty(batch_size, 1, device=device).uniform_(0.8, 1.0)
        dt_ms = torch.empty(batch_size, 1, device=device).uniform_(9.0, 11.0)
        
        is_thermal = (profile_choice == 1).float()
        cpu_h = cpu_h * (1.0 - 0.65 * is_thermal)
        gpu_h = gpu_h * (1.0 - 0.65 * is_thermal)
        buf_ms = buf_ms - 20.0 * is_thermal
        dt_ms = dt_ms + 18.0 * is_thermal
        
        is_gc_stall = (profile_choice == 2).float()
        dt_ms = dt_ms + 32.0 * is_gc_stall
        buf_ms = buf_ms - 22.0 * is_gc_stall
        
        is_bus_choke = (profile_choice == 3).float()
        cpu_h = cpu_h * (1.0 - 0.45 * is_bus_choke)
        gpu_h = gpu_h * (1.0 - 0.70 * is_bus_choke)
        
        is_game_interference = (profile_choice == 4).float()
        dt_ms = dt_ms + torch.empty(batch_size, 1, device=device).uniform_(5.0, 25.0) * is_game_interference
        gpu_h = gpu_h * (0.4 + 0.6 * torch.rand(batch_size, 1, device=device)) * is_game_interference + gpu_h * (1.0 - is_game_interference)
        
        is_bluetooth = (profile_choice == 5).float()
        buf_ms = buf_ms + 110.0 * is_bluetooth
        dt_ms = dt_ms + 5.0 * is_bluetooth
        
        is_eco = (profile_choice == 6).float()
        cpu_h = cpu_h * (1.0 - 0.85 * is_eco)
        gpu_h = gpu_h * (1.0 - 0.85 * is_eco)
        buf_ms = buf_ms - 15.0 * is_eco
        
        is_ecore = (profile_choice == 7).float()
        cpu_h = cpu_h * (1.0 - 0.75 * is_ecore)
        gpu_h = gpu_h * 1.0
        
        telemetry = torch.cat([torch.clamp(buf_ms, min=2.0), torch.clamp(cpu_h, 0.05, 1.0), torch.clamp(gpu_h, 0.05, 1.0), torch.clamp(dt_ms, min=1.0)], dim=-1)
        user_behavior_weights = torch.tensor([[0.7, 0.3, 40.0]], device=device).repeat(batch_size, 1)
        
        init_logits = torch.randn(batch_size, mamba_moe.num_experts, device=device)
        dummy_qual = torch.cat([enc_quality, torch.ones_like(enc_quality)], dim=-1)
        
        device_type = "cuda" if device.startswith("cuda") else "cpu"
        with torch.amp.autocast(device_type, enabled=use_amp):
            meta_decision = meta_controller(
                init_logits, 
                telemetry, 
                user_behavior_weights, 
                quality_scores=dummy_qual,
                active_slice_level=slice_level
            )
            expert_mask = meta_decision["expert_mask"]
            tau_moe = meta_decision["tau_moe"]
            
            mu, sigma, moe_logits, mamba_quality, log_var = mamba_moe(
                z_input, 
                u_eff, 
                expert_mask=expert_mask, 
                tau_moe=tau_moe,
                drift_scale=0.2,
                active_level=slice_level
            )
            
            traj_losses = traj_loss_fn(mu, log_var, z_target)
            traj_loss = traj_losses["total_traj_loss"]
            
            flow_dict = mamba_moe.compute_flow_matching_loss(z_target, u_eff, expert_mask, tau_moe)
            flow_loss = flow_dict["flow_matching_loss"]
            
            moe_loss_dict = moe_balancer(expert_mask)
            moe_aux_loss = moe_loss_dict["moe_aux_loss"]
            
            sparsity_loss = mamba_moe.get_activation_sparsity_loss()
            
            hwil_penalty = compute_slice_aware_hwil_penalty(
                expert_mask, 
                buf_ms, 
                gpu_h, 
                active_slice_level=slice_level
            )
            
            loss = (
                traj_loss + 
                0.3 * flow_loss + 
                hwil_penalty + 
                moe_aux_loss + 
                sparsity_loss
            ) / accumulation_steps
            
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
        raw_loss = loss.item() * accumulation_steps
        total_loss += raw_loss
        total_nll += traj_losses["nll_loss"].item()
        total_vel += traj_losses["velocity_loss"].item()
        total_turb += traj_losses["turbulence_loss"].item()
        total_flow += flow_loss.item()
        total_penalty += hwil_penalty.item()
        total_moe_aux += moe_aux_loss.item()
        active_experts_count += float(torch.mean(num_active).item())
        num_batches += 1
        
        slice_names = ["S0_Ternary", "S1_INT4", "S2_INT8_BF16", "S3_INT16_FP16", "S4_FP32_Master"]
        if batch_idx % 2 == 0 or (max_batches and batch_idx == max_batches):
            print(
                f"    [Batch {batch_idx:02d} | Slice {slice_level} ({slice_names[slice_level]})] "
                f"Loss: {raw_loss:.4f} (NLL: {traj_losses['nll_loss'].item():.3f}, "
                f"Flow: {flow_loss.item():.3f}, Vel: {traj_losses['velocity_loss'].item():.3f}, "
                f"Active: {torch.mean(num_active).item():.1f}/8)",
                flush=True
            )
            
    return {
        "loss": total_loss / max(num_batches, 1),
        "nll": total_nll / max(num_batches, 1),
        "vel": total_vel / max(num_batches, 1),
        "turb": total_turb / max(num_batches, 1),
        "flow": total_flow / max(num_batches, 1),
        "hwil_penalty": total_penalty / max(num_batches, 1),
        "moe_aux": total_moe_aux / max(num_batches, 1),
        "avg_active_experts": active_experts_count / max(num_batches, 1)
    }


@torch.no_grad()
def validate_mamba_epoch(
    encoder: nn.Module,
    mamba_moe: nn.Module,
    meta_controller: nn.Module,
    dataloader,
    traj_loss_fn: PhysicsTrajectoryLoss,
    device: str = "cpu",
    max_batches: Optional[int] = None
) -> Dict[str, float]:
    encoder.eval()
    mamba_moe.eval()
    meta_controller.eval()
    
    total_loss = 0.0
    total_nll = 0.0
    total_flow = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader, 1):
        if max_batches and batch_idx > max_batches:
            break
        audio = batch["audio"].to(device)
        u = batch["conditioning"].to(device)
        batch_size = audio.shape[0]
        
        z_q, _, enc_quality = encoder(audio, u, tau=0.05, active_level=4)
        z_seq = z_q.transpose(1, 2)
        z_input = z_seq[:, :-1, :]
        z_target = z_seq[:, 1:, :]
        
        telemetry = torch.tensor([[40.0, 1.0, 1.0, 10.0]], device=device).repeat(batch_size, 1)
        user_weights = torch.tensor([[0.7, 0.3, 40.0]], device=device).repeat(batch_size, 1)
        init_logits = torch.zeros(batch_size, mamba_moe.num_experts, device=device)
        dummy_qual = torch.cat([enc_quality, torch.ones_like(enc_quality)], dim=-1)
        
        meta_decision = meta_controller(init_logits, telemetry, user_weights, dummy_qual, active_slice_level=4)
        expert_mask = meta_decision["expert_mask"]
        tau_moe = meta_decision["tau_moe"]
        
        mu, _, _, _, log_var = mamba_moe(z_input, u, expert_mask=expert_mask, tau_moe=tau_moe, active_level=4)
        traj_losses = traj_loss_fn(mu, log_var, z_target)
        flow_dict = mamba_moe.compute_flow_matching_loss(z_target, u, expert_mask, tau_moe)
        
        total_loss += traj_losses["total_traj_loss"].item()
        total_nll += traj_losses["nll_loss"].item()
        total_flow += flow_dict["flow_matching_loss"].item()
        num_batches += 1
        
    return {
        "val_loss": total_loss / max(num_batches, 1),
        "val_nll": total_nll / max(num_batches, 1),
        "val_flow": total_flow / max(num_batches, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="RainAI Phase 3: Production Mamba-2 MoE & Meta-Controller Training")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--accumulation-steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--use-amp", action="store_true", default=True, help="Enable automatic mixed precision")
    parser.add_argument("--max-batches", type=int, default=0, help="Max batches per epoch (0 for full dataset)")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to processed audio directory")
    parser.add_argument("--manifest-path", type=str, default=None, help="Path to manifest JSON")
    parser.add_argument("--save-dir", type=str, default=None, help="Path to save checkpoints")
    parser.add_argument("--vae-checkpoint", type=str, default=None, help="Path to pretrained VAE checkpoint")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume training from mamba2_metacontroller_best.pt if available")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Disable checkpoint resumption")
    args = parser.parse_args()

    print("=" * 75)
    print("RainAI Phase 3: Production Mamba-2 SSD MoE & Hardware-in-the-Loop Meta-Control")
    print("Jamba Hybrid Attention | Flow Matching | MoE Load Balancing | Multi-Slice QAT")
    print("=" * 75)
    
    proc_cand = PROJECT_ROOT / "data" / "processed"
    if not proc_cand.exists():
        proc_cand = PROJECT_ROOT / "Data" / "processed"
    processed_dir = Path(args.data_dir) if args.data_dir else proc_cand

    manifest_cand = processed_dir / "manifest.json"
    if not manifest_cand.exists():
        manifest_cand = PROJECT_ROOT / "data" / "rain_corpus_manifest.json"
    manifest_path = Path(args.manifest_path) if args.manifest_path else manifest_cand

    save_dir = Path(args.save_dir) if args.save_dir else (PROJECT_ROOT / "checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute device: {device} | AMP: {args.use_amp} | Accumulation: {args.accumulation_steps}")
    
    train_loader = create_dataloader(
        data_dir=processed_dir,
        manifest_path=manifest_path,
        batch_size=args.batch_size,
        split="train",
        val_ratio=0.15,
        augment_yaw=True,
        shuffle=True
    )
    val_loader = create_dataloader(
        data_dir=processed_dir,
        manifest_path=manifest_path,
        batch_size=args.batch_size,
        split="val",
        val_ratio=0.15,
        augment_yaw=False,
        shuffle=False
    )
    print(f"[*] Loaded {len(train_loader.dataset)} train samples, {len(val_loader.dataset)} val samples.")

    encoder = SpatialAudioEncoder().to(device)
    mamba_moe = Mamba2MoETrajectory().to(device)
    meta_controller = InvasiveMetaController(num_experts=8).to(device)
    traj_loss_fn = PhysicsTrajectoryLoss().to(device)
    moe_balancer = MoELoadBalancingLoss(num_experts=8).to(device)

    vae_ckpt_path = Path(args.vae_checkpoint) if args.vae_checkpoint else (save_dir / "spatial_vae_ddsp_latest.pt")
    if vae_ckpt_path.exists():
        print(f"[*] Loading pretrained SpatialAudioEncoder from {vae_ckpt_path.name}...")
        try:
            ckpt = torch.load(vae_ckpt_path, map_location=device)
            encoder.load_state_dict(ckpt["encoder"])
            print("[+] Pretrained SpatialAudioEncoder loaded successfully.")
        except Exception as e:
            print(f"[!] Warning: Could not load encoder checkpoint ({e}). Using fresh weights.")
    else:
        print(f"[*] Pretrained VAE checkpoint not found at {vae_ckpt_path}. Using initial weights.")
    
    optimizer = torch.optim.AdamW(
        list(mamba_moe.parameters()) + list(meta_controller.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda" if device.startswith("cuda") else "cpu", enabled=args.use_amp)
    ema = ModelEMA(mamba_moe, decay=0.995)

    best_val_loss = float("inf")
    start_epoch = 1
    mamba_best_path = save_dir / "mamba2_metacontroller_best.pt"
    if args.resume and mamba_best_path.exists():
        print(f"[*] Resuming Mamba-2 & MetaController from {mamba_best_path.name}...")
        try:
            mamba_ckpt = torch.load(mamba_best_path, map_location=device)
            if "mamba_moe" in mamba_ckpt:
                mamba_moe.load_state_dict(mamba_ckpt["mamba_moe"])
            if "mamba_moe_ema" in mamba_ckpt:
                ema.load_state_dict(mamba_ckpt["mamba_moe_ema"])
            if "meta_controller" in mamba_ckpt:
                meta_controller.load_state_dict(mamba_ckpt["meta_controller"])
            if "best_val_loss" in mamba_ckpt:
                best_val_loss = mamba_ckpt["best_val_loss"]
            if "epoch" in mamba_ckpt:
                start_epoch = mamba_ckpt["epoch"] + 1
            print(f"[+] Resumed Mamba-2 checkpoint (prior best val loss: {best_val_loss:.4f})")
        except Exception as e:
            print(f"[!] Warning: Could not resume Mamba-2 checkpoint ({e}). Starting fresh.")

    total_target_epochs = start_epoch + args.epochs - 1
    print(f"[*] Starting Mamba-2 trajectory training loop for {args.epochs} epochs (epochs {start_epoch} -> {total_target_epochs})...")
    for epoch in range(start_epoch, total_target_epochs + 1):
        start = time.time()
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
        scheduler.step()
        
        val_metrics = validate_mamba_epoch(
            encoder=encoder,
            mamba_moe=mamba_moe,
            meta_controller=meta_controller,
            dataloader=val_loader,
            traj_loss_fn=traj_loss_fn,
            device=device,
            max_batches=args.max_batches
        )
        elapsed = time.time() - start
        
        print(
            f"Epoch {epoch:02d}/{total_target_epochs:02d} | "
            f"Train Loss: {train_metrics['loss']:.4f} (NLL: {train_metrics['nll']:.3f}, Flow: {train_metrics['flow']:.3f}) | "
            f"Val Loss: {val_metrics['val_loss']:.4f} (ValNLL: {val_metrics['val_nll']:.3f}, ValFlow: {val_metrics['val_flow']:.3f}) | "
            f"Active: {train_metrics['avg_active_experts']:.1f}/8 | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"Time: {elapsed:.2f}s",
            flush=True
        )
        
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            torch.save({
                "mamba_moe": mamba_moe.state_dict(),
                "mamba_moe_ema": ema.state_dict(),
                "meta_controller": meta_controller.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss
            }, save_dir / "mamba2_metacontroller_best.pt")
            print(f"  [*] Saved new best Mamba-2 checkpoint (Val Loss: {best_val_loss:.4f})")

    save_path = save_dir / "mamba2_metacontroller_latest.pt"
    torch.save({
        "mamba_moe": mamba_moe.state_dict(),
        "mamba_moe_ema": ema.state_dict(),
        "meta_controller": meta_controller.state_dict(),
        "epochs_completed": total_target_epochs
    }, save_path)
    print(f"[+] Checkpoint saved to {save_path}")


if __name__ == "__main__":
    main()
