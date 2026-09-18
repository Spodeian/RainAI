"""
Training script for Phase 2: Continuous Spatial VAE & HOA-DDSP Engine (Diagnostic Hardened & Compiled).
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Optional, Dict
import math
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import torch._dynamo

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import create_dataloader
from src.data.corruptions import AcousticCorruptionPipeline

from src.models.diff_autoencoder import SpatialAudioEncoder, HierarchicalMultiResLoss
from src.models.ddsp import ContinuousParametricFilter, DifferentiableReverbEngine
from src.models.physics_losses import (
    MultiScaleAmbisonicPhysicsLoss,
    MultiScaleSTFTDiscriminator,
    discriminator_hinge_loss,
    generator_adversarial_loss,
    feature_matching_loss,
    BetaVAEDisentanglementLoss
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
        verify_checkpoint_integrity(state_dict, "EMA Checkpoint")
        for k, v in state_dict.items():
            clean_k = k[len("_orig_mod.")] if k.startswith("_orig_mod.") else k
            if clean_k in self.shadow:
                self.shadow[clean_k].copy_(v)
            else:
                self.shadow[clean_k] = v.clone().detach()


def train_epoch(
    encoder: nn.Module,
    ddsp: nn.Module,
    discriminator: Optional[nn.Module],
    dataloader,
    opt_g,
    opt_d: Optional[torch.optim.Optimizer],
    scaler: GradScaler,
    physics_criterion: MultiScaleAmbisonicPhysicsLoss,
    disentangle_criterion: BetaVAEDisentanglementLoss,
    hierarchical_criterion: HierarchicalMultiResLoss,
    tau_quant: float,
    global_step: int,
    accumulation_steps: int = 1,
    use_amp: bool = False,
    device: str = "cpu",
    max_batches: Optional[int] = None,
    ema: Optional[ModelEMA] = None,
) -> Dict[str, float]:
    encoder.train()
    ddsp.train()
    if discriminator:
        discriminator.train()
    
    total_loss = 0.0
    total_spectral = 0.0
    total_energy = 0.0
    total_doa = 0.0
    total_transient = 0.0
    total_phase = 0.0
    total_d_loss = 0.0
    total_g_adv = 0.0
    num_batches = 0
    consecutive_nans = 0
    MAX_CONSECUTIVE_NANS = 3
    
    corruption_pipeline = AcousticCorruptionPipeline(sample_rate=48000)
    opt_g.zero_grad()
    if opt_d:
        opt_d.zero_grad()
        
    for batch_idx, batch in enumerate(dataloader, 1):
        if max_batches and max_batches > 0 and batch_idx > max_batches:
            break
            
        audio = batch["audio"].to(device)
        u = batch["conditioning"].to(device)
        noise = batch["noise"].to(device)
        
        # Lock slice level on device
        slice_level = torch.randint(0, 5, (1,), device=device).squeeze()
        corrupted_audio = corruption_pipeline(audio)
        device_type = "cuda" if device.startswith("cuda") else "cpu"
        
        with torch.amp.autocast(device_type, enabled=use_amp):
            z_q, z_align, quality_score = encoder(corrupted_audio, u, tau=tau_quant, active_level=slice_level)
            audio_rec, filter_params = ddsp(z_q, noise, ambisonic_order=1)
            
        with torch.amp.autocast(device_type, enabled=False):
            audio_f32 = audio.float()
            audio_rec_f32 = audio_rec.float()
            
            physics_losses = physics_criterion(audio_rec_f32, audio_f32, current_step=global_step)
            acoustic_loss = physics_losses["total_loss"]
            
            hier_losses = hierarchical_criterion(audio_rec_f32, audio_f32)
            hier_loss = hier_losses["hierarchical_loss"]
            
            disentangle = disentangle_criterion(z_align.float())
            dis_loss = torch.clamp(disentangle["disentangle_loss"], min=0.0, max=50.0)
            
            bit_reg = 0.001 * torch.mean(torch.clamp(encoder.quantizer.beta.float(), min=0.0, max=8.0))
            sparsity_loss = 0.005 * ddsp.get_sparsity_loss(filter_params["gains"].float())
            dampening_loss = 0.01 * ddsp.get_material_dampening_loss(filter_params["q"].float(), max_damped_q=8.0)
            
            g_adv_loss = torch.tensor(0.0, device=device)
            fm_loss = torch.tensor(0.0, device=device)
            real_fmaps_cached = None
            
            if discriminator and opt_d:
                fake_scores, fake_fmaps = discriminator(audio_rec_f32)
                g_adv_loss = generator_adversarial_loss(fake_scores)
                real_scores_d, real_fmaps_cached = discriminator(audio_f32.detach())
                fm_loss = feature_matching_loss(real_fmaps_cached, fake_fmaps)
                
            raw_loss_g = (
                acoustic_loss + 
                0.5 * hier_loss + 
                0.2 * dis_loss + 
                bit_reg + 
                sparsity_loss + 
                dampening_loss + 
                0.1 * g_adv_loss + 
                2.0 * fm_loss
            )
            
            if torch.isnan(raw_loss_g) or torch.isinf(raw_loss_g):
                consecutive_nans += 1
                print(f"\n[!] DIAGNOSTIC WARNING: NaN/Inf detected in VAE Loss at Batch {batch_idx} (Slice Level {slice_level})", flush=True)
                if consecutive_nans >= MAX_CONSECUTIVE_NANS:
                    raise RuntimeError("Aborting VAE training: Persistent NaN/Inf numerical instability encountered across consecutive batches.")
                
                opt_g.zero_grad()
                if opt_d:
                    opt_d.zero_grad()
                continue
            else:
                consecutive_nans = 0
                
            loss_g = raw_loss_g / accumulation_steps
            
        scaler.scale(loss_g).backward()
        
        d_loss_val = 0.0
        if discriminator and opt_d:
            with torch.amp.autocast(device_type, enabled=False):
                fake_scores_d, _ = discriminator(audio_rec_f32.detach())
                loss_d = discriminator_hinge_loss(real_scores_d, fake_scores_d) / accumulation_steps
            scaler.scale(loss_d).backward()
            d_loss_val = loss_d.item() * accumulation_steps
            
        is_last = (batch_idx == len(dataloader)) or (max_batches is not None and batch_idx == max_batches)
        if batch_idx % accumulation_steps == 0 or is_last:
            scaler.unscale_(opt_g)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(ddsp.parameters(), max_norm=1.0)
            scaler.step(opt_g)
            opt_g.zero_grad()
            
            if discriminator and opt_d:
                scaler.unscale_(opt_d)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                scaler.step(opt_d)
                opt_d.zero_grad()
                
            scaler.update()
            if ema:
                ema.update(encoder)
                
        global_step += 1
        num_batches += 1
        
        raw_g_loss = loss_g.item() * accumulation_steps
        total_loss += raw_g_loss
        total_spectral += physics_losses["spectral_loss"].item()
        total_energy += physics_losses["energy_density_loss"].item()
        total_doa += physics_losses["doa_intensity_loss"].item()
        total_transient += physics_losses["transient_envelope_loss"].item()
        total_phase += physics_losses["phase_loss"].item()
        total_d_loss += d_loss_val
        total_g_adv += g_adv_loss.item() if isinstance(g_adv_loss, torch.Tensor) else 0.0
        
    return {
        "loss": total_loss / max(num_batches, 1),
        "spectral_loss": total_spectral / max(num_batches, 1),
        "energy_loss": total_energy / max(num_batches, 1),
        "doa_loss": total_doa / max(num_batches, 1),
        "transient_loss": total_transient / max(num_batches, 1),
        "phase_loss": total_phase / max(num_batches, 1),
        "d_loss": total_d_loss / max(num_batches, 1),
        "g_adv_loss": total_g_adv / max(num_batches, 1),
        "global_step": global_step,
        "avg_bit_width": float(torch.mean(encoder.quantizer.beta).item()),
    }


@torch.no_grad()
def validate_epoch(encoder, ddsp, dataloader, physics_criterion, device="cpu", max_batches=None):
    encoder.eval()
    ddsp.eval()
    total_loss, total_spectral, total_phase, total_doa, num_batches = 0.0, 0.0, 0.0, 0.0, 0
    for batch_idx, batch in enumerate(dataloader, 1):
        if max_batches and batch_idx > max_batches:
            break
        audio = batch["audio"].to(device)
        u = batch["conditioning"].to(device)
        noise = batch["noise"].to(device)
        
        z_q, _, _ = encoder(audio, u, tau=0.05, active_level=4)
        audio_rec, _ = ddsp(z_q, noise, ambisonic_order=1)
        
        physics_losses = physics_criterion(audio_rec.float(), audio.float(), current_step=None)
        total_loss += physics_losses["total_loss"].item()
        total_spectral += physics_losses["spectral_loss"].item()
        total_phase += physics_losses["phase_loss"].item()
        total_doa += physics_losses["doa_intensity_loss"].item()
        num_batches += 1
        
    return {
        "val_loss": total_loss / max(num_batches, 1),
        "val_spectral": total_spectral / max(num_batches, 1),
        "val_phase": total_phase / max(num_batches, 1),
        "val_doa": total_doa / max(num_batches, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="RainAI Phase 2: Production VAE & DDSP Training")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-d", type=float, default=1e-4)
    parser.add_argument("--use-disc", action="store_true", default=False)
    parser.add_argument("--use-amp", action="store_true", default=True)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--chunk-curriculum", action="store_true", default=True)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()

    print("=" * 75)
    print("RainAI Phase 2: Diagnostic-Hardened Spatial VAE + HOA-DDSP Engine")
    print("=" * 75)
    
    proc_cand = PROJECT_ROOT / "data" / "processed"
    if not proc_cand.exists():
        proc_cand = PROJECT_ROOT / "Data" / "processed"
    processed_dir = Path(args.data_dir) if args.data_dir else proc_cand
    manifest_path = processed_dir / "manifest.json"
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    is_cuda = device.startswith("cuda")
    
    train_loader = create_dataloader(
        data_dir=processed_dir, manifest_path=manifest_path, batch_size=args.batch_size, 
        split="train", val_ratio=0.15, augment_yaw=True, shuffle=True, 
        chunk_curriculum=args.chunk_curriculum, pin_memory=is_cuda, num_workers=4
    )
    val_loader = create_dataloader(
        data_dir=processed_dir, manifest_path=manifest_path, batch_size=args.batch_size, 
        split="val", val_ratio=0.15, augment_yaw=False, shuffle=False, 
        chunk_curriculum=False, pin_memory=is_cuda, num_workers=4
    )

    encoder = SpatialAudioEncoder().to(device)
    ddsp = ContinuousParametricFilter(num_filters=16).to(device)
    discriminator = MultiScaleSTFTDiscriminator().to(device) if args.use_disc else None

    save_dir = PROJECT_ROOT / "checkpoints"
    save_dir.mkdir(exist_ok=True)
    best_ckpt_path = save_dir / "spatial_vae_ddsp_best.pt"
    
    start_epoch = 1
    best_val_loss = float("inf")
    ema = ModelEMA(encoder, decay=0.995)
    
    if args.resume and best_ckpt_path.exists():
        print(f"[*] Resuming from existing checkpoint: {best_ckpt_path}")
        try:
            ckpt_data = torch.load(best_ckpt_path, map_location=device)
            if "encoder" in ckpt_data:
                verify_checkpoint_integrity(ckpt_data["encoder"], "VAE Encoder")
                enc_sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in ckpt_data["encoder"].items()}
                encoder.load_state_dict(enc_sd, strict=False)
            if "encoder_ema" in ckpt_data:
                ema_sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in ckpt_data["encoder_ema"].items()}
                ema.load_state_dict(ema_sd)
            if "ddsp" in ckpt_data:
                verify_checkpoint_integrity(ckpt_data["ddsp"], "DDSP")
                ddsp_sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in ckpt_data["ddsp"].items()}
                ddsp.load_state_dict(ddsp_sd, strict=False)
            start_epoch = int(ckpt_data.get("epoch", 0)) + 1
            best_val_loss = float(ckpt_data.get("best_val_spectral", float("inf")))
            print(f"  [+] Resumed weights successfully! Resuming at Epoch {start_epoch:02d}")
        except Exception as e:
            print(f"[!] Warning: Checkpoint corrupted ({e}). Starting fresh.")

    if is_cuda and hasattr(torch, "compile"):
        print("[*] Applying torch.compile(mode='default', dynamic=True)...")
        encoder = torch.compile(encoder, mode="default", dynamic=True)
        ddsp = torch.compile(ddsp, mode="default", dynamic=True)
        if discriminator is not None:
            discriminator = torch.compile(discriminator, mode="default", dynamic=True)
    
    physics_criterion = MultiScaleAmbisonicPhysicsLoss().to(device)
    disentangle_criterion = BetaVAEDisentanglementLoss(beta=1.5).to(device)
    hierarchical_criterion = HierarchicalMultiResLoss().to(device)
    
    opt_g = torch.optim.AdamW(list(encoder.parameters()) + list(ddsp.parameters()), lr=args.lr, weight_decay=1e-4)
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=args.lr_d, betas=(0.5, 0.999)) if discriminator else None

    total_target_epochs = start_epoch + args.epochs - 1
    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=total_target_epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda" if is_cuda else "cpu", enabled=args.use_amp)
    tau_schedule = torch.linspace(1.0, 0.05, max(total_target_epochs, 1))
    global_step = (start_epoch - 1) * len(train_loader)
    
    for epoch in range(start_epoch, total_target_epochs + 1):
        tau = float(tau_schedule[min(epoch - 1, len(tau_schedule) - 1)])
        start = time.time()
        
        train_metrics = train_epoch(
            encoder=encoder, ddsp=ddsp, discriminator=discriminator, dataloader=train_loader,
            opt_g=opt_g, opt_d=opt_d, scaler=scaler, physics_criterion=physics_criterion,
            disentangle_criterion=disentangle_criterion, hierarchical_criterion=hierarchical_criterion,
            tau_quant=tau, global_step=global_step, accumulation_steps=args.accumulation_steps,
            use_amp=args.use_amp, device=device, max_batches=args.max_batches, ema=ema
        )
        global_step = int(train_metrics["global_step"])
        
        # Corrected order: optimizer.step() has completed inside train_epoch, now step scheduler_g safely
        if math.isfinite(train_metrics["loss"]) and train_metrics["loss"] > 0.0:
            scheduler_g.step()
        else:
            print(f"[!] Notice: Skipping scheduler step (Loss: {train_metrics['loss']})", flush=True)

        val_metrics = validate_epoch(encoder=encoder, ddsp=ddsp, dataloader=val_loader, physics_criterion=physics_criterion, device=device, max_batches=args.max_batches)

        print(f"Epoch {epoch:02d}/{total_target_epochs:02d} | Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['val_loss']:.4f}", flush=True)
        
        if val_metrics["val_spectral"] < best_val_loss:
            best_val_loss = val_metrics["val_spectral"]
            torch.save({
                "encoder": encoder.state_dict(),
                "encoder_ema": ema.state_dict(),
                "ddsp": ddsp.state_dict(),
                "epoch": epoch,
                "best_val_spectral": best_val_loss
            }, best_ckpt_path)

    torch.save({
        "encoder": encoder.state_dict(),
        "encoder_ema": ema.state_dict(),
        "ddsp": ddsp.state_dict(),
        "epochs_completed": total_target_epochs
    }, save_dir / "spatial_vae_ddsp_latest.pt")
    print("[+] Training complete and saved safely.")

if __name__ == "__main__":
    main()