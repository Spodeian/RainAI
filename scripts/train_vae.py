"""
Training script for Phase 2: Continuous Spatial VAE and HOA-DDSP Subtractive Synthesis Engine.
Includes Multi-Slice Residual QAT (S0..S4), Multi-Scale STFT Discriminator (GAN),
Instantaneous Phase & Physics Losses, Beta-VAE Disentanglement, EMA Shadow Tracking,
Automatic Mixed Precision (AMP), Gradient Accumulation, and Validation Monitoring.
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Optional, Dict
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import create_dataloader
from src.data.corruptions import AcousticCorruptionPipeline
from src.models.diff_autoencoder import SpatialAudioEncoder, HierarchicalMultiResLoss
from src.models.ddsp import ContinuousParametricFilter, DifferentiableReverbEngine
from src.dsp.physics_losses import (
    MultiScaleAmbisonicPhysicsLoss,
    MultiScaleSTFTDiscriminator,
    discriminator_hinge_loss,
    generator_adversarial_loss,
    feature_matching_loss,
    BetaVAEDisentanglementLoss
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
    corruption_pipeline = AcousticCorruptionPipeline(sample_rate=48000)
    
    opt_g.zero_grad()
    if opt_d:
        opt_d.zero_grad()
        
    for batch_idx, batch in enumerate(dataloader, 1):
        if max_batches and max_batches > 0 and batch_idx > max_batches:
            break
            
        audio = batch["audio"].to(device)         # (B, 4, samples)
        u = batch["conditioning"].to(device)     # (B, 554)
        noise = batch["noise"].to(device)         # (B, 1, samples)
        
        # Stochastic Slice Sampling (Biased 50% to Master FP32 Slice S4)
        if torch.rand(1).item() < 0.5:
            slice_level = 4
        else:
            slice_level = int(torch.randint(0, 4, (1,)).item())
            
        corrupted_audio = corruption_pipeline(audio)
        device_type = "cuda" if device.startswith("cuda") else "cpu"
        
        with torch.amp.autocast(device_type, enabled=use_amp):
            # 1. Encode into continuous latents
            z_q, z_align, quality_score = encoder(corrupted_audio, u, tau=tau_quant, active_level=slice_level)
            
            # 2. Differentiable DDSP reconstructs 4-channel FOA audio
            audio_rec, filter_params = ddsp(z_q, noise, ambisonic_order=1)
            
            # 3. Multi-Scale Ambisonic Physics Loss with dynamic step annealing
            physics_losses = physics_criterion(audio_rec, audio, current_step=global_step)
            acoustic_loss = physics_losses["total_loss"]
            
            # 4. Hierarchical Multi-Resolution Loss (16k, 32k, 48k)
            hier_losses = hierarchical_criterion(audio_rec, audio)
            hier_loss = hier_losses["hierarchical_loss"]
            
            # 5. Beta-VAE Latent Disentanglement & Mutual Information Penalty (FP32 stability)
            with torch.amp.autocast(device_type, enabled=False):
                disentangle = disentangle_criterion(z_align.float())
                dis_loss = torch.clamp(disentangle["disentangle_loss"], min=0.0, max=50.0)
            
            # 6. Sparsity and Material Dampening
            bit_reg = 0.001 * torch.mean(torch.clamp(encoder.quantizer.beta, min=0.0, max=8.0))
            sparsity_loss = 0.005 * ddsp.get_sparsity_loss(filter_params["gains"])
            dampening_loss = 0.01 * ddsp.get_material_dampening_loss(filter_params["q"], max_damped_q=8.0)
            
            g_adv_loss = torch.tensor(0.0, device=device)
            fm_loss = torch.tensor(0.0, device=device)
            real_fmaps_cached = None
            
            # 7. Adversarial Multi-Scale STFT Discriminator
            if discriminator and opt_d:
                fake_scores, fake_fmaps = discriminator(audio_rec)
                g_adv_loss = generator_adversarial_loss(fake_scores)
                # Compute real audio features once for both feature matching and discriminator training
                with torch.amp.autocast(device_type, enabled=use_amp):
                    real_scores_d, real_fmaps_cached = discriminator(audio.detach())
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
            
            if torch.isnan(raw_loss_g):
                raw_loss_g = acoustic_loss
                
            loss_g = raw_loss_g / accumulation_steps
            
        scaler.scale(loss_g).backward()
        
        # Train Discriminator
        d_loss_val = 0.0
        if discriminator and opt_d:
            with torch.amp.autocast(device_type, enabled=use_amp):
                fake_scores_d, _ = discriminator(audio_rec.detach())
                loss_d = discriminator_hinge_loss(real_scores_d, fake_scores_d) / accumulation_steps
                
            scaler.scale(loss_d).backward()
            d_loss_val = loss_d.item() * accumulation_steps
            
        # Accumulate gradients
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
        
        slice_names = ["S0_Ternary", "S1_INT4", "S2_INT8_BF16", "S3_INT16_FP16", "S4_FP32_Master"]
        if batch_idx % 2 == 0 or (max_batches and batch_idx == max_batches):
            print(
                f"    [Batch {batch_idx:02d} | Slice {slice_level} ({slice_names[slice_level]})] "
                f"Loss: {raw_g_loss:.4f} (Spec: {physics_losses['spectral_loss'].item():.3f}, "
                f"Phase: {physics_losses['phase_loss'].item():.3f}, "
                f"DoA: {physics_losses['doa_intensity_loss'].item():.3f})",
                flush=True
            )
            
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
def validate_epoch(
    encoder: nn.Module,
    ddsp: nn.Module,
    dataloader,
    physics_criterion: MultiScaleAmbisonicPhysicsLoss,
    device: str = "cpu",
    max_batches: Optional[int] = None
) -> Dict[str, float]:
    encoder.eval()
    ddsp.eval()
    
    total_loss = 0.0
    total_spectral = 0.0
    total_phase = 0.0
    total_doa = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader, 1):
        if max_batches and max_batches > 0 and batch_idx > max_batches:
            break
        audio = batch["audio"].to(device)
        u = batch["conditioning"].to(device)
        noise = batch["noise"].to(device)
        
        # Validate at full FP32 Master slice (k=4) without corruptions
        z_q, _, _ = encoder(audio, u, tau=0.05, active_level=4)
        audio_rec, _ = ddsp(z_q, noise, ambisonic_order=1)
        
        physics_losses = physics_criterion(audio_rec, audio, current_step=None)
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
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Mini-batch size")
    parser.add_argument("--accumulation-steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Generator learning rate")
    parser.add_argument("--lr-d", type=float, default=1e-4, help="Discriminator learning rate")
    parser.add_argument("--use-disc", action="store_true", default=False, help="Enable multi-scale STFT GAN discriminator")
    parser.add_argument("--use-amp", action="store_true", default=True, help="Enable automatic mixed precision")
    parser.add_argument("--max-batches", type=int, default=0, help="Max batches per epoch (0 for full dataset)")
    parser.add_argument("--device", type=str, default=None, help="Compute device (cuda/cpu)")
    parser.add_argument("--data-dir", type=str, default=None, help="Processed audio directory")
    parser.add_argument("--chunk-curriculum", action="store_true", default=True, help="Enable dynamic length bucketing and curriculum duration scheduling")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume training from spatial_vae_ddsp_best.pt if available")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Disable checkpoint resumption")
    args = parser.parse_args()

    print("=" * 75)
    print("RainAI Phase 2: Production Continuous Spatial VAE + HOA-DDSP Engine")
    print("Multi-Slice Residual QAT (S0..S4) | Jamba SSM | Phase & Physics Losses")
    if args.chunk_curriculum:
        print("[*] Multi-Scale Dynamic Chunk Curriculum & VAD Active Windowing: ACTIVE")
    print("=" * 75)
    
    proc_cand = PROJECT_ROOT / "data" / "processed"
    if not proc_cand.exists():
        proc_cand = PROJECT_ROOT / "Data" / "processed"
    processed_dir = Path(args.data_dir) if args.data_dir else proc_cand
    manifest_path = processed_dir / "manifest.json"
    
    if not processed_dir.exists() or not list(processed_dir.glob("*.flac")):
        print(f"[!] No processed audio in {processed_dir}. Run preprocessing first.")
        return

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    is_cuda = device.startswith("cuda")
    print(f"[*] Compute device: {device} | AMP: {args.use_amp} | Accumulation: {args.accumulation_steps}")
    
    # Train and Validation DataLoaders
    train_loader = create_dataloader(
        data_dir=processed_dir,
        manifest_path=manifest_path,
        batch_size=args.batch_size,
        split="train",
        val_ratio=0.15,
        augment_yaw=True,
        shuffle=True,
        chunk_curriculum=args.chunk_curriculum,
        pin_memory=is_cuda
    )
    val_loader = create_dataloader(
        data_dir=processed_dir,
        manifest_path=manifest_path,
        batch_size=args.batch_size,
        split="val",
        val_ratio=0.15,
        augment_yaw=False,
        shuffle=False,
        chunk_curriculum=False,
        pin_memory=is_cuda
    )
    print(f"[*] Loaded {len(train_loader.dataset)} train samples, {len(val_loader.dataset)} val samples.")

    encoder = SpatialAudioEncoder().to(device)
    ddsp = ContinuousParametricFilter(num_filters=16).to(device)
    discriminator = MultiScaleSTFTDiscriminator().to(device) if args.use_disc else None
    
    physics_criterion = MultiScaleAmbisonicPhysicsLoss().to(device)
    disentangle_criterion = BetaVAEDisentanglementLoss(beta=1.5).to(device)
    hierarchical_criterion = HierarchicalMultiResLoss().to(device)
    
    opt_g = torch.optim.AdamW(
        list(encoder.parameters()) + list(ddsp.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )
    opt_d = torch.optim.AdamW(
        discriminator.parameters(),
        lr=args.lr_d,
        betas=(0.5, 0.999)
    ) if discriminator else None
    
    save_dir = PROJECT_ROOT / "checkpoints"
    save_dir.mkdir(exist_ok=True)
    best_ckpt_path = save_dir / "spatial_vae_ddsp_best.pt"
    
    start_epoch = 1
    best_val_loss = float("inf")
    ema = ModelEMA(encoder, decay=0.995)
    
    if args.resume and best_ckpt_path.exists():
        print(f"[*] Resuming from existing checkpoint: {best_ckpt_path}")
        ckpt_data = torch.load(best_ckpt_path, map_location=device)
        if "encoder" in ckpt_data:
            encoder.load_state_dict(ckpt_data["encoder"])
        if "encoder_ema" in ckpt_data and ema is not None:
            ema.load_state_dict(ckpt_data["encoder_ema"])
        if "ddsp" in ckpt_data:
            ddsp.load_state_dict(ckpt_data["ddsp"])
        start_epoch = int(ckpt_data.get("epoch", 0)) + 1
        best_val_loss = float(ckpt_data.get("best_val_spectral", float("inf")))
        print(f"  [+] Resumed weights successfully! Resuming at Epoch {start_epoch:02d} (Best Val Spectral: {best_val_loss:.4f})")

    total_target_epochs = start_epoch + args.epochs - 1
    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=total_target_epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda" if is_cuda else "cpu", enabled=args.use_amp)
    
    tau_schedule = torch.linspace(1.0, 0.05, max(total_target_epochs, 1))
    global_step = (start_epoch - 1) * len(train_loader)
    
    print(f"[*] Starting training for {args.epochs} epochs (Epoch {start_epoch:02d} -> {total_target_epochs:02d})...")
    for epoch in range(start_epoch, total_target_epochs + 1):
        tau = float(tau_schedule[min(epoch - 1, len(tau_schedule) - 1)])
        start = time.time()
        
        # Schedule duration curriculum
        if hasattr(train_loader, "curriculum_collate") and train_loader.curriculum_collate is not None:
            train_loader.curriculum_collate.set_epoch(epoch)
        
        train_metrics = train_epoch(
            encoder=encoder,
            ddsp=ddsp,
            discriminator=discriminator,
            dataloader=train_loader,
            opt_g=opt_g,
            opt_d=opt_d,
            scaler=scaler,
            physics_criterion=physics_criterion,
            disentangle_criterion=disentangle_criterion,
            hierarchical_criterion=hierarchical_criterion,
            tau_quant=tau,
            global_step=global_step,
            accumulation_steps=args.accumulation_steps,
            use_amp=args.use_amp,
            device=device,
            max_batches=args.max_batches,
            ema=ema
        )
        global_step = int(train_metrics["global_step"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scheduler_g.step()
        
        # Validation Pass
        val_metrics = validate_epoch(
            encoder=encoder,
            ddsp=ddsp,
            dataloader=val_loader,
            physics_criterion=physics_criterion,
            device=device,
            max_batches=args.max_batches
        )
        elapsed = time.time() - start
        
        print(
            f"Epoch {epoch:02d}/{total_target_epochs:02d} | "
            f"Train Loss: {train_metrics['loss']:.4f} (Spec: {train_metrics['spectral_loss']:.3f}, Phase: {train_metrics['phase_loss']:.3f}) | "
            f"Val Loss: {val_metrics['val_loss']:.4f} (ValSpec: {val_metrics['val_spectral']:.3f}, ValPhase: {val_metrics['val_phase']:.3f}) | "
            f"LR: {scheduler_g.get_last_lr()[0]:.2e} | "
            f"Time: {elapsed:.2f}s",
            flush=True
        )
        
        # Save Best Checkpoint
        if val_metrics["val_spectral"] < best_val_loss:
            best_val_loss = val_metrics["val_spectral"]
            torch.save({
                "encoder": encoder.state_dict(),
                "encoder_ema": ema.state_dict(),
                "ddsp": ddsp.state_dict(),
                "epoch": epoch,
                "best_val_spectral": best_val_loss
            }, best_ckpt_path)
            print(f"  [*] Saved new best model checkpoint (Val Spec: {best_val_loss:.4f})")

    # Save Latest Checkpoint
    save_path = save_dir / "spatial_vae_ddsp_latest.pt"
    torch.save({
        "encoder": encoder.state_dict(),
        "encoder_ema": ema.state_dict(),
        "ddsp": ddsp.state_dict(),
        "learned_beta": encoder.quantizer.beta.detach().cpu(),
        "epochs_completed": total_target_epochs
    }, save_path)
    print(f"[+] Checkpoint saved to {save_path}")


if __name__ == "__main__":
    main()
