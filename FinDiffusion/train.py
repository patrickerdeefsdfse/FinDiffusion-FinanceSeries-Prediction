"""
FinDiffusion - Training
Training loop with logging, checkpointing, and early stopping.
Supports both FinDiffusion (diffusion model) and LSTM baseline.
"""
import os
import time
import torch
import numpy as np
from tqdm import tqdm
from typing import Optional

from config import Config
from model import FinDiffusion, LSTMGenerator, VAEGenerator


def train_findiffusion(model: FinDiffusion, train_loader, config: Config,
                       val_loader=None) -> dict:
    """
    Train the FinDiffusion model.

    Returns:
        history: dict with 'train_loss', 'mse_loss', 'fft_loss' lists
    """
    device = torch.device(config.device)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.lr * 0.01
    )

    history = {'train_loss': [], 'mse_loss': [], 'fft_loss': [], 'fide_loss': [], 'val_loss': []}
    best_loss = float('inf')
    patience_counter = 0

    print("\n" + "=" * 60)
    print("Training FinDiffusion")
    print(f"  Device     : {device}")
    print(f"  Parameters : {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Epochs     : {config.epochs}")
    print(f"  Batch size : {config.batch_size}")
    print(f"  LR         : {config.lr}")
    print("=" * 60)

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss, epoch_mse, epoch_fft, epoch_fide = 0.0, 0.0, 0.0, 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}",
                    leave=False, ncols=100)

        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()

            losses = model.compute_loss(
                batch,
                fft_weight=config.fft_weight,
                fide_weight=config.fide_weight,
                fide_gamma=config.fide_gamma,
                fide_scales=config.fide_scales,
            )
            losses['total'].backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            optimizer.step()

            epoch_loss += losses['total'].item()
            epoch_mse += losses['mse']
            epoch_fft += losses['fft']
            epoch_fide += losses.get('fide', 0.0)
            n_batches += 1

            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'mse': f"{losses['mse']:.4f}",
                'fft': f"{losses['fft']:.4f}",
                'fide': f"{losses.get('fide', 0.0):.4f}",
            })

        scheduler.step()

        # Average losses
        avg_loss = epoch_loss / n_batches
        avg_mse = epoch_mse / n_batches
        avg_fft = epoch_fft / n_batches
        avg_fide = epoch_fide / n_batches

        history['train_loss'].append(avg_loss)
        history['mse_loss'].append(avg_mse)
        history['fft_loss'].append(avg_fft)
        history['fide_loss'].append(avg_fide)

        # Validation
        val_loss = avg_loss  # Use train loss if no val_loader
        if val_loader is not None:
            model.eval()
            v_loss = 0.0
            v_n = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    losses = model.compute_loss(
                        batch,
                        fft_weight=config.fft_weight,
                        fide_weight=config.fide_weight,
                        fide_gamma=config.fide_gamma,
                        fide_scales=config.fide_scales,
                    )
                    v_loss += losses['total'].item()
                    v_n += 1
            val_loss = v_loss / v_n
        history['val_loss'].append(val_loss)

        # Logging
        if epoch % config.log_interval == 0 or epoch == 1:
            lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:3d} | Loss: {avg_loss:.5f} | "
                  f"MSE: {avg_mse:.5f} | FFT: {avg_fft:.5f} | "
                  f"FIDE: {history['fide_loss'][-1]:.5f} | "
                  f"Val: {val_loss:.5f} | LR: {lr:.2e}")

        # Checkpointing & early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            ckpt_path = os.path.join(config.save_dir, 'best_findiffusion.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': config,
            }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"\n  Early stopping at epoch {epoch} (patience={config.patience})")
                break

    # Load best model
    ckpt_path = os.path.join(config.save_dir, 'best_findiffusion.pt')
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\n  Loaded best model from epoch {checkpoint['epoch']} "
              f"(loss={checkpoint['loss']:.5f})")

    return history


def train_lstm_baseline(model: LSTMGenerator, train_loader, config: Config) -> dict:
    """Train the LSTM baseline generator."""
    device = torch.device(config.device)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr * 5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    history = {'train_loss': []}
    best_loss = float('inf')

    print("\n" + "=" * 60)
    print("Training LSTM Baseline")
    print(f"  Parameters : {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 60)

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            losses = model.compute_loss(batch)
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            epoch_loss += losses['total'].item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        history['train_loss'].append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(),
                       os.path.join(config.save_dir, 'best_lstm.pt'))

        if epoch % config.log_interval == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Loss: {avg_loss:.5f}")

    # Load best
    best_path = os.path.join(config.save_dir, 'best_lstm.pt')
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))

    return history


def train_vae_baseline(model: VAEGenerator, train_loader, config: Config) -> dict:
    """Train the VAE baseline generator."""
    device = torch.device(config.device)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr * 5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    history = {'train_loss': [], 'mse_loss': [], 'kl_loss': []}
    best_loss = float('inf')

    print("\n" + "=" * 60)
    vae_epochs = getattr(config, 'vae_epochs', config.epochs)
    print("Training VAE Baseline")
    print(f"  Parameters : {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Epochs     : {vae_epochs}")
    print("=" * 60)

    for epoch in range(1, vae_epochs + 1):
        model.train()
        epoch_loss, epoch_mse, epoch_kl = 0.0, 0.0, 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{vae_epochs}",
                    leave=False, ncols=100)
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()

            losses = model.compute_loss(batch)
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            epoch_loss += losses['total'].item()
            epoch_mse += losses['mse']
            epoch_kl += losses['kl']
            n_batches += 1

            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'mse': f"{losses['mse']:.4f}",
                'kl': f"{losses['kl']:.4f}",
            })

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        avg_mse = epoch_mse / n_batches
        avg_kl = epoch_kl / n_batches
        history['train_loss'].append(avg_loss)
        history['mse_loss'].append(avg_mse)
        history['kl_loss'].append(avg_kl)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(),
                       os.path.join(config.save_dir, 'best_vae.pt'))

        if epoch % config.log_interval == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Loss: {avg_loss:.5f} | "
                  f"MSE: {avg_mse:.5f} | KL: {avg_kl:.5f}")

    # Load best
    best_path = os.path.join(config.save_dir, 'best_vae.pt')
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))

    return history
