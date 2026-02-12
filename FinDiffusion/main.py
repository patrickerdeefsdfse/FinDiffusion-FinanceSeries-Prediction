"""
FinDiffusion - Main Entry Point

Transformer-based Diffusion Model for Synthetic Financial Time Series Generation.

Usage:
    python main.py                    # Full pipeline (train + evaluate)
    python main.py --mode train       # Train only
    python main.py --mode eval        # Evaluate only (requires trained model)
    python main.py --mode generate    # Generate samples only
    python main.py --mode ablation    # Ablation study
    python main.py --epochs 100       # Custom epochs
    python main.py --device cpu       # Force CPU
    python main.py --ddim             # Use DDIM fast sampling

References:
    [1] Diffusion-TS: Interpretable Diffusion for General Time Series Generation (ICLR 2024)
    [2] FIDE: Frequency-Inflated Conditional Diffusion Model (NeurIPS 2024)
    [3] Ho et al. Denoising Diffusion Probabilistic Models (NeurIPS 2020)
    [4] Song et al. Denoising Diffusion Implicit Models (ICLR 2021)
"""
import argparse
import json
import os
import sys
import numpy as np
import torch
import random

from config import Config
from data_loader import prepare_data
from model import FinDiffusion, LSTMGenerator, VAEGenerator
from train import train_findiffusion, train_lstm_baseline, train_vae_baseline
from evaluate import (run_full_evaluation, plot_training_curves,
                      plot_samples_comparison)


def parse_args():
    parser = argparse.ArgumentParser(description='FinDiffusion')
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'train', 'eval', 'generate', 'ablation'],
                        help='Run mode')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--n_gen', type=int, default=None,
                        help='Number of samples to generate')
    parser.add_argument('--ddim', action='store_true',
                        help='Use DDIM sampling (faster)')
    parser.add_argument('--ddim_steps', type=int, default=50)
    parser.add_argument('--no_baseline', action='store_true',
                        help='Skip baseline training')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: fewer epochs, smaller model for testing')
    # Data and reproducibility
    parser.add_argument('--data_source', type=str, default=None,
                        help='yfinance | stooq | alphavantage | local_csv')
    parser.add_argument('--local_csv', type=str, default=None,
                        help='Path to local CSV with Date and OHLCV')
    parser.add_argument('--alpha_vantage_function', type=str, default=None,
                        help='Alpha Vantage function, e.g. TIME_SERIES_INTRADAY')
    parser.add_argument('--alpha_vantage_interval', type=str, default=None,
                        help='Alpha Vantage intraday interval, e.g. 1min, 5min, 60min')
    parser.add_argument('--alpha_vantage_outputsize', type=str, default=None,
                        help='Alpha Vantage outputsize: compact or full')
    parser.add_argument('--feature_mode', type=str, default=None,
                        help='ohlcv | ohlcv_returns | returns')
    parser.add_argument('--scaler', type=str, default=None,
                        help='zscore | minmax | none | per_window')
    parser.add_argument('--start_date', type=str, default=None)
    parser.add_argument('--end_date', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--train_ratio', type=float, default=None)
    parser.add_argument('--val_ratio', type=float, default=None)
    parser.add_argument('--test_ratio', type=float, default=None)
    parser.add_argument('--fide_weight', type=float, default=None,
                        help='Weight for FIDE-inspired frequency inflation loss')
    parser.add_argument('--fide_gamma', type=float, default=None,
                        help='High-frequency emphasis exponent')
    parser.add_argument('--fide_scales', type=str, default=None,
                        help='Comma-separated scales, e.g. 1,2,4')
    parser.add_argument('--obs_ratio', type=float, default=None,
                        help='Observed ratio for conditional forecasting')
    parser.add_argument('--forecast_samples', type=int, default=None,
                        help='Number of samples for forecast metrics')
    parser.add_argument('--forecast_mc_samples', type=int, default=None,
                        help='Monte Carlo samples for forecast intervals')
    parser.add_argument('--forecast_interval', type=float, default=None,
                        help='Prediction interval coverage, e.g. 0.9')
    parser.add_argument('--vae_epochs', type=int, default=None,
                        help='Epochs for VAE baseline (independent of main epochs)')
    parser.add_argument('--only_vae', action='store_true',
                        help='Skip FinDiffusion/LSTM training and train only VAE baseline')
    return parser.parse_args()


def build_model(config: Config, **overrides) -> FinDiffusion:
    """Build FinDiffusion model with optional parameter overrides (for ablation)."""
    kwargs = dict(
        feature_dim=config.feature_dim,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        dropout=config.dropout,
        n_steps=config.n_diffusion_steps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        beta_schedule=config.beta_schedule,
        trend_kernel=config.trend_kernel,
        clip_range=config.clip_range,
    )
    kwargs.update(overrides)
    return FinDiffusion(**kwargs)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_samples(model, config, use_ddim=False, ddim_steps=50):
    """Generate synthetic samples from trained model."""
    device = torch.device(config.device)
    model = model.to(device)
    model.eval()

    print(f"\nGenerating {config.n_gen_samples} samples...")
    if use_ddim:
        print(f"  Using DDIM with {ddim_steps} steps")
        samples = model.sample_ddim(
            config.n_gen_samples, config.seq_len, device,
            n_ddim_steps=ddim_steps
        )
    else:
        print(f"  Using DDPM with {config.n_diffusion_steps} steps")
        batch_size = min(100, config.n_gen_samples)
        all_samples = []
        remaining = config.n_gen_samples
        while remaining > 0:
            n = min(batch_size, remaining)
            s = model.sample_ddpm(n, config.seq_len, device)
            all_samples.append(s.cpu())
            remaining -= n
            print(f"    Generated {config.n_gen_samples - remaining}/{config.n_gen_samples}")
        samples = torch.cat(all_samples, dim=0)

    samples = samples.cpu().numpy()
    if config.clip_range is not None:
        samples = np.clip(samples, -config.clip_range, config.clip_range)
    return samples


def _load_baseline(cls, path, config, device):
    """Helper to load a baseline model checkpoint."""
    if not os.path.exists(path):
        return None
    if cls == LSTMGenerator:
        m = cls(config.feature_dim, config.d_model, 2, config.dropout)
    else:
        m = cls(config.feature_dim, config.d_model, 32, 2, config.dropout)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    m.to(device)
    m.eval()
    return m


def _load_findiffusion(config, device):
    ckpt_path = os.path.join(config.save_dir, 'best_findiffusion.pt')
    if not os.path.exists(ckpt_path):
        return None
    model = build_model(config)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Loaded FinDiffusion from {ckpt_path}")
    return model


def _generate_baseline(model, config, device, name):
    """Generate samples from a baseline model."""
    try:
        with torch.no_grad():
            samples = model.sample(config.n_gen_samples, config.seq_len, device,
                                   clip_range=config.clip_range)
        samples = samples.cpu().numpy()
        if config.clip_range is not None:
            samples = np.clip(samples, -config.clip_range, config.clip_range)
        print(f"  Generated {name} samples shape: {samples.shape}")
        return samples
    except Exception as e:
        print(f"  {name} generation skipped: {e}")
        return None


# ======================================================================
#  Ablation Study
# ======================================================================

def run_ablation(config, train_loader, val_loader, test_loader, test_data, args):
    """
    Ablation study: train variants to measure contribution of each component.
      1. Full model (FinDiffusion)
      2. No FFT loss  (fft_weight=0)
      3. No decomposition (trend_kernel=1, degenerates to identity)
      4. Cosine schedule vs linear
    """
    from evaluate import discriminative_score, predictive_score

    ablation_dir = os.path.join(config.result_dir, 'ablation')
    os.makedirs(ablation_dir, exist_ok=True)

    full_beta = config.beta_schedule
    alt_beta = 'cosine' if full_beta != 'cosine' else 'linear'
    variants = {
        'Full Model': {'fft_weight': config.fft_weight, 'trend_kernel': config.trend_kernel,
                        'beta_schedule': full_beta},
        'No FFT Loss': {'fft_weight': 0.0, 'trend_kernel': config.trend_kernel,
                         'beta_schedule': full_beta},
        'No Decomposition': {'fft_weight': config.fft_weight, 'trend_kernel': 1,
                              'beta_schedule': full_beta},
        f'{alt_beta.capitalize()} Schedule': {'fft_weight': config.fft_weight,
                                              'trend_kernel': config.trend_kernel,
                                              'beta_schedule': alt_beta},
    }

    results = {}
    for name, params in variants.items():
        print(f"\n{'='*60}")
        print(f"  Ablation: {name}")
        print(f"{'='*60}")

        model = build_model(config,
                            trend_kernel=params['trend_kernel'],
                            beta_schedule=params['beta_schedule'])

        # Override fft_weight during training for the No FFT Loss variant
        original_fft = config.fft_weight
        config.fft_weight = params['fft_weight']

        history = train_findiffusion(
            model, train_loader, config, val_loader=val_loader
        )
        plot_training_curves(
            history,
            os.path.join(ablation_dir, f'loss_{name.lower().replace(" ", "_")}.png')
        )

        samples = generate_samples(model, config, use_ddim=args.ddim,
                                    ddim_steps=args.ddim_steps)
        config.fft_weight = original_fft

        d_score = discriminative_score(test_data, samples, device=config.device)
        p_score = predictive_score(test_data, samples, device=config.device)
        results[name] = {
            'discriminative_score': d_score,
            'predictive_score': p_score,
            'final_train_loss': history['train_loss'][-1],
        }
        print(f"  Disc: {d_score:.4f} | Pred: {p_score:.4f}")

    # Save ablation results
    with open(os.path.join(ablation_dir, 'ablation_results.json'), 'w') as f:
        json.dump({k: {kk: float(vv) for kk, vv in v.items()}
                   for k, v in results.items()}, f, indent=2)

    # Plot ablation comparison
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    names = list(results.keys())
    d_scores = [results[n]['discriminative_score'] for n in names]
    p_scores = [results[n]['predictive_score'] for n in names]

    colors = ['#FF5722', '#2196F3', '#4CAF50', '#9C27B0']
    axes[0].bar(names, d_scores, color=colors, alpha=0.8)
    axes[0].set_title('Discriminative Score (closer to 0.5 = better)')
    axes[0].axhline(y=0.5, color='gray', linestyle=':', alpha=0.7)
    axes[0].tick_params(axis='x', rotation=15, labelsize=8)

    axes[1].bar(names, p_scores, color=colors, alpha=0.8)
    axes[1].set_title('Predictive Score (lower = better)')
    axes[1].tick_params(axis='x', rotation=15, labelsize=8)

    plt.suptitle('Ablation Study Results', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(ablation_dir, 'ablation_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nAblation results saved to {ablation_dir}/")
    return results


# ======================================================================
#  Main
# ======================================================================

def main():
    args = parse_args()
    config = Config()

    # Override config with command-line args
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.lr = args.lr
    if args.device is not None:
        config.device = args.device
    if args.n_gen is not None:
        config.n_gen_samples = args.n_gen
    if args.data_source is not None:
        config.data_source = args.data_source
    if args.local_csv is not None:
        config.local_csv_path = args.local_csv
    if args.alpha_vantage_function is not None:
        config.alpha_vantage_function = args.alpha_vantage_function
    if args.alpha_vantage_interval is not None:
        config.alpha_vantage_interval = args.alpha_vantage_interval
    if args.alpha_vantage_outputsize is not None:
        config.alpha_vantage_outputsize = args.alpha_vantage_outputsize
    if args.feature_mode is not None:
        config.feature_mode = args.feature_mode
    if args.scaler is not None:
        config.scaler = args.scaler
    if args.start_date is not None:
        config.start_date = args.start_date
    if args.end_date is not None:
        config.end_date = args.end_date
    if args.seed is not None:
        config.seed = args.seed
    if args.train_ratio is not None:
        config.train_ratio = args.train_ratio
    if args.val_ratio is not None:
        config.val_ratio = args.val_ratio
    if args.test_ratio is not None:
        config.test_ratio = args.test_ratio
    if args.fide_weight is not None:
        config.fide_weight = args.fide_weight
    if args.fide_gamma is not None:
        config.fide_gamma = args.fide_gamma
    if args.fide_scales is not None:
        config.fide_scales = [int(x.strip()) for x in args.fide_scales.split(',') if x.strip()]
    if args.obs_ratio is not None:
        config.obs_ratio = args.obs_ratio
    if args.forecast_samples is not None:
        config.forecast_samples = args.forecast_samples
    if args.forecast_mc_samples is not None:
        config.forecast_mc_samples = args.forecast_mc_samples
    if args.forecast_interval is not None:
        config.forecast_interval = args.forecast_interval
    if args.vae_epochs is not None:
        config.vae_epochs = args.vae_epochs

    # Quick mode for testing
    if args.quick:
        config.epochs = 30
        config.n_diffusion_steps = 50
        config.d_model = 64
        config.n_layers = 2
        config.d_ff = 256
        config.n_gen_samples = 200
        config.patience = 15
        config.vae_epochs = min(getattr(config, 'vae_epochs', config.epochs), 10)

    # Reproducibility
    set_seed(config.seed)

    device = torch.device(config.device)

    print("=" * 60)
    print("  FinDiffusion: Transformer-based Diffusion Model")
    print("  for Financial Time Series Generation")
    print("=" * 60)
    print(f"  Mode   : {args.mode}")
    print(f"  Device : {config.device}")
    print(f"  Epochs : {config.epochs}")
    print()

    # ================================================================
    #  DATA
    # ================================================================
    train_loader, val_loader, test_loader, test_data, norm_info = prepare_data(config)

    # ================================================================
    #  ABLATION MODE
    # ================================================================
    if args.mode == 'ablation':
        run_ablation(config, train_loader, val_loader, test_loader, test_data, args)
        print("\nDone!")
        return

    # ================================================================
    #  TRAIN
    # ================================================================
    model = None
    lstm = None
    vae = None

    if args.mode in ('full', 'train'):
        if not args.only_vae:
            # --- FinDiffusion ---
            model = build_model(config)
            history = train_findiffusion(model, train_loader, config,
                                         val_loader=val_loader)
            plot_training_curves(
                history, os.path.join(config.result_dir, 'findiffusion_loss.png')
            )

            # --- Baselines ---
            if not args.no_baseline:
                # LSTM
                lstm = LSTMGenerator(
                    feature_dim=config.feature_dim,
                    hidden_dim=config.d_model,
                    n_layers=2, dropout=config.dropout
                )
                lstm_history = train_lstm_baseline(lstm, train_loader, config)
                plot_training_curves(
                    {'train_loss': lstm_history['train_loss'],
                     'mse_loss': lstm_history['train_loss'],
                     'fft_loss': [0] * len(lstm_history['train_loss']),
                     'val_loss': []},
                    os.path.join(config.result_dir, 'lstm_loss.png')
                )

                # VAE
                vae = VAEGenerator(
                    feature_dim=config.feature_dim,
                    hidden_dim=config.d_model,
                    latent_dim=32,
                    n_layers=2, dropout=config.dropout
                )
                vae_history = train_vae_baseline(vae, train_loader, config)
                plot_training_curves(
                    {'train_loss': vae_history['train_loss'],
                     'mse_loss': vae_history['mse_loss'],
                     'fft_loss': vae_history['kl_loss'],
                     'val_loss': []},
                    os.path.join(config.result_dir, 'vae_loss.png')
                )
        else:
            # Only train VAE; load existing FinDiffusion/LSTM checkpoints if available
            if args.no_baseline:
                print("WARNING: --only_vae with --no_baseline has no effect.")
            model = _load_findiffusion(config, device)
            if model is None:
                print("ERROR: FinDiffusion checkpoint not found. Train it once before using --only_vae.")
                sys.exit(1)
            if not args.no_baseline:
                lstm = _load_baseline(
                    LSTMGenerator, os.path.join(config.save_dir, 'best_lstm.pt'),
                    config, device
                )
            vae = VAEGenerator(
                feature_dim=config.feature_dim,
                hidden_dim=config.d_model,
                latent_dim=32,
                n_layers=2, dropout=config.dropout
            )
            vae_history = train_vae_baseline(vae, train_loader, config)
            plot_training_curves(
                {'train_loss': vae_history['train_loss'],
                 'mse_loss': vae_history['mse_loss'],
                 'fft_loss': vae_history['kl_loss'],
                 'val_loss': []},
                os.path.join(config.result_dir, 'vae_loss.png')
            )

    # ================================================================
    #  LOAD MODELS  (for eval / generate modes)
    # ================================================================
    if args.mode in ('eval', 'generate'):
        model = build_model(config)
        ckpt_path = os.path.join(config.save_dir, 'best_findiffusion.pt')
        if not os.path.exists(ckpt_path):
            print(f"ERROR: No checkpoint found at {ckpt_path}")
            print("  Run training first: python main.py --mode train")
            sys.exit(1)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)
        print(f"Loaded FinDiffusion from {ckpt_path}")

        if not args.no_baseline:
            lstm = _load_baseline(
                LSTMGenerator, os.path.join(config.save_dir, 'best_lstm.pt'),
                config, device
            )
            vae = _load_baseline(
                VAEGenerator, os.path.join(config.save_dir, 'best_vae.pt'),
                config, device
            )

    # ================================================================
    #  GENERATE
    # ================================================================
    if args.mode in ('full', 'eval', 'generate'):
        # FinDiffusion
        fake_findiff = generate_samples(model, config,
                                         use_ddim=args.ddim,
                                         ddim_steps=args.ddim_steps)
        np.save(os.path.join(config.result_dir, 'generated_findiffusion.npy'),
                fake_findiff)
        print(f"  Generated FinDiffusion samples shape: {fake_findiff.shape}")

        fake_dict = {'FinDiffusion': fake_findiff}

        # Baselines
        if not args.no_baseline:
            if lstm is not None:
                fake_lstm = _generate_baseline(lstm, config, device, 'LSTM')
                if fake_lstm is not None:
                    fake_dict['LSTM'] = fake_lstm

            if vae is not None:
                fake_vae = _generate_baseline(vae, config, device, 'VAE')
                if fake_vae is not None:
                    fake_dict['VAE'] = fake_vae

    # ================================================================
    #  EVALUATE
    # ================================================================
    if args.mode in ('full', 'eval'):
        results = run_full_evaluation(
            real_data=test_data,
            fake_dict=fake_dict,
            feature_names=config.features,
            result_dir=config.result_dir,
            model=model,
            device=config.device,
            norm_info=norm_info,
            obs_ratio=config.obs_ratio,
            forecast_samples=config.forecast_samples,
            forecast_mc_samples=config.forecast_mc_samples,
            forecast_interval=config.forecast_interval,
        )

        # Save numerical results
        results_serializable = {}
        for k, v in results.items():
            out = {}
            for kk, vv in v.items():
                if isinstance(vv, (int, float, np.floating)):
                    out[kk] = float(vv)
                else:
                    out[kk] = vv
            results_serializable[k] = out
        with open(os.path.join(config.result_dir, 'metrics.json'), 'w') as f:
            json.dump(results_serializable, f, indent=2)
        print(f"\nMetrics saved to {os.path.join(config.result_dir, 'metrics.json')}")

    # ================================================================
    #  GENERATE-ONLY mode: save and plot
    # ================================================================
    if args.mode == 'generate':
        plot_samples_comparison(
            test_data[:5], {'FinDiffusion': fake_findiff[:5]},
            config.features,
            os.path.join(config.result_dir, 'generated_preview.png'),
            n_show=5
        )
        print(f"\nGenerated samples saved to {config.result_dir}/")

    print("\nDone!")


if __name__ == '__main__':
    main()
