# FinDiffusion

**Transformer-based Denoising Diffusion Model with Interpretable Seasonal-Trend Decomposition for Financial Time Series Generation**

A course project for CDS521 Foundations of Artificial Intelligence.

## Overview

FinDiffusion generates realistic synthetic financial time series (stock OHLCV data) using a denoising diffusion probabilistic model (DDPM) with a Transformer backbone and an interpretable seasonal-trend decomposition decoder. The project also incorporates frequency-domain losses inspired by recent top-tier publications, and provides comprehensive evaluation against real market data.

### Key Technical Features

| Component | Technique | Reference |
|---|---|---|
| Generative framework | DDPM + DDIM fast sampling | Ho et al. (NeurIPS 2020), Song et al. (ICLR 2021) |
| Backbone network | Transformer encoder with Adaptive LayerNorm | Peebles & Xie (ICCV 2023) |
| Decoder | Interpretable seasonal-trend decomposition | Diffusion-TS (ICLR 2024) |
| Frequency loss | FFT amplitude + phase matching | Diffusion-TS (ICLR 2024) |
| High-freq emphasis | FIDE-inspired multi-scale frequency inflation | FIDE (NeurIPS 2024) |
| Conditional generation | Inpainting-style masking for forecasting/imputation | RePaint (CVPR 2022) |
| Signal prediction | Predict x_0 directly (not noise) | Diffusion-TS (ICLR 2024) |
| Baselines | LSTM autoregressive, LSTM-VAE | TimeGAN (NeurIPS 2019) |

## Architecture

```
Input x_0 ──► Forward Diffusion (add noise) ──► x_t
                                                  │
                                                  ▼
                              ┌─────────────────────────────────┐
                              │     Denoising Network           │
                              │                                 │
                              │  Linear Proj + Positional Enc   │
                              │  + Diffusion Step Embedding     │
                              │           │                     │
                              │     Transformer Encoder         │
                              │     (4 layers, AdaLN cond.)     │
                              │           │                     │
                              │    ┌──────┴──────┐              │
                              │    ▼             ▼              │
                              │  Trend        Seasonal          │
                              │  Decoder      Decoder           │
                              │  (AvgPool)   (CrossAttn on      │
                              │              de-trended residual)│
                              │    │             │              │
                              │    └──────┬──────┘              │
                              │           ▼                     │
                              │     x̂_0 = trend + seasonal     │
                              └─────────────────────────────────┘
                                                  │
Loss = MSE(x̂_0, x_0) + λ₁·FFT_loss + λ₂·FIDE_loss
```

## Project Structure

```
FinDiffusion/
├── config.py          # All hyperparameters and settings
├── data_loader.py     # Multi-source data download, feature engineering, scaling
├── model.py           # FinDiffusion model, LSTM baseline, VAE baseline
├── train.py           # Training loops with logging and early stopping
├── evaluate.py        # Evaluation metrics and visualization
├── main.py            # Entry point with CLI arguments
├── requirements.txt   # Python dependencies
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline (Train + Evaluate)

```bash
# Default: downloads 49 US stocks via yfinance, trains FinDiffusion + baselines, evaluates
python main.py

# Quick test mode (smaller model, fewer epochs)
python main.py --quick

# Use DDIM for faster sampling
python main.py --ddim --ddim_steps 50

# Specify device
python main.py --device cuda
```

### 3. Run Only Specific Stages

```bash
python main.py --mode train       # Train only
python main.py --mode eval        # Evaluate (requires trained checkpoint)
python main.py --mode generate    # Generate and save synthetic data
python main.py --mode ablation    # Run ablation study
```

### 4. Customize Data Source

```bash
# Use Stooq (no API key needed)
python main.py --data_source stooq

# Use local CSV file
python main.py --data_source local_csv --local_csv path/to/data.csv

# Use Alpha Vantage intraday data
export ALPHAVANTAGE_API_KEY=your_key
python main.py --data_source alphavantage
```

### 5. Customize Training

```bash
python main.py --epochs 200 --lr 2e-4 --batch_size 128
python main.py --scaler minmax --feature_mode returns
python main.py --fide_weight 0.2 --fide_gamma 2.0
```

## Data Pipeline

1. **Download**: 49 US large-cap stocks + ETFs (2010-present) via yfinance/stooq/Alpha Vantage, with local CSV fallback
2. **Feature engineering**: OHLCV + log returns + log volume change (`ohlcv_returns` mode, 7 features)
3. **Chronological split**: 70% train / 15% val / 15% test (no data leakage)
4. **Scaling**: z-score normalization fitted on train set only
5. **Windowing**: sliding windows of length 60 (approx. 3 months of trading days)

## Evaluation Metrics

| Metric | Description | Ideal Value |
|---|---|---|
| Discriminative Score | LSTM classifier accuracy on real vs. synthetic | 0.5 |
| Predictive Score (TSTR) | Train-on-synthetic, test-on-real next-step MAE | Lower is better |
| Feature Correlation Distance | Frobenius norm of correlation matrix difference | Lower is better |
| Excess Kurtosis | Fat-tail behavior of returns | Match real data |
| Volatility Autocorrelation | Volatility clustering at lag-1 | Match real data |
| VaR / ES (1%) | Tail risk metrics | Match real data |

### Generated Plots

- `samples_comparison.png` — Real vs. generated time series
- `distribution_comparison.png` — Marginal distribution histograms
- `autocorrelation.png` — ACF comparison
- `pca_tsne.png` — PCA and t-SNE embedding visualization
- `stylized_facts.png` — Financial stylized facts comparison
- `decomposition.png` — Interpretable trend-seasonal decomposition
- `conditional_forecasting.png` — Conditional generation (observe 50%, predict 50%)

## Ablation Study

Run `python main.py --mode ablation` to compare:

| Variant | Description |
|---|---|
| Full Model | All components enabled |
| No FFT Loss | fft_weight = 0 |
| No Decomposition | trend_kernel = 1 (identity) |
| Cosine Schedule | Cosine vs. linear beta schedule |

## References

1. Yuan & Qiao. *Diffusion-TS: Interpretable Diffusion for General Time Series Generation.* **ICLR 2024**
2. Ni et al. *FIDE: Frequency-Inflated Conditional Diffusion Model for Extreme-Aware Time Series Generation.* **NeurIPS 2024**
3. Ho et al. *Denoising Diffusion Probabilistic Models.* **NeurIPS 2020**
4. Song et al. *Denoising Diffusion Implicit Models.* **ICLR 2021**
5. Peebles & Xie. *Scalable Diffusion Models with Transformers (DiT).* **ICCV 2023**
6. Yoon et al. *Time-series Generative Adversarial Networks.* **NeurIPS 2019**
7. Lugmayr et al. *RePaint: Inpainting Using Denoising Diffusion Probabilistic Models.* **CVPR 2022**
8. Nichol & Dhariwal. *Improved Denoising Diffusion Probabilistic Models.* **ICML 2021**
