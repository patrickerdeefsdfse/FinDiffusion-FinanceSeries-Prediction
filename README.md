FinDiffusion
============

Overview
--------
FinDiffusion is a diffusion-based model for financial time series generation and
forecasting. It combines a Transformer backbone with seasonal-trend
decomposition and frequency-aware losses to produce realistic financial
sequences, and includes a conditional forecasting mode.

Key Features
------------
- DDPM/DDIM sampling with signal prediction parameterization.
- Seasonal-trend decoder for interpretability.
- Frequency-domain loss and FIDE-inspired multi-scale frequency inflation.
- Multi-source data loader (yfinance / stooq / Alpha Vantage / local CSV).
- Time-series safe preprocessing (chronological split, train-only scaling).
- Comprehensive evaluation: discriminative/predictive scores, ACF, PCA/t-SNE,
  stylized facts (VaR/ES), feature-correlation distance, and MMD.
- Conditional forecasting and forecast metrics (MAE/RMSE).
- Probabilistic forecast intervals via Monte Carlo conditional sampling
  (coverage/width + interval plots).

Project Structure
-----------------
- FinDiffusion/config.py: all hyperparameters and dataset settings.
- FinDiffusion/data_loader.py: data download, features, scaling, and windows.
- FinDiffusion/model.py: diffusion model + baselines + sampling.
- FinDiffusion/train.py: training loops and checkpoints.
- FinDiffusion/evaluate.py: metrics and visualization.
- FinDiffusion/main.py: CLI entry point.

Data Sources
------------
- stooq (default, direct CSV download; no extra packages)
- yfinance (optional; may be rate-limited)
- Alpha Vantage (daily or intraday; requires API key)
- local CSV (Date column required; optional Ticker column)

Notes on Stooq
--------------
- Some regions/networks may block Stooq or return HTML instead of CSV.
- The loader will try both stooq.pl and stooq.com and handle basic format issues.
- If it still fails, switch to yfinance or Alpha Vantage, or use local CSV.

Alpha Vantage key:
  Set ALPHAVANTAGE_API_KEY in your environment.

Quick Start
-----------
Install dependencies:
  pip install -r FinDiffusion/requirements.txt

Train + evaluate:
  python FinDiffusion/main.py --mode full

Forecasting (conditional generation metrics are computed during eval):
  python FinDiffusion/main.py --mode eval

Example: Alpha Vantage intraday
-------------------------------
  python FinDiffusion/main.py --mode full \
    --data_source alphavantage \
    --alpha_vantage_function TIME_SERIES_INTRADAY \
    --alpha_vantage_interval 5min \
    --alpha_vantage_outputsize full

Config Highlights
-----------------
- data_source: stooq | yfinance | alphavantage | local_csv
- feature_mode: ohlcv | ohlcv_returns | returns
- scaler: zscore | minmax | none
- epochs: default 100 (adjust in config.py or via --epochs)
- vae_epochs: default 50 (VAE baseline only; adjust in config.py or via --vae_epochs)
- obs_ratio: observed ratio for conditional forecasting
- forecast_samples: number of windows used for forecast metrics
- forecast_mc_samples: Monte Carlo samples for prediction intervals
- forecast_interval: target interval coverage (e.g., 0.9)
- fide_weight / fide_gamma / fide_scales: frequency inflation loss settings

Resume / Train Only VAE
-----------------------
If FinDiffusion and LSTM are already trained, you can skip them and train only VAE:
  python FinDiffusion/main.py --mode full --only_vae
Optionally set VAE epochs:
  python FinDiffusion/main.py --mode full --only_vae --vae_epochs 30

Outputs
-------
Generated samples and plots are saved to:
- checkpoints/
- results/
Key images include:
- samples_comparison.png
- distribution_comparison.png
- autocorrelation.png
- pca_tsne.png
- stylized_facts.png
- decomposition.png
- conditional_forecasting.png
- forecast_intervals.png

Notes
-----
- For time-series integrity, splitting is chronological and scalers are fit on
  the training set only.
- The FIDE component is a "FIDE-inspired" frequency-inflated loss (not a full
  reproduction of the original diffusion modulation).
