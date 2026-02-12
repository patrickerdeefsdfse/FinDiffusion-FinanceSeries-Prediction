"""
FinDiffusion - Configuration
All hyperparameters and settings for the project.
"""
from dataclasses import dataclass, field
from typing import List
import torch
import os


@dataclass
class Config:
    # ======================== Data ========================
    # Data source: 'stooq' (default), 'yfinance', 'alphavantage', or 'local_csv'
    data_source: str = 'stooq'
    alpha_vantage_key: str = os.getenv('ALPHAVANTAGE_API_KEY', '')
    alpha_vantage_function: str = 'TIME_SERIES_INTRADAY'
    alpha_vantage_interval: str = '60min'
    alpha_vantage_outputsize: str = 'full'
    local_csv_path: str = ''

    # Broader, more diverse universe (US large caps + ETFs)
    tickers: List[str] = field(default_factory=lambda: [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
        'BRK-B', 'JPM', 'V', 'MA', 'UNH', 'XOM', 'JNJ', 'PG',
        'HD', 'DIS', 'BAC', 'PFE', 'CSCO', 'INTC', 'KO', 'PEP',
        'ABBV', 'AVGO', 'COST', 'WMT', 'CVX', 'MRK', 'NFLX',
        'CRM', 'ORCL', 'ADBE', 'NKE', 'MCD', 'LLY', 'QCOM',
        'TXN', 'ABT', 'AMD', 'CAT', 'IBM', 'GE', 'GS',
        'SPY', 'QQQ', 'IWM', 'TLT', 'GLD'
    ])
    start_date: str = '2010-01-01'
    end_date: str = 'auto'       # 'auto' -> today
    seq_len: int = 60            # ~3 months of trading days

    # Feature engineering
    # feature_mode: 'ohlcv', 'ohlcv_returns', 'returns'
    feature_mode: str = 'ohlcv_returns'
    return_type: str = 'log'     # 'log' or 'simple'
    return_col: str = 'Adj Close'
    log_volume: bool = True
    features: List[str] = field(default_factory=list)
    feature_dim: int = 0

    # Normalization: 'zscore', 'minmax', 'none', 'per_window' (not recommended)
    scaler: str = 'zscore'
    cache_dir: str = 'data_cache'

    # Chronological split to avoid leakage
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    shuffle_windows: bool = True

    # ======================== Model ========================
    d_model: int = 128           # Transformer hidden dimension
    n_heads: int = 4             # Number of attention heads
    n_layers: int = 4            # Number of transformer encoder layers
    d_ff: int = 512              # Feed-forward dimension
    dropout: float = 0.1
    trend_kernel: int = 5        # Moving average kernel for trend decomposition
    clip_range: float = 5.0      # Clamp range for sampling stability (None to disable)

    # ======================== Diffusion ========================
    n_diffusion_steps: int = 200
    beta_start: float = 1e-4
    beta_end: float = 0.02
    beta_schedule: str = 'cosine'   # 'linear' or 'cosine'

    # ======================== Training ========================
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-6
    epochs: int = 100
    vae_epochs: int = 50
    grad_clip: float = 1.0
    fft_weight: float = 0.1     # Weight of FFT loss term
    # FIDE-inspired frequency inflation
    fide_weight: float = 0.1
    fide_gamma: float = 1.5
    fide_scales: List[int] = field(default_factory=lambda: [1, 2, 4])
    patience: int = 30           # Early stopping patience
    log_interval: int = 10       # Print every N epochs
    seed: int = 42

    # ======================== Evaluation ========================
    n_gen_samples: int = 1000    # Number of samples to generate for evaluation
    disc_hidden: int = 64        # Discriminator LSTM hidden size
    disc_epochs: int = 50        # Discriminator training epochs
    pred_hidden: int = 64        # Predictor LSTM hidden size
    pred_epochs: int = 50        # Predictor training epochs
    obs_ratio: float = 0.5       # Conditional forecasting observed ratio
    forecast_samples: int = 200  # Samples used for forecast metrics
    forecast_mc_samples: int = 5 # Monte Carlo samples for forecast intervals
    forecast_interval: float = 0.9  # Prediction interval coverage (e.g., 0.9)

    # ======================== Paths ========================
    save_dir: str = 'checkpoints'
    result_dir: str = 'results'

    # ======================== Device ========================
    device: str = 'auto'        # 'auto', 'cuda', 'cpu', 'mps'

    def __post_init__(self):
        if self.device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
