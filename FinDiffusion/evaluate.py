"""
FinDiffusion - Evaluation & Visualization
Comprehensive evaluation of generated financial time series:
  1. Discriminative Score  (classifier-based, from TimeGAN)
  2. Predictive Score      (train-on-synthetic test-on-real, from TimeGAN)
  3. Marginal Distribution comparison
  4. Temporal autocorrelation analysis
  5. PCA / t-SNE visualization
  6. Financial stylized facts (kurtosis, volatility clustering)
  7. Training loss curves
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Tuple
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import seaborn as sns

from data_loader import inverse_transform

sns.set_style('whitegrid')
COLORS = {'real': '#2196F3', 'findiffusion': '#FF5722', 'lstm': '#4CAF50', 'vae': '#9C27B0'}


# ======================================================================
#  1. Discriminative Score
# ======================================================================

class _Discriminator(nn.Module):
    """2-layer LSTM binary classifier for real vs. synthetic."""
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2,
                            batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def discriminative_score(real: np.ndarray, fake: np.ndarray,
                         hidden_dim: int = 64, epochs: int = 50,
                         device: str = 'cpu') -> float:
    """
    Train a classifier to distinguish real from synthetic.
    Returns accuracy (closer to 0.5 = better generator).
    """
    n = min(len(real), len(fake))
    real_sub = real[np.random.choice(len(real), n, replace=False)]
    fake_sub = fake[np.random.choice(len(fake), n, replace=False)]

    X = np.concatenate([real_sub, fake_sub], axis=0)
    y = np.concatenate([np.ones(n), np.zeros(n)])

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    # Split 80/20
    split = int(0.8 * len(X))
    X_train, X_test = torch.FloatTensor(X[:split]), torch.FloatTensor(X[split:])
    y_train, y_test = torch.FloatTensor(y[:split]), torch.FloatTensor(y[split:])

    train_ds = TensorDataset(X_train, y_train)
    loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    input_dim = X.shape[-1]
    model = _Discriminator(input_dim, hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb).squeeze(), yb).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(X_test.to(device)).squeeze()).cpu().numpy()
    acc = accuracy_score(y_test.numpy(), (preds > 0.5).astype(float))
    return acc


# ======================================================================
#  2. Predictive Score
# ======================================================================

class _Predictor(nn.Module):
    """Simple LSTM next-step predictor."""
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)


def predictive_score(real: np.ndarray, fake: np.ndarray,
                     hidden_dim: int = 64, epochs: int = 50,
                     device: str = 'cpu') -> float:
    """
    Train-on-Synthetic, Test-on-Real (TSTR).
    Train next-step predictor on fake data, evaluate MAE on real data.
    Lower = better.
    """
    input_dim = real.shape[-1]
    model = _Predictor(input_dim, hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()

    # Train on synthetic
    X_syn = torch.FloatTensor(fake[:, :-1, :])
    y_syn = torch.FloatTensor(fake[:, 1:, :])
    train_ds = TensorDataset(X_syn, y_syn)
    loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

    # Evaluate on real
    model.eval()
    X_real = torch.FloatTensor(real[:, :-1, :]).to(device)
    y_real = torch.FloatTensor(real[:, 1:, :]).to(device)
    with torch.no_grad():
        pred = model(X_real)
        mae = loss_fn(pred, y_real).item()
    return mae


# ======================================================================
#  3-7. Visualization Functions
# ======================================================================

def plot_training_curves(history: dict, save_path: str):
    """Plot training loss curves."""
    has_fide = 'fide_loss' in history and any(history.get('fide_loss', []))
    n_plots = 4 if has_fide else 3
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))

    axes[0].plot(history['train_loss'], color='#333', linewidth=1.5)
    if history.get('val_loss'):
        axes[0].plot(history['val_loss'], color='red', linewidth=1.5, alpha=0.7, label='Val')
        axes[0].legend()
    axes[0].set_title('Total Loss')
    axes[0].set_xlabel('Epoch')

    if 'mse_loss' in history:
        axes[1].plot(history['mse_loss'], color='#2196F3', linewidth=1.5)
        axes[1].set_title('MSE Loss (Reconstruction)')
        axes[1].set_xlabel('Epoch')

    if 'fft_loss' in history:
        axes[2].plot(history['fft_loss'], color='#FF5722', linewidth=1.5)
        axes[2].set_title('FFT Loss (Frequency)')
        axes[2].set_xlabel('Epoch')

    if has_fide:
        axes[3].plot(history['fide_loss'], color='#3F51B5', linewidth=1.5)
        axes[3].set_title('FIDE Loss (High-Freq)')
        axes[3].set_xlabel('Epoch')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_samples_comparison(real: np.ndarray, fake_dict: dict,
                            feature_names: list, save_path: str, n_show: int = 5):
    """Plot real vs. generated time series samples side by side."""
    n_features = real.shape[-1]
    n_methods = len(fake_dict) + 1  # +1 for real
    fig, axes = plt.subplots(n_show, n_features, figsize=(n_features * 3.5, n_show * 2))
    if n_show == 1:
        axes = axes[np.newaxis, :]

    for row in range(n_show):
        for col in range(n_features):
            ax = axes[row, col]
            # Real
            ax.plot(real[row, :, col], color=COLORS['real'],
                    linewidth=1.5, alpha=0.9, label='Real')
            # Generated
            for name, fake in fake_dict.items():
                color = COLORS.get(name, '#999')
                ax.plot(fake[row, :, col], color=color,
                        linewidth=1.2, alpha=0.7, linestyle='--', label=name)

            if row == 0:
                ax.set_title(feature_names[col], fontsize=10)
            if col == 0:
                ax.set_ylabel(f'Sample {row+1}', fontsize=9)
            if row == 0 and col == n_features - 1:
                ax.legend(fontsize=7, loc='upper right')
            ax.tick_params(labelsize=7)

    plt.suptitle('Real vs. Generated Time Series', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_distribution_comparison(real: np.ndarray, fake_dict: dict,
                                  feature_names: list, save_path: str):
    """Compare marginal distributions of each feature."""
    n_features = real.shape[-1]
    fig, axes = plt.subplots(1, n_features, figsize=(n_features * 3, 3.5))
    if n_features == 1:
        axes = [axes]

    for col in range(n_features):
        ax = axes[col]
        vals_real = real[:, :, col].flatten()
        ax.hist(vals_real, bins=50, density=True, alpha=0.5,
                color=COLORS['real'], label='Real')
        for name, fake in fake_dict.items():
            vals_fake = fake[:, :, col].flatten()
            ax.hist(vals_fake, bins=50, density=True, alpha=0.4,
                    color=COLORS.get(name, '#999'), label=name)
        ax.set_title(feature_names[col], fontsize=10)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    plt.suptitle('Marginal Distribution Comparison', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_autocorrelation(real: np.ndarray, fake_dict: dict,
                          feature_idx: int, save_path: str, max_lag: int = 20):
    """Compare autocorrelation functions."""
    fig, ax = plt.subplots(figsize=(8, 4))

    def compute_acf(data, max_lag):
        """Compute mean autocorrelation across all samples for one feature."""
        acfs = []
        for i in range(min(500, len(data))):
            series = data[i, :, feature_idx]
            n = len(series)
            mean = np.mean(series)
            var = np.var(series)
            if var < 1e-10:
                continue
            acf = []
            for lag in range(max_lag + 1):
                c = np.mean((series[:n-lag] - mean) * (series[lag:] - mean)) / var
                acf.append(c)
            acfs.append(acf)
        return np.mean(acfs, axis=0) if acfs else np.zeros(max_lag + 1)

    lags = np.arange(max_lag + 1)
    if feature_idx < 0 or feature_idx >= real.shape[-1]:
        feature_idx = 0
    acf_real = compute_acf(real, max_lag)
    ax.plot(lags, acf_real, 'o-', color=COLORS['real'], linewidth=2, label='Real', markersize=4)

    for name, fake in fake_dict.items():
        acf_fake = compute_acf(fake, max_lag)
        ax.plot(lags, acf_fake, 's--', color=COLORS.get(name, '#999'),
                linewidth=1.5, label=name, markersize=3, alpha=0.8)

    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Autocorrelation Function Comparison')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def _mmd_rbf(real: np.ndarray, fake: np.ndarray, max_samples: int = 300,
             gamma: Optional[float] = None, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    n = min(max_samples, len(real), len(fake))
    if n < 2:
        return 0.0
    idx_r = rng.choice(len(real), n, replace=False)
    idx_f = rng.choice(len(fake), n, replace=False)
    X = real[idx_r].reshape(n, -1)
    Y = fake[idx_f].reshape(n, -1)

    if gamma is None:
        gamma = 1.0 / max(1, X.shape[1])

    def _cdist2(A, B):
        A_norm = np.sum(A ** 2, axis=1)[:, None]
        B_norm = np.sum(B ** 2, axis=1)[None, :]
        return A_norm + B_norm - 2 * A @ B.T

    Kxx = np.exp(-gamma * _cdist2(X, X))
    Kyy = np.exp(-gamma * _cdist2(Y, Y))
    Kxy = np.exp(-gamma * _cdist2(X, Y))
    return float(Kxx.mean() + Kyy.mean() - 2 * Kxy.mean())


def plot_pca_tsne(real: np.ndarray, fake_dict: dict, save_path: str):
    """PCA and t-SNE 2D projection of real vs. generated."""
    n_samples = min(500, len(real))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Flatten windows for dimensionality reduction
    real_flat = real[:n_samples].reshape(n_samples, -1)
    all_labels = ['Real'] * n_samples
    all_data = [real_flat]

    for name, fake in fake_dict.items():
        n = min(n_samples, len(fake))
        fake_flat = fake[:n].reshape(n, -1)
        all_data.append(fake_flat)
        all_labels.extend([name] * n)

    combined = np.concatenate(all_data, axis=0)

    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(combined)

    offset = 0
    axes[0].scatter(pca_result[:n_samples, 0], pca_result[:n_samples, 1],
                    c=COLORS['real'], alpha=0.4, s=10, label='Real')
    offset = n_samples
    for name in fake_dict:
        n = min(n_samples, len(fake_dict[name]))
        axes[0].scatter(pca_result[offset:offset+n, 0], pca_result[offset:offset+n, 1],
                        c=COLORS.get(name, '#999'), alpha=0.4, s=10, label=name)
        offset += n
    axes[0].set_title('PCA Projection')
    axes[0].legend(fontsize=8)

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    tsne_result = tsne.fit_transform(combined)

    offset = 0
    axes[1].scatter(tsne_result[:n_samples, 0], tsne_result[:n_samples, 1],
                    c=COLORS['real'], alpha=0.4, s=10, label='Real')
    offset = n_samples
    for name in fake_dict:
        n = min(n_samples, len(fake_dict[name]))
        axes[1].scatter(tsne_result[offset:offset+n, 0], tsne_result[offset:offset+n, 1],
                        c=COLORS.get(name, '#999'), alpha=0.4, s=10, label=name)
        offset += n
    axes[1].set_title('t-SNE Projection')
    axes[1].legend(fontsize=8)

    plt.suptitle('Embedding Space Visualization', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def _select_return_feature(feature_names: list) -> Tuple[int, str, bool]:
    candidates = ['LogReturn', 'Return', 'log_return', 'ret']
    for name in candidates:
        if name in feature_names:
            return feature_names.index(name), name, True
    # fallback to price-like series
    for name in ['Close', 'Adj Close', 'Open', 'High', 'Low']:
        if name in feature_names:
            return feature_names.index(name), name, False
    return 0, feature_names[0], False


def _pick_feature_idx(feature_names: Optional[list], candidates: list,
                      fallback: int = 0) -> Tuple[int, str]:
    if feature_names:
        for name in candidates:
            if name in feature_names:
                return feature_names.index(name), name
        idx = min(max(0, fallback), len(feature_names) - 1)
        return idx, feature_names[idx]
    return fallback, f"feat_{fallback}"


def _extract_returns(data: np.ndarray, feature_idx: int, is_return_feature: bool) -> np.ndarray:
    series = data[:, :, feature_idx]
    if is_return_feature:
        returns = series
    else:
        # Use log-diff when prices are positive; fallback to simple diff
        with np.errstate(divide='ignore', invalid='ignore'):
            log_series = np.log(np.clip(series, 1e-12, None))
        returns = np.diff(log_series, axis=1)
    flat = returns.reshape(-1)
    return flat[np.isfinite(flat)]


def compute_stylized_facts(data: np.ndarray, feature_idx: int, is_return_feature: bool) -> dict:
    """
    Compute financial stylized facts for the given feature.
    Uses returns directly if feature is already returns; otherwise log-diff prices.
    """
    flat_returns = _extract_returns(data, feature_idx, is_return_feature)
    if len(flat_returns) == 0:
        return {'kurtosis': 0.0, 'skewness': 0.0, 'vol_autocorr': 0.0,
                'var_1': 0.0, 'es_1': 0.0}

    kurtosis = stats.kurtosis(flat_returns, fisher=True)
    skewness = stats.skew(flat_returns)

    # Volatility clustering: autocorrelation of |returns| at lag 1
    abs_returns = np.abs(flat_returns)
    if len(abs_returns) > 10:
        vol_autocorr = np.corrcoef(abs_returns[:-1], abs_returns[1:])[0, 1]
    else:
        vol_autocorr = 0.0

    # Tail risk metrics (1% VaR & Expected Shortfall)
    var_1 = np.quantile(flat_returns, 0.01)
    es_1 = flat_returns[flat_returns <= var_1].mean() if np.any(flat_returns <= var_1) else var_1

    return {
        'kurtosis': kurtosis,
        'skewness': skewness,
        'vol_autocorr': vol_autocorr,
        'var_1': var_1,
        'es_1': es_1,
    }


def _feature_corr_distance(real: np.ndarray, fake: np.ndarray) -> float:
    real_flat = real.reshape(-1, real.shape[-1])
    fake_flat = fake.reshape(-1, fake.shape[-1])
    corr_real = np.corrcoef(real_flat.T)
    corr_fake = np.corrcoef(fake_flat.T)
    corr_real = np.nan_to_num(corr_real, nan=0.0)
    corr_fake = np.nan_to_num(corr_fake, nan=0.0)
    return float(np.linalg.norm(corr_real - corr_fake, ord='fro') / real.shape[-1])


def plot_stylized_facts(real: np.ndarray, fake_dict: dict, save_path: str,
                        feature_idx: int, is_return_feature: bool):
    """Bar chart comparing financial stylized facts."""
    real_facts = compute_stylized_facts(real, feature_idx, is_return_feature)
    all_facts = {'Real': real_facts}
    for name, fake in fake_dict.items():
        all_facts[name] = compute_stylized_facts(fake, feature_idx, is_return_feature)

    metrics = ['kurtosis', 'skewness', 'vol_autocorr', 'var_1', 'es_1']
    metric_labels = [
        'Excess Kurtosis\n(fat tails)',
        'Skewness',
        'Volatility Autocorr.\n(clustering)',
        'VaR 1%',
        'ES 1%'
    ]

    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    x_labels = list(all_facts.keys())
    colors = [COLORS.get(k.lower(), '#999') for k in x_labels]
    colors[0] = COLORS['real']

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [all_facts[k][metric] for k in x_labels]
        bars = axes[i].bar(x_labels, values, color=colors, alpha=0.8, edgecolor='white')
        axes[i].set_title(label, fontsize=10)
        axes[i].tick_params(labelsize=8)
        for bar, val in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                         f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    plt.suptitle('Financial Stylized Facts Comparison', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_trend_seasonal_decomposition(model, real_sample: np.ndarray,
                                       device: str, save_path: str,
                                       feature_names: Optional[list] = None):
    """Visualize the interpretable trend-seasonal decomposition."""
    model.eval()
    x = torch.FloatTensor(real_sample[:4]).to(device)  # 4 samples

    # Add noise at t=50 and predict
    t = torch.full((4,), 50, device=torch.device(device), dtype=torch.long)
    x_noised = model.q_sample(x, t)

    with torch.no_grad():
        x_pred, trend, seasonal = model.net(x_noised, t)

    x_pred = x_pred.cpu().numpy()
    trend = trend.cpu().numpy()
    seasonal = seasonal.cpu().numpy()
    x_orig = real_sample[:4]

    fig, axes = plt.subplots(4, 4, figsize=(16, 10))
    titles = ['Original', 'Reconstructed', 'Trend', 'Seasonal']
    feat_idx, feat_name = _pick_feature_idx(feature_names, ['Close', 'Adj Close'], fallback=0)

    for row in range(4):
        axes[row, 0].plot(x_orig[row, :, feat_idx], color='#333', linewidth=1.5)
        axes[row, 1].plot(x_pred[row, :, feat_idx], color='#FF5722', linewidth=1.5)
        axes[row, 2].plot(trend[row, :, feat_idx], color='#2196F3', linewidth=1.5)
        axes[row, 3].plot(seasonal[row, :, feat_idx], color='#4CAF50', linewidth=1.5)
        axes[row, 3].axhline(y=0, color='gray', linestyle=':', alpha=0.5)

        if row == 0:
            for col, title in enumerate(titles):
                axes[0, col].set_title(title, fontsize=11)
        axes[row, 0].set_ylabel(f'Sample {row+1}', fontsize=9)

    plt.suptitle(f'Interpretable Seasonal-Trend Decomposition ({feat_name})',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ======================================================================
#  Conditional Forecasting Visualization
# ======================================================================

def plot_conditional_forecasting(model, real_data: np.ndarray, device: str,
                                  save_path: str, n_show: int = 4,
                                  obs_ratio: float = 0.5,
                                  feature_names: Optional[list] = None):
    """
    Visualize conditional generation (forecasting):
    Given the first obs_ratio portion, generate the rest.
    """
    model.eval()
    n_show = min(n_show, len(real_data))
    x_full = torch.FloatTensor(real_data[:n_show]).to(device)
    B, L, C = x_full.shape
    obs_len = int(L * obs_ratio)

    # Build mask: 1 for observed, 0 for to-generate
    mask = torch.zeros(B, L, C, device=torch.device(device))
    mask[:, :obs_len, :] = 1.0

    with torch.no_grad():
        x_gen = model.conditional_sample(x_full, mask, torch.device(device))
    x_gen = x_gen.cpu().numpy()
    x_orig = real_data[:n_show]

    fig, axes = plt.subplots(n_show, 2, figsize=(12, n_show * 2.5))
    if n_show == 1:
        axes = axes[np.newaxis, :]

    close_idx, close_name = _pick_feature_idx(feature_names, ['Close', 'Adj Close'], fallback=0)
    open_idx, open_name = _pick_feature_idx(feature_names, ['Open'], fallback=0)

    for row in range(n_show):
        for col, (feat_idx, feat_name) in enumerate([(close_idx, close_name),
                                                     (open_idx, open_name)]):
            ax = axes[row, col]
            time_axis = np.arange(L)
            ax.plot(time_axis, x_orig[row, :, feat_idx], color=COLORS['real'],
                    linewidth=2, label='Ground Truth')
            ax.plot(time_axis[obs_len:], x_gen[row, obs_len:, feat_idx],
                    color=COLORS['findiffusion'], linewidth=2,
                    linestyle='--', label='Generated')
            ax.axvline(x=obs_len - 0.5, color='gray', linestyle=':', alpha=0.7)
            if row == 0:
                ax.set_title(f'{feat_name} Price', fontsize=11)
                ax.legend(fontsize=8)
            ax.set_ylabel(f'Sample {row+1}', fontsize=9)
            ax.tick_params(labelsize=7)

    plt.suptitle(f'Conditional Forecasting (observe first {obs_ratio:.0%}, predict rest)',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def _forecast_mc_bundle(model, real_data: np.ndarray, device: str,
                        obs_ratio: float, norm_info: Optional[dict],
                        max_samples: int = 200, mc_samples: int = 5) -> dict:
    if len(real_data) == 0:
        return {}
    n = min(max_samples, len(real_data))
    x = torch.FloatTensor(real_data[:n]).to(device)
    B, L, C = x.shape
    obs_len = max(1, int(L * obs_ratio))
    if obs_len >= L:
        return {}

    mask = torch.zeros(B, L, C, device=torch.device(device))
    mask[:, :obs_len, :] = 1.0

    mc_samples = max(1, int(mc_samples))
    preds = []
    with torch.no_grad():
        for _ in range(mc_samples):
            x_gen = model.conditional_sample(x, mask, torch.device(device))
            preds.append(x_gen.cpu().numpy())
    preds = np.stack(preds, axis=0)  # [S, B, L, C]
    x_true = x.cpu().numpy()

    can_inverse = norm_info and norm_info.get('scaler') in ('zscore', 'minmax')
    if can_inverse:
        preds_shape = preds.shape
        preds_flat = preds.reshape(preds_shape[0] * preds_shape[1], preds_shape[2], preds_shape[3])
        preds_flat = inverse_transform(preds_flat, norm_info)
        preds = preds_flat.reshape(preds_shape)
        x_true = inverse_transform(x_true, norm_info)

    return {
        'preds': preds,
        'target': x_true,
        'obs_len': obs_len,
    }


def _forecast_metrics(model, real_data: np.ndarray, device: str,
                      obs_ratio: float, feature_names: Optional[list],
                      norm_info: Optional[dict], max_samples: int = 200,
                      mc_samples: int = 5, interval: float = 0.9,
                      bundle: Optional[dict] = None) -> dict:
    if bundle is None:
        bundle = _forecast_mc_bundle(
            model, real_data, device, obs_ratio, norm_info,
            max_samples=max_samples, mc_samples=mc_samples
        )
    if not bundle:
        return {}

    preds = bundle['preds']
    target_full = bundle['target']
    obs_len = bundle['obs_len']

    pred = preds[:, :, obs_len:, :]
    target = target_full[:, obs_len:, :]
    mean_pred = pred.mean(axis=0)
    err = mean_pred - target
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    feat_idx, feat_name = _pick_feature_idx(
        feature_names, ['Close', 'Adj Close', 'LogReturn'], fallback=0
    )
    feat_err = mean_pred[:, :, feat_idx] - target[:, :, feat_idx]
    feat_mae = float(np.mean(np.abs(feat_err)))
    feat_rmse = float(np.sqrt(np.mean(feat_err ** 2)))

    metrics = {
        'obs_ratio': obs_ratio,
        'forecast_mae': mae,
        'forecast_rmse': rmse,
        'forecast_feature': feat_name,
        'forecast_mae_feature': feat_mae,
        'forecast_rmse_feature': feat_rmse,
    }

    if preds.shape[0] > 1:
        alpha = (1.0 - interval) / 2.0
        lower = np.quantile(pred, alpha, axis=0)
        upper = np.quantile(pred, 1.0 - alpha, axis=0)
        covered = (target >= lower) & (target <= upper)
        picp = float(covered.mean())
        pinaw = float((upper - lower).mean())

        feat_lower = lower[:, :, feat_idx]
        feat_upper = upper[:, :, feat_idx]
        feat_target = target[:, :, feat_idx]
        feat_covered = (feat_target >= feat_lower) & (feat_target <= feat_upper)
        feat_picp = float(feat_covered.mean())
        feat_pinaw = float((feat_upper - feat_lower).mean())

        metrics.update({
            'forecast_interval': float(interval),
            'forecast_picp': picp,
            'forecast_pinaw': pinaw,
            'forecast_picp_feature': feat_picp,
            'forecast_pinaw_feature': feat_pinaw,
        })

    return metrics


def plot_forecast_intervals(bundle: dict, feature_names: Optional[list],
                            save_path: str, interval: float = 0.9,
                            n_show: int = 4):
    """Plot forecast mean and prediction interval for a key feature."""
    if not bundle:
        return
    preds = bundle['preds']
    target = bundle['target']
    obs_len = bundle['obs_len']

    n_show = min(n_show, target.shape[0])
    feat_idx, feat_name = _pick_feature_idx(
        feature_names, ['Close', 'Adj Close', 'LogReturn'], fallback=0
    )

    alpha = (1.0 - interval) / 2.0
    pred = preds[:, :, obs_len:, :]
    mean_pred = pred.mean(axis=0)
    lower = np.quantile(pred, alpha, axis=0)
    upper = np.quantile(pred, 1.0 - alpha, axis=0)

    fig, axes = plt.subplots(n_show, 1, figsize=(10, n_show * 2.5))
    if n_show == 1:
        axes = [axes]

    for i in range(n_show):
        ax = axes[i]
        full_time = np.arange(target.shape[1])
        ax.plot(full_time, target[i, :, feat_idx], color=COLORS['real'],
                linewidth=1.5, label='Ground Truth')

        forecast_time = np.arange(obs_len, target.shape[1])
        ax.plot(forecast_time, mean_pred[i, :, feat_idx],
                color=COLORS['findiffusion'], linewidth=1.5, label='Forecast Mean')
        ax.fill_between(forecast_time,
                        lower[i, :, feat_idx],
                        upper[i, :, feat_idx],
                        color=COLORS['findiffusion'], alpha=0.2,
                        label=f'{int(interval*100)}% PI' if i == 0 else None)
        ax.axvline(x=obs_len - 0.5, color='gray', linestyle=':', alpha=0.7)
        ax.set_ylabel(f'Sample {i+1}', fontsize=9)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.set_title(f'Forecast Interval ({feat_name})', fontsize=11)
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ======================================================================
#  Full Evaluation Pipeline
# ======================================================================

def run_full_evaluation(real_data: np.ndarray, fake_dict: dict,
                        feature_names: list, result_dir: str,
                        model=None, device: str = 'cpu',
                        norm_info: Optional[dict] = None,
                        obs_ratio: float = 0.5,
                        forecast_samples: int = 200,
                        forecast_mc_samples: int = 5,
                        forecast_interval: float = 0.9):
    """
    Run all evaluation metrics and generate all plots.

    Args:
        real_data:     [N, L, C]  real test data
        fake_dict:     {'FinDiffusion': array, 'LSTM': array, ...}
        feature_names: list of feature names
        result_dir:    directory to save plots
        model:         trained FinDiffusion model (for decomposition plot)
        device:        torch device string
    """
    os.makedirs(result_dir, exist_ok=True)
    print("\n" + "=" * 60)
    print("Running Full Evaluation")
    print("=" * 60)

    # ---- Quantitative Metrics ----
    print("\n--- Quantitative Metrics ---")
    feature_idx, feature_name, is_return_feature = _select_return_feature(feature_names)
    print(f"  Stylized facts feature: {feature_name} "
          f"({'returns' if is_return_feature else 'price log-diff'})")

    # Inverse transform for financial metrics if possible
    can_inverse = norm_info and norm_info.get('scaler') in ('zscore', 'minmax')
    raw_real = inverse_transform(real_data, norm_info) if can_inverse else real_data
    raw_fake = {k: inverse_transform(v, norm_info) for k, v in fake_dict.items()} if can_inverse else fake_dict
    results = {}
    for name, fake in fake_dict.items():
        print(f"\n  [{name}]")
        d_score = discriminative_score(real_data, fake, device=device)
        p_score = predictive_score(real_data, fake, device=device)
        s_facts = compute_stylized_facts(raw_fake[name], feature_idx, is_return_feature)
        corr_dist = _feature_corr_distance(raw_real, raw_fake[name])
        mmd_score = _mmd_rbf(raw_real, raw_fake[name])
        results[name] = {
            'discriminative_score': d_score,
            'predictive_score': p_score,
            'feature_corr_dist': corr_dist,
            'mmd_rbf': mmd_score,
            **s_facts,
        }
        print(f"    Discriminative Score : {d_score:.4f}  (ideal: 0.5)")
        print(f"    Predictive Score     : {p_score:.4f}  (lower is better)")
        print(f"    Feature Corr Dist    : {corr_dist:.4f}  (lower is better)")
        print(f"    MMD (RBF)            : {mmd_score:.4f}  (lower is better)")
        print(f"    Kurtosis             : {s_facts['kurtosis']:.4f}")
        print(f"    Skewness             : {s_facts['skewness']:.4f}")
        print(f"    Vol. Autocorrelation : {s_facts['vol_autocorr']:.4f}")
        print(f"    VaR 1%               : {s_facts['var_1']:.4f}")
        print(f"    ES 1%                : {s_facts['es_1']:.4f}")

    real_facts = compute_stylized_facts(raw_real, feature_idx, is_return_feature)
    print(f"\n  [Real Data Reference]")
    print(f"    Kurtosis             : {real_facts['kurtosis']:.4f}")
    print(f"    Skewness             : {real_facts['skewness']:.4f}")
    print(f"    Vol. Autocorrelation : {real_facts['vol_autocorr']:.4f}")
    print(f"    VaR 1%               : {real_facts['var_1']:.4f}")
    print(f"    ES 1%                : {real_facts['es_1']:.4f}")

    # ---- Visualizations ----
    print("\n--- Generating Plots ---")
    plot_samples_comparison(
        real_data, fake_dict, feature_names,
        os.path.join(result_dir, 'samples_comparison.png')
    )
    plot_distribution_comparison(
        real_data, fake_dict, feature_names,
        os.path.join(result_dir, 'distribution_comparison.png')
    )
    plot_autocorrelation(
        raw_real, raw_fake, feature_idx=feature_idx,
        save_path=os.path.join(result_dir, 'autocorrelation.png')
    )
    plot_pca_tsne(
        real_data, fake_dict,
        os.path.join(result_dir, 'pca_tsne.png')
    )
    plot_stylized_facts(
        raw_real, raw_fake,
        os.path.join(result_dir, 'stylized_facts.png'),
        feature_idx=feature_idx,
        is_return_feature=is_return_feature
    )

    if model is not None:
        plot_trend_seasonal_decomposition(
            model, real_data, device,
            os.path.join(result_dir, 'decomposition.png'),
            feature_names=feature_names
        )
        plot_conditional_forecasting(
            model, real_data, device,
            os.path.join(result_dir, 'conditional_forecasting.png'),
            obs_ratio=obs_ratio,
            feature_names=feature_names
        )
        forecast_bundle = _forecast_mc_bundle(
            model, real_data, device,
            obs_ratio=obs_ratio,
            norm_info=norm_info,
            max_samples=forecast_samples,
            mc_samples=forecast_mc_samples,
        )
        forecast = _forecast_metrics(
            model, real_data, device,
            obs_ratio=obs_ratio,
            feature_names=feature_names,
            norm_info=norm_info,
            max_samples=forecast_samples,
            mc_samples=forecast_mc_samples,
            interval=forecast_interval,
            bundle=forecast_bundle,
        )
        if forecast:
            results['Forecast'] = {k: float(v) if isinstance(v, (int, float)) else v
                                   for k, v in forecast.items()}
            print("\n  [Forecast Metrics]")
            print(f"    Obs Ratio           : {forecast['obs_ratio']:.2f}")
            print(f"    MAE (all feats)     : {forecast['forecast_mae']:.4f}")
            print(f"    RMSE (all feats)    : {forecast['forecast_rmse']:.4f}")
            print(f"    MAE ({forecast['forecast_feature']}) : {forecast['forecast_mae_feature']:.4f}")
            print(f"    RMSE ({forecast['forecast_feature']}) : {forecast['forecast_rmse_feature']:.4f}")
            if 'forecast_interval' in forecast:
                print(f"    PI Coverage         : {forecast['forecast_picp']:.4f}")
                print(f"    PI Width            : {forecast['forecast_pinaw']:.4f}")
                print(f"    PI Coverage ({forecast['forecast_feature']}) : "
                      f"{forecast['forecast_picp_feature']:.4f}")
                print(f"    PI Width ({forecast['forecast_feature']}) : "
                      f"{forecast['forecast_pinaw_feature']:.4f}")
        plot_forecast_intervals(
            forecast_bundle,
            feature_names=feature_names,
            save_path=os.path.join(result_dir, 'forecast_intervals.png'),
            interval=forecast_interval,
            n_show=4,
        )

    print("\n" + "=" * 60)
    print("Evaluation complete! All results saved to:", result_dir)
    print("=" * 60)

    return results
